"""Evaluation, always at the level of individual test images.

Everything returns a per-image vector rather than a single average, because the
statistical comparison between arms is *paired*: arm A and arm B are scored on the same
1,000 shared NSD images, so the right test operates on the per-image differences. Group
means throw away exactly the information that makes the comparison powerful.

Two tiers of metric:

* **CLIP-space** (no image decoder needed) — cosine similarity to the target embedding,
  forward/backward retrieval, and per-image two-way identification. These measure the
  brain-to-CLIP mapping, which is the only part LoRA touches, so they are the primary
  outcome.
* **Pixel/feature space** (needs reconstructions) — the eight metrics reported in the
  MindEye papers. The decoder is frozen and shared across arms, so differences here are
  attributable to the encoder.
"""
from __future__ import annotations

import contextlib
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import eta_string, log, progress, read_json, write_json

from .metric_meta import (  # re-exported for convenience
    EMBEDDING_METRICS,
    HIGHER_IS_BETTER,
    IMAGE_METRICS,
)


# --------------------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------------------
def _call_prior_sampler(prior, backbone_out, timesteps: int = 20, cond_scale: float = 1.0):
    """Call the diffusion prior's sampler, tolerating upstream signature drift."""
    if hasattr(prior, "p_sample_loop"):
        fn = prior.p_sample_loop
        kwargs = {"text_cond": dict(text_embed=backbone_out), "cond_scale": cond_scale}
        sig = inspect.signature(fn)
        if "timesteps" in sig.parameters:
            kwargs["timesteps"] = timesteps
        return fn(backbone_out.shape, **kwargs)
    if hasattr(prior, "sample"):
        return prior.sample(text_embed=backbone_out, timesteps=timesteps, cond_scale=cond_scale)
    raise AttributeError("Diffusion prior exposes neither p_sample_loop nor sample.")


class PredictionStore:
    """Disk-backed store for one arm's test-set predictions.

    Holding predictions in RAM does not scale: 1,000 test images x 256 tokens x 1664
    dims is ~1.7 GB per tensor in float32, and an earlier version accumulated four of
    them in Python lists and then `torch.cat`-ed the lot, briefly doubling it. That is
    ~13 GB on a 12 GB Colab VM, which appears as a silent session kill rather than a
    traceback.

    So each tensor is a float16 memmap written batch by batch. Memory stays flat
    whatever the test-set size, the file on disk *is* the resume checkpoint, and
    downstream stages map it read-only instead of loading it.
    """

    FIELDS = ("prior", "clip_voxels", "target")

    def __init__(self, directory, n: int, seq: int, dim: int, use_prior: bool):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n, self.seq, self.dim = n, seq, dim
        # `clip_voxels` is only ever read as the fallback when there is no diffusion
        # prior, so storing both wastes ~850 MB of disk and page cache per arm.
        self.fields = ["prior", "target"] if use_prior else ["clip_voxels", "target"]
        self.arrays: dict = {}
        self.meta_path = self.dir / "meta.json"

    def _path(self, field: str) -> Path:
        return self.dir / f"{field}.npy"

    def open(self, resume: bool = True) -> int:
        """Create or reattach the memmaps. Returns the number of completed batches."""
        meta = read_json(self.meta_path, default=None) if resume else None
        compatible = bool(
            meta
            and meta.get("shape") == [self.n, self.seq, self.dim]
            and sorted(meta.get("fields", [])) == sorted(self.fields)
            and all(self._path(f).exists() for f in self.fields)
            and (self.dir / "rows.npy").exists()
        )
        for field in self.fields:
            self.arrays[field] = (
                np.lib.format.open_memmap(self._path(field), mode="r+")
                if compatible
                else np.lib.format.open_memmap(
                    self._path(field), mode="w+", dtype=np.float16,
                    shape=(self.n, self.seq, self.dim))
            )
        rows_path = self.dir / "rows.npy"
        self.arrays["rows"] = (
            np.lib.format.open_memmap(rows_path, mode="r+")
            if compatible
            else np.lib.format.open_memmap(rows_path, mode="w+", dtype=np.int64,
                                           shape=(self.n,))
        )
        return int(meta.get("batches_done", 0)) if compatible else 0

    def write(self, start: int, **tensors) -> None:
        for field, value in tensors.items():
            arr = self.arrays.get(field)
            if arr is None:
                continue
            data = value.detach().cpu().numpy()
            arr[start : start + len(data)] = data.astype(arr.dtype)

    def commit(self, batches_done: int, samples_done: int) -> None:
        for arr in self.arrays.values():
            arr.flush()
        write_json(self.meta_path, {
            "shape": [self.n, self.seq, self.dim], "fields": self.fields,
            "batches_done": batches_done, "samples_done": samples_done,
        })

    def close(self) -> None:
        for arr in self.arrays.values():
            with contextlib.suppress(Exception):
                arr.flush()
        self.arrays.clear()

    @classmethod
    def load(cls, directory) -> dict:
        """Map an existing store read-only. Nothing is copied into RAM."""
        directory = Path(directory)
        meta = read_json(directory / "meta.json")
        if meta is None:
            raise FileNotFoundError(f"No prediction store at {directory}")
        out = {}
        for field in list(meta["fields"]) + ["rows"]:
            path = directory / f"{field}.npy"
            if path.exists():
                out[field] = np.load(path, mmap_mode="r")
        out["_meta"] = meta
        return out

    @classmethod
    def is_complete(cls, directory) -> bool:
        meta = read_json(Path(directory) / "meta.json", default=None)
        if not meta:
            return False
        return int(meta.get("samples_done", 0)) >= int(meta["shape"][0])


@torch.no_grad()
def predict(
    model,
    loader,
    store_dir,
    device: str = "cuda",
    precision: str = "fp16",
    use_prior: bool = True,
    prior_timesteps: int = 20,
    desc: str = "predict",
    checkpoint_every: int = 5,
    resume: bool = True,
) -> "PredictionStore":
    """Run the encoder over the test set, streaming results straight to disk."""
    import time

    model.eval()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    n_total = len(loader.dataset)
    probe = next(iter(loader))[1]
    seq, dim = int(probe.shape[-2]), int(probe.shape[-1])
    del probe

    store = PredictionStore(store_dir, n_total, seq, dim, use_prior=use_prior)
    start_batch = store.open(resume=resume)
    if start_batch:
        log.info("resuming %s from batch %d", desc, start_batch)

    total = len(loader)
    t0 = time.time()
    bar = progress(total=total, desc=desc, unit="batch")
    if start_batch:
        bar.update(start_batch)

    written = 0
    for i, (voxel, clip_target, _img, row) in enumerate(loader):
        if i < start_batch:
            written += len(row)
            continue
        voxel = voxel.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
            latent = model.ridge(voxel, 0)
            out = model.backbone(latent)
        backbone_out = out[0] if isinstance(out, (tuple, list)) else out
        cv = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out

        payload = {"target": clip_target.float()}
        if use_prior and hasattr(model, "diffusion_prior"):
            with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
                payload["prior"] = _call_prior_sampler(
                    model.diffusion_prior, backbone_out, timesteps=prior_timesteps
                ).float()
        else:
            payload["clip_voxels"] = cv.float()

        store.write(written, **payload)
        store.arrays["rows"][written : written + len(row)] = row.numpy()
        written += len(row)
        del payload, backbone_out, cv, out, latent, voxel

        bar.update(1)
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(eta_string(i + 1 - start_batch, total - start_batch,
                                           time.time() - t0))
        if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
            store.commit(i + 1, written)

    bar.close()
    store.commit(total, written)
    return store


# --------------------------------------------------------------------------------------
# CLIP-space metrics
# --------------------------------------------------------------------------------------
def _as_tensor(x) -> torch.Tensor:
    """Accept a torch tensor or a numpy memmap. Memmaps arrive from PredictionStore."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.ascontiguousarray(x))


def _flat_norm(x, chunk: int = 256) -> torch.Tensor:
    """L2-normalised flattened embeddings, materialised in chunks.

    A 1,000 x 256 x 1664 memmap is 1.7 GB in float32, so converting it wholesale
    reintroduces exactly the memory blow-up the store exists to avoid. Normalising in
    chunks keeps the peak to one chunk at a time.
    """
    n = len(x)
    out = None
    for start in range(0, n, chunk):
        block = _as_tensor(x[start : start + chunk]).flatten(1).float()
        block = F.normalize(block, dim=-1)
        if out is None:
            out = torch.empty((n, block.shape[1]), dtype=torch.float32)
        out[start : start + len(block)] = block
    return out


def two_way_identification(pred, target, chunk: int = 256) -> np.ndarray:
    """Per-sample probability that the true target scores higher than a random distractor."""
    p, t = _flat_norm(pred), _flat_norm(target)
    n = p.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = p[start : start + chunk]
        sim = block @ t.T                                    # [b, n]
        idx = torch.arange(start, min(start + chunk, n))
        true_sim = sim[torch.arange(len(idx)), idx].unsqueeze(1)
        wins = (true_sim > sim).float().sum(1) - 0.0         # excludes the tie with itself
        out[start : start + chunk] = (wins / (n - 1)).numpy()
    return out


def retrieval_percentile(pred, target) -> np.ndarray:
    """1.0 means the correct image was ranked first among all test images."""
    p, t = _flat_norm(pred), _flat_norm(target)
    sim = p @ t.T
    n = sim.shape[0]
    ranks = (sim > sim.diagonal().unsqueeze(1)).sum(1).numpy()
    return 1.0 - ranks / max(1, n - 1)


def cosine_per_sample(pred, target) -> np.ndarray:
    return (_flat_norm(pred) * _flat_norm(target)).sum(-1).numpy()


def embedding_metrics(pred, target) -> dict[str, np.ndarray]:
    return {
        "cosine": cosine_per_sample(pred, target),
        "two_way_clip": two_way_identification(pred, target),
        "retrieval_percentile": retrieval_percentile(pred, target),
    }


def retrieval_summary(pred, target, pool: int = 300, seed: int = 0) -> dict:
    """Top-1 retrieval within random pools of `pool` candidates (the paper's protocol)."""
    rng = np.random.default_rng(seed)
    p, t = _flat_norm(pred), _flat_norm(target)
    n = p.shape[0]
    pool = min(pool, n)
    fwd, bwd = [], []
    order = rng.permutation(n)
    for start in range(0, n - pool + 1, pool):
        idx = torch.from_numpy(order[start : start + pool])
        sim = p[idx] @ t[idx].T
        labels = torch.arange(len(idx))
        fwd.append((sim.argmax(1) == labels).float().mean().item())
        bwd.append((sim.T.argmax(1) == labels).float().mean().item())
    return {"fwd_top1": float(np.mean(fwd)), "bwd_top1": float(np.mean(bwd)), "pool": pool}


# --------------------------------------------------------------------------------------
# image-space metrics
# --------------------------------------------------------------------------------------
class ImageMetricSuite:
    """The eight metrics reported by MindEye1/2, computed per image."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._cache: dict = {}

    def _feature_model(self, name: str):
        if name in self._cache:
            return self._cache[name]
        import torchvision.models as tvm

        if name == "alexnet":
            m = tvm.alexnet(weights=tvm.AlexNet_Weights.IMAGENET1K_V1)
        elif name == "inception":
            m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
            m.fc = torch.nn.Identity()
        elif name == "efficientnet":
            m = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.IMAGENET1K_V1)
            m.classifier = torch.nn.Identity()
        elif name == "swav":
            m = torch.hub.load("facebookresearch/swav:main", "resnet50")
            m.fc = torch.nn.Identity()
        else:
            raise ValueError(name)
        m = m.to(self.device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self._cache[name] = m
        return m

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def _alexnet_layer(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        m = self._feature_model("alexnet")
        h = self._norm(x)
        # layer 2 -> after the 2nd conv stage; layer 5 -> after the 5th
        stops = {2: 5, 5: 12}
        for i, mod in enumerate(m.features):
            h = mod(h)
            if i == stops[layer]:
                break
        return h.flatten(1)

    @torch.no_grad()
    def _generic_features(self, x: torch.Tensor, name: str, size: int = 224) -> torch.Tensor:
        m = self._feature_model(name)
        h = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        return m(self._norm(h)).flatten(1)

    @torch.no_grad()
    def _clip_features(self, x: torch.Tensor) -> torch.Tensor:
        from .embeddings import CLIPTokenEmbedder

        if "clip" not in self._cache:
            self._cache["clip"] = CLIPTokenEmbedder(device=self.device)
        emb = self._cache["clip"](x)
        return emb.flatten(1).float()

    @staticmethod
    def _pairwise_two_way(rec: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
        """Per-sample two-way accuracy using correlation distance in feature space."""
        rec = rec - rec.mean(1, keepdim=True)
        gt = gt - gt.mean(1, keepdim=True)
        rec = rec / (rec.norm(dim=1, keepdim=True) + 1e-8)
        gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
        sim = rec @ gt.T
        n = sim.shape[0]
        true = sim.diagonal().unsqueeze(1)
        wins = (true > sim).float().sum(1)
        return (wins / max(1, n - 1)).cpu().numpy()

    @staticmethod
    def _pixcorr(rec: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
        a = rec.flatten(1)
        b = gt.flatten(1)
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        num = (a * b).sum(1)
        den = a.norm(dim=1) * b.norm(dim=1) + 1e-8
        return (num / den).cpu().numpy()

    @staticmethod
    def _ssim(rec: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
        from skimage.color import rgb2gray
        from skimage.metrics import structural_similarity

        r = rec.permute(0, 2, 3, 1).cpu().numpy()
        g = gt.permute(0, 2, 3, 1).cpu().numpy()
        out = np.empty(len(r))
        for i in range(len(r)):
            out[i] = structural_similarity(
                rgb2gray(g[i]), rgb2gray(r[i]), data_range=1.0, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False,
            )
        return out

    @torch.no_grad()
    def __call__(self, recons: torch.Tensor, targets: torch.Tensor, batch: int = 32) -> dict:
        """recons/targets: [N, 3, H, W] float in [0, 1]."""
        recons = recons.clamp(0, 1)
        targets = targets.clamp(0, 1)
        out: dict[str, np.ndarray] = {
            "pixcorr": self._pixcorr(recons, targets),
            "ssim": self._ssim(recons, targets),
        }

        def feats(fn, label: str = "") -> tuple[torch.Tensor, torch.Tensor]:
            rs, gs = [], []
            bar = progress(total=len(recons), desc=f"metric {label}", unit="img",
                           leave=False)
            for s in range(0, len(recons), batch):
                rs.append(fn(recons[s : s + batch].to(self.device)).cpu())
                gs.append(fn(targets[s : s + batch].to(self.device)).cpu())
                bar.update(min(batch, len(recons) - s))
            bar.close()
            return torch.cat(rs), torch.cat(gs)

        for key, fn in [
            ("alexnet2", lambda x: self._alexnet_layer(x, 2)),
            ("alexnet5", lambda x: self._alexnet_layer(x, 5)),
            ("inception", lambda x: self._generic_features(x, "inception", 299)),
            ("clip", self._clip_features),
        ]:
            try:
                r, g = feats(fn, key)
                out[key] = self._pairwise_two_way(r, g)
            except Exception as exc:  # a missing model shouldn't kill the whole report
                log.warning("metric %s failed: %s", key, exc)

        for key, name in [("effnet", "efficientnet"), ("swav", "swav")]:
            try:
                r, g = feats(lambda x, n=name: self._generic_features(x, n), key)
                rn = F.normalize(r, dim=1)
                gn = F.normalize(g, dim=1)
                out[key] = (1 - (rn * gn).sum(1)).numpy()   # correlation distance
            except Exception as exc:
                log.warning("metric %s failed: %s", key, exc)
        return out


def save_per_sample(path: str | Path, metrics: dict[str, np.ndarray], rows: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=rows, **{k: np.asarray(v) for k, v in metrics.items()})
    return path


def load_per_sample(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
