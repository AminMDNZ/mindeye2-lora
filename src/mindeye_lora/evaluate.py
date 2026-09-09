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

import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import eta_string, log, progress, robust_load, robust_save

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


@torch.no_grad()
def predict(
    model,
    loader,
    device: str = "cuda",
    precision: str = "fp16",
    use_prior: bool = True,
    prior_timesteps: int = 20,
    desc: str = "predict",
    checkpoint_path=None,
    checkpoint_every: int = 5,
) -> dict[str, torch.Tensor]:
    """Run the encoder over the test set. Returns CPU float32 tensors.

    Sampling 1,000 images through the diffusion prior takes ~12 minutes per arm, which
    is long enough that a Colab disconnect used to throw the whole arm away. Partial
    results are now written every `checkpoint_every` batches and resumed automatically.
    """
    import time

    model.eval()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    backbones, clip_voxels, priors, targets, rows = [], [], [], [], []

    start_batch = 0
    if checkpoint_path is not None:
        def _validate(part):
            required = ("backbone", "clip_voxels", "target", "rows", "batches_done")
            missing = [k for k in required if k not in part]
            if missing:
                raise ValueError(f"missing keys {missing}")
            n = int(part["rows"].shape[0])
            for k in ("backbone", "clip_voxels", "target"):
                if part[k].shape[0] != n:
                    raise ValueError(f"{k} disagrees with rows on length")
            if "prior" in part and part["prior"].shape[0] != n:
                raise ValueError("prior disagrees with rows on length")

        part = robust_load(checkpoint_path, validate=_validate)
        if part is not None:
            backbones = [part["backbone"]]
            clip_voxels = [part["clip_voxels"]]
            targets = [part["target"]]
            rows = [part["rows"]]
            if "prior" in part:
                priors = [part["prior"]]
            start_batch = int(part["batches_done"])
            log.info("resuming %s from batch %d (%d samples recovered)",
                     desc, start_batch, int(part["rows"].shape[0]))
            del part

    total = len(loader)
    t0 = time.time()
    bar = progress(total=total, desc=desc, unit="batch")
    if start_batch:
        bar.update(start_batch)

    for i, (voxel, clip_target, _img, row) in enumerate(loader):
        if i < start_batch:          # already covered by the checkpoint
            continue
        voxel = voxel.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
            latent = model.ridge(voxel, 0)
            out = model.backbone(latent)
        backbone_out = out[0] if isinstance(out, (tuple, list)) else out
        cv = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
        backbones.append(backbone_out.float().cpu())
        clip_voxels.append(cv.float().cpu())
        targets.append(clip_target.float())
        rows.append(row)
        if use_prior and hasattr(model, "diffusion_prior"):
            with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
                pr = _call_prior_sampler(model.diffusion_prior, backbone_out,
                                         timesteps=prior_timesteps)
            priors.append(pr.float().cpu())

        bar.update(1)
        done = i + 1
        if hasattr(bar, "set_postfix"):
            bar.set_postfix_str(eta_string(done - start_batch, total - start_batch,
                                           time.time() - t0))

        if checkpoint_path is not None and done % checkpoint_every == 0 and done < total:
            partial = {
                "backbone": torch.cat(backbones), "clip_voxels": torch.cat(clip_voxels),
                "target": torch.cat(targets), "rows": torch.cat(rows),
                "batches_done": done,
            }
            if priors:
                partial["prior"] = torch.cat(priors)
            try:
                robust_save(partial, checkpoint_path)
            except IOError as exc:
                # A failed checkpoint must never kill a run that is otherwise fine.
                log.warning("could not write %s (%s); continuing without a resume point",
                            Path(checkpoint_path).name, exc)

    bar.close()
    result = {
        "backbone": torch.cat(backbones),
        "clip_voxels": torch.cat(clip_voxels),
        "target": torch.cat(targets),
        "rows": torch.cat(rows),
    }
    if priors:
        result["prior"] = torch.cat(priors)
    return result


# --------------------------------------------------------------------------------------
# CLIP-space metrics
# --------------------------------------------------------------------------------------
def _flat_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.flatten(1).float(), dim=-1)


def two_way_identification(pred: torch.Tensor, target: torch.Tensor, chunk: int = 256) -> np.ndarray:
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


def retrieval_percentile(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """1.0 means the correct image was ranked first among all test images."""
    p, t = _flat_norm(pred), _flat_norm(target)
    sim = p @ t.T
    n = sim.shape[0]
    ranks = (sim > sim.diagonal().unsqueeze(1)).sum(1).numpy()
    return 1.0 - ranks / max(1, n - 1)


def cosine_per_sample(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (_flat_norm(pred) * _flat_norm(target)).sum(-1).numpy()


def embedding_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "cosine": cosine_per_sample(pred, target),
        "two_way_clip": two_way_identification(pred, target),
        "retrieval_percentile": retrieval_percentile(pred, target),
    }


def retrieval_summary(pred: torch.Tensor, target: torch.Tensor, pool: int = 300, seed: int = 0) -> dict:
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
