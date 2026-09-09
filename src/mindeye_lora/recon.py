"""Turn predicted CLIP embeddings into images with MindEye2's SDXL unCLIP decoder.

This stage is **optional and heavy**. The decoder checkpoint is 18 GB, and it is
completely frozen — it is the same weights for every arm, so it contributes no
between-arm variance. Every statistical conclusion in this project is therefore
available with `--decoder none`; reconstructions exist to answer the "show me the
pictures" half of the question and to compute the eight published image metrics.

Storage strategy on Colab: download to *local* disk (`/content`), optionally slim to
fp16 once and keep the slim copy on Drive so later sessions skip the 18 GB pull.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .assets import UNCLIP_CKPT, hf_download
from .utils import human_bytes, log, progress, robust_load, robust_save

GENERATIVE_MODELS_URL = "https://github.com/Stability-AI/generative-models.git"


# --------------------------------------------------------------------------------------
# decoder setup
# --------------------------------------------------------------------------------------
def ensure_generative_models(upstream_dir: Path) -> Path:
    """Make `generative_models` importable, preferring the copy inside MindEyeV2."""
    candidates = [
        Path(upstream_dir) / "MindEyeV2" / "src" / "generative_models",
        Path(upstream_dir) / "generative-models",
    ]
    for c in candidates:
        if c.exists():
            parent = str(c.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            if c.name == "generative-models" and str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    dest = Path(upstream_dir) / "generative-models"
    log.info("Cloning Stability generative-models -> %s", dest)
    subprocess.run(["git", "clone", "--depth", "1", GENERATIVE_MODELS_URL, str(dest)], check=True)
    sys.path.insert(0, str(dest))
    return dest


def find_unclip_config(upstream_dir: Path) -> Path:
    """Locate the `unclip6.yaml` config MindEye2 ships with."""
    for pattern in ("**/configs/unclip6.yaml", "**/unclip6.yaml"):
        for base in (Path(upstream_dir) / "MindEyeV2", Path(upstream_dir)):
            hits = sorted(base.glob(pattern))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        "unclip6.yaml not found. It ships inside MindEyeV2/src/generative_models/configs/. "
        "If the upstream layout changed, pass --unclip_config explicitly."
    )


def download_unclip_checkpoint(target_dir: Path, slim_path: Path | None = None) -> Path:
    """Fetch the 18 GB unCLIP checkpoint, reusing a slimmed fp16 copy when present."""
    if slim_path and Path(slim_path).exists():
        log.info("using slimmed decoder checkpoint %s (%s)", slim_path,
                 human_bytes(Path(slim_path).stat().st_size))
        return Path(slim_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return hf_download(UNCLIP_CKPT, target_dir)


def slim_unclip_checkpoint(src: Path, dst: Path) -> Path:
    """Cast the decoder weights to fp16 (roughly halves 18 GB) for cheap re-loading."""
    dst = Path(dst)
    if dst.exists():
        return dst
    log.info("slimming decoder checkpoint to fp16 ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    half = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in sd.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": half}, dst)
    log.info("  -> %s (%s)", dst.name, human_bytes(dst.stat().st_size))
    return dst


def build_diffusion_engine(
    ckpt_path: Path, config_path: Path, device: str = "cuda", num_steps: int = 38
):
    from omegaconf import OmegaConf
    from generative_models.sgm.models.diffusion import DiffusionEngine  # type: ignore

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    params = cfg["model"]["params"]
    params["sampler_config"]["params"]["num_steps"] = num_steps
    params["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"

    engine = DiffusionEngine(
        network_config=params["network_config"],
        denoiser_config=params["denoiser_config"],
        first_stage_config=params["first_stage_config"],
        conditioner_config=params["conditioner_config"],
        sampler_config=params["sampler_config"],
        scale_factor=params["scale_factor"],
        disable_first_stage_autocast=params["disable_first_stage_autocast"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = engine.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    if missing:
        log.warning("decoder: %d missing keys (e.g. %s)", len(missing), list(missing)[:3])
    engine = engine.to(device).eval().requires_grad_(False)
    del ckpt
    torch.cuda.empty_cache()
    return engine


@torch.no_grad()
def _vector_suffix(engine, device: str, size: int = 768):
    """The size/crop conditioning vector SDXL expects, computed once."""
    batch = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * size,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
    }
    out = engine.conditioner(batch)
    return out["vector"].to(device)


@torch.no_grad()
def unclip_reconstruct(
    engine,
    clip_embeddings: torch.Tensor,
    vector_suffix: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 4,
    upstream_utils=None,
    checkpoint_path=None,
) -> torch.Tensor:
    """Decode [N, 256, 1664] CLIP token embeddings into [N, 3, H, W] images in [0, 1].

    Diffusion decoding runs at 3-5 s/image, so a 200-image arm is a 15-minute stage.
    Results are checkpointed every few batches and resumed, because re-decoding from
    scratch after a disconnect is the most expensive restart in the pipeline.
    """
    if upstream_utils is not None and hasattr(upstream_utils, "unclip_recon"):
        outs, start_at = [], 0
        if checkpoint_path is not None:
            def _validate(part):
                if "images" not in part or "done" not in part:
                    raise ValueError("missing keys")
                if int(part["done"]) != int(part["images"].shape[0]):
                    raise ValueError("count disagrees with tensor length")

            part = robust_load(checkpoint_path, validate=_validate)
            if part is not None:
                outs = [part["images"]]
                start_at = int(part["done"])
                log.info("resuming decode from image %d/%d", start_at, len(clip_embeddings))

        bar = progress(total=len(clip_embeddings), desc="decode", unit="img")
        if start_at:
            bar.update(start_at)
        for start in range(start_at, len(clip_embeddings), batch_size):
            chunk = clip_embeddings[start : start + batch_size].to(device)
            samples = upstream_utils.unclip_recon(chunk, engine, vector_suffix, num_samples=1)
            outs.append(samples.float().cpu())
            done = min(start + batch_size, len(clip_embeddings))
            bar.update(done - start)
            if checkpoint_path is not None and (start // batch_size) % 3 == 0:
                try:
                    robust_save({"images": torch.cat(outs), "done": done}, checkpoint_path)
                except IOError as exc:
                    log.warning("decode checkpoint failed (%s); continuing", exc)
        bar.close()
        return torch.cat(outs).clamp(0, 1)

    raise RuntimeError(
        "Upstream `utils.unclip_recon` is unavailable, and reimplementing the sgm sampling "
        "loop here would risk diverging from the published decoder. Either make upstream "
        "utils importable, or run with --decoder none (all statistics still work)."
    )


# --------------------------------------------------------------------------------------
# stage entry point
# --------------------------------------------------------------------------------------
def reconstruct_for_run(
    predictions_path: Path,   # directory holding the streamed PredictionStore
    out_path: Path,
    ws,
    n_images: int,
    decoder: str = "sdxl_unclip",
    device: str = "cuda",
    num_steps: int = 38,
    batch_size: int = 4,
    unclip_dir: Path | None = None,
    slim_decoder: bool = True,
) -> Path | None:
    """Decode the first `n_images` predicted embeddings of one run."""
    if decoder == "none":
        log.info("decoder disabled — skipping reconstruction.")
        return None

    from .evaluate import PredictionStore

    preds = PredictionStore.load(predictions_path)
    # Slice first, then materialise: the store is a memmap and the full array is ~1.7 GB.
    source = preds.get("prior", preds["clip_voxels"])
    emb = torch.from_numpy(np.ascontiguousarray(source[:n_images])).float()
    rows = torch.from_numpy(np.asarray(preds["rows"][:n_images]))

    from .upstream import load_upstream

    up = load_upstream(ws["upstream"])
    ensure_generative_models(ws["upstream"])
    config_path = find_unclip_config(ws["upstream"])

    unclip_dir = Path(unclip_dir or "/content/unclip_cache")
    slim = ws["assets"] / "unclip6_fp16.ckpt" if slim_decoder else None
    ckpt = download_unclip_checkpoint(unclip_dir, slim_path=slim)
    if slim_decoder and slim is not None and ckpt != slim:
        ckpt = slim_unclip_checkpoint(ckpt, slim)

    engine = build_diffusion_engine(ckpt, config_path, device=device, num_steps=num_steps)
    suffix = _vector_suffix(engine, device)
    out_path = Path(out_path)
    partial = out_path.with_name(out_path.stem + ".partial.pt")
    images = unclip_reconstruct(engine, emb, suffix, device=device, batch_size=batch_size,
                                upstream_utils=up.utils, checkpoint_path=partial)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    robust_save({"recons": images, "rows": rows}, out_path)
    partial.unlink(missing_ok=True)
    Path(str(partial) + ".bak").unlink(missing_ok=True)
    log.info("saved %d reconstructions -> %s", len(images), out_path.name)

    del engine
    torch.cuda.empty_cache()
    return out_path


def load_ground_truth(images_path: Path, rows: np.ndarray, size: int | None = None) -> torch.Tensor:
    imgs = np.load(images_path, mmap_mode="r")
    gt = torch.from_numpy(np.asarray(imgs[np.asarray(rows, dtype=np.int64)], dtype=np.float32))
    if size and gt.shape[-1] != size:
        gt = torch.nn.functional.interpolate(gt, size=(size, size), mode="bilinear",
                                             align_corners=False, antialias=True)
    return gt.clamp(0, 1)
