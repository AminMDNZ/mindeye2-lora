"""Precompute OpenCLIP ViT-bigG/14 token embeddings for the stimulus images.

MindEye2 regresses brain activity onto 256x1664 OpenCLIP ViT-bigG/14 *token* embeddings
(patch tokens after `ln_post`, before the projection head). Recomputing those every epoch
would mean keeping a 2.5 GB vision tower resident next to the model — the fastest way to
OOM a T4. Since the stimulus set is fixed and the encoder is frozen, we compute the
embeddings once and memory-map them thereafter.

Cost for a 1-session experiment: ~1,750 unique images -> ~1.5 GB in fp16, computed in a
couple of minutes, then reused forever.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import human_bytes, log

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ARCH = "ViT-bigG-14"
PRETRAINED = "laion2b_s39b_b160k"


class CLIPTokenEmbedder(torch.nn.Module):
    """Frozen bigG image tower returning [B, 256, 1664] token embeddings.

    Two backends:
      * ``open_clip``  – always available, no Stability dependencies (default).
      * ``sgm``        – the exact class upstream uses, if `generative_models` is on the
                         path. Kept so the paper-faithful decoder path can share it.
    """

    def __init__(self, backend: str = "open_clip", device: str = "cuda", dtype=torch.float16):
        super().__init__()
        self.backend = backend
        self.device = device
        self.dtype = dtype
        if backend == "sgm":
            from generative_models.sgm.modules.encoders.modules import (  # type: ignore
                FrozenOpenCLIPImageEmbedder,
            )

            self.model = FrozenOpenCLIPImageEmbedder(
                arch=ARCH, version=PRETRAINED, output_tokens=True, only_tokens=True
            ).to(device, dtype).eval()
        elif backend == "open_clip":
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
            self.visual = model.visual
            self.visual.output_tokens = True
            self.visual = self.visual.to(device, dtype).eval()
            del model
        else:
            raise ValueError(f"unknown CLIP backend: {backend}")

        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        """images_01: [B, 3, 224, 224] float in [0, 1]."""
        x = images_01.to(self.device, torch.float32)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
            )
        if self.backend == "sgm":
            return self.model(x.to(self.dtype))
        x = ((x - self.mean.to(x.device)) / self.std.to(x.device)).to(self.dtype)
        out = self.visual(x)
        tokens = out[1] if isinstance(out, (tuple, list)) else out
        return tokens


@torch.no_grad()
def precompute_embeddings(
    images_path: str | Path,
    out_path: str | Path,
    backend: str = "open_clip",
    batch_size: int = 16,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Embed every image in the cached subset -> fp16 memmap [N, 256, 1664]."""
    out_path = Path(out_path)
    images = np.load(images_path, mmap_mode="r")
    n = len(images)
    if out_path.exists() and not force:
        existing = np.load(out_path, mmap_mode="r")
        if len(existing) == n:
            log.info("CLIP embeddings already cached: %s %s", out_path.name, existing.shape)
            return out_path
        log.warning("Cached embeddings have %d rows but images have %d — recomputing.",
                    len(existing), n)

    embedder = CLIPTokenEmbedder(backend=backend, device=device)
    probe = embedder(torch.from_numpy(np.asarray(images[:1], dtype=np.float32)))
    seq, dim = probe.shape[-2], probe.shape[-1]
    log.info("embedding %d images -> [%d, %d, %d] fp16 (%s)", n, n, seq, dim,
             human_bytes(n * seq * dim * 2))

    tmp = out_path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, seq, dim))
    for start in range(0, n, batch_size):
        chunk = np.asarray(images[start : start + batch_size], dtype=np.float32)
        emb = embedder(torch.from_numpy(chunk))
        out[start : start + batch_size] = emb.float().cpu().numpy().astype(np.float16)
        if (start // batch_size) % 10 == 0:
            log.info("  %d/%d", min(start + batch_size, n), n)
    out.flush()
    del out, embedder
    torch.cuda.empty_cache()
    tmp.replace(out_path)
    log.info("wrote %s (%s)", out_path.name, human_bytes(out_path.stat().st_size))
    return out_path


def embedding_stats(path: str | Path) -> dict:
    e = np.load(path, mmap_mode="r")
    sample = np.asarray(e[: min(64, len(e))], dtype=np.float32)
    return {
        "shape": list(e.shape),
        "dtype": str(e.dtype),
        "mean": float(sample.mean()),
        "std": float(sample.std()),
        "norm_per_token": float(np.linalg.norm(sample, axis=-1).mean()),
    }
