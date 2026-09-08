"""Retrieval-based qualitative output — the fallback when the SDXL decoder is unavailable.

**These images are retrieved, not generated.** For each test scan we take the model's
predicted CLIP embedding and find the nearest test-set image in the same space. The
result is a real photograph the model was scored against, not something it synthesised.
Everything downstream labels it that way, and the output is deliberately written to
`retrieval.pt` rather than `reconstructions.pt` so `evaluate.py` cannot pick it up and
compute PixCorr or SSIM on it — those metrics are meaningless here (a correct retrieval
scores perfectly, a wrong one scores like an unrelated photo, and neither tells you
anything about reconstruction quality).

Why top-k rather than top-1: when an arm retrieves correctly, its top-1 image is
*pixel-identical* to the ground truth, which looks like a flawless reconstruction and is
the most misleading thing this module could produce. Showing ranks 2 and 3 makes the
model's actual confusions visible, and the figure annotates where the true image ranked.

This measures the same quantity as the `retrieval_percentile` column already in the
report — it just renders it instead of tabulating it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import log


def retrieve_topk(
    pred: torch.Tensor, target: torch.Tensor, k: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest neighbours of each prediction among all target embeddings.

    Returns (indices [N, k], similarities [N, k], rank_of_truth [N]) where a rank of 0
    means the correct image was retrieved first.
    """
    p = F.normalize(pred.flatten(1).float(), dim=-1)
    t = F.normalize(target.flatten(1).float(), dim=-1)
    sim = p @ t.T
    k = min(k, sim.shape[1])
    top = sim.topk(k, dim=-1)
    truth = sim.diagonal().unsqueeze(1)
    rank = (sim > truth).sum(1)  # how many candidates beat the correct one
    return top.indices, top.values, rank


def build_retrieval_output(
    predictions_path: str | Path,
    out_path: str | Path,
    k: int = 3,
    n_items: int | None = None,
) -> Path:
    """Produce a retrieval result file from a run's cached predictions."""
    preds = torch.load(str(predictions_path), map_location="cpu", weights_only=False)
    pred_emb = preds.get("prior", preds["clip_voxels"])
    target = preds["target"]
    rows = preds["rows"]

    indices, sims, rank = retrieve_topk(pred_emb, target, k=k)
    pool_size = int(target.shape[0])

    if n_items is not None:
        indices, sims, rank = indices[:n_items], sims[:n_items], rank[:n_items]

    payload = {
        "kind": "retrieval",
        "indices": indices,                      # positions within the test set
        "similarities": sims,
        "rank_of_truth": rank,                   # 0 == correct image retrieved first
        "image_rows": rows[indices],             # rows into the cached image array
        "rows": rows[: len(indices)],            # ground-truth rows, same order
        "pool_size": pool_size,
        "k": int(indices.shape[1]),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    top1 = float((rank == 0).float().mean())
    log.info(
        "retrieval: %d items, top-1 = %.1f%% out of a %d-image pool -> %s",
        len(indices), 100 * top1, pool_size, out_path.name,
    )
    return out_path


def load_retrieved_images(images_path: str | Path, image_rows: torch.Tensor) -> np.ndarray:
    """Gather the retrieved photographs -> [N, k, 3, H, W] float32 in [0, 1]."""
    imgs = np.load(str(images_path), mmap_mode="r")
    flat = np.asarray(image_rows, dtype=np.int64).reshape(-1)
    gathered = np.asarray(imgs[flat], dtype=np.float32)
    n, k = image_rows.shape
    return gathered.reshape(n, k, *gathered.shape[1:]).clip(0, 1)
