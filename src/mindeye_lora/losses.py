"""Contrastive objectives used by MindEye2.

Training follows the published recipe: BiMixCo (mixup + bidirectional InfoNCE) for the
first `mixup_pct` of training, then SoftCLIP (soft targets from the CLIP-CLIP similarity
matrix) with an annealed temperature. If upstream's `utils.py` imports cleanly we defer
to it so the arms are trained with byte-identical objectives to the reference model;
these local versions are the fallback.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mixco(
    voxels: torch.Tensor, beta: float = 0.15, s_thresh: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix a random subset of samples with a permutation of the batch."""
    voxels = voxels.clone()
    n = voxels.shape[0]
    perm = torch.randperm(n, device=voxels.device)
    shuffled = voxels[perm]
    betas = torch.distributions.Beta(beta, beta).sample([n]).to(voxels.device)
    select = (torch.rand(n, device=voxels.device) <= s_thresh)
    shape = [-1] + [1] * (voxels.ndim - 1)
    voxels[select] = (
        voxels[select] * betas[select].reshape(*shape)
        + shuffled[select] * (1 - betas[select]).reshape(*shape)
    )
    betas[~select] = 1.0
    return voxels, perm, betas, select


def mixco_nce(
    preds: torch.Tensor,
    targs: torch.Tensor,
    temp: float = 0.006,
    perm: torch.Tensor | None = None,
    betas: torch.Tensor | None = None,
    select: torch.Tensor | None = None,
    bidirectional: bool = True,
) -> torch.Tensor:
    logits = (preds @ targs.T) / temp
    n = preds.shape[0]
    if perm is None or betas is None:
        target = torch.arange(n, device=preds.device)
        loss = F.cross_entropy(logits, target)
        if bidirectional:
            loss = (loss + F.cross_entropy(logits.T, target)) / 2
        return loss

    probs = torch.diag(betas)
    probs[torch.arange(n, device=preds.device), perm] = 1 - betas
    loss = -(logits.log_softmax(-1) * probs).sum(-1).mean()
    if bidirectional:
        loss2 = -(logits.T.log_softmax(-1) * probs.T).sum(-1).mean()
        loss = (loss + loss2) / 2
    return loss


def soft_clip_loss(preds: torch.Tensor, targs: torch.Tensor, temp: float = 0.125) -> torch.Tensor:
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) / 2


def resolve_losses(upstream_utils=None) -> dict:
    """Prefer upstream implementations; fall back to the local ones above."""
    out = {"mixco": mixco, "mixco_nce": mixco_nce, "soft_clip_loss": soft_clip_loss,
           "source": "local"}
    if upstream_utils is None:
        return out
    have = all(hasattr(upstream_utils, n) for n in ("mixco", "mixco_nce", "soft_clip_loss"))
    if have:
        out.update(
            mixco=upstream_utils.mixco,
            mixco_nce=upstream_utils.mixco_nce,
            soft_clip_loss=upstream_utils.soft_clip_loss,
            source="upstream",
        )
    return out


@torch.no_grad()
def topk_retrieval_accuracy(preds: torch.Tensor, targs: torch.Tensor, k: int = 1) -> float:
    """Top-k accuracy of matching predicted embeddings to their true image embeddings."""
    sim = preds @ targs.T
    labels = torch.arange(sim.shape[0], device=sim.device)
    topk = sim.topk(min(k, sim.shape[1]), dim=-1).indices
    return (topk == labels[:, None]).any(-1).float().mean().item()
