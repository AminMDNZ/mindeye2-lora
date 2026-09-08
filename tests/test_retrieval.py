"""Tests for the retrieval fallback and the memory preflight."""
import numpy as np
import pytest

from mindeye_lora.capacity import estimate_memory, preflight


# --------------------------------------------------------------------------------------
# capacity (torch-free)
# --------------------------------------------------------------------------------------
def test_lora_is_far_cheaper_than_full_at_paper_scale():
    total = 2_100_000_000
    full = estimate_memory(total, total, "adamw")
    lora = estimate_memory(total, 70_000_000, "adamw")
    assert full.total_bytes > 3 * lora.total_bytes
    # the gap is optimiser state, not parameters — that is the whole point
    assert full.optimizer_bytes > 20 * lora.optimizer_bytes


def test_8bit_optimizer_cuts_moment_memory_fourfold():
    total = 2_100_000_000
    fp32 = estimate_memory(total, total, "adamw")
    int8 = estimate_memory(total, total, "adamw8bit")
    assert fp32.optimizer_bytes == pytest.approx(4 * int8.optimizer_bytes)
    assert int8.total_bytes < fp32.total_bytes


def test_preflight_passes_when_no_gpu_is_present():
    # gpu_capacity_bytes() returns None off-GPU, so preflight must not raise
    est = preflight("lora_r16", 2_100_000_000, 70_000_000, strict=True)
    assert est.total_bytes > 0


# --------------------------------------------------------------------------------------
# retrieval (needs torch)
# --------------------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from mindeye_lora.retrieval import retrieve_topk  # noqa: E402


def test_perfect_predictions_retrieve_rank_zero():
    torch.manual_seed(0)
    target = torch.randn(20, 4, 8)
    idx, sims, rank = retrieve_topk(target.clone(), target, k=3)
    assert (rank == 0).all()
    assert (idx[:, 0] == torch.arange(20)).all()


def test_noise_predictions_rank_poorly():
    torch.manual_seed(1)
    target = torch.randn(64, 4, 8)
    pred = torch.randn(64, 4, 8)
    _, _, rank = retrieve_topk(pred, target, k=3)
    assert rank.float().mean() > 5   # nowhere near the top of a 64-image pool


def test_topk_is_ordered_and_shaped():
    torch.manual_seed(2)
    target = torch.randn(30, 2, 6)
    pred = target + 0.3 * torch.randn_like(target)
    idx, sims, rank = retrieve_topk(pred, target, k=3)
    assert idx.shape == (30, 3) and sims.shape == (30, 3)
    assert (sims[:, 0] >= sims[:, 1]).all() and (sims[:, 1] >= sims[:, 2]).all()
    assert rank.shape == (30,)


def test_k_is_clamped_to_pool_size():
    target = torch.randn(2, 3, 4)
    idx, _, _ = retrieve_topk(target.clone(), target, k=10)
    assert idx.shape[1] == 2
