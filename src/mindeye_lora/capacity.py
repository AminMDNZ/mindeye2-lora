"""Predict whether an arm will fit on the current GPU, before wasting an hour finding out.

The MindEye2 backbone is lopsided: at `hidden_dim=4096` the single `backbone_linear`
layer mapping 4096 -> 256x1664 is ~1.75B of the ~2.1B total parameters. That makes full
fine-tuning **optimiser-state bound**, not activation bound — gradient checkpointing and
smaller batches barely help, because the memory is sitting in AdamW's two moment buffers.

    full fine-tune, fp32 AdamW:  params + grads + 2 moments  ≈ 4 x param_bytes
    LoRA:                        params + tiny grads/moments ≈ 1 x param_bytes

This is precisely the cost LoRA removes, and it is why the paper used an 8xA100-80GB node
with DeepSpeed ZeRO-2 (which shards the optimiser state) rather than a single card.

These are estimates, not measurements. The trainer records real `peak_memory_bytes` per
arm; treat this module as a pre-flight check, not as a result.
"""
from __future__ import annotations

from dataclasses import dataclass

from .utils import human_bytes, log

BYTES_PER_FP32 = 4
BYTES_PER_FP16 = 2
# Allowance for autocast copies, workspace buffers and allocator fragmentation, on top
# of the explicitly modelled activation cost below.
OVERHEAD_FRACTION = 0.15
MIN_OVERHEAD_BYTES = 1 * 1024**3

# Empirical correction on the analytic activation figure below. The first version of
# this module predicted 4.9 GB for the frozen arm at batch_size=24 and the run then OOMed
# on a 14.6 GB T4, so the analytic term was low by roughly 5x. The gap comes from things
# that are tedious to model exactly: dalle2-pytorch's attention keeps several
# intermediates alive, `p_losses` runs an extra self-conditioning forward pass, and
# autocast holds fp32 masters alongside fp16 activations.
#
# Calibrated against that single observation, so treat it as a guard rail rather than a
# measurement -- the trainer records real peak memory, and that is what the report uses.
# Deliberately biased toward over-estimating: refusing a run that would have fit costs a
# config edit, while a run that OOMs 40 minutes in costs the session.
ACTIVATION_SAFETY = 5.0


@dataclass
class MemoryEstimate:
    total_params: int
    trainable_params: int
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return (self.param_bytes + self.grad_bytes + self.optimizer_bytes
                + self.activation_bytes + self.overhead_bytes)

    def describe(self) -> str:
        return (
            f"params {human_bytes(self.param_bytes)} + grads {human_bytes(self.grad_bytes)} "
            f"+ optimiser {human_bytes(self.optimizer_bytes)} "
            f"+ activations {human_bytes(self.activation_bytes)} "
            f"+ overhead {human_bytes(self.overhead_bytes)} "
            f"= {human_bytes(self.total_bytes)}"
        )


def estimate_activations(
    batch_size: int,
    clip_seq_dim: int = 256,
    clip_emb_dim: int = 1664,
    prior_depth: int = 6,
    dim_head: int = 52,
) -> int:
    """Activation memory, dominated by the diffusion prior's attention matrices.

    This is the term that actually decides whether a batch fits, and the reason a
    parameter-only estimate is useless here. The prior runs `prior_depth` attention
    layers over `clip_seq_dim + 1` tokens with `clip_emb_dim // dim_head` heads, and
    each layer materialises a [batch, heads, tokens, tokens] similarity matrix -- kept
    alive for the backward pass, and transiently doubled by the softmax.

    At batch 24 with 32 heads and 257 tokens that is several GB on its own, which is
    exactly the T4 OOM this function exists to predict.
    """
    heads = max(1, clip_emb_dim // dim_head)
    tokens = clip_seq_dim + 1
    attn_matrix = batch_size * heads * tokens * tokens * BYTES_PER_FP16
    attn_total = attn_matrix * prior_depth * 2      # sim + softmax output, retained
    # token streams through the transformer (qkv projections, feed-forward inner dim)
    token_stream = batch_size * tokens * clip_emb_dim * BYTES_PER_FP16
    stream_total = token_stream * prior_depth * 8
    # the backbone's flattened 256x1664 output and its CLIP target
    backbone_out = batch_size * clip_seq_dim * clip_emb_dim * BYTES_PER_FP16 * 4
    return int((attn_total + stream_total + backbone_out) * ACTIVATION_SAFETY)


def estimate_memory(
    total_params: int,
    trainable_params: int,
    optimizer: str = "adamw",
    batch_size: int = 24,
    clip_seq_dim: int = 256,
    clip_emb_dim: int = 1664,
    prior_depth: int = 6,
) -> MemoryEstimate:
    """Steady-state training memory for one arm."""
    param_bytes = total_params * BYTES_PER_FP32
    grad_bytes = trainable_params * BYTES_PER_FP32
    per_param_state = {
        "adamw": 2 * BYTES_PER_FP32,   # exp_avg + exp_avg_sq, fp32
        "adamw8bit": 2 * 1,            # both moments quantised to int8
        "sgd": 1 * BYTES_PER_FP32,     # momentum only
    }.get(optimizer, 2 * BYTES_PER_FP32)
    optimizer_bytes = trainable_params * per_param_state
    activation_bytes = estimate_activations(
        batch_size, clip_seq_dim, clip_emb_dim, prior_depth
    )
    base = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
    overhead = max(int(base * OVERHEAD_FRACTION), MIN_OVERHEAD_BYTES)
    return MemoryEstimate(total_params, trainable_params, param_bytes, grad_bytes,
                          optimizer_bytes, activation_bytes, overhead)


def gpu_capacity_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # pragma: no cover
        return None


def suggest_batch_size(
    total_params: int,
    trainable_params: int,
    optimizer: str = "adamw",
    capacity: int | None = None,
    headroom: float = 0.85,
    effective_batch: int = 24,
) -> tuple[int, int]:
    """Largest batch that fits, plus the grad_accum needed to preserve `effective_batch`."""
    capacity = capacity or gpu_capacity_bytes() or 0
    if not capacity:
        return effective_batch, 1
    budget = capacity * headroom
    # Only divisors: batch_size x grad_accum must land exactly on the effective batch,
    # or the arms are no longer trained on a comparable schedule.
    candidates = sorted(
        (b for b in range(1, effective_batch + 1) if effective_batch % b == 0), reverse=True
    )
    for bs in candidates:
        if estimate_memory(total_params, trainable_params, optimizer, bs).total_bytes <= budget:
            return bs, effective_batch // bs
    return 1, effective_batch


def preflight(
    arm_name: str,
    total_params: int,
    trainable_params: int,
    optimizer: str = "adamw",
    headroom: float = 0.90,
    strict: bool = True,
    batch_size: int = 24,
) -> MemoryEstimate:
    """Warn, or refuse, when an arm cannot plausibly fit on the attached GPU."""
    est = estimate_memory(total_params, trainable_params, optimizer, batch_size)
    capacity = gpu_capacity_bytes()
    log.info("memory estimate [%s]: %s", arm_name, est.describe())
    if capacity is None:
        return est

    budget = capacity * headroom
    if est.total_bytes <= budget:
        log.info("  fits: %s of %s usable", human_bytes(est.total_bytes), human_bytes(budget))
        return est

    bs, accum = suggest_batch_size(total_params, trainable_params, optimizer, capacity)
    advice = [
        f"Estimated {human_bytes(est.total_bytes)} needed at batch_size={batch_size}, "
        f"but only {human_bytes(budget)} usable on this GPU ({human_bytes(capacity)} total).",
        f"Activations alone are {human_bytes(est.activation_bytes)} -- try "
        f"`batch_size: {bs}` with `grad_accum: {accum}` in the config, which keeps the "
        f"effective batch at {bs * accum}.",
    ]
    if optimizer == "adamw" and trainable_params > 0.5 * total_params:
        saving = est.optimizer_bytes - estimate_memory(
            total_params, trainable_params, "adamw8bit"
        ).optimizer_bytes
        advice.append(
            f"Set `optimizer: adamw8bit` in the config to save about "
            f"{human_bytes(saving)} of optimiser state (requires bitsandbytes). "
            "Apply it to EVERY arm, or you introduce an optimiser confound."
        )
    advice.append(
        "Or switch to `pretrain: multisubject_1024`, which is roughly 4x smaller and is "
        "upstream's own reduced-memory configuration."
    )
    if est.optimizer_bytes > est.activation_bytes:
        advice.append(
            "Optimiser state dominates here, so a smaller batch helps less than "
            "switching optimiser or model size."
        )
    message = f"Arm '{arm_name}' probably will not fit.\n  " + "\n  ".join(advice)

    if strict:
        raise RuntimeError(message + "\n  Pass --ignore-memory-check to try anyway.")
    log.warning(message)
    return est
