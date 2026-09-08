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
# Rough allowance for activations, autocast copies, the CLIP target batch and allocator
# fragmentation. Empirical fudge factor, intentionally generous.
OVERHEAD_FRACTION = 0.20
MIN_OVERHEAD_BYTES = 2 * 1024**3


@dataclass
class MemoryEstimate:
    total_params: int
    trainable_params: int
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes + self.overhead_bytes

    def describe(self) -> str:
        return (
            f"params {human_bytes(self.param_bytes)} + grads {human_bytes(self.grad_bytes)} "
            f"+ optimiser {human_bytes(self.optimizer_bytes)} "
            f"+ overhead {human_bytes(self.overhead_bytes)} "
            f"= {human_bytes(self.total_bytes)}"
        )


def estimate_memory(
    total_params: int,
    trainable_params: int,
    optimizer: str = "adamw",
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
    base = param_bytes + grad_bytes + optimizer_bytes
    overhead = max(int(base * OVERHEAD_FRACTION), MIN_OVERHEAD_BYTES)
    return MemoryEstimate(total_params, trainable_params, param_bytes, grad_bytes,
                          optimizer_bytes, overhead)


def gpu_capacity_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # pragma: no cover
        return None


def preflight(
    arm_name: str,
    total_params: int,
    trainable_params: int,
    optimizer: str = "adamw",
    headroom: float = 0.90,
    strict: bool = True,
) -> MemoryEstimate:
    """Warn, or refuse, when an arm cannot plausibly fit on the attached GPU."""
    est = estimate_memory(total_params, trainable_params, optimizer)
    capacity = gpu_capacity_bytes()
    log.info("memory estimate [%s]: %s", arm_name, est.describe())
    if capacity is None:
        return est

    budget = capacity * headroom
    if est.total_bytes <= budget:
        log.info("  fits: %s of %s usable", human_bytes(est.total_bytes), human_bytes(budget))
        return est

    advice = [
        f"Estimated {human_bytes(est.total_bytes)} needed but only "
        f"{human_bytes(budget)} usable on this GPU ({human_bytes(capacity)} total).",
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
    advice.append("Smaller batches will not help much: this is optimiser state, not activations.")
    message = f"Arm '{arm_name}' probably will not fit.\n  " + "\n  ".join(advice)

    if strict:
        raise RuntimeError(message + "\n  Pass --ignore-memory-check to try anyway.")
    log.warning(message)
    return est
