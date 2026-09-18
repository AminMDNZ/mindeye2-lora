"""A small, dependency-free LoRA implementation targeted at MindEye2.

Why not `peft`? MindEye2's modules are hand-written `nn.Module`s (a residual MLP
backbone plus a DALLE-2 style diffusion prior), not a HuggingFace `PreTrainedModel`.
Wrapping them ourselves is ~200 lines, keeps the injection rules explicit, and lets us
report exactly which layers were adapted — which matters when the whole point of the
experiment is to attribute performance to specific parameter groups.

Key design points
-----------------
* `B` is zero-initialised, so an injected model is *numerically identical* to the
  pretrained one at step 0. This is what makes the LoRA arm and the full fine-tune arm
  share an identical starting point.
* Targeting is by regex on the fully-qualified module path, so nothing is hard-coded to
  upstream attribute names.
* `merge_()` folds the adapter into the base weight, so inference costs exactly the same
  as the original model — a real advantage over the frozen-plus-side-network baselines.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .utils import log


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: Sequence[str] = field(default_factory=lambda: [r"^backbone\.", r"^diffusion_prior\."])
    exclude_modules: Sequence[str] = field(default_factory=lambda: [r"^ridge\."])
    variant: str = "lora"            # "lora" | "dora"
    use_rslora: bool = False         # scale by alpha/sqrt(r) instead of alpha/r
    init_scale: float = 1.0

    @property
    def scaling(self) -> float:
        if self.r == 0:
            return 0.0
        denom = math.sqrt(self.r) if self.use_rslora else self.r
        return self.alpha / denom


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = cfg.r
        self.scaling = cfg.scaling
        self.variant = cfg.variant
        self.merged = False

        in_f, out_f = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(cfg.r, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, cfg.r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        with torch.no_grad():
            self.lora_A.mul_(cfg.init_scale)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        if self.variant == "dora":
            # weight-decomposed LoRA: learn the column magnitudes separately
            with torch.no_grad():
                norm = base.weight.norm(p=2, dim=1, keepdim=True)
            self.lora_magnitude = nn.Parameter(norm.clone())
        else:
            self.register_parameter("lora_magnitude", None)

    # -- maths ---------------------------------------------------------------------
    def delta_w(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def effective_weight(self) -> torch.Tensor:
        w = self.base.weight + self.delta_w()
        if self.variant == "dora":
            w = self.lora_magnitude * w / (w.norm(p=2, dim=1, keepdim=True) + 1e-8)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return self.base(x)
        if self.variant == "dora":
            return nn.functional.linear(x, self.effective_weight().to(x.dtype), self.base.bias)
        out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T
        return out + lora_out * self.scaling

    # -- merging -------------------------------------------------------------------
    @torch.no_grad()
    def merge_(self) -> None:
        if self.merged or self.r == 0:
            return
        self.base.weight.data.copy_(self.effective_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge_(self) -> None:
        if not self.merged:
            return
        if self.variant == "dora":
            raise RuntimeError("DoRA merges are not invertible; reload the base checkpoint.")
        self.base.weight.data -= self.delta_w()
        self.merged = False

    def extra_repr(self) -> str:  # pragma: no cover
        return (f"in={self.base.in_features}, out={self.base.out_features}, r={self.r}, "
                f"scaling={self.scaling:.3f}, variant={self.variant}")


# --------------------------------------------------------------------------------------
# injection
# --------------------------------------------------------------------------------------
def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


def find_target_linears(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _matches(name, cfg.target_modules):
            continue
        if cfg.exclude_modules and _matches(name, cfg.exclude_modules):
            continue
        names.append(name)
    return names


def _set_module(root: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> dict:
    """Replace every targeted `nn.Linear` with a `LoRALinear`. Returns a summary dict."""
    if cfg.r <= 0:
        log.info("LoRA rank is 0 — no adapters injected.")
        return {"injected": [], "adapter_params": 0}

    targets = find_target_linears(model, cfg)
    if not targets:
        raise ValueError(
            "LoRA targeting matched zero Linear layers. Check `target_modules` regexes "
            f"against the module tree: {[n for n, _ in model.named_modules()][:20]} ..."
        )
    injected, adapter_params = [], 0
    for name in targets:
        base = model.get_submodule(name)
        wrapped = LoRALinear(base, cfg)
        _set_module(model, name, wrapped)
        injected.append(name)
        adapter_params += wrapped.lora_A.numel() + wrapped.lora_B.numel()
        if wrapped.lora_magnitude is not None:
            adapter_params += wrapped.lora_magnitude.numel()

    log.info("LoRA: wrapped %d Linear layers (r=%d, alpha=%g) -> %s adapter params",
             len(injected), cfg.r, cfg.alpha, f"{adapter_params:,}")
    return {"injected": injected, "adapter_params": adapter_params, "config": cfg}


def iter_lora_modules(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield name, m


def merge_lora(model: nn.Module) -> nn.Module:
    for _, m in iter_lora_modules(model):
        m.merge_()
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    for _, m in iter_lora_modules(model):
        m.unmerge_()
    return model


# --------------------------------------------------------------------------------------
# freezing policy
# --------------------------------------------------------------------------------------
def set_trainable(
    model: nn.Module,
    mode: str,
    always_trainable: Sequence[str] = (r"^ridge\.",),
    train_bias: str = "none",       # none | all | lora_only
    train_norms: bool = False,
) -> dict:
    """Configure `requires_grad` for one experimental arm.

    mode:
      "full"   – everything trainable (the paper's fine-tuning recipe)
      "lora"   – LoRA adapters + `always_trainable` only
      "frozen" – `always_trainable` only (shared model completely frozen)
      "bitfit" – biases (+ LayerNorms if `train_norms`) + `always_trainable`
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def unfreeze_named(pred) -> None:
        for name, param in model.named_parameters():
            if pred(name, param):
                param.requires_grad_(True)

    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "lora":
        unfreeze_named(lambda n, p: ".lora_" in n)
    elif mode == "frozen":
        pass
    elif mode == "bitfit":
        unfreeze_named(lambda n, p: n.endswith(".bias"))
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")

    # always-trainable groups (the subject-specific ridge layer is randomly initialised
    # for a held-out subject, so it must be learned in every arm — there is no
    # pretrained weight for LoRA to be "low-rank relative to").
    unfreeze_named(lambda n, p: _matches(n, always_trainable))

    if train_bias == "all":
        unfreeze_named(lambda n, p: n.endswith(".bias"))
    elif train_bias == "lora_only":
        unfreeze_named(lambda n, p: n.endswith(".bias") and ".base." in n)

    if train_norms:
        norm_paths = {
            name for name, m in model.named_modules()
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d))
        }
        unfreeze_named(lambda n, p: any(n.startswith(np_ + ".") for np_ in norm_paths))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "mode": mode,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / max(total, 1),
    }
    log.info("arm=%s | trainable %s / %s (%.3f%%)", mode, f"{trainable:,}", f"{total:,}",
             100 * summary["trainable_fraction"])
    return summary


# --------------------------------------------------------------------------------------
# adapter checkpoints
# --------------------------------------------------------------------------------------
def adapter_state_dict(model: nn.Module, extra: Sequence[str] = (r"^ridge\.",)) -> dict:
    """Only the tensors this arm actually learned — a few MB instead of a gigabyte."""
    sd = {}
    for name, param in model.named_parameters():
        if ".lora_" in name or _matches(name, extra) or param.requires_grad:
            sd[name] = param.detach().cpu()
    # LayerNorm/BatchNorm buffers that training may have touched
    for name, buf in model.named_buffers():
        if _matches(name, extra):
            sd[name] = buf.detach().cpu()
    return sd


def load_adapter_state_dict(model: nn.Module, sd: dict, strict: bool = False) -> dict:
    missing, unexpected = [], []
    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    for k, v in sd.items():
        if k not in own:
            unexpected.append(k)
            continue
        with torch.no_grad():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    for k in own:
        if (".lora_" in k) and k not in sd:
            missing.append(k)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Adapter load mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    return {"missing": missing, "unexpected": unexpected}


def adapter_size_bytes(sd: dict) -> int:
    return sum(v.numel() * v.element_size() for v in sd.values())
