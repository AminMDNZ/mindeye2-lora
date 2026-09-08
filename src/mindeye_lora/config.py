"""Typed configuration for the experiment.

One `ExperimentConfig` describes the shared setup (subject, data budget, optimiser,
seeds); each `ArmConfig` describes one adaptation strategy. Everything an arm can change
is listed explicitly so the comparison stays controlled: the arms differ in *which
parameters are trainable*, not in data, schedule, or seed.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from .lora import LoRAConfig


@dataclass
class ArmConfig:
    name: str
    mode: str = "lora"                    # full | lora | frozen | bitfit
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_variant: str = "lora"            # lora | dora
    lora_targets: Sequence[str] = field(
        default_factory=lambda: [r"^backbone\.", r"^diffusion_prior\."]
    )
    lora_excludes: Sequence[str] = field(default_factory=lambda: [r"^ridge\."])
    train_bias: str = "none"              # none | all | lora_only
    train_norms: bool = False
    lr: float | None = None               # None -> inherit; PEFT arms usually want more
    label: str | None = None              # pretty name for plots

    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            r=self.lora_r,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_modules=list(self.lora_targets),
            exclude_modules=list(self.lora_excludes),
            variant=self.lora_variant,
        )

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass
class ExperimentConfig:
    # data ---------------------------------------------------------------------------
    subj: int = 1
    num_sessions: int = 1                 # 1 session ~= 1 hour of scanning
    pretrain: str = "multisubject_1024"
    average_test_repeats: bool = True
    zscore_voxels: bool = True

    # model --------------------------------------------------------------------------
    use_prior: bool = True
    blurry_recon: bool = False            # low-level branch off: halves memory on a T4
    upstream_ref: str = "main"
    clip_backend: str = "open_clip"

    # optimisation -------------------------------------------------------------------
    epochs: int = 150
    batch_size: int = 24
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    mixup_pct: float = 0.33
    prior_scale: float = 30.0
    clip_scale: float = 1.0
    max_grad_norm: float = 1.0
    precision: str = "fp16"               # fp16 | bf16 | fp32
    optimizer: str = "adamw"              # adamw | adamw8bit | sgd
                                          # NOTE: applies to every arm. Changing it for
                                          # only one arm would confound the comparison.
    num_workers: int = 2
    warmup_pct: float = 0.02

    # experiment ---------------------------------------------------------------------
    seeds: Sequence[int] = field(default_factory=lambda: [0, 1, 2])
    eval_every: int = 10
    save_every_min: float = 10.0
    time_budget_min: float | None = None
    arms: Sequence[ArmConfig] = field(default_factory=list)

    # evaluation ---------------------------------------------------------------------
    recon_decoder: str = "none"           # none | sdxl_unclip
    recon_n_images: int = 100             # how many test images to reconstruct per arm
    recon_seed: int = 0
    qualitative_n: int = 8
    qualitative_arms: int = 3             # image grids stay readable with <= 3 rows/arm
    retrieval_fallback: bool = True       # nearest-neighbour panel when no decoder
    retrieval_k: int = 3                  # show top-3, so confusions are visible

    def arm(self, name: str) -> ArmConfig:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"No arm named {name!r}. Known: {[a.name for a in self.arms]}")

    def run_name(self, arm: str, seed: int) -> str:
        return f"subj{self.subj:02d}_{self.num_sessions}sess_{arm}_seed{seed}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arms"] = [asdict(a) for a in self.arms]
        return d


DEFAULT_ARMS = [
    ArmConfig(name="frozen", mode="frozen", label="Frozen shared model (ridge only)"),
    ArmConfig(name="bitfit", mode="bitfit", train_norms=True, lr=1e-3, label="BitFit + norms"),
    ArmConfig(name="lora_r4", mode="lora", lora_r=4, lora_alpha=8, lr=1e-3, label="LoRA r=4"),
    ArmConfig(name="lora_r16", mode="lora", lora_r=16, lora_alpha=32, lr=1e-3, label="LoRA r=16"),
    ArmConfig(name="lora_r64", mode="lora", lora_r=64, lora_alpha=128, lr=1e-3, label="LoRA r=64"),
    ArmConfig(name="full", mode="full", label="Full fine-tune (paper recipe)"),
]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    data: dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        data = _merge(data, {k: v for k, v in overrides.items() if v is not None})

    arms_raw = data.pop("arms", None)
    cfg = ExperimentConfig(**{k: v for k, v in data.items() if k in ExperimentConfig.__annotations__})
    cfg.arms = [ArmConfig(**a) for a in arms_raw] if arms_raw else list(DEFAULT_ARMS)
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return path
