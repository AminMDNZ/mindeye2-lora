"""Assemble a MindEye2 model from upstream classes and load the pretrained weights.

Two things here are worth reading carefully.

**Architecture inference.** Rather than trusting hard-coded hyperparameters, we read the
shapes out of the checkpoint (`hidden_dim`, `n_blocks`, `clip_emb_dim`, `clip_seq_dim`,
whether the low-level branch exists, how many subjects the ridge covers). If upstream
ever republishes with different settings, this still does the right thing.

**Ridge handling.** The published `multisubject_subj01_*` checkpoint contains ridge
layers for subjects 2–8 only. Adapting to subject 1 means constructing a *fresh* ridge
layer sized to subject 1's voxel count. That layer has no pretrained weights, so it is
fully trained in every arm, including the LoRA and frozen arms. LoRA is a statement
about how far pretrained weights need to move; it says nothing about a randomly
initialised layer.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .upstream import Upstream, load_upstream
from .utils import log


@dataclass
class ArchSpec:
    hidden_dim: int
    n_blocks: int
    clip_emb_dim: int
    clip_seq_dim: int
    blurry_recon: bool
    pretrain_subjects: int
    pretrain_voxel_dims: list[int]

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def normalise_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for prefix in ("module.", "_orig_mod.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def load_checkpoint_state(path: str | Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "module", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return normalise_state_dict(ckpt[key])
    return normalise_state_dict(ckpt)


def infer_arch(sd: dict) -> ArchSpec:
    """Recover the architecture hyperparameters from tensor shapes."""
    ridge_keys = sorted(k for k in sd if k.startswith("ridge.") and k.endswith("weight"))
    voxel_dims = [sd[k].shape[1] for k in ridge_keys]
    hidden_dim = sd[ridge_keys[0]].shape[0] if ridge_keys else None

    # residual blocks: backbone.mlp.<i>....
    block_ids = set()
    for k in sd:
        if k.startswith("backbone.mlp."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                block_ids.add(int(parts[2]))
    n_blocks = (max(block_ids) + 1) if block_ids else 4

    # CLIP token geometry from the diffusion prior's width and the backbone output size
    clip_emb_dim = None
    for key in ("diffusion_prior.net.to_time_embeds.0.weight",
                "diffusion_prior.net.null_image_embed",
                "diffusion_prior.net.learned_query"):
        if key in sd:
            clip_emb_dim = sd[key].shape[-1]
            break
    if clip_emb_dim is None:
        for k, v in sd.items():
            if k.startswith("diffusion_prior.net.causal_transformer") and v.ndim == 2:
                clip_emb_dim = v.shape[-1]
                break
    clip_emb_dim = int(clip_emb_dim or 1664)

    out_key = next((k for k in ("backbone.backbone_linear.weight", "backbone.lin1.weight")
                    if k in sd), None)
    if out_key is not None:
        clip_seq_dim = int(sd[out_key].shape[0] // clip_emb_dim)
    else:
        clip_seq_dim = 256

    blurry = any(k.startswith("backbone.blin1") or "bupsampler" in k or "b_maps" in k for k in sd)
    if hidden_dim is None:
        hidden_dim = int(sd[out_key].shape[1]) if out_key is not None else 1024

    spec = ArchSpec(
        hidden_dim=int(hidden_dim),
        n_blocks=int(n_blocks),
        clip_emb_dim=clip_emb_dim,
        clip_seq_dim=clip_seq_dim,
        blurry_recon=bool(blurry),
        pretrain_subjects=len(voxel_dims),
        pretrain_voxel_dims=[int(v) for v in voxel_dims],
    )
    log.info("inferred architecture: %s", spec.as_dict())
    return spec


def _filtered_kwargs(cls, candidates: dict[str, Any]) -> dict[str, Any]:
    """Pass only the kwargs a class actually accepts (upstream signatures drift)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover
        return candidates
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return candidates
    allowed = set(sig.parameters) - {"self"}
    dropped = set(candidates) - allowed
    if dropped:
        log.debug("dropping unsupported kwargs for %s: %s", cls.__name__, sorted(dropped))
    return {k: v for k, v in candidates.items() if k in allowed}


class MindEyeModule(nn.Module):
    """Container matching upstream's `MindEyeModule` so state-dict keys line up."""

    def __init__(self):
        super().__init__()

    def forward(self, x):  # pragma: no cover - upstream calls submodules directly
        return x


def build_model(
    arch: ArchSpec,
    num_voxels_list: list[int],
    up: Upstream,
    use_prior: bool = True,
    blurry_recon: bool | None = None,
    prior_depth: int = 6,
    prior_dim_head: int = 52,
    prior_timesteps: int = 100,
) -> nn.Module:
    M = up.models
    blurry = arch.blurry_recon if blurry_recon is None else blurry_recon

    model = MindEyeModule()
    model.ridge = M.RidgeRegression(
        **_filtered_kwargs(
            M.RidgeRegression,
            dict(input_sizes=num_voxels_list, out_features=arch.hidden_dim, seq_len=1),
        )
    )
    model.backbone = M.BrainNetwork(
        **_filtered_kwargs(
            M.BrainNetwork,
            dict(
                h=arch.hidden_dim,
                in_dim=arch.hidden_dim,
                seq_len=1,
                n_blocks=arch.n_blocks,
                clip_size=arch.clip_emb_dim,
                out_dim=arch.clip_emb_dim * arch.clip_seq_dim,
                blurry_recon=blurry,
                clip_scale=1.0,
            ),
        )
    )
    if use_prior:
        net = M.PriorNetwork(
            **_filtered_kwargs(
                M.PriorNetwork,
                dict(
                    dim=arch.clip_emb_dim,
                    depth=prior_depth,
                    dim_head=prior_dim_head,
                    heads=max(1, arch.clip_emb_dim // prior_dim_head),
                    causal=False,
                    num_tokens=arch.clip_seq_dim,
                    learned_query_mode="pos_emb",
                ),
            )
        )
        model.diffusion_prior = M.BrainDiffusionPrior(
            **_filtered_kwargs(
                M.BrainDiffusionPrior,
                dict(
                    net=net,
                    image_embed_dim=arch.clip_emb_dim,
                    condition_on_text_encodings=False,
                    timesteps=prior_timesteps,
                    cond_drop_prob=0.2,
                    image_embed_scale=None,
                ),
            )
        )
    return model


@dataclass
class LoadReport:
    loaded: int
    skipped_ridge: int
    missing: list[str]
    unexpected: list[str]
    shape_mismatch: list[str]

    @property
    def ok(self) -> bool:
        return not self.shape_mismatch and not self.unexpected

    def describe(self) -> str:
        parts = [f"loaded {self.loaded} tensors", f"skipped {self.skipped_ridge} pretrained ridge tensors"]
        if self.missing:
            parts.append(f"{len(self.missing)} freshly initialised (e.g. {self.missing[:3]})")
        if self.unexpected:
            parts.append(f"UNEXPECTED {len(self.unexpected)} (e.g. {self.unexpected[:3]})")
        if self.shape_mismatch:
            parts.append(f"SHAPE MISMATCH {self.shape_mismatch[:3]}")
        return "; ".join(parts)


def load_pretrained_weights(
    model: nn.Module,
    sd: dict,
    skip_prefixes: tuple[str, ...] = ("ridge.",),
    strict: bool = True,
) -> LoadReport:
    """Load shared-subject weights, deliberately skipping subject-specific layers."""
    own = model.state_dict()
    to_load, skipped, mismatch, unexpected = {}, 0, [], []
    for k, v in sd.items():
        if k.startswith(skip_prefixes):
            skipped += 1
            continue
        if k not in own:
            unexpected.append(k)
            continue
        if tuple(own[k].shape) != tuple(v.shape):
            mismatch.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(own[k].shape)}")
            continue
        to_load[k] = v

    missing = [k for k in own if k not in to_load and not k.startswith(skip_prefixes)]
    model.load_state_dict(to_load, strict=False)
    report = LoadReport(len(to_load), skipped, missing, unexpected, mismatch)
    log.info("checkpoint load: %s", report.describe())

    if strict and not report.ok:
        raise RuntimeError(
            "Pretrained weights did not map cleanly onto the constructed model.\n"
            f"{report.describe()}\n"
            "This usually means the upstream architecture changed. Pin a revision with "
            "--upstream_ref, or inspect with `mindeye-lora verify`."
        )
    if strict and missing:
        # Missing keys that are *not* ridge are a red flag: they would silently train
        # from scratch and destroy the comparison.
        non_ridge = [m for m in missing if not m.startswith("ridge.")]
        if non_ridge:
            raise RuntimeError(
                f"{len(non_ridge)} non-ridge parameters had no pretrained value, e.g. "
                f"{non_ridge[:5]}. Refusing to run: the 'pretrained' baseline would be fake."
            )
    return report


def create_mindeye(
    ckpt_path: str | Path,
    num_voxels: int,
    upstream_dir: str | Path,
    upstream_ref: str = "main",
    use_prior: bool = True,
    blurry_recon: bool | None = False,
    device: str | torch.device = "cuda",
    strict: bool = True,
) -> tuple[nn.Module, ArchSpec, LoadReport]:
    """One-call model construction: infer architecture, build, load, move to device."""
    up = load_upstream(Path(upstream_dir), ref=upstream_ref)
    sd = load_checkpoint_state(ckpt_path)
    arch = infer_arch(sd)
    model = build_model(
        arch, num_voxels_list=[num_voxels], up=up, use_prior=use_prior, blurry_recon=blurry_recon
    )
    report = load_pretrained_weights(model, sd, strict=strict)
    model.to(device)
    return model, arch, report


def module_tree(model: nn.Module, max_lines: int = 400) -> str:
    """Human-readable listing of every Linear layer — useful for tuning LoRA regexes."""
    lines = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            lines.append(f"{name:70s} {tuple(m.weight.shape)}  bias={m.bias is not None}")
    return "\n".join(lines[:max_lines])
