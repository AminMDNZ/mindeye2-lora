"""Fine-tune MindEye2 on a held-out subject under one adaptation strategy.

Every arm shares the same data order, schedule, batch size, seed, and loss. The *only*
difference is which parameters carry gradients, which is what makes the downstream
statistics interpretable.

Colab survivability
-------------------
Training state (weights or adapter, optimiser, scheduler, RNG, epoch/step) is written to
Drive on a wall-clock interval and at every epoch boundary. If the runtime dies, rerun
the same command: it resumes from the last checkpoint. `time_budget_min` stops cleanly
before a session is likely to be reclaimed so you never lose a partial epoch.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ArmConfig, ExperimentConfig
from .data import MindEyeDataset, build_subsets, make_loaders, normalise_voxels
from .losses import resolve_losses, topk_retrieval_accuracy
from .lora import (
    adapter_size_bytes,
    adapter_state_dict,
    apply_lora,
    load_adapter_state_dict,
    set_trainable,
)
from .model_factory import create_mindeye
from .upstream import load_upstream
from .utils import (
    CSVLogger,
    TimeBudget,
    human_bytes,
    log,
    parameter_report,
    read_json,
    seed_everything,
    write_json,
)


@dataclass
class TrainResult:
    run_name: str
    arm: str
    seed: int
    epochs_completed: int
    trainable_params: int
    total_params: int
    checkpoint_bytes: int
    peak_memory_bytes: int
    train_seconds: float
    seconds_per_epoch: float
    final_metrics: dict
    completed: bool


def _autocast_dtype(precision: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def build_arm_model(cfg: ExperimentConfig, arm: ArmConfig, num_voxels: int, ws, device: str):
    """Construct the pretrained model and configure it for one arm."""
    from .assets import load_asset_meta

    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    ckpt = Path(meta["files"]["pretrain_ckpt"])
    model, arch, report = create_mindeye(
        ckpt_path=ckpt,
        num_voxels=num_voxels,
        upstream_dir=ws["upstream"],
        upstream_ref=cfg.upstream_ref,
        use_prior=cfg.use_prior,
        blurry_recon=cfg.blurry_recon,
        device=device,
        strict=True,
    )
    lora_info = {}
    if arm.mode == "lora":
        lora_info = apply_lora(model, arm.lora_config())
        model.to(device)
    freeze_info = set_trainable(
        model,
        mode=arm.mode,
        train_bias=arm.train_bias,
        train_norms=arm.train_norms,
    )
    info = {
        "arch": arch.as_dict(),
        "load_report": report.describe(),
        "freeze": freeze_info,
        "lora": {k: v for k, v in lora_info.items() if k != "config"},
        "param_groups": parameter_report(model, group_depth=1),
    }
    return model, info


def build_optimizer(model, cfg: ExperimentConfig, arm: ArmConfig, steps_per_epoch: int):
    lr = arm.lr if arm.lr is not None else cfg.lr
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or name.endswith(".bias")) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = getattr(cfg, "optimizer", "adamw")
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer: adamw8bit needs bitsandbytes (`pip install bitsandbytes`). "
                "It cuts AdamW's moment buffers ~4x, which is what makes a paper-scale "
                "full fine-tune fit on a single 40GB A100."
            ) from exc
        opt = bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    elif name == "sgd":
        opt = torch.optim.SGD(groups, lr=lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    total_steps = max(1, int(math.ceil(steps_per_epoch * cfg.epochs / cfg.grad_accum)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_pct,
        anneal_strategy="cos",
        final_div_factor=1000,
        last_epoch=-1,
    )
    return opt, sched, lr, total_steps


@torch.no_grad()
def evaluate_embedding_space(model, loader, device: str, precision: str, max_batches: int | None = None) -> dict:
    """Cheap in-training metric: retrieval and cosine similarity in CLIP space."""
    model.eval()
    dtype = _autocast_dtype(precision)
    sims, fwd, bwd, n = [], [], [], 0
    for i, (voxel, clip_target, _img, _row) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        voxel = voxel.to(device, non_blocking=True).unsqueeze(1)
        clip_target = clip_target.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
            latent = model.ridge(voxel, 0)
            out = model.backbone(latent)
        clip_voxels = out[1] if isinstance(out, (tuple, list)) else out
        pv = F.normalize(clip_voxels.flatten(1).float(), dim=-1)
        pt = F.normalize(clip_target.flatten(1).float(), dim=-1)
        sims.append((pv * pt).sum(-1).cpu())
        if pv.shape[0] > 1:
            fwd.append(topk_retrieval_accuracy(pv, pt))
            bwd.append(topk_retrieval_accuracy(pt, pv))
        n += pv.shape[0]
    model.train()
    return {
        "cosine": float(torch.cat(sims).mean()) if sims else float("nan"),
        "fwd_top1": float(np.mean(fwd)) if fwd else float("nan"),
        "bwd_top1": float(np.mean(bwd)) if bwd else float("nan"),
        "n": n,
    }


def train_arm(
    cfg: ExperimentConfig,
    arm: ArmConfig,
    seed: int,
    ws,
    device: str = "cuda",
    resume: bool = True,
    ignore_memory_check: bool = False,
) -> TrainResult:
    from .assets import load_asset_meta

    run_name = cfg.run_name(arm.name, seed)
    run_dir = ws.run_dir(run_name)
    state_path = run_dir / "state.pt"
    result_path = run_dir / "result.json"

    if result_path.exists() and resume:
        done = read_json(result_path)
        if done and done.get("completed"):
            log.info("%s already finished — skipping.", run_name)
            return TrainResult(**done)

    seed_everything(seed, deterministic_algorithms=False)
    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    paths_meta = meta["files"]

    # ---- data -----------------------------------------------------------------------
    from .assets import AssetPaths

    paths = AssetPaths(**{k: Path(v) for k, v in paths_meta.items()})
    train_sub, test_sub, _ = build_subsets(paths, average_test_repeats=cfg.average_test_repeats)
    if cfg.zscore_voxels:
        normalise_voxels(train_sub, test_sub)
    emb_path = ws["data"] / f"subj{cfg.subj:02d}_clip_{cfg.num_sessions}sess.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} missing — run `mindeye-lora precompute` before training."
        )
    train_ds = MindEyeDataset(train_sub, paths.images, emb_path)
    test_ds = MindEyeDataset(test_sub, paths.images, emb_path)
    train_loader, test_loader = make_loaders(
        train_ds, test_ds, cfg.batch_size, seed, num_workers=cfg.num_workers, eval_batch_size=32
    )
    num_voxels = train_sub.voxels.shape[1]
    steps_per_epoch = max(1, len(train_loader))

    # ---- model ----------------------------------------------------------------------
    model, info = build_arm_model(cfg, arm, num_voxels, ws, device)

    from .capacity import preflight

    est = preflight(
        arm.name,
        total_params=info["freeze"]["total_params"],
        trainable_params=info["freeze"]["trainable_params"],
        optimizer=getattr(cfg, "optimizer", "adamw"),
        strict=not ignore_memory_check,
        batch_size=cfg.batch_size,
    )
    info["memory_estimate"] = est.__dict__ | {"total_bytes": est.total_bytes}

    opt, sched, lr, total_steps = build_optimizer(model, cfg, arm, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
    dtype = _autocast_dtype(cfg.precision)

    up = load_upstream(ws["upstream"], ref=cfg.upstream_ref)
    L = resolve_losses(up.utils)
    log.info("loss implementations: %s", L["source"])

    start_epoch, global_step, train_seconds = 0, 0, 0.0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if arm.mode == "full":
            model.load_state_dict(state["model"], strict=False)
        else:
            load_adapter_state_dict(model, state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"].cpu().to(torch.uint8))
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        train_seconds = state.get("train_seconds", 0.0)
        log.info("resumed %s from epoch %d (step %d)", run_name, start_epoch, global_step)

    csv = CSVLogger(
        run_dir / "train_log.csv",
        ["epoch", "step", "lr", "loss", "loss_prior", "loss_clip", "cosine", "fwd_top1",
         "bwd_top1", "seconds"],
    )
    write_json(run_dir / "setup.json", {
        "run_name": run_name, "arm": asdict(arm), "config": cfg.to_dict(),
        "num_voxels": num_voxels, "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps, "lr": lr, "model_info": info,
        "upstream_sha": up.sha,
    })

    budget = TimeBudget(cfg.time_budget_min)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    last_save = time.time()
    mixup_epochs = int(cfg.mixup_pct * cfg.epochs)
    completed = False

    log.info("── %s | trainable %s params | %d epochs × %d steps ──", run_name,
             f"{info['freeze']['trainable_params']:,}", cfg.epochs, steps_per_epoch)

    model.train()
    _failed = False
    for epoch in range(start_epoch, cfg.epochs):
        t_epoch = time.time()
        use_mixco = epoch < mixup_epochs
        # SoftCLIP temperature anneal, matching the published schedule
        if use_mixco:
            temp = 0.006
        else:
            frac = (epoch - mixup_epochs) / max(1, cfg.epochs - mixup_epochs)
            temp = 0.004 * (1 - frac) + 0.0075 * frac

        running = {"loss": 0.0, "prior": 0.0, "clip": 0.0, "n": 0}
        opt.zero_grad(set_to_none=True)

        for it, (voxel, clip_target, _img, _row) in enumerate(train_loader):
            voxel = voxel.to(device, non_blocking=True).unsqueeze(1)   # [B, 1, V]
            clip_target = clip_target.to(device, non_blocking=True)

            perm = betas = select = None
            if use_mixco:
                voxel, perm, betas, select = L["mixco"](voxel)

            with torch.autocast("cuda", dtype=dtype, enabled=cfg.precision != "fp32"):
                latent = model.ridge(voxel, 0)
                out = model.backbone(latent)
                backbone_out = out[0] if isinstance(out, (tuple, list)) else out
                clip_voxels = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out

                loss_prior = torch.zeros((), device=device)
                if cfg.use_prior:
                    loss_prior, _ = model.diffusion_prior(
                        text_embed=backbone_out, image_embed=clip_target
                    )

                pv = F.normalize(clip_voxels.flatten(1), dim=-1)
                pt = F.normalize(clip_target.flatten(1), dim=-1)
                if use_mixco:
                    loss_clip = L["mixco_nce"](pv, pt, temp=temp, perm=perm, betas=betas,
                                               select=select)
                else:
                    loss_clip = L["soft_clip_loss"](pv, pt, temp=temp)

                loss = cfg.prior_scale * loss_prior + cfg.clip_scale * loss_clip
                loss = loss / cfg.grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % cfg.grad_accum == 0:
                if cfg.max_grad_norm:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
                    )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < total_steps - 1:
                    sched.step()
                global_step += 1

            bs = voxel.shape[0]
            running["loss"] += loss.item() * cfg.grad_accum * bs
            running["prior"] += float(loss_prior) * bs
            running["clip"] += float(loss_clip) * bs
            running["n"] += bs

        train_seconds += time.time() - t_epoch
        n = max(1, running["n"])
        row = {
            "epoch": epoch, "step": global_step, "lr": sched.get_last_lr()[0],
            "loss": running["loss"] / n, "loss_prior": running["prior"] / n,
            "loss_clip": running["clip"] / n, "seconds": train_seconds,
        }

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            row.update(evaluate_embedding_space(model, test_loader, device, cfg.precision,
                                                max_batches=10))
            log.info("epoch %3d | loss %.4f | cos %.4f | fwd@1 %.3f",
                     epoch, row["loss"], row.get("cosine", float("nan")),
                     row.get("fwd_top1", float("nan")))
        csv.log(**row)

        should_save = (time.time() - last_save) / 60 >= cfg.save_every_min or epoch == cfg.epochs - 1
        if should_save:
            _save_state(state_path, model, opt, sched, scaler, epoch, global_step,
                        train_seconds, arm)
            last_save = time.time()

        if budget.exceeded and epoch < cfg.epochs - 1:
            _save_state(state_path, model, opt, sched, scaler, epoch, global_step,
                        train_seconds, arm)
            log.warning(
                "Time budget reached at epoch %d/%d. State saved — rerun the same command "
                "to continue.", epoch + 1, cfg.epochs
            )
            break
    else:
        completed = True

    # ---- final artefacts --------------------------------------------------------------
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    final_metrics = evaluate_embedding_space(model, test_loader, device, cfg.precision)
    weights_path = run_dir / ("weights_full.pt" if arm.mode == "full" else "adapter.pt")
    if completed:
        payload = (
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if arm.mode == "full"
            else adapter_state_dict(model)
        )
        torch.save({"state_dict": payload, "arm": asdict(arm), "seed": seed}, weights_path)
        ckpt_bytes = weights_path.stat().st_size
        log.info("saved %s (%s)", weights_path.name, human_bytes(ckpt_bytes))
    else:
        ckpt_bytes = adapter_size_bytes(adapter_state_dict(model))

    epochs_done = (epoch + 1) if "epoch" in dir() else 0
    result = TrainResult(
        run_name=run_name,
        arm=arm.name,
        seed=seed,
        epochs_completed=epochs_done,
        trainable_params=info["freeze"]["trainable_params"],
        total_params=info["freeze"]["total_params"],
        checkpoint_bytes=int(ckpt_bytes),
        peak_memory_bytes=int(peak),
        train_seconds=float(train_seconds),
        seconds_per_epoch=float(train_seconds / max(1, epochs_done)),
        final_metrics=final_metrics,
        completed=completed,
    )
    write_json(result_path, asdict(result))
    if completed:
        log.info("✔ %s done: cos=%.4f fwd@1=%.3f peak=%s", run_name,
                 final_metrics["cosine"], final_metrics["fwd_top1"], human_bytes(peak))
    del model, opt
    torch.cuda.empty_cache()
    return result


def release_cuda(*objects) -> None:
    """Drop references and empty the allocator cache.

    A CUDA OOM inside a notebook is sticky: the exception's traceback holds every local
    in every frame, including the model that just failed, so the next attempt starts
    with the GPU still full. Freeing explicitly at the point of failure means a smaller
    batch size can actually be tried without restarting the runtime.
    """
    import gc

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _save_state(path, model, opt, sched, scaler, epoch, global_step, train_seconds, arm) -> None:
    payload = (
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if arm.mode == "full"
        else adapter_state_dict(model)
    )
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "model": payload,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "train_seconds": train_seconds,
            "cpu_rng": torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)
