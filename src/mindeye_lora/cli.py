"""Command line entry point.

Every stage is idempotent and records completion in the workspace manifest, so
`run-all` can be re-issued after a Colab disconnect and it will pick up where it
stopped rather than starting over.

    python -m mindeye_lora.cli assets
    python -m mindeye_lora.cli precompute
    python -m mindeye_lora.cli train --arm lora_r16 --seed 0
    python -m mindeye_lora.cli run-all --config configs/colab_t4.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, load_config, save_config
from .env import get_workspace, setup_environment
from .utils import RunTracker, human_bytes, log, read_json, robust_save, write_json


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _ws(args, mirror_data: bool = False):
    ws = setup_environment(args.root, use_drive=not args.no_drive)
    if mirror_data and not getattr(args, "no_local_cache", False):
        from .env import mirror_data_locally

        ws = mirror_data_locally(ws)
    return ws


def _cfg(args) -> ExperimentConfig:
    overrides = {}
    for key in ("subj", "num_sessions", "epochs", "batch_size", "lr", "precision",
                "time_budget_min", "recon_decoder", "recon_n_images", "pretrain",
                "upstream_ref", "num_workers"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if getattr(args, "seeds", None):
        overrides["seeds"] = [int(s) for s in args.seeds]
    return load_config(args.config, overrides)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _emb_path(ws, cfg):
    return ws["data"] / f"subj{cfg.subj:02d}_clip_{cfg.num_sessions}sess.npy"


def safe_torch_load(path, **kw):
    """torch.load that reports a corrupt file clearly instead of a miniz stack trace.

    Files on a Drive mount can be truncated if the session died mid-upload, and the
    default error ("failed finding central directory ... internal miniz error") gives no
    hint about which file or what to do about it.
    """
    import torch

    try:
        return torch.load(str(path), **kw)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}. It is most likely truncated — Google Drive uploads "
            f"asynchronously, so a file written just before a session was reclaimed can "
            f"be incomplete. Delete it and re-run the stage that produces it:\n"
            f"    from pathlib import Path; Path({str(path)!r}).unlink()\n"
            f"Original error: {exc}"
        ) from exc


def _selected_arms(cfg: ExperimentConfig, names):
    if not names:
        return list(cfg.arms)
    return [cfg.arm(n) for n in names]


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------
def cmd_setup(args):
    ws = _ws(args)
    from .utils import gpu_info

    print(ws.summary())
    print("GPU:", json.dumps(gpu_info(), indent=2))
    cfg = _cfg(args)
    save_config(cfg, ws.root / "config_resolved.yaml")
    print(f"\nresolved config -> {ws.root / 'config_resolved.yaml'}")
    print("arms:", ", ".join(a.name for a in cfg.arms))


# Rough model size per pretrained checkpoint, used only to refuse an obviously
# impossible run before spending ten minutes downloading it.
PRETRAIN_MIN_VRAM_GB = {"multisubject_1024": 12.0, "multisubject_4096": 24.0,
                        "reference_1sess": 12.0}


def _check_gpu_fits_config(cfg, force: bool = False) -> None:
    """Refuse a paper-scale config on a small GPU before the download, not after.

    The 4096 checkpoint is ~10 GB to fetch and the model needs ~23 GB to fine-tune even
    with 8-bit moments. Discovering that after the download wastes both time and, on a
    paid runtime, compute units.
    """
    from .capacity import gpu_capacity_bytes

    need = PRETRAIN_MIN_VRAM_GB.get(cfg.pretrain)
    capacity = gpu_capacity_bytes()
    if need is None or capacity is None:
        return
    have = capacity / 1024**3
    if have + 0.5 < need:
        message = (
            f"`pretrain: {cfg.pretrain}` needs roughly {need:.0f} GB of VRAM but this "
            f"GPU has {have:.1f} GB.\n"
            f"  Switch to configs/colab_t4.yaml (pretrain: multisubject_1024), or "
            f"request an A100 runtime.\n"
            f"  Pass --ignore-memory-check to proceed anyway."
        )
        if force:
            log.warning(message)
        else:
            raise RuntimeError(message)


def cmd_assets(args):
    ws, cfg = _ws(args), _cfg(args)
    _check_gpu_fits_config(cfg, force=args.ignore_memory_check)
    from .assets import prepare_assets

    paths = prepare_assets(
        ws, subj=cfg.subj, num_sessions=cfg.num_sessions, pretrain=cfg.pretrain,
        full_local_hdf5=args.full_hdf5, force=args.force,
    )
    print("assets ready:")
    for k, v in asdict(paths).items():
        p = Path(v)
        size = human_bytes(p.stat().st_size) if p.exists() else "missing"
        print(f"  {k:16s} {size:>10s}  {p.name}")


def cmd_precompute(args):
    ws, cfg = _ws(args), _cfg(args)
    from .assets import load_asset_meta
    from .embeddings import embedding_stats, precompute_embeddings

    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    out = precompute_embeddings(
        images_path=meta["files"]["images"],
        out_path=_emb_path(ws, cfg),
        backend=cfg.clip_backend,
        batch_size=args.batch_size or 16,
        device=_device(),
        force=args.force,
    )
    print(json.dumps(embedding_stats(out), indent=2))
    ws.mark_done(f"precompute_subj{cfg.subj:02d}_{cfg.num_sessions}sess")


def cmd_verify(args):
    """Sanity-check that the pretrained weights map onto the constructed model."""
    ws, cfg = _ws(args), _cfg(args)
    from .assets import load_asset_meta
    from .lora import LoRAConfig, find_target_linears
    from .model_factory import create_mindeye, module_tree

    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    model, arch, report = create_mindeye(
        ckpt_path=meta["files"]["pretrain_ckpt"], num_voxels=meta["num_voxels"],
        upstream_dir=ws["upstream"], upstream_ref=cfg.upstream_ref,
        use_prior=cfg.use_prior, blurry_recon=cfg.blurry_recon,
        device="cpu", strict=not args.lenient,
    )
    print("architecture:", json.dumps(arch.as_dict(), indent=2))
    print("checkpoint load:", report.describe())
    targets = find_target_linears(model, LoRAConfig())
    print(f"\nLoRA would wrap {len(targets)} Linear layers:")
    for t in targets[:25]:
        print("  ", t)
    if len(targets) > 25:
        print(f"   ... and {len(targets) - 25} more")
    if args.tree:
        print("\nAll Linear layers:\n" + module_tree(model))


def cmd_train(args):
    ws, cfg = _ws(args, mirror_data=True), _cfg(args)
    import torch

    from .train import train_arm

    device = _device()
    arms = _selected_arms(cfg, args.arm)
    seeds = [int(s) for s in (args.seed or cfg.seeds)]
    results = []
    tracker = RunTracker(len(seeds) * len(arms), label="training runs")
    for seed in seeds:
        for arm in arms:
            run_name = cfg.run_name(arm.name, seed)
            prior = read_json(ws.run_dir(run_name) / "result.json")
            already = bool(prior and prior.get("completed"))
            tracker.start(run_name)
            try:
                results.append(asdict(train_arm(
                    cfg, arm, seed, ws, device=device, resume=not args.restart,
                    ignore_memory_check=args.ignore_memory_check)))
                tracker.finish(run_name, skipped=already)
            except torch.cuda.OutOfMemoryError:
                # Release before re-raising: otherwise the traceback pins the failed
                # model and every later arm OOMs too, for no reason.
                from .train import release_cuda

                release_cuda()
                log.error(
                    "%s ran out of memory. Lower `batch_size` and raise `grad_accum` to "
                    "match in the config, then re-run — finished arms are skipped.",
                    cfg.run_name(arm.name, seed),
                )
                raise
    log.info("── %s ──", tracker.summary())
    write_json(ws["results"] / "training_summary.json", results)
    for r in results:
        print(f"{r['run_name']:44s} trainable={r['trainable_params']:>12,} "
              f"cos={r['final_metrics']['cosine']:.4f} "
              f"{'done' if r['completed'] else 'PARTIAL'}")


def cmd_predict(args):
    """Run each trained model over the test set and cache its predicted embeddings."""
    ws, cfg = _ws(args), _cfg(args)
    import torch

    from .assets import AssetPaths, load_asset_meta
    from .data import MindEyeDataset, build_subsets, make_loaders, normalise_voxels
    from .evaluate import PredictionStore, predict
    from .lora import load_adapter_state_dict
    from .train import build_arm_model

    device = _device()
    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    paths = AssetPaths(**{k: Path(v) for k, v in meta["files"].items()})
    train_sub, test_sub, _ = build_subsets(paths, cfg.average_test_repeats)
    if cfg.zscore_voxels:
        normalise_voxels(train_sub, test_sub)
    emb = _emb_path(ws, cfg)
    test_ds = MindEyeDataset(test_sub, paths.images, emb)
    _, test_loader = make_loaders(
        MindEyeDataset(train_sub, paths.images, emb), test_ds, cfg.batch_size, 0,
        num_workers=cfg.num_workers, eval_batch_size=args.batch_size or 16,
    )

    pred_seeds = [int(s) for s in (args.seed or cfg.seeds)]
    pred_arms = _selected_arms(cfg, args.arm)
    tracker = RunTracker(len(pred_seeds) * len(pred_arms), label="predict runs")
    for seed in pred_seeds:
        for arm in pred_arms:
            run_dir = ws.run_dir(cfg.run_name(arm.name, seed))
            store_dir = run_dir / "predictions"
            if PredictionStore.is_complete(store_dir) and not args.force:
                tracker.finish(run_dir.name, skipped=True)
                continue
            tracker.start(run_dir.name)
            weights = run_dir / ("weights_full.pt" if arm.mode == "full" else "adapter.pt")
            if not weights.exists():
                log.warning("no trained weights for %s — train it first.", run_dir.name)
                continue
            model, _ = build_arm_model(cfg, arm, train_sub.voxels.shape[1], ws, device)
            payload = safe_torch_load(weights, map_location=device,
                                      weights_only=False)["state_dict"]
            if arm.mode == "full":
                model.load_state_dict(payload, strict=False)
            else:
                load_adapter_state_dict(model, payload)
            store = predict(
                model, test_loader, store_dir, device=device, precision=cfg.precision,
                use_prior=cfg.use_prior, prior_timesteps=args.prior_timesteps,
                desc=f"predict {arm.name} seed{seed}",
                resume=not args.force,
            )
            store.close()
            log.info("predictions streamed to %s", store_dir)
            tracker.finish(run_dir.name)
            from .train import release_cuda

            release_cuda(model)
    log.info("── %s ──", tracker.summary())


def cmd_recon(args):
    """Generate images with the SDXL decoder, falling back to retrieval if unavailable."""
    ws, cfg = _ws(args), _cfg(args)
    from .evaluate import PredictionStore
    from .recon import reconstruct_for_run
    from .retrieval import build_retrieval_output

    decoder = args.decoder or cfg.recon_decoder
    n_images = args.n_images or cfg.recon_n_images
    seeds = [int(s) for s in (args.seed or [cfg.recon_seed])]
    recon_arms = _selected_arms(cfg, args.arm)
    tracker = RunTracker(len(seeds) * len(recon_arms), label="recon runs")

    for seed in seeds:
        for arm in recon_arms:
            run_dir = ws.run_dir(cfg.run_name(arm.name, seed))
            tracker.start(run_dir.name)
            store_dir = run_dir / "predictions"
            if not PredictionStore.is_complete(store_dir):
                log.warning("no predictions for %s — run `predict` first.", run_dir.name)
                continue

            decoded = False
            if decoder != "none":
                out = run_dir / "reconstructions.pt"
                if out.exists() and not args.force:
                    log.info("reconstructions exist for %s", run_dir.name)
                    decoded = True
                else:
                    try:
                        reconstruct_for_run(
                            store_dir, out, ws, n_images=n_images, decoder=decoder,
                            device=_device(), num_steps=args.num_steps,
                            batch_size=1, unclip_dir=args.unclip_dir,
                        )
                        decoded = True
                    except Exception as exc:
                        # The 18 GB sgm decoder is the most fragile piece of the project.
                        # Losing it should cost the qualitative figure, not the whole run --
                        # but log the traceback, because a bare message like
                        # "No data left in file" is impossible to act on.
                        import traceback

                        log.warning("SDXL decoder unavailable for %s (%s: %s)",
                                    run_dir.name, type(exc).__name__, exc)
                        log.debug("decoder traceback:\n%s", traceback.format_exc())
                        if args.strict_decoder:
                            raise

            if not decoded and cfg.retrieval_fallback:
                target = run_dir / "retrieval.pt"
                if target.exists() and not args.force:
                    log.info("retrieval output exists for %s", run_dir.name)
                    tracker.finish(run_dir.name, skipped=True)
                    continue
                build_retrieval_output(store_dir, target, k=cfg.retrieval_k,
                                       n_items=n_images)
                tracker.finish(run_dir.name)
            elif decoded:
                tracker.finish(run_dir.name)
    log.info("── %s ──", tracker.summary())


def cmd_evaluate(args):
    ws, cfg = _ws(args, mirror_data=True), _cfg(args)
    import torch

    from .assets import AssetPaths, load_asset_meta
    from .evaluate import (
        ImageMetricSuite, PredictionStore, embedding_metrics, retrieval_summary,
        save_per_sample,
    )
    from .recon import load_ground_truth

    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    paths = AssetPaths(**{k: Path(v) for k, v in meta["files"].items()})
    suite = None

    for seed in [int(s) for s in (args.seed or cfg.seeds)]:
        for arm in _selected_arms(cfg, args.arm):
            run_dir = ws.run_dir(cfg.run_name(arm.name, seed))
            store_dir = run_dir / "predictions"
            if not PredictionStore.is_complete(store_dir):
                continue
            out = run_dir / "per_sample_metrics.npz"
            if out.exists() and not args.force:
                log.info("metrics exist for %s", run_dir.name)
                continue
            preds = PredictionStore.load(store_dir)
            pred_emb = PredictionStore.embedding(preds)
            metrics = embedding_metrics(pred_emb, preds["target"])
            summary = retrieval_summary(pred_emb, preds["target"])

            recon_path = run_dir / "reconstructions.pt"
            if recon_path.exists():
                rec = torch.load(recon_path, map_location="cpu", weights_only=False)
                gt = load_ground_truth(paths.images, rec["rows"].numpy(),
                                       size=rec["recons"].shape[-1])
                suite = suite or ImageMetricSuite(device=_device())
                img_metrics = suite(rec["recons"], gt)
                n = len(rec["recons"])
                # image metrics cover a prefix of the test set; pad so arrays stay aligned
                for k, v in img_metrics.items():
                    padded = np.full(len(pred_emb), np.nan)
                    padded[:n] = v
                    metrics[k] = padded

            save_per_sample(out, metrics, np.asarray(preds["rows"]))
            write_json(run_dir / "retrieval_summary.json", summary)
            log.info("%s | cos=%.4f two-way=%.4f fwd@1=%.3f", run_dir.name,
                     float(np.mean(metrics["cosine"])),
                     float(np.mean(metrics["two_way_clip"])), summary["fwd_top1"])


def cmd_compare(args):
    ws, cfg = _ws(args), _cfg(args)
    from .evaluate import load_per_sample
    from .metric_meta import HIGHER_IS_BETTER
    from .stats import (
        comparisons_to_records, compare_against_reference, retention_table, seed_variability,
    )

    per_seed: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for seed in [int(s) for s in cfg.seeds]:
        for arm in cfg.arms:
            path = ws.run_dir(cfg.run_name(arm.name, seed)) / "per_sample_metrics.npz"
            if path.exists():
                per_seed.setdefault(arm.name, {})[seed] = load_per_sample(path)
    if not per_seed:
        raise SystemExit("No per-sample metrics found. Run `evaluate` first.")

    # Pool seeds by averaging each image's score, keeping the pairing across arms intact.
    per_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm, seeds in per_seed.items():
        metrics = set().union(*[set(v) for v in seeds.values()]) - {"rows"}
        pooled = {}
        for m in metrics:
            stack = [seeds[s][m] for s in sorted(seeds) if m in seeds[s]]
            if not stack:
                continue
            L = min(len(x) for x in stack)
            arr = np.nanmean(np.stack([x[:L] for x in stack]), axis=0)
            if np.isfinite(arr).any():
                pooled[m] = arr
        per_arm[arm] = pooled

    # keep only images every arm scored (image metrics may cover a shorter prefix)
    for m in set().union(*[set(v) for v in per_arm.values()]):
        lengths = [len(v[m]) for v in per_arm.values() if m in v]
        if not lengths:
            continue
        L = min(lengths)
        mask = np.ones(L, bool)
        for v in per_arm.values():
            if m in v:
                mask &= np.isfinite(v[m][:L])
        for v in per_arm.values():
            if m in v:
                v[m] = v[m][:L][mask]

    reference = args.reference or "full"
    comparisons = compare_against_reference(
        per_arm, reference=reference, higher_is_better=HIGHER_IS_BETTER,
        equivalence_fraction=args.equivalence_fraction, frozen_arm=args.frozen_arm,
        n_boot=args.n_boot,
    )
    retention = retention_table(per_arm, full_arm=reference, frozen_arm=args.frozen_arm,
                                higher_is_better=HIGHER_IS_BETTER, n_boot=args.n_boot)
    seeds_rows = seed_variability(per_seed)

    payload = {
        "comparisons": comparisons_to_records(comparisons),
        "retention": retention,
        "seed_variability": seeds_rows,
        "means": {a: {m: float(np.mean(v)) for m, v in d.items()} for a, d in per_arm.items()},
    }
    write_json(ws["results"] / "statistics.json", payload)
    np.savez_compressed(
        ws["results"] / "pooled_per_sample.npz",
        **{f"{a}::{m}": v for a, d in per_arm.items() for m, v in d.items()},
    )
    for c in payload["comparisons"]:
        print(f"{c['metric']:>20s} {c['arm']:>10s}  Δ={c['diff']:+.4f} "
              f"[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]  p_holm={c['p_holm']:.3g}  "
              f"equiv={c['equivalent']}")
    print(f"\nwrote {ws['results'] / 'statistics.json'}")


def cmd_report(args):
    ws, cfg = _ws(args), _cfg(args)
    import torch

    from .report import (
        build_report, figure_forest, figure_metric_bars, figure_pareto,
        figure_qualitative, figure_retrieval_grid, figure_training_curves,
        select_qualitative_arms,
    )
    from .retrieval import load_retrieved_images
    from .recon import load_ground_truth
    from .assets import AssetPaths, load_asset_meta

    stats = read_json(ws["results"] / "statistics.json")
    if stats is None:
        raise SystemExit("No statistics.json — run `compare` first.")
    with np.load(ws["results"] / "pooled_per_sample.npz") as z:
        per_arm: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            arm, metric = key.split("::")
            per_arm.setdefault(arm, {})[metric] = z[key]

    training = read_json(ws["results"] / "training_summary.json", default=[]) or []
    efficiency: dict[str, dict] = {}
    for r in training:
        e = efficiency.setdefault(r["arm"], {
            "arm": r["arm"], "trainable_params": r["trainable_params"],
            "total_params": r["total_params"], "checkpoint_bytes": r["checkpoint_bytes"],
            "peak_memory_bytes": 0, "seconds_per_epoch": 0.0, "_n": 0,
        })
        e["peak_memory_bytes"] = max(e["peak_memory_bytes"], r["peak_memory_bytes"])
        e["seconds_per_epoch"] += r["seconds_per_epoch"]
        e["_n"] += 1
    for e in efficiency.values():
        e["seconds_per_epoch"] /= max(1, e.pop("_n"))
    efficiency_rows = list(efficiency.values())

    figdir = ws["figures"]
    figs = {}
    primary = [m for m in ("two_way_clip", "cosine", "retrieval_percentile")
               if any(m in v for v in per_arm.values())]
    figs["Per-image metrics"] = figure_metric_bars(per_arm, primary, figdir / "metric_bars.png")
    figs["Paired differences vs full fine-tune"] = figure_forest(
        stats["comparisons"], figdir / "forest.png", metric_filter=primary)
    if efficiency_rows and primary:
        figs["Cost / quality trade-off"] = figure_pareto(
            efficiency_rows, per_arm, primary[0], figdir / "pareto.png")
    run_dirs = {a.name: ws.run_dir(cfg.run_name(a.name, int(cfg.seeds[0]))) for a in cfg.arms}
    figs["Training loss"] = figure_training_curves(run_dirs, figdir / "training_loss.png")

    # qualitative grid, if any arm produced reconstructions
    recons_by_arm, gt = {}, None
    meta = load_asset_meta(ws, cfg.subj, cfg.num_sessions)
    paths = AssetPaths(**{k: Path(v) for k, v in meta["files"].items()})
    for a in cfg.arms:
        p = ws.run_dir(cfg.run_name(a.name, int(cfg.recon_seed))) / "reconstructions.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            recons_by_arm[a.name] = d["recons"][: cfg.qualitative_n].numpy()
            if gt is None:
                gt = load_ground_truth(paths.images, d["rows"][: cfg.qualitative_n].numpy(),
                                       size=d["recons"].shape[-1]).numpy()
    if recons_by_arm and gt is not None:
        keep = select_qualitative_arms(list(recons_by_arm), limit=cfg.qualitative_arms)
        figs["Reconstructions"] = figure_qualitative(
            gt, {a: recons_by_arm[a] for a in keep}, figdir / "qualitative.png",
            n=cfg.qualitative_n)

    # Retrieval panel: either the fallback when no decoder ran, or a supplement to it.
    retrieved, ranks, pool_size, rgt = {}, {}, 0, None
    for a in cfg.arms:
        p = ws.run_dir(cfg.run_name(a.name, int(cfg.recon_seed))) / "retrieval.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = cfg.qualitative_n
        retrieved[a.name] = load_retrieved_images(paths.images, d["image_rows"][:n])
        ranks[a.name] = d["rank_of_truth"][:n].numpy()
        pool_size = int(d["pool_size"])
        if rgt is None:
            rgt = load_ground_truth(paths.images, d["rows"][:n].numpy()).numpy()
    if retrieved and rgt is not None:
        keep = select_qualitative_arms(list(retrieved), limit=cfg.qualitative_arms)
        figs["Retrieval (nearest test-set image)"] = figure_retrieval_grid(
            rgt, {a: retrieved[a] for a in keep}, {a: ranks[a] for a in keep},
            figdir / "retrieval_grid.png", pool_size=pool_size, n=cfg.qualitative_n)

    cfg_summary = {
        "subject": cfg.subj,
        "training data": f"{cfg.num_sessions} session(s) (~{cfg.num_sessions} hour)",
        "pretrained init": cfg.pretrain,
        "epochs": cfg.epochs,
        "batch size": cfg.batch_size,
        "seeds": ", ".join(str(s) for s in cfg.seeds),
        "decoder": cfg.recon_decoder,
        "primary metric": primary[0] if primary else "–",
    }
    out = build_report(
        ws["reports"] / "REPORT.md", cfg_summary, per_arm, stats["comparisons"],
        stats["retention"], efficiency_rows, stats["seed_variability"],
        {k: v for k, v in figs.items() if v},
        primary_metric=primary[0] if primary else "cosine",
    )
    # keep figures next to the report so the markdown renders standalone
    for f in figdir.glob("*.png"):
        target = out.parent / f.name
        if not target.exists() or target.stat().st_mtime < f.stat().st_mtime:
            target.write_bytes(f.read_bytes())
    print(f"report -> {out}")


def cmd_run_all(args):
    ws, cfg = _ws(args), _cfg(args)
    save_config(cfg, ws.root / "config_resolved.yaml")
    stages = [
        ("assets", cmd_assets),
        ("precompute", cmd_precompute),
        ("train", cmd_train),
        ("predict", cmd_predict),
        ("recon", cmd_recon),
        ("evaluate", cmd_evaluate),
        ("compare", cmd_compare),
        ("report", cmd_report),
    ]
    skip = set(args.skip or [])
    active = [(n, f) for n, f in stages if n not in skip]
    t0 = time.time()
    for i, (name, fn) in enumerate(active, start=1):
        log.info("═══ stage %d/%d: %s ═══ (%.1f min elapsed)",
                 i, len(active), name, (time.time() - t0) / 60)
        fn(args)
    log.info("═══ all %d stages complete in %.1f min ═══",
             len(active), (time.time() - t0) / 60)


def cmd_status(args):
    ws, cfg = _ws(args), _cfg(args)
    print(ws.summary())
    print("\nruns:")
    for seed in cfg.seeds:
        for arm in cfg.arms:
            d = ws["runs"] / cfg.run_name(arm.name, seed)
            r = read_json(d / "result.json")
            state = "—"
            if r:
                state = "done" if r.get("completed") else f"partial(ep {r['epochs_completed']})"
            elif (d / "state.pt").exists():
                state = "in progress"
            extras = []
            from .evaluate import PredictionStore

            if PredictionStore.is_complete(d / "predictions"):
                extras.append("predictions")
            extras += [n for n in ("reconstructions.pt", "retrieval.pt",
                                   "per_sample_metrics.npz") if (d / n).exists()]
            print(f"  {d.name:46s} {state:18s} {' '.join(extras)}")


# --------------------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------------------
GLOBAL_DEFAULTS = {
    "root": None, "no_drive": False, "config": None, "subj": None, "num_sessions": None,
    "epochs": None, "batch_size": None, "lr": None, "time_budget_min": None,
    "recon_n_images": None, "num_workers": None, "precision": None, "pretrain": None,
    "upstream_ref": None, "recon_decoder": None, "seeds": None, "no_local_cache": False,
}


def _add_global_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Global options, attached to both the top level and every subcommand.

    Defaults are SUPPRESS rather than None on purpose: with `parents=`, a subparser
    that stored a None default would overwrite a value the top-level parser already
    parsed, so `--config x run-all` would silently lose the config. Suppressed
    defaults leave the namespace untouched when the flag is absent, and `main()`
    backfills whatever is missing.
    """
    S = argparse.SUPPRESS
    parser.add_argument("--root", default=S, help="workspace root (defaults to Drive)")
    parser.add_argument("--no-drive", dest="no_drive", action="store_true", default=S,
                        help="do not mount Google Drive")
    parser.add_argument("--config", default=S, help="YAML config path")
    parser.add_argument("--no-local-cache", dest="no_local_cache", action="store_true",
                        default=S,
                        help="read caches straight from Drive instead of mirroring them "
                             "to local disk (slower, but uses no local storage)")
    for name, kind in [("subj", int), ("num_sessions", int), ("epochs", int),
                       ("batch_size", int), ("lr", float), ("time_budget_min", float),
                       ("recon_n_images", int), ("num_workers", int)]:
        parser.add_argument(f"--{name}", type=kind, default=S)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=S)
    parser.add_argument("--pretrain", default=S,
                        choices=["multisubject_1024", "multisubject_4096", "reference_1sess"])
    parser.add_argument("--upstream_ref", default=S)
    parser.add_argument("--recon_decoder", default=S, choices=["none", "sdxl_unclip"])
    parser.add_argument("--seeds", nargs="*", default=S)
    return parser


def build_parser() -> argparse.ArgumentParser:
    # `common` is inherited by every subcommand so that global options work on either
    # side of the subcommand name: `--config c.yaml train` and `train --config c.yaml`
    # are equivalent. The latter is what everyone types first.
    common = _add_global_args(argparse.ArgumentParser(add_help=False))

    p = argparse.ArgumentParser("mindeye-lora", description=__doc__,
                                parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    add("setup").set_defaults(func=cmd_setup)

    a = add("assets")
    a.add_argument("--ignore-memory-check", action="store_true",
                   dest="ignore_memory_check",
                   help="download even if this GPU cannot hold the chosen model")
    a.add_argument("--full-hdf5", action="store_true",
                   help="download the complete HDF5 files instead of streaming row subsets")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_assets)

    pc = add("precompute")
    pc.add_argument("--force", action="store_true")
    pc.set_defaults(func=cmd_precompute)

    v = add("verify")
    v.add_argument("--tree", action="store_true", help="print every Linear layer")
    v.add_argument("--lenient", action="store_true", help="do not fail on load mismatch")
    v.set_defaults(func=cmd_verify)

    t = add("train")
    t.add_argument("--arm", nargs="*", default=None)
    t.add_argument("--seed", nargs="*", default=None)
    t.add_argument("--restart", action="store_true", help="ignore saved state and start over")
    t.add_argument("--ignore-memory-check", action="store_true",
                   dest="ignore_memory_check",
                   help="attempt the run even if it is predicted not to fit")
    t.set_defaults(func=cmd_train)

    pr = add("predict")
    pr.add_argument("--arm", nargs="*", default=None)
    pr.add_argument("--seed", nargs="*", default=None)
    pr.add_argument("--prior_timesteps", type=int, default=20)
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(func=cmd_predict)

    rc = add("recon")
    rc.add_argument("--arm", nargs="*", default=None)
    rc.add_argument("--seed", nargs="*", default=None)
    rc.add_argument("--decoder", default=None, choices=["none", "sdxl_unclip"])
    rc.add_argument("--n_images", type=int, default=None)
    rc.add_argument("--num_steps", type=int, default=38)
    rc.add_argument("--unclip_dir", default=None)
    rc.add_argument("--force", action="store_true")
    rc.add_argument("--strict-decoder", dest="strict_decoder", action="store_true",
                    help="re-raise decoder errors instead of falling back to retrieval")
    rc.set_defaults(func=cmd_recon)

    ev = add("evaluate")
    ev.add_argument("--arm", nargs="*", default=None)
    ev.add_argument("--seed", nargs="*", default=None)
    ev.add_argument("--force", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    cp = add("compare")
    cp.add_argument("--reference", default="full")
    cp.add_argument("--frozen_arm", default="frozen")
    cp.add_argument("--equivalence_fraction", type=float, default=0.2)
    cp.add_argument("--n_boot", type=int, default=10000)
    cp.set_defaults(func=cmd_compare)

    add("report").set_defaults(func=cmd_report)

    ra = add("run-all")
    ra.add_argument("--skip", nargs="*", default=None)
    ra.add_argument("--arm", nargs="*", default=None)
    ra.add_argument("--seed", nargs="*", default=None)
    ra.add_argument("--force", action="store_true")
    ra.add_argument("--restart", action="store_true")
    ra.add_argument("--ignore-memory-check", action="store_true",
                    dest="ignore_memory_check")
    ra.add_argument("--full-hdf5", action="store_true")
    ra.add_argument("--decoder", default=None, choices=["none", "sdxl_unclip"])
    ra.add_argument("--n_images", type=int, default=None)
    ra.add_argument("--num_steps", type=int, default=38)
    ra.add_argument("--unclip_dir", default=None)
    ra.add_argument("--prior_timesteps", type=int, default=20)
    ra.add_argument("--reference", default="full")
    ra.add_argument("--frozen_arm", default="frozen")
    ra.add_argument("--equivalence_fraction", type=float, default=0.2)
    ra.add_argument("--n_boot", type=int, default=10000)
    ra.set_defaults(func=cmd_run_all)

    add("status").set_defaults(func=cmd_status)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Suppressed global defaults never land in the namespace, so fill them in here.
    for name, default in GLOBAL_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    # defaults for options only defined on some subparsers
    for name, default in [("force", False), ("arm", None),
                          ("seed", None), ("restart", False), ("full_hdf5", False),
                          ("decoder", None), ("n_images", None), ("num_steps", 38),
                          ("unclip_dir", None), ("prior_timesteps", 20),
                          ("reference", "full"), ("frozen_arm", "frozen"),
                          ("equivalence_fraction", 0.2), ("n_boot", 10000),
                          ("tree", False), ("lenient", False), ("skip", None),
                          ("ignore_memory_check", False), ("strict_decoder", False)]:
        if not hasattr(args, name):
            setattr(args, name, default)
    args.func(args)


if __name__ == "__main__":
    main()
