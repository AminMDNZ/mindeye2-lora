"""Figures and the written report.

Nothing here invents numbers: every table and panel is rendered from the per-sample
metric files and the run manifests produced by earlier stages. If a stage was skipped
(no decoder, fewer seeds) the corresponding section is omitted rather than faked.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .metric_meta import ALL_METRICS, HIGHER_IS_BETTER
from .utils import human_bytes, log

ARM_ORDER = ["frozen", "bitfit", "lora_r4", "lora_r16", "lora_r64", "full"]


def _order(arms: Sequence[str]) -> list[str]:
    known = [a for a in ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in known)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25,
    })
    return plt


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figure_metric_bars(per_arm: dict[str, dict[str, np.ndarray]], metrics: Sequence[str],
                       out: Path) -> Path:
    plt = _mpl()
    arms = _order(list(per_arm))
    metrics = [m for m in metrics if any(m in per_arm[a] for a in arms)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.2))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        means, errs, labels = [], [], []
        for a in arms:
            if metric not in per_arm[a]:
                continue
            v = np.asarray(per_arm[a][metric], float)
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
            labels.append(a)
        colors = ["#b0b0b0" if l == "frozen" else "#2f6fb2" if l.startswith("lora")
                  else "#c1440e" if l == "full" else "#7a9e5e" for l in labels]
        ax.bar(range(len(means)), means, yerr=errs, capsize=3, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(metric + ("  ↑" if HIGHER_IS_BETTER.get(metric, True) else "  ↓"))
    fig.suptitle("Per-image mean ± 95% CI on the held-out NSD test images", y=1.03)
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_forest(comparisons: list[dict], out: Path, metric_filter: Sequence[str] | None = None) -> Path:
    """Paired difference vs the full fine-tune, with bootstrap CIs."""
    plt = _mpl()
    rows = [c for c in comparisons if not metric_filter or c["metric"] in metric_filter]
    if not rows:
        return out
    metrics = sorted({r["metric"] for r in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.5 * len(rows) / len(metrics) + 2.2))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        sub = [r for r in rows if r["metric"] == metric]
        sub = sorted(sub, key=lambda r: _order([x["arm"] for x in sub]).index(r["arm"]))
        y = np.arange(len(sub))
        centers = [r["diff"] for r in sub]
        lo = [r["diff"] - r["ci_low"] for r in sub]
        hi = [r["ci_high"] - r["diff"] for r in sub]
        ax.errorbar(centers, y, xerr=[lo, hi], fmt="o", color="#2f6fb2", capsize=3)
        ax.axvline(0, color="#c1440e", lw=1)
        margin = next((r["tost_margin"] for r in sub if np.isfinite(r.get("tost_margin", np.nan))), None)
        if margin:
            ax.axvspan(-margin, margin, color="#7a9e5e", alpha=0.12,
                       label="equivalence zone")
            ax.legend(loc="lower right", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([r["arm"] for r in sub])
        ax.set_xlabel(f"Δ {metric} vs full fine-tune")
    fig.suptitle("Paired differences (negative = worse than full fine-tuning)", y=1.02)
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_pareto(efficiency: list[dict], per_arm: dict[str, dict[str, np.ndarray]],
                  metric: str, out: Path) -> Path:
    """Trainable parameters against quality — the actual decision surface."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for row in efficiency:
        arm = row["arm"]
        if arm not in per_arm or metric not in per_arm[arm]:
            continue
        x = max(row["trainable_params"], 1)
        y = float(np.mean(per_arm[arm][metric]))
        color = "#b0b0b0" if arm == "frozen" else "#c1440e" if arm == "full" else "#2f6fb2"
        ax.scatter(x, y, s=48, color=color, zorder=3)
        ax.annotate(arm, (x, y), textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel(metric)
    ax.set_title("Cost / quality trade-off")
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_training_curves(run_dirs: dict[str, Path], out: Path, key: str = "loss") -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for arm, d in run_dirs.items():
        path = Path(d) / "train_log.csv"
        if not path.exists():
            continue
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        xs = [float(r["epoch"]) for r in rows if r.get(key)]
        ys = [float(r[key]) for r in rows if r.get(key)]
        if xs:
            ax.plot(xs, ys, label=arm, lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.set_title(f"Training {key}")
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_qualitative(
    ground_truth, recons_by_arm: dict[str, "np.ndarray"], out: Path, n: int = 8
) -> Path:
    """Ground truth on top, one row per adaptation strategy underneath."""
    plt = _mpl()
    arms = _order(list(recons_by_arm))
    n = min(n, len(ground_truth))
    rows = 1 + len(arms)
    fig, axes = plt.subplots(rows, n, figsize=(1.35 * n, 1.45 * rows))
    axes = np.atleast_2d(axes)
    for j in range(n):
        axes[0, j].imshow(np.transpose(np.asarray(ground_truth[j]), (1, 2, 0)).clip(0, 1))
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("ground truth")
    for i, arm in enumerate(arms, start=1):
        imgs = recons_by_arm[arm]
        for j in range(n):
            axes[i, j].axis("off")
            if j < len(imgs):
                axes[i, j].imshow(np.transpose(np.asarray(imgs[j]), (1, 2, 0)).clip(0, 1))
    for i, label in enumerate(["ground truth"] + arms):
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return out


def select_qualitative_arms(arms: Sequence[str], limit: int = 3) -> list[str]:
    """Pick a readable subset for image grids: the two endpoints plus one LoRA arm.

    Stacking six arms makes the panel unreadable, and the comparison that carries the
    argument is frozen (floor) vs. a LoRA arm vs. full (ceiling).
    """
    ordered = _order(arms)
    if len(ordered) <= limit:
        return ordered
    picked = [a for a in ("frozen",) if a in ordered]
    loras = [a for a in ordered if a.startswith("lora")]
    if loras:
        picked.append(loras[len(loras) // 2])   # a middle rank, not the extremes
    if "full" in ordered:
        picked.append("full")
    for a in ordered:
        if len(picked) >= limit:
            break
        if a not in picked:
            picked.append(a)
    return _order(picked)[:limit]


def figure_retrieval_grid(
    ground_truth,
    retrieved_by_arm: dict[str, np.ndarray],
    ranks_by_arm: dict[str, np.ndarray],
    out: Path,
    pool_size: int,
    n: int = 6,
) -> Path:
    """Nearest-neighbour images per arm, with correct hits outlined and ranks annotated.

    Deliberately loud about what it is: these are photographs retrieved from the test
    pool, not model output. A correct top-1 is pixel-identical to the ground-truth row
    above it, so without the outline and the caption this figure reads as a perfect
    reconstruction, which would badly overstate the result.
    """
    plt = _mpl()
    arms = _order(list(retrieved_by_arm))
    if not arms:
        return out
    k = retrieved_by_arm[arms[0]].shape[1]
    n = int(min(n, len(ground_truth), *(len(v) for v in retrieved_by_arm.values())))
    rows = 1 + len(arms) * k

    fig, axes = plt.subplots(rows, n, figsize=(1.35 * n, 1.45 * rows))
    axes = np.atleast_2d(axes)
    if axes.shape[0] != rows:
        axes = axes.reshape(rows, n)

    def _blank(ax, label=None):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if label is not None:
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7.5)

    for j in range(n):
        axes[0, j].imshow(np.transpose(np.asarray(ground_truth[j]), (1, 2, 0)).clip(0, 1))
        _blank(axes[0, j])
    _blank(axes[0, 0], "stimulus")

    r = 1
    for arm in arms:
        imgs = retrieved_by_arm[arm]
        ranks = np.asarray(ranks_by_arm.get(arm, np.full(n, -1)))
        for rank_slot in range(k):
            for j in range(n):
                ax = axes[r, j]
                ax.imshow(np.transpose(imgs[j, rank_slot], (1, 2, 0)).clip(0, 1))
                _blank(ax)
                correct = ranks[j] == rank_slot
                if correct:
                    for s in ax.spines.values():
                        s.set_visible(True)
                        s.set_color("#2e8b57")
                        s.set_linewidth(2.5)
                if rank_slot == 0:
                    ax.set_title(
                        f"true rank {int(ranks[j]) + 1}" if ranks[j] >= 0 else "",
                        fontsize=6.5, pad=2,
                        color="#2e8b57" if correct else "#666666",
                    )
            label = f"{arm}\ntop-{rank_slot + 1}" if k > 1 else arm
            axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7.5)
            r += 1

    fig.subplots_adjust(hspace=0.28, wspace=0.05, top=0.94)
    fig.suptitle(
        "RETRIEVED, NOT GENERATED — nearest test-set image to each predicted CLIP "
        f"embedding\n(pool = {pool_size} images; green outline = correct image retrieved "
        "at this rank)",
        y=0.985, fontsize=9,
    )
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------
def _table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, float):
        if abs(x) < 1e-3 and x != 0:
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def build_report(
    out_path: Path,
    cfg_summary: dict,
    per_arm: dict[str, dict[str, np.ndarray]],
    comparisons: list[dict],
    retention: list[dict],
    efficiency: list[dict],
    seed_rows: list[dict],
    figures: dict[str, Path],
    primary_metric: str = "two_way_clip",
) -> Path:
    arms = _order(list(per_arm))
    metrics = [m for m in ALL_METRICS if any(m in per_arm[a] for a in arms)]

    lines = ["# LoRA vs. full fine-tuning for MindEye2 subject adaptation", ""]
    lines += ["## Setup", ""]
    lines += [_table(["setting", "value"], [[k, v] for k, v in cfg_summary.items()]), ""]

    lines += ["## Headline result", ""]
    head = [r for r in retention if r["metric"] == primary_metric]
    if head:
        for r in sorted(head, key=lambda x: -x["retention"]):
            lines.append(
                f"- **{r['arm']}** recovers **{r['retention']*100:.1f}%** "
                f"(95% CI {r['ci_low']*100:.1f}–{r['ci_high']*100:.1f}%) of the improvement "
                f"that full fine-tuning achieves over the frozen shared model on "
                f"`{primary_metric}`."
            )
        lines.append("")
        lines.append(
            "> Retention is measured against the frozen-backbone baseline, so it isolates "
            "what adaptation of the *pretrained* weights buys. A value near 1.0 means the "
            "low-rank update captured essentially all of the useful adaptation."
        )
        lines.append("")

    lines += ["## Per-image scores", ""]
    rows = []
    for a in arms:
        row = [a]
        for m in metrics:
            v = per_arm[a].get(m)
            row.append(_fmt(float(np.mean(v))) if v is not None else "–")
        rows.append(row)
    lines += [_table(["arm"] + [m + (" ↑" if HIGHER_IS_BETTER.get(m, True) else " ↓")
                                for m in metrics], rows), ""]

    lines += ["## Paired comparison against full fine-tuning", ""]
    lines += [
        "Each arm is scored on the *same* test images as the full fine-tune, so the tests "
        "are paired. `p (Holm)` is corrected across the arms compared within a metric. "
        "`TOST` asks the opposite question from the significance test: is the difference "
        "credibly *smaller* than the equivalence margin (20% of the frozen→full headroom)?",
        "",
    ]
    crows = []
    for c in comparisons:
        crows.append([
            c["metric"], c["arm"], _fmt(c["mean_arm"]), _fmt(c["mean_reference"]),
            f"{_fmt(c['diff'])} [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]",
            _fmt(c["cohens_dz"], 3), _fmt(c["p_holm"]),
            "yes" if c["significant"] else "no",
            _fmt(c["tost_p"]), "yes" if c["equivalent"] else "no",
        ])
    lines += [_table(
        ["metric", "arm", "mean", "mean (full)", "Δ [95% CI]", "d_z", "p (Holm)",
         "differs?", "p (TOST)", "equivalent?"], crows), ""]

    if retention:
        lines += ["## Retention of the achievable gain", ""]
        rrows = [[r["metric"], r["arm"], _fmt(r["retention"], 3),
                  f"[{_fmt(r['ci_low'],3)}, {_fmt(r['ci_high'],3)}]",
                  _fmt(r["headroom"], 4)] for r in retention]
        lines += [_table(["metric", "arm", "retention", "95% CI", "frozen→full headroom"],
                         rrows), ""]

    if efficiency:
        lines += ["## Cost", ""]
        erows = []
        for e in sorted(efficiency, key=lambda x: _order([y["arm"] for y in efficiency]).index(x["arm"])):
            erows.append([
                e["arm"],
                f"{e['trainable_params']:,}",
                f"{100 * e['trainable_params'] / max(e['total_params'], 1):.3f}%",
                human_bytes(e["checkpoint_bytes"]),
                human_bytes(e["peak_memory_bytes"]),
                f"{e['seconds_per_epoch']:.1f}s",
            ])
        lines += [_table(
            ["arm", "trainable params", "share of model", "checkpoint", "peak VRAM",
             "sec/epoch"], erows), ""]

    if seed_rows:
        lines += ["## Seed variability", "",
                  "Between-seed standard deviation of each arm's mean score. Any "
                  "between-arm difference smaller than this is not worth interpreting.", ""]
        srows = [[r["arm"], r["metric"], _fmt(r["mean"]), _fmt(r["sd"]), r["n_seeds"]]
                 for r in seed_rows if r["metric"] in metrics]
        lines += [_table(["arm", "metric", "mean", "sd across seeds", "seeds"], srows), ""]

    if figures:
        lines += ["## Figures", ""]
        for name, path in figures.items():
            if not path or not Path(path).exists():
                continue
            lines.append(f"### {name}\n\n![{name}]({Path(path).name})\n")
            if "Retrieval" in name:
                lines += [
                    "> **These images were retrieved, not generated.** Each panel is the "
                    "real test-set photograph whose CLIP embedding is closest to the "
                    "model's prediction, so a correct top-1 is pixel-identical to the "
                    "stimulus above it — that is a retrieval hit, not a reconstruction. "
                    "Retrieval is also a coarse discriminator: it is a hit-or-miss "
                    "threshold, so two arms a few percent apart often produce identical "
                    "rows. Read the statistics tables, not this panel, for the size of "
                    "the difference between arms.",
                    "",
                ]

    lines += [
        "## How to read this",
        "",
        "- A significant `differs?` with a tiny `d_z` and a CI hugging zero means the "
        "difference is real but negligible; that is the expected outcome for a "
        "well-chosen LoRA rank on a test set of ~1,000 images.",
        "- `equivalent? = yes` is the positive claim. It says the gap is *bounded* below "
        "the margin, which a non-significant p-value alone never establishes.",
        "- If `frozen` already scores close to `full`, the shared-subject model was "
        "already nearly sufficient and neither LoRA nor full fine-tuning has much room "
        "to differ. Check the `frozen→full headroom` column before interpreting anything "
        "else.",
        "",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    log.info("report written -> %s", out_path)
    return out_path
