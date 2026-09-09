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
# Metrics with a known chance level, used to frame the axis honestly when zooming in.
CHANCE_LEVEL = {"two_way_clip": 0.5, "retrieval_percentile": 0.5, "cosine": 0.0}


def figure_metric_bars(per_arm: dict[str, dict[str, np.ndarray]], metrics: Sequence[str],
                       out: Path) -> Path:
    """Per-arm means with 95% CIs.

    The y-axis is zoomed to the data rather than anchored at zero. Anchoring at zero is
    the usual advice for bar charts, but here the arms differ by ~0.02 on a metric whose
    floor is chance (0.5), so a 0-1 axis renders three visually identical bars and hides
    the entire result. To keep that honest the axis is broken explicitly: bars start at
    the zoom floor, and a caption states the chance level so nobody reads bar height as
    proportional to performance.
    """
    plt = _mpl()
    arms = _order(list(per_arm))
    metrics = [m for m in metrics if any(m in per_arm[a] for a in arms)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.4))
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
        if not means:
            continue
        colors = ["#b0b0b0" if l == "frozen" else "#2f6fb2" if l.startswith("lora")
                  else "#c1440e" if l == "full" else "#7a9e5e" for l in labels]

        lo = min(m - e for m, e in zip(means, errs))
        hi = max(m + e for m, e in zip(means, errs))
        span = max(hi - lo, 1e-9)
        floor, ceil = lo - 0.6 * span, hi + 0.3 * span
        chance = CHANCE_LEVEL.get(metric)
        if chance is not None and chance > floor:
            floor = min(floor, chance - 0.1 * span)   # keep chance visible if it is close

        ax.bar(range(len(means)), [m - floor for m in means], bottom=floor,
               yerr=errs, capsize=3, color=colors)
        if chance is not None and floor < chance < ceil:
            ax.axhline(chance, color="#555", ls="--", lw=1)
            ax.text(len(means) - 0.5, chance, " chance", va="bottom", ha="right",
                    fontsize=7, color="#555")
        ax.set_ylim(floor, ceil)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(metric + ("  ↑" if HIGHER_IS_BETTER.get(metric, True) else "  ↓"))

    fig.suptitle("Per-image mean ± 95% CI on the held-out NSD test images\n"
                 "y-axis is zoomed to the data — bar height is not proportional to score",
                 y=1.02, fontsize=9)
    fig.tight_layout()
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


def _embed_png(path: Path) -> str | None:
    """Inline a PNG as a base64 data URI.

    Relative image links only resolve if the viewer's working directory happens to be
    the report's folder — which is not the case for `display(Markdown(...))` in a
    notebook, nor for the file viewed from anywhere else. Embedding makes REPORT.md
    self-contained and portable at the cost of ~33% size overhead.
    """
    import base64

    try:
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except Exception as exc:  # pragma: no cover
        log.warning("could not embed %s (%s)", path, exc)
        return None
    return f"data:image/png;base64,{data}"


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
    embed_figures: bool = True,
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
            src = _embed_png(path) if embed_figures else Path(path).name
            lines.append(f"### {name}\n\n![{name}]({src or Path(path).name})\n")
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
    markdown = "\n".join(lines)
    out_path.write_text(markdown)
    log.info("report written -> %s", out_path)

    # A self-contained HTML twin: opens in any browser straight off Drive, with the
    # tables and embedded figures intact, and needs no markdown renderer.
    try:
        html_path = out_path.with_suffix(".html")
        html_path.write_text(_markdown_to_html(markdown))
        log.info("html report  -> %s", html_path)
    except Exception as exc:  # pragma: no cover
        log.warning("could not write HTML report (%s)", exc)
    return out_path


def _markdown_to_html(md: str) -> str:
    """Minimal markdown -> HTML. Handles what build_report actually emits: headings,
    tables, images, blockquotes, bullets and paragraphs. No dependency required."""
    import html as _html
    import re

    out, in_table = [], False
    for raw in md.split("\n"):
        line = raw.rstrip()
        img = re.match(r"!\[(.*?)\]\((.*?)\)$", line.strip())
        if img:
            if in_table:
                out.append("</table>"); in_table = False
            out.append(f'<img alt="{_html.escape(img.group(1))}" src="{img.group(2)}">')
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells) and in_table:
                continue                       # separator row
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>"); in_table = True
            row = "".join(f"<{tag}>{_html.escape(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{_html.escape(line.lstrip('# ').strip())}</h{level}>")
        elif line.startswith(">"):
            out.append(f"<blockquote>{_html.escape(line.lstrip('> '))}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<li>{_html.escape(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{_html.escape(line)}</p>")
    if in_table:
        out.append("</table>")
    style = (
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:1000px;"
        "margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}"
        "table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}"
        "th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}"
        "th{background:#f5f5f5}img{max-width:100%;margin:1rem 0}"
        "blockquote{border-left:3px solid #ccc;margin:1rem 0;padding:.2rem 1rem;color:#555}"
        "h1,h2,h3{margin-top:1.6rem}"
    )
    return f"<!doctype html><meta charset='utf-8'><style>{style}</style>\n" + "\n".join(out)
