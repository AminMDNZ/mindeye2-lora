"""Statistical comparison of adaptation strategies.

The analysis is deliberately built around three questions, because "LoRA scored 0.31 and
full fine-tuning scored 0.32" answers none of them:

1. **Is there a difference at all?** Paired tests on per-image scores (the arms see the
   same test images, so pairing is the correct design and is far more powerful than
   comparing two group means).
2. **How big is it, and how certain are we?** Bias-corrected accelerated (BCa) bootstrap
   confidence intervals on the paired mean difference, plus Cohen's d_z.
3. **Is LoRA *good enough*?** A null result from (1) is not evidence of equivalence. So
   we also run two one-sided tests (TOST) against an equivalence margin, and report a
   retention ratio: the fraction of full fine-tuning's improvement over the frozen
   baseline that LoRA recovers, with its own bootstrap interval.

Multiple metrics are compared, so p-values are Holm-corrected within each family.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps

from .metric_meta import REDUNDANT_METRICS
from .utils import log, progress


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------
def _bca_interval(
    boot: np.ndarray, theta_hat: float, jackknife: np.ndarray, alpha: float
) -> tuple[float, float]:
    prop = np.mean(boot < theta_hat)
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = sps.norm.ppf(prop)
    jk_mean = jackknife.mean()
    num = np.sum((jk_mean - jackknife) ** 3)
    den = 6.0 * (np.sum((jk_mean - jackknife) ** 2) ** 1.5) + 1e-12
    acc = num / den

    def adjust(z):
        return sps.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))

    lo_q = adjust(sps.norm.ppf(alpha / 2))
    hi_q = adjust(sps.norm.ppf(1 - alpha / 2))
    return float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q))


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
) -> dict:
    """Bootstrap the paired difference a - b (resampling image pairs, not images)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(d[idx], axis=1)
    theta = float(statistic(d))
    # Jackknife for the BCa acceleration term. Vectorised for the mean (the common
    # case): a Python loop over np.delete is O(n^2) copies and dominates runtime at
    # n=1000 across dozens of arm/metric pairs.
    jack = None
    if n <= 5000:
        if statistic is np.mean:
            total = d.sum()
            jack = (total - d) / (n - 1)
        else:
            jack = np.array([statistic(np.delete(d, i)) for i in range(n)])
    if jack is not None:
        lo, hi = _bca_interval(boot, theta, jack, alpha)
        method = "BCa"
    else:  # percentile fallback keeps very large test sets fast
        lo, hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
        method = "percentile"
    return {"diff": theta, "ci_low": lo, "ci_high": hi, "ci_method": method,
            "boot_std": float(boot.std(ddof=1)), "n": n}


# --------------------------------------------------------------------------------------
# hypothesis tests
# --------------------------------------------------------------------------------------
def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised paired effect size. A perfectly constant non-zero difference has an
    infinite standardised effect; report that honestly rather than collapsing it to 0."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    if sd > 0:
        return float(d.mean() / sd)
    return 0.0 if d.mean() == 0 else float(np.sign(d.mean()) * np.inf)


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict:
    # np.asarray(..., float) alone can pass through an exotic array type; ascontiguousarray
    # forces a plain float64 buffer so SciPy takes its NumPy path rather than dispatching
    # on whichever array namespace happens to be importable.
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    d = a - b
    t_stat, t_p = sps.ttest_rel(a, b)
    if not np.any(d):
        # identical arms: Wilcoxon is undefined, and the honest answer is "no evidence"
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_stat, w_p = sps.wilcoxon(a, b, alternative="two-sided")
        except Exception as exc:
            # A failure here must not take down the whole comparison stage; the paired
            # t-test and the bootstrap interval still carry the result.
            log.warning("Wilcoxon failed (%s); reporting p=nan for this pair", exc)
            w_stat, w_p = float("nan"), float("nan")
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohens_dz": cohens_dz(a, b),
        "n": int(len(d)),
    }


def tost_equivalence(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests: is |mean(a) - mean(b)| credibly smaller than `margin`?

    A significant TOST result is positive evidence *for* practical equivalence, unlike a
    non-significant difference test which is merely an absence of evidence.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
    margin = abs(float(margin))
    if se == 0 or not np.isfinite(se):
        equivalent = abs(d.mean()) < margin
        return {"margin": margin, "p_lower": 0.0 if equivalent else 1.0,
                "p_upper": 0.0 if equivalent else 1.0,
                "p_tost": 0.0 if equivalent else 1.0, "equivalent": bool(equivalent), "n": n}
    df = n - 1
    t_lower = (d.mean() + margin) / se        # H0: diff <= -margin
    t_upper = (d.mean() - margin) / se        # H0: diff >= +margin
    p_lower = float(sps.t.sf(t_lower, df))
    p_upper = float(sps.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    return {"margin": margin, "p_lower": p_lower, "p_upper": p_upper, "p_tost": p_tost,
            "equivalent": bool(p_tost < alpha), "n": n}


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> dict:
    """Step-down Holm correction: strong FWER control without Bonferroni's conservatism."""
    p = np.asarray(pvalues, float)
    p = np.where(np.isfinite(p), p, 1.0)   # a failed test is not a significant one
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return {"p_adjusted": adjusted, "reject": adjusted < alpha, "alpha": alpha, "n_tests": m}


# --------------------------------------------------------------------------------------
# retention ratio: the headline number
# --------------------------------------------------------------------------------------
def retention_ratio(
    method: np.ndarray,
    full: np.ndarray,
    frozen: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    higher_is_better: bool = True,
) -> dict:
    """(method - frozen) / (full - frozen): how much of the achievable gain LoRA keeps.

    1.0 means LoRA matches full fine-tuning; 0.0 means it did nothing beyond training the
    subject-specific ridge layer. Reported with a bootstrap interval because it is a
    ratio of two noisy differences and its uncertainty is easy to underestimate.
    """
    m, f, z = (np.asarray(x, float) for x in (method, full, frozen))
    sign = 1.0 if higher_is_better else -1.0
    denom = sign * (f.mean() - z.mean())
    numer = sign * (m.mean() - z.mean())
    point = float(numer / denom) if abs(denom) > 1e-12 else float("nan")

    rng = np.random.default_rng(seed)
    n = len(m)
    idx = rng.integers(0, n, size=(n_boot, n))
    d_num = sign * (m[idx].mean(1) - z[idx].mean(1))
    d_den = sign * (f[idx].mean(1) - z[idx].mean(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = d_num / d_den
    ratios = ratios[np.isfinite(ratios)]
    lo, hi = (float(np.quantile(ratios, alpha / 2)), float(np.quantile(ratios, 1 - alpha / 2))) \
        if len(ratios) else (float("nan"), float("nan"))
    return {
        "retention": point,
        "ci_low": lo,
        "ci_high": hi,
        "headroom": float(denom),
        "gain": float(numer),
        "n_boot": int(len(ratios)),
    }


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
@dataclass
class Comparison:
    metric: str
    arm: str
    reference: str
    mean_arm: float
    mean_reference: float
    diff: float
    ci_low: float
    ci_high: float
    cohens_dz: float
    t_p: float
    wilcoxon_p: float
    p_holm: float = float("nan")
    significant: bool = False
    tost_margin: float = float("nan")
    tost_p: float = float("nan")
    equivalent: bool = False
    n: int = 0


def compare_against_reference(
    per_arm: dict[str, dict[str, np.ndarray]],
    reference: str,
    metrics: Sequence[str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
    equivalence_fraction: float = 0.2,
    frozen_arm: str | None = "frozen",
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[Comparison]:
    """Compare every arm to `reference` (normally the full fine-tune) on every metric.

    The equivalence margin is derived from the data rather than picked arbitrarily: it is
    `equivalence_fraction` of the headroom that full fine-tuning creates over the frozen
    baseline. Declaring "LoRA is equivalent" then means "LoRA is within 20% of the gain
    that full fine-tuning was able to buy", which is a statement about practical
    relevance rather than about raw metric units.
    """
    higher_is_better = higher_is_better or {}
    ref = per_arm[reference]
    metrics = list(metrics or ref.keys())

    # Drop metrics that are algebraically the same as one already present. Reporting
    # both makes a single result look like two independent confirmations.
    dropped = [m for m in metrics
               if m in REDUNDANT_METRICS and REDUNDANT_METRICS[m] in metrics]
    if dropped:
        log.info("omitting %s: identical to %s", ", ".join(dropped),
                 ", ".join(REDUNDANT_METRICS[m] for m in dropped))
        metrics = [m for m in metrics if m not in dropped]

    results: list[Comparison] = []

    bar = progress(total=len(metrics), desc="paired statistics", unit="metric")
    for metric in metrics:
        bar.update(1)
        if metric not in ref:
            continue
        b = np.asarray(ref[metric], float)
        margin = float("nan")
        if frozen_arm and frozen_arm in per_arm and metric in per_arm[frozen_arm]:
            headroom = abs(b.mean() - np.asarray(per_arm[frozen_arm][metric], float).mean())
            margin = equivalence_fraction * headroom

        block: list[Comparison] = []
        for arm, metrics_dict in per_arm.items():
            if arm == reference or metric not in metrics_dict:
                continue
            a = np.asarray(metrics_dict[metric], float)
            if a.shape != b.shape:
                continue
            boot = paired_bootstrap(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
            tests = paired_tests(a, b)
            tost = tost_equivalence(a, b, margin, alpha) if np.isfinite(margin) else None
            block.append(
                Comparison(
                    metric=metric, arm=arm, reference=reference,
                    mean_arm=tests["mean_a"], mean_reference=tests["mean_b"],
                    diff=boot["diff"], ci_low=boot["ci_low"], ci_high=boot["ci_high"],
                    cohens_dz=tests["cohens_dz"], t_p=tests["t_p"],
                    wilcoxon_p=tests["wilcoxon_p"], n=tests["n"],
                    tost_margin=margin,
                    tost_p=tost["p_tost"] if tost else float("nan"),
                    equivalent=bool(tost["equivalent"]) if tost else False,
                )
            )
        if block:
            holm = holm_bonferroni([c.wilcoxon_p for c in block], alpha=alpha)
            for c, padj, rej in zip(block, holm["p_adjusted"], holm["reject"]):
                c.p_holm = float(padj)
                c.significant = bool(rej)
        results.extend(block)
    bar.close()
    return results


def retention_table(
    per_arm: dict[str, dict[str, np.ndarray]],
    full_arm: str = "full",
    frozen_arm: str = "frozen",
    metrics: Sequence[str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict]:
    higher_is_better = higher_is_better or {}
    rows = []
    metrics = list(metrics or per_arm[full_arm].keys())
    bar = progress(total=len(metrics), desc="retention ratios", unit="metric")
    for metric in metrics:
        bar.update(1)
        if metric not in per_arm.get(full_arm, {}) or metric not in per_arm.get(frozen_arm, {}):
            continue
        for arm, md in per_arm.items():
            if arm in (full_arm, frozen_arm) or metric not in md:
                continue
            r = retention_ratio(
                md[metric], per_arm[full_arm][metric], per_arm[frozen_arm][metric],
                n_boot=n_boot, seed=seed,
                higher_is_better=higher_is_better.get(metric, True),
            )
            rows.append({"metric": metric, "arm": arm, **r})
    bar.close()
    return rows


def seed_variability(per_seed: dict[str, dict[int, dict[str, np.ndarray]]]) -> list[dict]:
    """Between-seed spread of each arm's mean score — the noise floor for comparisons."""
    rows = []
    for arm, seeds in per_seed.items():
        metrics = set().union(*[set(v.keys()) for v in seeds.values()]) if seeds else set()
        for metric in sorted(metrics):
            means = [float(np.mean(seeds[s][metric])) for s in sorted(seeds) if metric in seeds[s]]
            if len(means) < 1:
                continue
            rows.append({
                "arm": arm, "metric": metric, "n_seeds": len(means),
                "mean": float(np.mean(means)),
                "sd": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                "min": float(np.min(means)), "max": float(np.max(means)),
            })
    return rows


def comparisons_to_records(comparisons: Sequence[Comparison]) -> list[dict]:
    return [asdict(c) for c in comparisons]
