"""Tests for the statistical comparison layer.

These run without torch, so they can be used to check the analysis logic anywhere.
"""
import numpy as np

from mindeye_lora.stats import (
    cohens_dz,
    compare_against_reference,
    holm_bonferroni,
    paired_bootstrap,
    paired_tests,
    retention_ratio,
    seed_variability,
    tost_equivalence,
)


def _fixture(seed=0, n=800):
    rng = np.random.default_rng(seed)
    frozen = rng.normal(0.20, 0.05, n)
    full = frozen + rng.normal(0.10, 0.03, n)
    near = frozen + rng.normal(0.095, 0.03, n)     # ~95% of the achievable gain
    weak = frozen + rng.normal(0.02, 0.03, n)      # ~20%
    return {"frozen": {"m": frozen}, "full": {"m": full},
            "lora": {"m": near}, "bitfit": {"m": weak}}


def test_bootstrap_ci_brackets_the_point_estimate():
    per = _fixture()
    out = paired_bootstrap(per["lora"]["m"], per["full"]["m"], n_boot=2000)
    assert out["ci_low"] < out["diff"] < out["ci_high"]
    assert out["ci_method"] in {"BCa", "percentile"}


def test_paired_tests_detect_a_real_gap():
    per = _fixture()
    res = paired_tests(per["frozen"]["m"], per["full"]["m"])
    assert res["mean_diff"] < 0
    assert res["t_p"] < 1e-10
    assert res["cohens_dz"] < -1


def test_cohens_dz_sign():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 200)
    assert cohens_dz(a + rng.normal(1.0, 0.5, 200), a) > 0
    assert cohens_dz(a - rng.normal(1.0, 0.5, 200), a) < 0
    # a (near-)constant non-zero difference is an enormous standardised effect, not zero
    assert cohens_dz(a + 1.0, a) > 1e6
    assert cohens_dz(np.ones(5) + 1.0, np.ones(5)) == np.inf
    assert cohens_dz(a, a) == 0.0


def test_holm_is_monotone_and_conservative():
    raw = [0.001, 0.02, 0.04, 0.5]
    out = holm_bonferroni(raw)
    adj = out["p_adjusted"]
    assert np.all(adj >= np.array(raw) - 1e-12)
    assert np.all(np.diff(adj[np.argsort(raw)]) >= -1e-12)
    assert out["reject"][0] and not out["reject"][-1]


def test_tost_declares_equivalence_only_when_close():
    per = _fixture()
    close = tost_equivalence(per["lora"]["m"], per["full"]["m"], margin=0.02)
    far = tost_equivalence(per["bitfit"]["m"], per["full"]["m"], margin=0.02)
    assert close["equivalent"]
    assert not far["equivalent"]


def test_retention_recovers_the_planted_ratio():
    per = _fixture()
    r = retention_ratio(per["lora"]["m"], per["full"]["m"], per["frozen"]["m"], n_boot=2000)
    assert 0.85 < r["retention"] < 1.05
    assert r["ci_low"] < r["retention"] < r["ci_high"]

    weak = retention_ratio(per["bitfit"]["m"], per["full"]["m"], per["frozen"]["m"], n_boot=2000)
    assert weak["retention"] < r["retention"]


def test_compare_against_reference_shapes():
    per = _fixture()
    cmps = compare_against_reference(per, reference="full", n_boot=1000)
    arms = {c.arm for c in cmps}
    assert arms == {"frozen", "lora", "bitfit"}
    assert all(np.isfinite(c.p_holm) for c in cmps)
    lora = next(c for c in cmps if c.arm == "lora")
    frozen = next(c for c in cmps if c.arm == "frozen")
    assert abs(lora.diff) < abs(frozen.diff)
    assert lora.equivalent and not frozen.equivalent


def test_seed_variability():
    rng = np.random.default_rng(1)
    per_seed = {"lora": {s: {"m": rng.normal(0.3, 0.05, 100)} for s in range(3)}}
    rows = seed_variability(per_seed)
    assert rows[0]["n_seeds"] == 3
    assert rows[0]["sd"] >= 0
