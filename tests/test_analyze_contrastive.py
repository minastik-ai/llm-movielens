"""scripts/analyze_contrastive.py: these tests pin each rule its docstring states.

The statistics are checked against scipy and against hand-worked cases; the real-data test reads
the shipped per-seed results and pins the five-seed means of the paper's main table.
"""
from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import pytest
from scipy import stats

HERE = Path(__file__).resolve().parents[1]


def _mod():
    p = HERE / "scripts" / "analyze_contrastive.py"
    if not p.exists():
        pytest.skip("scripts/analyze_contrastive.py not present")
    spec = importlib.util.spec_from_file_location("_analyze_contrastive", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod()


def test_paired_matches_scipy_and_uses_four_degrees_of_freedom_for_five_seeds():
    rng = random.Random(7)
    a = [0.117 + rng.gauss(0, 0.001) for _ in range(5)]
    b = [0.114 + rng.gauss(0, 0.001) for _ in range(5)]
    r = A.paired(a, b)
    ref = stats.ttest_rel(a, b)
    assert r["df"] == 4 and r["n"] == 5
    assert r["t"] == pytest.approx(ref.statistic, rel=1e-9)
    assert r["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    lo, hi = r["ci95"]
    assert lo < r["delta"] < hi
    assert (hi - lo) / 2 == pytest.approx(stats.t.ppf(0.975, 4) * r["sd"] / math.sqrt(5), rel=1e-9)


def test_paired_difference_has_the_sign_of_a_minus_b():
    assert A.paired([2, 3, 4, 5, 6.5], [1, 1.2, 2, 2.1, 3])["delta"] > 0
    assert A.paired([1, 1.2, 2, 2.1, 3], [2, 3, 4, 5, 6.5])["delta"] < 0


def test_critical_value_for_four_degrees_of_freedom_gives_p_of_five_percent():
    # the textbook value: t = 2.7764 at df 4 is p = 0.05 two-sided
    d = [1.0, 0.0, 1.0, 0.0, 1.0]            # not used; build diffs with a known t instead
    base = [0.0, 0.0, 0.0, 0.0, 0.0]
    n, target_t = 5, 2.7764451
    sd = 1.0
    # shift a constant so that mean/(sd/sqrt(n)) = target_t with the sample sd fixed at 1
    xs = [-1.2649111, -0.6324555, 0.0, 0.6324555, 1.2649111]    # mean 0, sample sd 1
    mean = target_t / math.sqrt(n) * sd
    diffs = [x + mean for x in xs]
    r = A.paired(diffs, base)
    assert r["t"] == pytest.approx(target_t, abs=1e-4)
    assert r["p"] == pytest.approx(0.05, abs=1e-4)


def test_intersection_union_p_is_the_largest_and_needs_every_difference_positive():
    ok = {"m1": {"delta": 0.003, "p": 0.004}, "m1b": {"delta": 0.009, "p": 0.0001},
          "m1c": {"delta": 0.006, "p": 0.001}, "m1d": {"delta": 0.028, "p": 0.00001}}
    r = A.intersection_union(ok)
    assert r["p"] == 0.004 and r["binding"] == "m1" and r["rejected_at_alpha"] is True
    # one comparison not rejected: the test is not rejected and its p is that largest p
    weak = {**ok, "m1": {"delta": 0.001, "p": 0.2}}
    assert A.intersection_union(weak)["p"] == 0.2
    assert A.intersection_union(weak)["rejected_at_alpha"] is False
    # a baseline that beats M4 (negative difference) can never be rejected, however small its p
    beaten = {**ok, "m1c": {"delta": -0.004, "p": 0.00001}}
    r = A.intersection_union(beaten)
    assert r["p"] == 1.0 and r["all_positive"] is False and r["rejected_at_alpha"] is False


def test_holm_hand_worked_example():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adj = A.holm(p, 4)
    # sorted 0.005, 0.01, 0.03, 0.04 -> 4*0.005, 3*0.01, 2*0.03, 1*0.04 = 0.02, 0.03, 0.06, 0.04 -> monotone
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})


def test_holm_with_a_fifth_hypothesis_that_cannot_be_rejected_is_stricter():
    p = {"a": 0.004, "b": 0.001, "c": 0.003, "d": 0.002}
    a4, a5 = A.holm(p, 4), A.holm(p, 5)
    assert a5 == pytest.approx({"b": 0.005, "d": 0.008, "c": 0.009, "a": 0.009})
    assert all(a5[k] >= a4[k] for k in p) and any(a5[k] > a4[k] for k in p)
    with pytest.raises(ValueError):
        A.holm(p, 3)


def test_sensitivity_drops_seed_42_and_has_three_degrees_of_freedom(tmp_path, monkeypatch):
    seeds = A.SEEDS
    assert 42 in seeds and len(seeds) == 5
    rng = random.Random(3)

    def fake(cfg, root=None):
        base = {"m4": 0.1173, "m1": 0.1139, "m1b": 0.1078, "m1c": 0.1108, "m1d": 0.0888,
                "m2": 0.1145, "m2b": 0.1140, "m3": 0.1136}[cfg]
        return {s: {"NDCG@10": base + 0.0005 * (s % 7) / 7 + rng.gauss(0, 0.0003), "Recall@10": 0.05,
                    "MRR": 0.23, "n_eval_users": 2643.0} for s in A.SEEDS}

    monkeypatch.setattr(A, "load_runs", fake)
    out = A.analyse()
    assert set(out["sensitivity_without_seed42"]) == set(A.PURE_CF)
    assert all(r["df"] == 3 and r["n"] == 4 for r in out["sensitivity_without_seed42"].values())
    assert all(r["df"] == 4 for r in out["full"].values())
    assert set(out["family"]) == {"IUT pure-CF", "M4 vs M2 genome PCA", "M4 vs M2b genome raw", "M4 vs M3 title"}
    text = A.render(out)
    assert "Holm" in text and "four confirmatory hypotheses" in text and "holm_5" not in out
    assert set(out["holm"]) == set(out["family"]) and len(out["holm"]) == 4
    assert "5 hypotheses" not in text and "Holm-adjusted, 5" not in text


def test_a_missing_seed_is_an_error_not_a_skip(tmp_path):
    with pytest.raises(FileNotFoundError):
        A.load_runs("m1b", root=tmp_path)


def test_different_user_sets_are_refused(monkeypatch):
    def fake(cfg, root=None):
        return {s: {"NDCG@10": 0.1, "Recall@10": 0.05, "MRR": 0.2,
                    "n_eval_users": 983.0 if (cfg == "m3" and s == 42) else 2643.0} for s in A.SEEDS}
    monkeypatch.setattr(A, "load_runs", fake)
    with pytest.raises(ValueError):
        A.analyse()


@pytest.mark.skipif(not (A.RESULTS / "m4").is_dir(), reason="per-seed results not present")
def test_real_data_five_seed_means_equal_the_paper_table():
    out = A.analyse()
    mean = lambda c: sum(out["per_seed_ndcg10"][c]) / 5
    assert mean("m1b") == pytest.approx(0.1078, abs=5e-5)
    assert mean("m1c") == pytest.approx(0.1108, abs=5e-5)
    assert mean("m1d") == pytest.approx(0.0888, abs=5e-5)
    assert mean("m1") == pytest.approx(0.1138, abs=5e-5)
    assert mean("m4") == pytest.approx(0.1175, abs=5e-5)
    assert out["n_eval_users"] == 2643
