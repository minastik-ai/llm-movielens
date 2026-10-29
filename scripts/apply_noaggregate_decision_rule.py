#!/usr/bin/env python3
"""Apply the decision rule of the paper's no-aggregate control (section 6).

The control regenerates every profile with the one prompt line carrying the labelled
aggregates removed (PROFILE_PROMPT=no-aggregates in src/profile_generator) and trains
M4 and M7 on the result. The rule was fixed before those runs existed; its thresholds
are defined from the reference effect, so they follow the baselines:

  reference   Delta_agg   = NDCG@10(M4) - NDCG@10(M2), paired over five seeds
  primary     Delta_noagg = NDCG@10(M4, no-aggregate) - NDCG@10(M2), paired
  PASS    : Delta_noagg > 0, p < 0.05, Delta_noagg >= lower bound of Delta_agg's 95% CI
  PARTIAL : Delta_noagg > 0, p < 0.05, 40% of Delta_agg <= Delta_noagg < that bound
  FAIL    : otherwise

Secondary (an interval, not a null acceptance): the 95% CI of M4 - M4(no-aggregate)
and whether it excludes +/- Delta_agg. Tertiary, uncorrected: no-aggregate M4 against
M3 and M1, and no-aggregate M7 against M2.

Usage:
    python3 scripts/apply_noaggregate_decision_rule.py [--noagg-dir results_noaggregate]
                                                       [--out verdict.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO, code  # noqa: E402

SEEDS = [42, 123, 456, 789, 2026]
ENCODER = "bge-large-en-v1.5"
PARTIAL_FRACTION = 0.40


def ndcg(root: Path, cfg: str, seed: int) -> float:
    p = root / cfg / ENCODER / f"seed-{seed}" / "results.json"
    if not p.exists():
        raise SystemExit(f"missing per-seed result: {p}")
    return float(json.loads(p.read_text())["test_metrics"]["NDCG@10"])


def series(root: Path, cfg: str) -> list:
    return [ndcg(root, cfg, s) for s in SEEDS]


def paired(a: list, b: list) -> dict:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se
    tc = stats.t.ppf(0.975, n - 1)
    return {"delta": m, "sd": sd, "se": se, "t": t,
            "p": float(2 * stats.t.sf(abs(t), n - 1)), "ci": [m - tc * se, m + tc * se]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--noagg-dir", type=Path, default=REPO / "results_noaggregate",
                    help="per-seed results of the no-aggregate runs")
    ap.add_argument("--out", type=Path, default=None, help="also write the verdict as JSON")
    a = ap.parse_args()
    main_root = code("benchmark", "results")

    m1, m2, m3, m4 = (series(main_root, c) for c in ("m1", "m2", "m3", "m4"))
    m4_na, m7_na = series(a.noagg_dir, "m4"), series(a.noagg_dir, "m7")

    ref = paired(m4, m2)
    pass_min, partial_min = ref["ci"][0], PARTIAL_FRACTION * ref["delta"]
    prim = paired(m4_na, m2)
    if prim["delta"] > 0 and prim["p"] < 0.05 and prim["delta"] >= pass_min:
        verdict = "PASS"
    elif prim["delta"] > 0 and prim["p"] < 0.05 and prim["delta"] >= partial_min:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    sec = paired(m4, m4_na)
    sec["ci_excludes_full_effect"] = bool(-ref["delta"] < sec["ci"][0] and sec["ci"][1] < ref["delta"])
    tert = {"M4_noagg-M3": paired(m4_na, m3), "M4_noagg-M1": paired(m4_na, m1),
            "M7_noagg-M2": paired(m7_na, m2)}

    print(f"reference  Delta_agg   = {ref['delta']:+.5f}  95% CI [{ref['ci'][0]:+.5f}, {ref['ci'][1]:+.5f}]")
    print(f"thresholds PASS >= {pass_min:+.5f}   PARTIAL >= {partial_min:+.5f}")
    print(f"primary    Delta_noagg = {prim['delta']:+.5f}  t(4) = {prim['t']:.3f}  p = {prim['p']:.4f}  "
          f"95% CI [{prim['ci'][0]:+.5f}, {prim['ci'][1]:+.5f}]")
    print(f"           retention   = {prim['delta'] / ref['delta'] * 100:.1f}% of Delta_agg")
    print(f"VERDICT    {verdict}")
    print(f"secondary  M4 - M4_noagg = {sec['delta']:+.5f}  95% CI [{sec['ci'][0]:+.5f}, "
          f"{sec['ci'][1]:+.5f}]  excludes +/-Delta_agg: {sec['ci_excludes_full_effect']}")
    for k, r in tert.items():
        print(f"tertiary   {k:12s} {r['delta']:+.5f}  t(4) = {r['t']:.3f}  p = {r['p']:.4f}")

    if a.out:
        a.out.write_text(json.dumps({
            "seeds": SEEDS, "reference": ref,
            "thresholds": {"pass_min": pass_min, "partial_min": partial_min,
                           "partial_fraction": PARTIAL_FRACTION},
            "primary": dict(prim, retention_pct=prim["delta"] / ref["delta"] * 100),
            "verdict": verdict, "secondary": sec, "tertiary": tert,
            "per_seed": {"M1": m1, "M2": m2, "M3": m3, "M4": m4,
                         "M4_noagg": m4_na, "M7_noagg": m7_na},
        }, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
