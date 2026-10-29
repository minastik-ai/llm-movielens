"""The paper's comparison of M4 with the purely collaborative and content baselines.

  * Confirmatory metric: test NDCG@10 (every other metric is descriptive: no correction, no claim).
  * Every comparison is M4 against a baseline, a paired t-test over the five seeds (4 degrees of
    freedom), two-sided, with a 95% CI of the difference.
  * The pure-CF comparison is an intersection-union test: M4 against each of M1 (LightGCN), M1b
    (SimGCL), M1c (XSimGCL) and M1d (LightGCL). It is rejected only if all four differences are
    positive and all four are rejected; its p-value is the largest of the four (1 if any difference
    is not positive).
  * Holm at alpha 0.05 over the four confirmatory hypotheses: the intersection-union test, M4 vs M2,
    M4 vs M2b, M4 vs M3.
  * Sensitivity: each pure-CF comparison is repeated without seed 42, the seed at which the
    contrastive baselines' settings were selected (four paired seeds, 3 degrees of freedom). It is
    judged on the sign and the 95% CI of the difference, not on significance.

Inputs: the per-seed results in results/<config>/bge-large-en-v1.5/seed-<s>/results.json for M1,
M1b, M1c, M1d, M2, M2b, M3 and M4.

    python3 scripts/analyze_contrastive.py [--out contrastive_analysis]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import code  # noqa: E402
ENC = "bge-large-en-v1.5"
SEEDS = (42, 123, 456, 789, 2026)
ALPHA = 0.05
RESULTS = code("benchmark", "results")
PURE_CF = ("m1", "m1b", "m1c", "m1d")
OTHER_CONFIRMATORY = ("m2", "m2b", "m3")
NAMES = {"m1": "M1 LightGCN", "m1b": "M1b SimGCL", "m1c": "M1c XSimGCL", "m1d": "M1d LightGCL",
         "m2": "M2 genome PCA", "m2b": "M2b genome raw", "m3": "M3 title", "m4": "M4 LLM profile"}
METRICS = ("NDCG@10", "Recall@10", "MRR")


def result_path(cfg: str, seed: int, root: Path | None = None) -> Path:
    return (root or RESULTS) / cfg / ENC / f"seed-{seed}" / "results.json"


def load_runs(cfg: str, root: Path | None = None) -> dict[int, dict]:
    """{seed: test_metrics} for one configuration; a missing file is an error, never a skipped seed."""
    out = {}
    for s in SEEDS:
        p = result_path(cfg, s, root)
        if not p.is_file():
            raise FileNotFoundError(f"{cfg} seed {s}: {p} is missing")
        out[s] = json.loads(p.read_text())["test_metrics"]
    return out


def paired(a: list[float], b: list[float]) -> dict:
    """Paired two-sided t-test of a - b: mean difference, t, df, p and the 95% CI."""
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("paired samples must have the same length, at least 3")
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        t, p = (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0), (0.0 if mean else 1.0)
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df))
    crit = float(stats.t.ppf(1 - ALPHA / 2, df))
    return {"n": n, "df": df, "delta": mean, "sd": sd, "t": t, "p": p,
            "ci95": (mean - crit * se, mean + crit * se)}


def intersection_union(comparisons: dict[str, dict]) -> dict:
    """Rejected only if every difference is positive and every comparison is rejected at alpha.

    The p-value is the largest of the individual p-values; it is 1 when any difference is not positive.
    """
    all_positive = all(c["delta"] > 0 for c in comparisons.values())
    p = max(c["p"] for c in comparisons.values()) if all_positive else 1.0
    binding = max(comparisons, key=lambda k: comparisons[k]["p"])
    return {"p": p, "all_positive": all_positive, "binding": binding,
            "rejected_at_alpha": all_positive and p < ALPHA}


def holm(pvals: dict[str, float], m: int | None = None) -> dict[str, float]:
    """Holm step-down adjusted p-values; with m larger than len(pvals) the missing hypotheses cannot be rejected."""
    m = m or len(pvals)
    if m < len(pvals):
        raise ValueError("m is smaller than the number of p-values")
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def means(runs: dict[int, dict], metric: str, seeds=SEEDS) -> list[float]:
    return [runs[s][metric] for s in seeds]


def analyse() -> dict:
    runs = {c: load_runs(c) for c in ("m4",) + PURE_CF + OTHER_CONFIRMATORY}
    users = {r["n_eval_users"] for c in runs.values() for r in c.values()}
    if len(users) != 1:
        raise ValueError(f"the configurations were evaluated on different user sets: {sorted(users)}")
    ndcg = {c: means(r, "NDCG@10") for c, r in runs.items()}
    full = {c: paired(ndcg["m4"], ndcg[c]) for c in PURE_CF + OTHER_CONFIRMATORY}
    without42 = tuple(s for s in SEEDS if s != 42)
    sens = {c: paired(means(runs["m4"], "NDCG@10", without42), means(runs[c], "NDCG@10", without42))
            for c in PURE_CF}
    iut = intersection_union({c: full[c] for c in PURE_CF})
    family = {"IUT pure-CF": iut["p"], **{f"M4 vs {NAMES[c]}": full[c]["p"] for c in OTHER_CONFIRMATORY}}
    descr = {c: {m: {"mean": sum(means(r, m)) / 5,
                     "sd": math.sqrt(sum((x - sum(means(r, m)) / 5) ** 2 for x in means(r, m)) / 4)}
                 for m in METRICS} for c, r in runs.items()}
    return {"n_eval_users": users.pop(), "per_seed_ndcg10": ndcg, "full": full, "iut": iut,
            "sensitivity_without_seed42": sens,
            "family": family, "holm": holm(family, len(family)), "descriptive": descr}


def render(a: dict) -> str:
    L = ["# M4 against the collaborative and content baselines", "",
         f"Test NDCG@10, {a['n_eval_users']:.0f} evaluated users, five seeds per configuration.", "",
         "## Descriptive (mean, sd over five seeds)", "",
         "| Configuration | NDCG@10 | Recall@10 | MRR |", "|---|---|---|---|"]
    for c in ("m1", "m1b", "m1c", "m1d", "m2", "m2b", "m3", "m4"):
        d = a["descriptive"][c]
        L.append(f"| {NAMES[c]} | " + " | ".join(f"{d[m]['mean']:.4f} ({d[m]['sd']:.4f})" for m in METRICS) + " |")
    L += ["", "## M4 against each baseline, NDCG@10, paired over five seeds (4 df, two-sided)", "",
          "| Comparison | Delta | 95% CI | t | p |", "|---|---|---|---|---|"]
    for c, r in a["full"].items():
        L.append(f"| M4 - {NAMES[c]} | {r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                 f"| {r['t']:.2f} | {r['p']:.4f} |")
    i = a["iut"]
    L += ["", f"**Intersection-union test (M4 vs M1, M1b, M1c, M1d):** every difference positive: "
          f"{i['all_positive']}; p = {i['p']:.4f}; binding comparison: {NAMES[i['binding']]}; "
          f"rejected at 0.05 unadjusted: {i['rejected_at_alpha']}.", "",
          "## Holm (alpha 0.05) over the four confirmatory hypotheses", "",
          "| Hypothesis | p | Holm-adjusted p |", "|---|---|---|"]
    for k, p in a["family"].items():
        L.append(f"| {k} | {p:.4f} | {a['holm'][k]:.4f} |")
    adj = a["holm"]
    L += ["", "All four rejected." if all(v < ALPHA for v in adj.values())
          else "Not rejected: " + ", ".join(k for k, v in adj.items() if v >= ALPHA) + "."]
    L += ["", "## Sensitivity without seed 42 (4 paired seeds, 3 df); judged on sign and CI, not significance", "",
          "| Comparison | Delta | 95% CI | same sign as five-seed | CI excludes 0 |", "|---|---|---|---|---|"]
    for c, r in a["sensitivity_without_seed42"].items():
        lo, hi = r["ci95"]
        same = (r["delta"] > 0) == (a["full"][c]["delta"] > 0)
        L.append(f"| M4 - {NAMES[c]} | {r['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {same} | {lo > 0 or hi < 0} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("contrastive_analysis"))
    out = ap.parse_args().out
    a = analyse()
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(a, indent=2, default=str) + "\n")
    text = render(a)
    (out / "analysis.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
