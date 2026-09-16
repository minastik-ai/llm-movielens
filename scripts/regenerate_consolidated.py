#!/usr/bin/env python3
"""Regenerate human_eval/consolidated/ from the filled annotator sheets.

Written 2026-08-29, after the GPT-4o-mini ratings were corrected. Emits the two
files that depend on those ratings, with their existing schemas:

  per_annotator_ratings.csv   long form, one row per (item, annotator, source, axis)
  summary_table5.csv          per (source, axis): mean, std, Fleiss kappa,
                              linearly-weighted kappa, >=4 rate

The mood files are NOT regenerated: the mood sheets were not part of the
correction.

Validate before trusting: `--sheets _old` must reproduce the superseded
consolidated files exactly. That is the negative test for this generator.
"""
import argparse, csv, itertools, statistics as st
from collections import Counter
from pathlib import Path

HE = Path(__file__).resolve().parents[1] / "human_eval"
AXES = ["thematic_accuracy", "discriminativeness", "rule_compliance",
        "factual_consistency", "coherence_fluency"]
SOURCES = ["claude", "gpt4o"]
RATERS = ["A1", "A2", "A3"]
CATS = [1, 2, 3, 4, 5]


def load(suffix=""):
    return {a: list(csv.DictReader(open(HE / f"annotator_sheet_filled_{a}{suffix}.csv")))
            for a in RATERS}


def fleiss(sheets, src, axis):
    n = len(RATERS)
    N = len(sheets["A1"])
    M = [[Counter(int(sheets[a][i][f"{src}_{axis}"]) for a in RATERS).get(j, 0)
          for j in CATS] for i in range(N)]
    Pbar = sum((sum(x * x for x in r) - n) / (n * (n - 1)) for r in M) / N
    pj = [sum(r[j] for r in M) / (N * n) for j in range(len(CATS))]
    Pe = sum(p * p for p in pj)
    return (Pbar - Pe) / (1 - Pe)


def weighted_kappa(sheets, src, axis):
    """Linearly-weighted Cohen's kappa, averaged over the three rater pairs."""
    N = len(sheets["A1"])
    w = lambda p, q: 1 - abs(p - q) / (len(CATS) - 1)
    out = []
    for a, b in itertools.combinations(RATERS, 2):
        x = [int(sheets[a][i][f"{src}_{axis}"]) for i in range(N)]
        y = [int(sheets[b][i][f"{src}_{axis}"]) for i in range(N)]
        Po = sum(w(p, q) for p, q in zip(x, y)) / N
        cx, cy = Counter(x), Counter(y)
        Pe = sum(w(p, q) * cx[p] / N * cy[q] / N for p in CATS for q in CATS)
        out.append((Po - Pe) / (1 - Pe))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="", help="sheet suffix, e.g. _old")
    ap.add_argument("--outdir", default=str(HE / "consolidated"))
    args = ap.parse_args()
    S = load(args.sheets)
    N = len(S["A1"])
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    with open(out / "per_annotator_ratings.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["eval_id", "movieId", "primary_genre", "annotator_id",
                     "source", "axis", "rating"])
        for a in RATERS:
            for i in range(N):
                base = S["A1"][i]
                for ax in AXES:
                    for src in SOURCES:
                        wr.writerow([base["eval_id"], base["movieId"],
                                     base["primary_genre"], a, src, ax,
                                     S[a][i][f"{src}_{ax}"]])

    with open(out / "summary_table5.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["source", "axis", "mean", "std", "fleiss_kappa",
                     "weighted_kappa", "ge4_rate"])
        for src in SOURCES:
            for ax in AXES:
                v = [int(S[a][i][f"{src}_{ax}"]) for i in range(N) for a in RATERS]
                wr.writerow([src, ax, round(st.mean(v), 3), round(st.stdev(v), 3),
                             round(fleiss(S, src, ax), 3),
                             round(weighted_kappa(S, src, ax), 3),
                             round(sum(1 for x in v if x >= 4) / len(v), 3)])
    print(f"wrote {out}/per_annotator_ratings.csv  ({N * len(RATERS) * len(SOURCES) * len(AXES)} rows)")
    print(f"wrote {out}/summary_table5.csv")


if __name__ == "__main__":
    main()
