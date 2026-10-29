#!/usr/bin/env python3
"""Recompute the paper's mood-pairwise human-evaluation numbers from the released sheets.

    python3 scripts/download_artifacts.py --human-eval    # fetches data/human_eval/
    python3 tools/verify_mood_agreement.py

Reads the three filled pairwise sheets (`mood_pairwise_sheet_filled_A{1,2,3}.csv`) and
the key (`mood_pairwise_key.json`) that the data repository ships, recomputes every
number the paper prints about the pairwise mood task, and compares each with
`paper/generated/mood_macros.tex`:

  judgements   3 annotators x 100 pairs x 10 axes, and the share that picked a side
               rather than "too close" -- overall and per axis (best and worst)
  agreement    over cells where the model's two values differ: the published rule
               (majority over all three votes, "too close" included, ties to the first
               vote seen), and the stricter statement the paper leads with -- over cells
               where at least two annotators chose the SAME side, how often that side is
               the model's, with a Wilson 95% interval. The first counts a "too close"
               majority as disagreement, so 50% is not its chance line; the second has
               one, since a side is binary.
  per axis     the same majority-side agreement on each axis separately: its highest and
               lowest rate, on how many axes it exceeds one half at all, and on how
               many it is distinguishable from chance (one-sided exact binomial test
               against 1/2, p < 0.05). The pooled
               rate is carried by some axes more than others, and a reader should see
               that rather than infer it.

Exit 0 every number matched; 1 the sheets and the paper disagree; 2 cannot judge (a
file is missing, or the paper no longer defines a number this checks).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATORS = ("A1", "A2", "A3")
# Where the sheets usually are: beside a clone, or where download_artifacts.py puts
# them. The third place is this workspace's data tree; it is offered only where it
# exists, so a clone is never told to look in a directory it cannot have.
CANDIDATES = [ROOT / "human_eval", ROOT / "data" / "human_eval"] + [
    d for d in [ROOT / "release" / "hf" / "human_eval"] if d.is_dir()]
PRETTY = lambda axis: "\\texttt{" + axis.replace("_", "\\_") + "}"   # noqa: E731


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def read_votes(he: Path, axes: list) -> dict:
    """(pair_id, axis) -> votes in annotator order, each 'A', 'B' or 'T' (too close)."""
    votes = defaultdict(list)
    for a in ANNOTATORS:
        with open(he / f"mood_pairwise_sheet_filled_{a}.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if not r["pair_id"].strip():
                    continue                                    # the legend row
                pid = int(float(r["pair_id"]))
                for ax in axes:
                    v = (r.get(ax) or "").strip().upper()[:1]
                    if v in ("A", "B", "T"):
                        votes[(pid, ax)].append(v)
    return votes


def compute(he: Path) -> dict:
    """Every pairwise number, formatted exactly as the paper's macros print it."""
    key = json.loads((he / "mood_pairwise_key.json").read_text(encoding="utf-8-sig"))
    axes = key["axes"]
    votes = read_votes(he, axes)
    flat = [(ax, v) for (_, ax), vs in votes.items() for v in vs]
    definite = sum(v != "T" for _, v in flat)
    rate = {ax: sum(v != "T" for a, v in flat if a == ax) / sum(a == ax for a, _ in flat)
            for ax in axes}
    best, worst = max(rate, key=rate.get), min(rate, key=rate.get)

    model = {}
    for pr in key["pairs"]:
        for ax in axes:
            x, y = pr["mood_a"][ax], pr["mood_b"][ax]
            if x != y:
                model[(pr["pair_id"], ax)] = "A" if x > y else "B"
    published = sum(1 for c, s in model.items()
                    if votes.get(c) and Counter(votes[c]).most_common()[0][0] == s)
    side = {c: s for c in model
            if (s := next((x for x in "AB" if votes.get(c, []).count(x) >= 2), None))}
    agree = sum(s == model[c] for c, s in side.items())
    lo, hi = wilson(agree, len(side))
    per = {ax: [0, 0] for ax in axes}
    for c, s in side.items():
        per[c[1]][0] += s == model[c]
        per[c[1]][1] += 1
    tail = lambda k, m: sum(math.comb(m, i) for i in range(k, m + 1)) / 2 ** m  # noqa: E731
    arate = {ax: k / m for ax, (k, m) in per.items()}
    a_hi, a_lo = max(arate, key=arate.get), min(arate, key=arate.get)
    n = lambda k: f"{k:,}".replace(",", "{,}")   # noqa: E731
    return {
        "mdPairTotalN": n(len(flat)),
        "mdPairDefiniteN": n(definite),
        "mdPairDefinite": f"{definite / len(flat) * 100:.1f}\\%",
        "mdPairBestAxis": PRETTY(best), "mdPairBestRate": f"{rate[best] * 100:.1f}\\%",
        "mdPairWorstAxis": PRETTY(worst), "mdPairWorstRate": f"{rate[worst] * 100:.1f}\\%",
        "mdPairNonTiedN": str(len(model)),
        "mdPairAgreeN": str(published),
        "mdPairAgree": f"{published / len(model) * 100:.1f}\\%",
        "mdPairSideN": str(len(side)),
        "mdPairSideAgreeN": str(agree),
        "mdPairSideAgree": f"{agree / len(side) * 100:.1f}\\%",
        "mdPairSideCIlo": f"{lo * 100:.1f}\\%",
        "mdPairSideCIhi": f"{hi * 100:.1f}\\%",
        "mdPairAxesAboveN": str(sum(tail(k, m) < 0.05 for k, m in per.values())),
        # how many axes lean the model's way at all: "significant on five" read as
        # "agreed on five", when the majority matched more often than not on every axis
        "mdPairAxesLeanN": str(sum(k / m > 0.5 for k, m in per.values())),
        "mdPairAxisHiAxis": PRETTY(a_hi), "mdPairAxisHi": f"{arate[a_hi] * 100:.1f}\\%",
        "mdPairAxisLoAxis": PRETTY(a_lo), "mdPairAxisLo": f"{arate[a_lo] * 100:.1f}\\%",
    }


def paper_macros() -> dict:
    f = ROOT / "paper" / "generated" / "mood_macros.tex"
    if not f.exists():
        return {}
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", f.read_text()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--human-eval-dir", type=Path, default=None,
                    help="the data repository's human_eval/ directory "
                         "(default: the first of human_eval/, data/human_eval/ that holds the sheets)")
    a = ap.parse_args()
    # The first place that HOLDS the sheets, not the first that exists:
    # scripts/prepare_human_eval.py creates human_eval/, which would otherwise hide
    # the data/human_eval/ that download_artifacts.py fills.
    need = [f"mood_pairwise_sheet_filled_{x}.csv" for x in ANNOTATORS] + ["mood_pairwise_key.json"]
    places = [a.human_eval_dir] if a.human_eval_dir else CANDIDATES
    he = next((d for d in places if all((d / n).is_file() for n in need)), None)
    if he is None:
        print(f"CANNOT JUDGE: {', '.join(need)} are not all in any of:", file=sys.stderr)
        for d in places:
            print(f"    {d}", file=sys.stderr)
        print("They ship on the data repository under human_eval/. Fetch them with\n"
              "    python3 scripts/download_artifacts.py --human-eval\n"
              "which writes data/human_eval/, where this looks by default.", file=sys.stderr)
        return 2
    paper = paper_macros()
    got = compute(he)
    absent = sorted(k for k in got if k not in paper)
    if absent:
        print("CANNOT JUDGE: paper/generated/mood_macros.tex defines none of "
              + ", ".join("\\" + k for k in absent), file=sys.stderr)
        return 2
    bad = 0
    for k, v in got.items():
        ok = paper[k] == v
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {k:<17} computed {v:<34} | paper {paper[k]}")
    print(f"\n  {len(got) - bad} of {len(got)} mood-pairwise numbers reproduce from {he}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
