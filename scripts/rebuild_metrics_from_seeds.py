#!/usr/bin/env python3
"""Rebuild each results*/<cfg>/metrics.json from its own seed-*/results.json files.

WHY THIS EXISTS. `metrics.json` is an aggregate of the per-seed evaluations that sit
beside it, and for 35 of 47 configurations it is exactly that. For the other 12 it
was produced some other way, and the two disagree:

  * 9 configs carry `_provenance: "aggregated from reproducibility/
    verify_headline_numbers.py PER_SEED_M_ML20M"` -- i.e. from a spreadsheet
    transcribed into the verifier as literals, ROUNDED TO 4 DECIMALS. So
    `results/m7/metrics.json` stores seed 42 as 0.1177 where the per-seed file says
    0.11766675. Every Δ computed against that denominator inherits the rounding,
    which is how the paper came to print +29.5% where full precision gives +29.34%.
    It also made the verifier circular: it compared the paper against a file
    generated from its own constants, so it could not fail.

The per-seed files are canonical: each records `"eval-only full-ranking
re-evaluation of the released best_model.pt (no retraining)"`, which is what a
reproducer downloading the release actually recomputes. This script makes the
aggregate agree with them.

It rebuilds ONLY the metrics already present in the file, preserves key order and
every other field, keeps the seed order from `seeds_evaluated`, and uses the sample
standard deviation (ddof=1) that the existing files use. Run with --write to apply;
default is a dry run.
"""
import argparse, glob, json, shutil, statistics as st
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

ROOT = code("benchmark")


def per_seed_values(cfgdir: Path) -> dict:
    """{seed: {metric: value}} from the per-seed evaluation files."""
    out = {}
    for f in cfgdir.rglob("seed-*/results.json"):
        seed = f.parent.name.split("-", 1)[1]
        tm = json.loads(f.read_text()).get("test_metrics", {})
        if tm:
            out[seed] = tm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    changed, skipped = [], []
    for mp in sorted(ROOT.glob("results*/*/metrics.json")):
        j = json.loads(mp.read_text())
        seeds = per_seed_values(mp.parent)
        if not seeds:
            skipped.append((mp, "no per-seed files"))
            continue
        summary = j.get("summary")
        if not isinstance(summary, dict):
            skipped.append((mp, "no summary block"))
            continue
        # Seed order: prefer the file's own declared order so nothing reshuffles.
        declared = [str(s) for s in j.get("seeds_evaluated", [])]
        order = [s for s in declared if s in seeds] or sorted(seeds)
        if declared and set(declared) != set(seeds):
            skipped.append((mp, f"seed sets differ: file {sorted(declared)} vs "
                                f"on-disk {sorted(seeds)}"))
            continue

        deltas = []
        for metric, block in summary.items():
            if not (isinstance(block, dict) and "values" in block):
                continue
            vals = [seeds[s][metric] for s in order if metric in seeds[s]]
            if len(vals) != len(order):
                skipped.append((mp, f"{metric} missing from some per-seed files"))
                continue
            old_mean = block.get("mean")
            new_mean = st.mean(vals)
            block["values"] = vals
            block["mean"] = new_mean
            block["std"] = st.stdev(vals) if len(vals) > 1 else 0.0
            if old_mean is not None and abs(old_mean - new_mean) > 1e-12:
                deltas.append((metric, old_mean, new_mean))
        # per_seed mirrors the same numbers
        if isinstance(j.get("per_seed"), dict):
            for s in order:
                if s in j["per_seed"]:
                    for metric in list(j["per_seed"][s]):
                        if metric in seeds[s]:
                            j["per_seed"][s][metric] = seeds[s][metric]

        # `vs_baseline` stores a Δ against another config, plus the baseline's own
        # value ROUNDED TO 4 DP right beside it -- the rounding cascade in
        # miniature. 17 of 19 of these were stale, having been computed against
        # the pre-canonicalisation baseline. Recompute the whole block from the
        # per-seed files and store the baseline at full precision.
        vb = j.get("vs_baseline")
        if isinstance(vb, dict) and "delta_pct" in vb:
            base = str(vb.get("baseline", "M7")).lower().replace("/", "").replace("-", "")
            bseeds = per_seed_values(mp.parent.parent / base)
            if bseeds:
                arm = [seeds[x]["NDCG@10"] for x in order if "NDCG@10" in seeds[x]]
                bl = [bseeds[x]["NDCG@10"] for x in order
                      if x in bseeds and "NDCG@10" in bseeds[x]]
                if len(arm) == len(bl) == len(order):
                    old_d = vb["delta_pct"]
                    new_d = 100.0 * (st.mean(arm) - st.mean(bl)) / st.mean(bl)
                    same = len({(a - b) > 0 for a, b in zip(arm, bl)}) == 1
                    if (abs(old_d - new_d) > 0.005
                            or vb.get("baseline_NDCG10") != st.mean(bl)
                            or vb.get("all_seeds_same_sign") != same):
                        vb["baseline_NDCG10"] = st.mean(bl)
                        vb["delta_pct"] = round(new_d, 2)
                        vb["all_seeds_same_sign"] = same
                        deltas.append(("vs_baseline Δ%", old_d, new_d))

        if not deltas:
            continue
        prior = str(j.get("_provenance", ""))
        j["_provenance"] = (
            "aggregated from the on-disk per-seed seed-*/results.json beside this "
            "file (full precision; sample std, ddof=1) by "
            "scripts/rebuild_metrics_from_seeds.py. Supersedes: " + prior
        )
        changed.append((mp, deltas))
        if a.write:
            bak = mp.with_suffix(".json.orig")
            if not bak.exists():
                shutil.copy2(mp, bak)
            mp.write_text(json.dumps(j, indent=2) + "\n")

    for mp, deltas in changed:
        rel = mp.relative_to(ROOT)
        print(f"  {'REWROTE' if a.write else 'would rewrite'} {rel}")
        for metric, o, n in deltas:
            print(f"      {metric:<12} {o:.8f} -> {n:.8f}   ({n-o:+.8f})")
    for mp, why in skipped:
        print(f"  skipped {mp.relative_to(ROOT)}: {why}")
    print(f"\n  {len(changed)} file(s) {'rewritten' if a.write else 'to rewrite'}, "
          f"{len(skipped)} skipped")
    if not a.write and changed:
        print("  (dry run -- pass --write to apply; originals kept as .json.orig)")


if __name__ == "__main__":
    main()
