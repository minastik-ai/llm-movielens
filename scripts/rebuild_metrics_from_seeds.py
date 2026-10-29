#!/usr/bin/env python3
"""Rebuild each results*/<cfg>/metrics.json from its own seed-*/results.json files.

`metrics.json` is the per-configuration aggregate of the per-seed result files beside
it: the mean and sample standard deviation (ddof=1) of every metric, at full precision,
and any `vs_baseline` delta recomputed from the two configurations' per-seed values.
The per-seed files are canonical; this script makes the aggregate agree with them.

It rebuilds ONLY the metrics already present in the file, preserves key order and
every other field, and keeps the seed order from `seeds_evaluated`. Run with --write
to apply; default is a dry run.
"""
import argparse, glob, json, shutil, statistics as st
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

ROOT = code("benchmark")
# Every results tree: in the development checkout they sit under the benchmark package
# (results*/), in the published release at the repository root (results/,
# results_gpt4omini/), which code() resolves.
RESULT_ROOTS = sorted({p for p in ROOT.glob("results*") if p.is_dir()}
                      | {p for p in (code("benchmark", "results"),
                                     code("benchmark", "results_ml20m_gpt4omini")) if p.is_dir()})
PROVENANCE = ("aggregated from the per-seed seed-*/results.json beside this file "
              "(full precision; sample std, ddof=1) by scripts/rebuild_metrics_from_seeds.py")


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
    if not RESULT_ROOTS:
        raise SystemExit(f"no results tree found under {ROOT} or at the repository root")
    for mp in sorted(m for r in RESULT_ROOTS for m in r.glob("*/metrics.json")):
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

        if j.get("_provenance") != PROVENANCE:
            j["_provenance"] = PROVENANCE
            deltas.append(("_provenance", 0.0, 0.0))
        if not deltas:
            continue
        changed.append((mp, deltas))
        if a.write:
            bak = mp.with_suffix(".json.orig")
            if not bak.exists():
                shutil.copy2(mp, bak)
            mp.write_text(json.dumps(j, indent=2) + "\n")

    for mp, deltas in changed:
        rel = mp.relative_to(ROOT.parent.parent)
        print(f"  {'REWROTE' if a.write else 'would rewrite'} {rel}")
        for metric, o, n in deltas:
            print(f"      {metric:<12} {o:.8f} -> {n:.8f}   ({n-o:+.8f})")
    for mp, why in skipped:
        print(f"  skipped {mp.relative_to(ROOT.parent.parent)}: {why}")
    print(f"\n  {len(changed)} file(s) {'rewritten' if a.write else 'to rewrite'}, "
          f"{len(skipped)} skipped")
    if not a.write and changed:
        print("  (dry run -- pass --write to apply; originals kept as .json.orig)")


if __name__ == "__main__":
    main()
