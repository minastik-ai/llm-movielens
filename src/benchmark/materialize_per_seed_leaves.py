#!/usr/bin/env python3
"""
Backfill per-seed leaf `results<suffix>/<config>/<encoder>/seed-<s>/results.json`
for configs that have a canonical aggregate but no per-seed leaves, by restating the
aggregate's per-seed (NDCG@10/Recall@10/MRR).

This is the inverse of `aggregate_results_metrics.py` (leaves -> aggregate). It exists
for R2/R3/SASRec: those are produced by upstream eval harnesses (KAR / HypernetReplacer /
pmixer-sequential) that emit config-level aggregates rather than the per-seed
`results.json` the M-config harness writes. The aggregate stays the source of truth; the
leaves are a derived view for structural parity with the M-configs. NEVER overwrites
existing real leaves; no re-evaluation (uses the canonical per-seed verbatim).

    python3 materialize_per_seed_leaves.py            # r2/r3 (all datasets) + base SASRec
    python3 materialize_per_seed_leaves.py --check     # report only (no writes)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
ENC = "bge-large-en-v1.5"
SUFFIXES = ["", "amazon", "ml1m", "ml20m_sub163"]
# (config dir, paper label) whose leaves we materialize from the aggregate's per_seed.
RCONFIGS = [("r2", "R2"), ("r3", "R3")]
PROV = ("per-seed leaf materialized from this config's canonical aggregate "
        "(NDCG@10/Recall@10/MRR); the aggregate is the source of truth — R2/R3 are "
        "produced by their upstream eval harnesses and SASRec by the pmixer sequential "
        "harness, which emit config-level aggregates, not per-seed leaves.")


def rd(suffix: str) -> Path:
    return B / ("results" if suffix == "" else f"results_{suffix}")


def has_leaves(cfgdir: Path) -> bool:
    return bool(list(cfgdir.glob(f"{ENC}/seed-*/results.json")))


def write_leaves(cfgdir: Path, label: str, per_seed: dict, check: bool) -> int:
    for seed, metrics in per_seed.items():
        if check:
            continue
        out = cfgdir / ENC / f"seed-{seed}" / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "experiment": f"{cfgdir.name}/{ENC}/seed-{seed}",
            "config": label, "seed": int(seed),
            "test_metrics": {k: float(v) for k, v in metrics.items()},
            "_provenance": PROV,
        }, indent=2) + "\n")
    return len(per_seed)


def main() -> int:
    check = "--check" in sys.argv
    total_dirs = total_files = 0
    # R2 / R3: per_seed lives in <config>/metrics.json
    for s in SUFFIXES:
        for cfg, label in RCONFIGS:
            cfgdir = rd(s) / cfg
            mj = cfgdir / "metrics.json"
            if not mj.exists():
                continue
            if has_leaves(cfgdir):
                print(f"   skip (real leaves present): {cfgdir.relative_to(B)}")
                continue
            ps = json.load(open(mj)).get("per_seed")
            if not ps:
                print(f"   skip (no per_seed): {mj.relative_to(B)}")
                continue
            n = write_leaves(cfgdir, label, ps, check)
            total_dirs += 1; total_files += n
            print(f"   {'would write' if check else '✓'} {cfgdir.relative_to(B)}  ({n} leaves)")
    # SASRec (base): per_seed_test lives in sasrec_pmixer/<ENC>/ml20m_test_5seeds.json
    sas = rd("") / "sasrec_pmixer"
    agg = sas / ENC / "ml20m_test_5seeds.json"
    if agg.exists() and not has_leaves(sas):
        ps = json.load(open(agg)).get("per_seed_test")
        if ps:
            n = write_leaves(sas, "SASRec", ps, check)
            total_dirs += 1; total_files += n
            print(f"   {'would write' if check else '✓'} {sas.relative_to(B)}  ({n} leaves)")
    else:
        print("   skip sasrec base (leaves present or no aggregate)")
    print(f"\n   {'(dry-run) ' if check else ''}{total_dirs} config dirs, {total_files} leaf files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
