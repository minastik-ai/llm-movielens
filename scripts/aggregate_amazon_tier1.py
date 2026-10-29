#!/usr/bin/env python3
"""Aggregate Amazon-Books Tier-1 results (M1, M4, M7 × 5 seeds).

Reads:  <results_dir>/{m1,m4,m7}/bge-large-en-v1.5/seed-<seed>/results.json
Writes: <results_dir>/amazon_tier1_summary.json
        <results_dir>/amazon_tier1_summary.csv
Prints: a markdown table for the paper.

Usage:
    python scripts/aggregate_amazon_tier1.py <results_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

CONFIGS = ["m1", "m4", "m7"]
ENCODER = "bge-large-en-v1.5"
SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["ndcg@10", "recall@10", "mrr"]


def load_test_metrics(p: Path) -> dict | None:
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    if "test_metrics" not in obj:
        raise SystemExit(f"{p}: no test_metrics (a validation-only run?)")
    return obj["test_metrics"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path, help="the Amazon-Books per-seed results tree")
    results_dir = ap.parse_args().results_dir

    summary = {}
    for cfg in CONFIGS:
        per_metric = {m: [] for m in METRICS}
        for seed in SEEDS:
            r = load_test_metrics(
                results_dir / cfg / ENCODER / f"seed-{seed}" / "results.json"
            )
            if r is None:
                continue
            for m in METRICS:
                # results.json stores keys like "NDCG@10", "Recall@10", "MRR"
                # — match case-insensitively
                v = next((r[k] for k in r if k.lower() == m), None)
                if v is not None:
                    per_metric[m].append(float(v))
        summary[cfg.upper()] = {
            m: {
                "n_seeds": len(per_metric[m]),
                "mean": mean(per_metric[m]) if per_metric[m] else None,
                "std": stdev(per_metric[m]) if len(per_metric[m]) > 1 else 0.0,
                "values": per_metric[m],
            }
            for m in METRICS
        }

    out_json = results_dir / "amazon_tier1_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    out_csv = results_dir / "amazon_tier1_summary.csv"
    with out_csv.open("w") as f:
        f.write("config," + ",".join(f"{m}_mean,{m}_std,n" for m in METRICS) + "\n")
        for cfg in CONFIGS:
            row = [cfg.upper()]
            for m in METRICS:
                s = summary[cfg.upper()][m]
                row += [
                    f"{s['mean']:.4f}" if s["mean"] is not None else "",
                    f"{s['std']:.4f}",
                    str(s["n_seeds"]),
                ]
            f.write(",".join(row) + "\n")

    # Markdown table for the paper
    print("\n=== Amazon-Books Tier-1 Results (mean ± std over seeds) ===\n")
    header = "| Config | " + " | ".join(METRICS) + " |"
    print(header)
    print("|" + "---|" * (1 + len(METRICS)))
    for cfg in CONFIGS:
        cells = [cfg.upper()]
        for m in METRICS:
            s = summary[cfg.upper()][m]
            if s["mean"] is None:
                cells.append("—")
            else:
                cells.append(f"{s['mean']:.4f} ± {s['std']:.4f} (n={s['n_seeds']})")
        print("| " + " | ".join(cells) + " |")
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
