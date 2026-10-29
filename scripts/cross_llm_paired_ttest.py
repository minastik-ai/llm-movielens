#!/usr/bin/env python3
"""Provider sensitivity: paired t-tests of GPT-4o-mini vs Claude profiles for M4 and M7.

Reads the per-seed results of both arms -- Claude from results/<cfg>/, GPT-4o-mini from
results_gpt4omini/<cfg>/ (bge-large-en-v1.5, seeds 42, 123, 456, 789, 2026) -- and for
each (config, metric) computes:
  - 5-seed mean and sample std for each arm
  - the paired difference GPT - Claude (absolute and as a percentage of Claude)
  - the paired t-test p-value and the 95% CI of the difference

Writes results_gpt4omini/analysis/cross_llm_summary.json and a Markdown table beside it.

Usage:
    python3 scripts/cross_llm_paired_ttest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.stats as st
from _paths import code  # dual-layout paths: dev tree or published release

ARMS = {"claude": code("benchmark", "results"),
        "gpt": code("benchmark", "results_ml20m_gpt4omini")}
SUMMARY = ARMS["gpt"] / "analysis" / "cross_llm_summary.json"
MD_OUT = ARMS["gpt"] / "analysis" / "cross_llm_paper_table.md"
ENCODER = "bge-large-en-v1.5"

CONFIGS = ["m4", "m7"]
SEEDS = [42, 123, 456, 789, 2026]
METRICS = ["NDCG@10", "Recall@10", "MRR"]


def load_cell(config: str, llm: str, seed: int) -> dict | None:
    """test_metrics of one (config, llm, seed) cell, or None if it is absent."""
    p = ARMS[llm] / config / ENCODER / f"seed-{seed}" / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def aggregate():
    out = {"protocol": ("Paired GPT-4o-mini vs Claude downstream comparison: M4 and M7 "
                        "trained on each provider's profile features (bge-large-en-v1.5, "
                        "the same 10,381 items), five seeds [42, 123, 456, 789, 2026] "
                        "paired by seed; ML-20M test split, full ranking. Produced by "
                        "scripts/cross_llm_paired_ttest.py from the per-seed results."),
           "configs": {}}

    missing = []
    for cfg in CONFIGS:
        cfg_summary = {"per_metric": {}}
        for metric in METRICS:
            claude_vals, gpt_vals = [], []
            paired_seeds = []
            for s in SEEDS:
                c = load_cell(cfg, "claude", s)
                g = load_cell(cfg, "gpt", s)
                if c is None or g is None:
                    missing.append((cfg, s, "claude" if c is None else "gpt"))
                    continue
                cm, gm = c["test_metrics"].get(metric), g["test_metrics"].get(metric)
                if cm is None or gm is None:
                    continue
                claude_vals.append(cm); gpt_vals.append(gm); paired_seeds.append(s)
            if len(claude_vals) < 2:
                cfg_summary["per_metric"][metric] = {"error": f"insufficient seeds ({len(claude_vals)})"}
                continue
            cv = np.array(claude_vals); gv = np.array(gpt_vals)
            diff = gv - cv
            t, pval = st.ttest_rel(gv, cv)
            ci_lo, ci_hi = st.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=st.sem(diff))
            cfg_summary["per_metric"][metric] = {
                "n_seeds": len(claude_vals),
                "claude_mean": float(cv.mean()), "claude_std": float(cv.std(ddof=1)),
                "gpt_mean":    float(gv.mean()), "gpt_std":    float(gv.std(ddof=1)),
                "diff_mean":   float(diff.mean()),
                "diff_pct_of_claude": float(diff.mean() / cv.mean() * 100),
                "paired_t":    float(t),
                "p_value":     float(pval),
                "ci95_lower":  float(ci_lo),
                "ci95_upper":  float(ci_hi),
                "claude_per_seed": cv.tolist(),
                "gpt_per_seed":    gv.tolist(),
                "paired_seeds": paired_seeds,
            }
        out["configs"][cfg] = cfg_summary

    out["missing_cells"] = missing
    out["completeness"] = f"{20 - len(set((c, s) for c, s, _ in missing))}/20 cells with both LLMs"
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")
    return out


def write_markdown_table(summary: dict):
    lines = []
    lines.append("# cross-LLM downstream NDCG comparison\n")
    lines.append(f"_{summary['protocol']}_\n")
    lines.append(f"**Completeness:** {summary['completeness']}\n")
    if summary["missing_cells"]:
        lines.append(f"_Missing: {summary['missing_cells']}_\n")

    for cfg in CONFIGS:
        if cfg not in summary["configs"]:
            continue
        lines.append(f"\n## {cfg.upper()} — LightGCN-SF + LLM features\n")
        lines.append("| Metric | Claude (5-seed mean ± std) | GPT-4o-mini | Δ (abs) | Δ (%) | paired-t p | 95% CI |")
        lines.append("|---|---|---|---|---|---|---|")
        for metric in METRICS:
            r = summary["configs"][cfg]["per_metric"].get(metric)
            if r is None or "error" in r:
                continue
            sig = ""
            if r["p_value"] < 0.001: sig = " ***"
            elif r["p_value"] < 0.01: sig = " **"
            elif r["p_value"] < 0.05: sig = " *"
            elif r["p_value"] < 0.1:  sig = " ."
            lines.append(
                f"| {metric} "
                f"| {r['claude_mean']:.4f} ± {r['claude_std']:.4f} "
                f"| {r['gpt_mean']:.4f} ± {r['gpt_std']:.4f} "
                f"| {r['diff_mean']:+.4f} "
                f"| {r['diff_pct_of_claude']:+.2f}% "
                f"| {r['p_value']:.3f}{sig} "
                f"| [{r['ci95_lower']:+.4f}, {r['ci95_upper']:+.4f}] |"
            )

    lines.append("\n_Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05, . p<0.1 (paired t-test, GPT vs Claude)._\n")
    MD_OUT.write_text("\n".join(lines))
    print(f"\n=== Wrote {MD_OUT} ===\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    summary = aggregate()
    write_markdown_table(summary)
