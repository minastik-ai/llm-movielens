#!/usr/bin/env python3
"""
Export benchmark results to CSV and LaTeX table.

Reads per-seed result JSONs from results/ directory,
computes mean ± std, and outputs:
  1. results/main_results.csv — machine-readable
  2. results/main_results.tex — copy-paste into paper

Usage:
    python scripts/export_results_table.py
    python scripts/export_results_table.py --results-dir results/
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20", "HR@10", "HR@20", "MRR"]

CONFIG_ORDER = [
    "M0", "M1", "M1b", "M1c", "M1d",  # Tier 1
    "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",  # Tier 2
    "R2", "R3",  # Tier 3
]

CONFIG_LABELS = {
    "M0": "BPR-MF",
    "M1": "LightGCN",
    "M1b": "SimGCL",
    "M1c": "XSimGCL",
    "M1d": "LightGCL",
    "M2": "+Genome(128)",
    "M3": "+BERT Title",
    "M4": "+LLM Profile",
    "M5": "+LLM Mood(10)",
    "M6": "+LLM Themes(528)",
    "M7": "+Profile+Mood",
    "M8": "+All LLM",
    "M9": "+Genome+Mood+Themes",
    "R2": "RLMRec-gene",
    "R3": "KAR",
}


def collect_results(results_dir: Path) -> dict:
    """Collect per-seed results into {config: {metric: [values]}}."""
    data = defaultdict(lambda: defaultdict(list))

    for result_file in sorted(results_dir.glob("*/results.json")):
        exp_name = result_file.parent.name  # e.g., "bpr_mf__none__seed42"

        with open(result_file) as f:
            result = json.load(f)

        config = result.get("config_name", exp_name)
        test_metrics = result.get("test_metrics", {})

        for metric in METRICS:
            if metric in test_metrics:
                data[config][metric].append(test_metrics[metric])

    return dict(data)


def export_csv(data: dict, output_path: Path):
    """Export to CSV: config, metric, mean, std, n_seeds."""
    lines = ["config,label,metric,mean,std,n_seeds"]

    for config in CONFIG_ORDER:
        if config not in data:
            continue
        label = CONFIG_LABELS.get(config, config)
        for metric in METRICS:
            values = data[config].get(metric, [])
            if values:
                mean = np.mean(values)
                std = np.std(values)
                lines.append(f"{config},{label},{metric},{mean:.4f},{std:.4f},{len(values)}")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"CSV saved to {output_path}")


def export_latex(data: dict, output_path: Path):
    """Export LaTeX table matching paper Table 4 format."""
    display_metrics = ["NDCG@10", "Recall@10", "NDCG@20", "Recall@20"]

    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& \textbf{Config} & \textbf{NDCG@10} & \textbf{Recall@10} "
        r"& \textbf{NDCG@20} & \textbf{Recall@20} \\"
    )
    lines.append(r"\midrule")

    # Find best values for bolding
    best = {}
    for metric in display_metrics:
        all_means = []
        for config in CONFIG_ORDER:
            if config in data and metric in data[config]:
                all_means.append((config, np.mean(data[config][metric])))
        if all_means:
            best[metric] = max(all_means, key=lambda x: x[1])[0]

    tier_headers = {
        "M0": r"\multicolumn{6}{l}{\textit{Tier 1: Pure Collaborative Filtering}} \\",
        "M2": r"\midrule" + "\n" + r"\multicolumn{6}{l}{\textit{Tier 2: Content-Augmented LightGCN-SF}} \\",
        "R2": r"\midrule" + "\n" + r"\multicolumn{6}{l}{\textit{Tier 3: LLM-for-RecSys Methods}} \\",
    }

    for config in CONFIG_ORDER:
        if config in tier_headers:
            lines.append(tier_headers[config])

        if config not in data:
            continue

        label = CONFIG_LABELS[config]
        cells = []
        for metric in display_metrics:
            values = data[config].get(metric, [])
            if values:
                mean = np.mean(values)
                std = np.std(values)
                cell = f"{mean:.3f}$\\pm${std:.3f}"
                if best.get(metric) == config:
                    cell = r"\textbf{" + cell + "}"
            else:
                cell = "---"
            cells.append(cell)

        lines.append(f"{config} & {label} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"LaTeX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export results table")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory with per-seed results"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = collect_results(results_dir)

    if not data:
        logger.warning(f"No results found in {results_dir}/. Looking in per_seed/...")
        data = collect_results(results_dir / "per_seed")

    if not data:
        logger.error("No results found. Run experiments first.")
        return

    logger.info(f"Found results for {len(data)} configs")

    export_csv(data, results_dir / "main_results.csv")
    export_latex(data, results_dir / "main_results.tex")


if __name__ == "__main__":
    main()
