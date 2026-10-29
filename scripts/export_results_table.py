#!/usr/bin/env python3
"""
Export benchmark results to CSV and LaTeX, in the layout of the paper's main table.

Reads the per-seed results.json files of one encoder, computes the mean and the
sample standard deviation (ddof=1) over seeds, and writes:
  1. <out-dir>/main_results.csv -- machine-readable
  2. <out-dir>/main_results.tex -- NDCG@10, Recall@10 and MRR, four decimals

Run on the shipped results/ tree it reproduces the paper's main table. M8 and M9 are
sensitivity configurations: they are exported, but the paper's table does not list
them.

Usage:
    python scripts/export_results_table.py                              # shipped results/
    python scripts/export_results_table.py --results-dir src/benchmark/results
    python scripts/export_results_table.py --encoder e5-large-v2
"""

import argparse
import json
import sys
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import code  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20", "HR@10", "HR@20", "MRR"]

CONFIG_ORDER = [
    "M0", "M1", "M1b", "M1c", "M1d",  # Tier 1
    "M2", "M2b", "M3", "M4", "M5", "M6", "M7", "M8", "M9",  # Tier 2
]

CONFIG_LABELS = {
    "M0": "BPR-MF",
    "M1": "LightGCN",
    "M1b": "SimGCL",
    "M1c": "XSimGCL",
    "M1d": "LightGCL",
    "M2": "+Genome PCA(128)",
    "M2b": "+Genome raw(1128)",
    "M3": "+Title embedding",
    "M4": "+LLM Profile",
    "M5": "+LLM Mood(10)",
    "M6": "+LLM Themes(528)",
    "M7": "+Profile+Mood",
    "M8": "+All LLM",
    "M9": "+Genome+Mood+Themes",
}


def collect_results(results_dir: Path, encoder: str) -> dict:
    """Collect per-seed results into {config: {metric: [values]}}.

    The layout is <config>/<encoder>/seed-<n>/results.json, as experiment_path() in
    benchmark/config.py writes it. ID-only configurations are filed under the default
    encoder's name, so the default encoder yields every configuration and any other
    encoder yields only the configurations that use it.
    """
    data = defaultdict(lambda: defaultdict(list))
    for result_file in sorted(results_dir.glob(f"*/{encoder}/seed-*/results.json")):
        result = json.loads(result_file.read_text())
        stem = (result.get("experiment") or result_file.relative_to(results_dir).parts[0]).split("/")[0]
        config = stem[0].upper() + stem[1:]
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
                std = np.std(values, ddof=1)
                lines.append(f"{config},{label},{metric},{mean:.4f},{std:.4f},{len(values)}")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"CSV saved to {output_path}")


def export_latex(data: dict, output_path: Path):
    """Export a LaTeX table in the paper's main-table layout."""
    display_metrics = ["NDCG@10", "Recall@10", "MRR"]
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"& \textbf{Configuration} & \textbf{NDCG@10} & \textbf{Recall@10} & \textbf{MRR} \\",
             r"\midrule"]
    for config in CONFIG_ORDER:
        if config == "M2":
            lines.append(r"\midrule")
        if config not in data:
            continue
        cells = []
        for metric in display_metrics:
            values = data[config].get(metric, [])
            cells.append(f"${np.mean(values):.4f}\\pm{np.std(values, ddof=1):.4f}$" if values else "---")
        lines.append(f"{config} & {CONFIG_LABELS[config]} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"LaTeX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export the results table")
    parser.add_argument("--results-dir", type=Path, default=code("benchmark", "results"),
                        help="per-seed results tree (default: the shipped results/)")
    parser.add_argument("--encoder", default="bge-large-en-v1.5",
                        help="sentence encoder whose runs to export (default bge-large-en-v1.5)")
    parser.add_argument("--out-dir", type=Path, default=Path("main_results"),
                        help="where to write main_results.csv/.tex (default ./main_results)")
    args = parser.parse_args()

    data = collect_results(args.results_dir, args.encoder)
    if not data:
        logger.error(
            "No %s results found under %s. Train something first, e.g.\n"
            "        bash scripts/reproduce_all.sh --config M0\n"
            "    or point --results-dir at the shipped per-seed tree (results/).",
            args.encoder, args.results_dir)
        sys.exit(2)          # cannot judge: nothing to export, not a clean run

    logger.info(f"Found results for {len(data)} configs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_csv(data, args.out_dir / "main_results.csv")
    export_latex(data, args.out_dir / "main_results.tex")


if __name__ == "__main__":
    main()
