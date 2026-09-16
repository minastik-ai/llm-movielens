#!/usr/bin/env python3
"""
Run full ablation table: all model × feature configs × seeds.

Usage:
  python run_ablation.py                    # run all experiments
  python run_ablation.py --quick            # quick test (1 seed, limited epochs)
  python run_ablation.py --model lightgcn_sf  # only LightGCN-SF ablations

  # Custom paths (CLI args override env vars override defaults):
  python run_ablation.py --embedding-dir /path/to/embeddings
  python run_ablation.py --results-dir /path/to/results --checkpoint-dir /path/to/ckpts
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURE_CONFIGS, SEEDS, RESULTS_DIR, EMBEDDING_DIR, DATA_DIR, CHECKPOINT_DIR, experiment_path
from features.loader import FeatureLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ablation experiments: (model, features, label)
ABLATION_TABLE = [
    # Tier 1: Pure CF baselines
    ("bpr_mf",      "none",          "M0: BPR-MF (ID only)"),
    ("lightgcn",    "none",          "M1: LightGCN (ID only)"),

    ("simgcl",      "none",          "M1b: SimGCL (ID only)"),
    ("xsimgcl",     "none",          "M1c: XSimGCL (ID only)"),
    ("lightgcl",    "none",          "M1d: LightGCL (ID only)"),

    # Tier 2: Content-augmented LightGCN-SF
    ("lightgcn_sf", "genome",        "M2: + genome PCA"),
    ("lightgcn_sf", "bert_title",    "M3: + BERT title"),
    ("lightgcn_sf", "llm_profile",   "M4: + LLM profile"),
    ("lightgcn_sf", "llm_mood",      "M5: + LLM mood"),
    ("lightgcn_sf", "llm_themes",    "M6: + LLM themes"),
    ("lightgcn_sf", "llm_prof_mood", "M7: + LLM profile+mood"),
    ("lightgcn_sf", "llm_all",       "M8: + LLM all"),
    ("lightgcn_sf", "genome_llm",    "M9: + genome+mood+themes"),

    # Tier 3: LLM-for-RecSys methods
    ("kar",         "llm_prof_mood", "R2: KAR + LLM profile+mood"),
]


def run_single(model, features, seed, extra_args=None):
    """Run a single experiment as a subprocess."""
    cmd = [
        sys.executable, "run_experiment.py",
        "--model", model,
        "--features", features,
        "--seed", str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(Path(__file__).parent))


def _write_cross_encoder_summary(results_dir):
    """Merge all per-encoder summaries into encoder_sensitivity_summary.json.

    Reads every `ablation_summary_{encoder}.json` at the top of results_dir and
    produces a combined view keyed by config label, with one entry per encoder.
    Safe to call whenever: if only one encoder has been run, the output will
    simply contain that one encoder.
    """
    per_encoder = {}
    for p in sorted(Path(results_dir).glob("ablation_summary_*.json")):
        encoder = p.stem.replace("ablation_summary_", "", 1)
        try:
            per_encoder[encoder] = json.loads(p.read_text())
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")

    if not per_encoder:
        return

    combined = {}  # config label → { encoder → metrics }
    for encoder, summary in per_encoder.items():
        for entry in summary:
            label = entry.get("label", entry.get("model"))
            combined.setdefault(label, {})[encoder] = {
                "model": entry.get("model"),
                "features": entry.get("features"),
                "n_seeds": entry.get("n_seeds"),
                "metrics": entry.get("metrics"),
            }

    out = Path(results_dir) / "encoder_sensitivity_summary.json"
    with open(out, "w") as f:
        json.dump({
            "encoders": sorted(per_encoder.keys()),
            "configs": combined,
        }, f, indent=2, default=str)
    logger.info(f"Cross-encoder summary saved to {out}")


def collect_results(results_dir, embedding_dir):
    """Collect all results into a summary table."""
    summary = []
    for model, features, label in ABLATION_TABLE:
        seed_results = []
        for seed in SEEDS:
            exp_name = experiment_path(model, features, seed, embedding_dir)
            result_file = results_dir / exp_name / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    r = json.load(f)
                seed_results.append(r["test_metrics"])

        if seed_results:
            # Aggregate across seeds
            agg = {}
            metric_keys = [k for k in seed_results[0] if isinstance(seed_results[0][k], float)]
            for key in metric_keys:
                vals = [r[key] for r in seed_results]
                agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}

            summary.append({
                "label": label,
                "model": model,
                "features": features,
                "n_seeds": len(seed_results),
                "metrics": agg,
            })

    return summary


def print_summary(summary):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print(f"{'Experiment':<35} {'NDCG@10':>12} {'NDCG@20':>12} {'Recall@10':>12} {'Recall@20':>12} {'HR@10':>12} {'MRR':>12} {'Seeds':>6}")
    print("=" * 120)

    for entry in summary:
        m = entry["metrics"]
        def fmt(key):
            if key in m:
                return f"{m[key]['mean']:.4f}±{m[key]['std']:.4f}"
            return "N/A"

        print(f"{entry['label']:<35} {fmt('NDCG@10'):>12} {fmt('NDCG@20'):>12} "
              f"{fmt('Recall@10'):>12} {fmt('Recall@20'):>12} {fmt('HR@10'):>12} "
              f"{fmt('MRR'):>12} {entry['n_seeds']:>6}")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 seed, 20 epochs")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model")
    parser.add_argument("--collect-only", action="store_true", help="Only collect and print results")
    parser.add_argument("--embedding-dir", type=str, default=None,
                        help="Path to embedding directory (overrides EMBEDDING_DIR env var)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to processed data directory (overrides DATA_DIR env var)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (overrides RESULTS_DIR env var)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoint directory (overrides CHECKPOINT_DIR env var)")
    args = parser.parse_args()

    # Resolve paths: CLI args > env vars > defaults
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else EMBEDDING_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR

    if args.collect_only:
        summary = collect_results(results_dir, embedding_dir)
        print_summary(summary)
        # Under the config-first layout, each encoder's summary lives at the
        # top of results/ as ablation_summary_{encoder}.json. The optional
        # cross-encoder merge step below produces encoder_sensitivity_summary.json.
        from config import get_encoder_name
        encoder = get_encoder_name(embedding_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"ablation_summary_{encoder}.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {out}")
        _write_cross_encoder_summary(results_dir)
        return

    seeds = [SEEDS[0]] if args.quick else SEEDS
    extra_args = ["--epochs", "20"] if args.quick else []

    # Pass path overrides to subprocess
    if args.embedding_dir:
        extra_args.extend(["--embedding-dir", args.embedding_dir])
    if args.data_dir:
        extra_args.extend(["--data-dir", args.data_dir])
    if args.results_dir:
        extra_args.extend(["--results-dir", args.results_dir])
    if args.checkpoint_dir:
        extra_args.extend(["--checkpoint-dir", args.checkpoint_dir])

    experiments = ABLATION_TABLE
    if args.model:
        experiments = [(m, f, l) for m, f, l in experiments if m == args.model]

    total = len(experiments) * len(seeds)
    completed = 0

    # Use FeatureLoader to check feature availability
    feature_loader = FeatureLoader(data_dir=data_dir, embedding_dir=embedding_dir)

    for model, features, label in experiments:
        feature_names = FEATURE_CONFIGS[features]

        # Check all required feature files exist
        missing = [fn for fn in feature_names if not feature_loader.has_feature(fn)]
        if missing:
            logger.warning(f"Skipping {label} — missing feature files: {missing}")
            completed += len(seeds)
            continue

        for seed in seeds:
            completed += 1
            exp_name = experiment_path(model, features, seed, embedding_dir)
            result_file = results_dir / exp_name / "results.json"
            if result_file.exists():
                logger.info(f"[{completed}/{total}] {label} (seed={seed}) — SKIPPED (complete)")
                continue
            logger.info(f"[{completed}/{total}] {label} (seed={seed})")
            result = run_single(model, features, seed, extra_args)
            if result.returncode != 0:
                logger.error(f"  FAILED: {label} seed={seed}")

    # Collect and print summary
    logger.info("Collecting results...")
    summary = collect_results(results_dir, embedding_dir)
    print_summary(summary)

    from config import get_encoder_name
    encoder = get_encoder_name(embedding_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"ablation_summary_{encoder}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {out}")
    _write_cross_encoder_summary(results_dir)


if __name__ == "__main__":
    main()
