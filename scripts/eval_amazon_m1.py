#!/usr/bin/env python3
"""Backfill per-seed test-metric JSONs for Amazon-Books M1 (LightGCN, ID-only).

Loads each saved `best_model.pt` from `checkpoints_amazon/m1/.../seed-*/`,
re-runs `evaluate_model(split="test")` with our standard full-ranking protocol,
and writes a `results.json` next to where the original training would have
written one (`results_amazon/m1/.../seed-*/`).

This is eval-only — no training, no checkpoint mutation. The 5 best_model.pt
files were produced by the original M1 training runs; we just persist their
test metrics for paper-claim traceability (Q6: SASRec vs M1 = −48.5% on Amazon).

Usage:
    python3 scripts/eval_amazon_m1.py
    python3 scripts/eval_amazon_m1.py --device mps --seeds 42 123
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
BENCH = code("benchmark")
sys.path.insert(0, str(BENCH))

from data.dataset import InteractionData  # noqa: E402
from features.loader import FeatureLoader  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from run_experiment import build_model, set_seed  # noqa: E402
from config import experiment_path  # noqa: E402

AMAZON_DATA_DIR = BENCH / "data" / "processed_amazon"
AMAZON_EMB_DIR = code("embedding_generator", "output_amazon", "bge-large-en-v1.5")
AMAZON_CKPT_DIR = BENCH / "checkpoints_amazon"
AMAZON_RESULTS_DIR = BENCH / "results_amazon"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_amazon_m1")


def eval_one_seed(seed: int, device: str) -> dict:
    set_seed(seed)
    data = InteractionData(data_dir=AMAZON_DATA_DIR)
    norm_adj = data.get_norm_adj().to(device)

    net = build_model("lightgcn", data.n_users, data.n_items, feature_dim=0, norm_adj=norm_adj)
    net = net.to(device)

    exp_name = experiment_path("lightgcn", "none", seed, AMAZON_EMB_DIR)
    ckpt_path = AMAZON_CKPT_DIR / exp_name / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    logger.info(f"  Loaded checkpoint: {ckpt_path.relative_to(REPO)}")

    test_metrics = evaluate_model(net, data, split="test", device=device)
    val_metrics = evaluate_model(net, data, split="val", device=device)
    results = {
        "experiment": exp_name,
        "model": "lightgcn",
        "features": "none",
        "seed": seed,
        "evaluator": "evaluate_model (full-ranking, eval-only backfill)",
        "checkpoint": str(ckpt_path.relative_to(REPO)),
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    out_dir = AMAZON_RESULTS_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Wrote: {out_path.relative_to(REPO)}")
    logger.info(f"  test NDCG@10 = {test_metrics['NDCG@10']:.4f}  "
                f"Recall@10 = {test_metrics['Recall@10']:.4f}  "
                f"MRR = {test_metrics['MRR']:.4f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    all_results = {}
    for seed in args.seeds:
        logger.info(f"═══ Amazon M1 eval-only seed={seed} ═══")
        all_results[seed] = eval_one_seed(seed, device)

    # 5-seed summary
    if len(all_results) >= 2:
        import statistics
        ndcg = [r["test_metrics"]["NDCG@10"] for r in all_results.values()]
        recall = [r["test_metrics"]["Recall@10"] for r in all_results.values()]
        mrr = [r["test_metrics"]["MRR"] for r in all_results.values()]
        logger.info("")
        logger.info(f"═══ {len(all_results)}-seed summary (Amazon M1) ═══")
        logger.info(f"  NDCG@10  : {statistics.mean(ndcg):.4f} ± {statistics.stdev(ndcg):.4f}")
        logger.info(f"  Recall@10: {statistics.mean(recall):.4f} ± {statistics.stdev(recall):.4f}")
        logger.info(f"  MRR      : {statistics.mean(mrr):.4f} ± {statistics.stdev(mrr):.4f}")


if __name__ == "__main__":
    main()
