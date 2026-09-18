"""Shared helper for ML-1M single-experiment runners.
Mirrors `_amazon_run_helper.py` with paths pinned to ML-1M outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
BENCHMARK = code("benchmark")
sys.path.insert(0, str(BENCHMARK))

ML1M_DATA_DIR = BENCHMARK / "data" / "processed_ml1m"
ML1M_EMB_DIR = (
    code("embedding_generator", "output_ml1m", "bge-large-en-v1.5")
)
ML1M_RESULTS_DIR = BENCHMARK / "results_ml1m"
ML1M_CKPT_DIR = BENCHMARK / "checkpoints_ml1m"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml1m_runner")


def _check_inputs(model: str, features: str) -> None:
    required = [
        ML1M_DATA_DIR / "train.csv",
        ML1M_DATA_DIR / "item_map.json",
    ]
    if features != "none":
        required += [
            ML1M_EMB_DIR / "profile_embeddings.npy",
            ML1M_EMB_DIR / "movie_id_index.json",
        ]
    if features == "llm_prof_mood":
        required.append(ML1M_EMB_DIR / "combined_features.npy")
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            logger.error(f"missing required input: {p}")
        logger.error("Run prepare_ml1m.py + build_ml1m_embeddings.py first.")
        sys.exit(2)


def run_ml1m_config(model: str, features: str, label: str) -> dict:
    parser = argparse.ArgumentParser(
        description=f"ML-1M single run: {label} ({model} + {features})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.info(f"Pinned to GPU {args.gpu}")

    _check_inputs(model, features)

    import torch
    from config import (
        LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
        FEATURE_CONFIGS, experiment_path,
    )
    from data.dataset import InteractionData
    from features.loader import FeatureLoader
    from train import train_model
    sys.path.insert(0, str(BENCHMARK))
    from run_experiment import build_model, set_seed

    out_dir = ML1M_RESULTS_DIR / experiment_path(model, features, args.seed, ML1M_EMB_DIR)
    if (out_dir / "results.json").exists():
        logger.info(f"[skip] {label} seed={args.seed} already complete.")
        return json.loads((out_dir / "results.json").read_text())

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    experiment_name = experiment_path(model, features, args.seed, ML1M_EMB_DIR)
    logger.info(f"═══ ML-1M: {label} (seed={args.seed}) ═══")
    logger.info(f"  experiment : {experiment_name}")
    logger.info(f"  device     : {device}")
    logger.info(f"  data       : {ML1M_DATA_DIR}")
    logger.info(f"  embeddings : {ML1M_EMB_DIR}")

    set_seed(args.seed)

    data = InteractionData(data_dir=ML1M_DATA_DIR)

    feature_names = FEATURE_CONFIGS[features]
    feature_loader = FeatureLoader(
        data_dir=ML1M_DATA_DIR, embedding_dir=ML1M_EMB_DIR,
    )
    feature_dim = (
        feature_loader.get_feature_dim(feature_names) if feature_names else 0
    )
    logger.info(f"  features   : {features} → {feature_names} (dim={feature_dim})")

    norm_adj = None
    if model in ("lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl"):
        norm_adj = data.get_norm_adj().to(device)

    net = build_model(model, data.n_users, data.n_items, feature_dim, norm_adj)
    if model in ("lightgcn_sf",) and feature_names:
        item_features = feature_loader.get_combined_tensor(feature_names, device=device)
        net.set_features(item_features)

    results = train_model(
        net, data,
        device=device,
        lr=args.lr if args.lr is not None else LEARNING_RATE,
        weight_decay=args.wd if args.wd is not None else WEIGHT_DECAY,
        num_epochs=args.epochs if args.epochs is not None else NUM_EPOCHS,
        patience=PATIENCE,
        experiment_name=experiment_name,
        checkpoint_dir=ML1M_CKPT_DIR,
        results_dir=ML1M_RESULTS_DIR,
        seed=args.seed,
    )

    test = results.get("test_metrics") or results.get("best_val_metrics") or {}
    logger.info(f"═══ Done: {label} seed={args.seed} ═══")
    for k in ("NDCG@10", "Recall@10", "MRR", "HR@10"):
        if k in test:
            logger.info(f"  {k:10s} {test[k]:.4f}")
    logger.info(f"  best_epoch  {results.get('best_epoch')}")
    return results
