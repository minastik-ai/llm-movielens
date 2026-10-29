#!/usr/bin/env python3
"""
Run a single experiment: model + feature config + seed.

Usage:
  python run_experiment.py --model lightgcn --features none --seed 42
  python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42
  python run_experiment.py --model bpr_mf --features none --seed 42

  # Custom paths (CLI args override env vars override defaults):
  python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
      --embedding-dir /path/to/embeddings \
      --results-dir /path/to/results
"""

import os
import sys
import json
import random
import hashlib
import logging
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBED_DIM, LIGHTGCN_LAYERS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, BATCH_SIZE, FEATURE_CONFIGS, SEEDS,
    LIGHTGCL_SVD_Q,
    EMBEDDING_DIR, DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    experiment_path,
    resolve_config,
)
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.bpr_mf import BPRMF
from models.lightgcn import LightGCN, LightGCNSF
from models.xsimgcl import XSimGCL
from models.simgcl import SimGCL
from models.lightgcl import LightGCL as LightGCLModel
from train import train_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Seed every RNG touched by the pipeline: Python's random, NumPy, PyTorch
    (CPU + CUDA on all visible GPUs), and cuDNN's algorithm selection.

    This fixes every random draw, but runs are still not bit-identical, even
    back to back on one machine: sparse matrix products and scatter-adds
    accumulate floating-point sums in a nondeterministic order. Measured on a CPU,
    two 10-epoch SimGCL runs with the same seed differed by at most 0.003 in any
    weight and by 1.8e-5 in test NDCG@10, far below the spread between seeds.
    docs/REPRODUCIBILITY.md says what to expect from a re-run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CONTRASTIVE = ("simgcl", "xsimgcl", "lightgcl")


def build_model(
    model_name: str,
    n_users: int,
    n_items: int,
    feature_dim: int = 0,
    norm_adj=None,
    overrides: dict | None = None,
):
    """Instantiate model by name.

    `overrides` sets a contrastive model's own hyperparameters (cl_weight,
    noise_eps, cl_temp, layer_cl, init) and is refused for any other model, so a
    grid flag can never silently do nothing.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if overrides and model_name not in CONTRASTIVE:
        raise ValueError(f"{sorted(overrides)} only apply to {CONTRASTIVE}, not {model_name}")
    if model_name == "bpr_mf":
        return BPRMF(n_users, n_items, EMBED_DIM)

    elif model_name == "lightgcn":
        model = LightGCN(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "lightgcn_sf":
        if feature_dim == 0:
            raise ValueError("lightgcn_sf requires features (feature_dim > 0)")
        model = LightGCNSF(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, feature_dim)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "xsimgcl":
        model = XSimGCL(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, **overrides)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "simgcl":
        model = SimGCL(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, **overrides)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "lightgcl":
        bad = sorted({"noise_eps", "layer_cl"} & overrides.keys())
        if bad:
            raise ValueError(f"{bad} do not apply to lightgcl")
        model = LightGCLModel(
            n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS,
            svd_q=LIGHTGCL_SVD_Q, **overrides,
        )
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model


    else:
        raise ValueError(f"Unknown model: {model_name}")


CONTRASTIVE_SETTINGS = Path(__file__).resolve().parent / "contrastive_settings.json"
CONTRASTIVE_LABEL = {"simgcl": "m1b", "xsimgcl": "m1c", "lightgcl": "m1d"}
CL_ARG = {"cl_weight": "cl_weight", "cl_eps": "cl_eps", "cl_temp": "cl_temp",
          "layer_cl": "layer_cl"}


def apply_contrastive_settings(args):
    """Fill every contrastive setting the command line left unset from the frozen
    record, so `--config M1b` trains the run the paper reports. A flag on the
    command line always wins (contrastive_grid.py passes all of them)."""
    if args.model in CONTRASTIVE_LABEL:
        record = json.loads(CONTRASTIVE_SETTINGS.read_text())
        digest = hashlib.sha256(json.dumps(record["chosen"], sort_keys=True).encode()).hexdigest()
        if digest != record["chosen_sha256"]:
            raise SystemExit(f"{CONTRASTIVE_SETTINGS.name}: the frozen choices do not match "
                             f"their recorded SHA-256")
        for key, value in record["chosen"][CONTRASTIVE_LABEL[args.model]]["hp"].items():
            if getattr(args, CL_ARG[key]) is None:
                setattr(args, CL_ARG[key], value)
        for key in ("init", "min_epochs", "dense_eval_until"):
            if getattr(args, key) is None:
                setattr(args, key, record["run"][key])
    if args.min_epochs is None:
        args.min_epochs = 0
    if args.dense_eval_until is None:
        args.dense_eval_until = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Paper label (M0, M4, M7, ...): sets --model and "
                             "--features together. The paper and "
                             "reproduce_all.sh address configurations this way.")
    parser.add_argument("--model", type=str, required=False, default=None, choices=["bpr_mf", "lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl"])
    parser.add_argument("--features", type=str, default="none", choices=list(FEATURE_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Epoch budget (default {NUM_EPOCHS}). Recorded in the "
                             f"checkpoint and results.json so a later resume knows whether "
                             f"the run finished or was interrupted.")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early-stopping patience in epochs (default {PATIENCE}). "
                             f"Recorded alongside the budget; the resume guard judges a "
                             f"checkpoint by the patience it was TRAINED with, not by "
                             f"whatever the config says later.")
    parser.add_argument("--dense-eval-until", type=int, default=None,
                        help="also validate every epoch up to this one (default 0: epoch 1 "
                             "and every 5th; 20 for the contrastive baselines, from "
                             "contrastive_settings.json). Recorded in results.json.")
    parser.add_argument("--no-test", dest="evaluate_test", action="store_false",
                        help="skip the test evaluation (hyperparameter-selection runs: no "
                             "test value may exist before the choice is frozen)")
    parser.add_argument("--min-epochs", type=int, default=None,
                        help="early stopping may not end the run before this epoch "
                             "(default 0: no floor; 45 for the contrastive baselines, from "
                             "contrastive_settings.json). Recorded in results.json.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"BPR training batch size (default {BATCH_SIZE}). The loader "
                             f"drops the final partial batch, so a dataset with fewer "
                             f"interaction pairs than this yields no batches at all -- "
                             f"lower it when running on a small or subsampled dataset. "
                             f"The published numbers were produced at the default; any "
                             f"other value changes the optimisation.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--embedding-dir", type=str, default=None,
                        help="Path to embedding directory (overrides EMBEDDING_DIR env var)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to processed data directory (overrides DATA_DIR env var)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (overrides RESULTS_DIR env var)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from training_state.pt if present (default).")
    parser.add_argument("--ignore-early-stop", action="store_true",
                        help="Continue training a run that already FINISHED — either by "
                             "early stopping or by exhausting its epoch budget. Default is "
                             "to refuse, because continuing can find a new best, overwrite "
                             "best_model.pt and silently change a published result. When "
                             "passed, best_model.pt is backed up to .pre-continue first.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any saved state and train from scratch.")
    # A contrastive baseline's own hyperparameters (SimGCL, XSimGCL, LightGCL only).
    # Unset means the frozen setting in contrastive_settings.json, which
    # scripts/contrastive_grid.py selected on the validation split; every value
    # used is recorded in results.json.
    parser.add_argument("--cl-weight", type=float, default=None,
                        help="contrastive weight lambda (lambda_1 for LightGCL)")
    parser.add_argument("--cl-eps", type=float, default=None,
                        help="noise magnitude eps (SimGCL, XSimGCL)")
    parser.add_argument("--cl-temp", type=float, default=None,
                        help="InfoNCE temperature tau")
    parser.add_argument("--layer-cl", type=int, default=None,
                        help="XSimGCL: contrast the final layer with layer l* (1-indexed)")
    parser.add_argument("--init", type=str, default=None, choices=["normal", "uniform"],
                        help="embedding initialisation for the contrastive models "
                             "(the authors use Xavier-uniform)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to checkpoint directory (overrides CHECKPOINT_DIR env var)")
    args = parser.parse_args()
    if args.config:
        if args.model is not None or args.features != "none":
            parser.error("--config already sets --model/--features")
        args.model, args.features = resolve_config(args.config)
    if args.model is None:
        parser.error("one of --config or --model is required")
    apply_contrastive_settings(args)

    # Resolve paths: CLI args > env vars > defaults (from config.py)
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else EMBEDDING_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    experiment_name = experiment_path(args.model, args.features, args.seed, embedding_dir)
    logger.info(f"═══ Experiment: {experiment_name} ═══")
    logger.info(f"Device: {device}")
    logger.info(f"Embedding dir: {embedding_dir}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    set_seed(args.seed)

    # Load data
    logger.info("Loading interaction data...")
    data = InteractionData(data_dir=data_dir)

    # Load features
    feature_names = FEATURE_CONFIGS[args.features]
    # Only when the config actually uses features. M0 and M1 are the pure
    # collaborative baselines -- feature_names is [] -- and constructing the
    # loader regardless made them read movie_id_index.json and the embedding
    # arrays they never touch. A reviewer running the cheapest, most obvious
    # first config had to download 566 MB to train an ID-only model, and the
    # guard the next line already applies says the code knew better.
    feature_loader = (FeatureLoader(data_dir=data_dir, embedding_dir=embedding_dir)
                      if feature_names else None)
    feature_dim = feature_loader.get_feature_dim(feature_names) if feature_names else 0
    logger.info(f"Features: {args.features} → {feature_names} (dim={feature_dim})")

    # Build adjacency for GNN models
    norm_adj = None
    if args.model in ("lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl"):
        logger.info("Building normalized adjacency matrix...")
        norm_adj = data.get_norm_adj().to(device)

    # Build model
    overrides = {"cl_weight": args.cl_weight, "noise_eps": args.cl_eps,
                 "cl_temp": args.cl_temp, "layer_cl": args.layer_cl, "init": args.init}
    if args.model == "lightgcl" and (args.cl_eps is not None or args.layer_cl is not None):
        parser.error("--cl-eps and --layer-cl do not apply to lightgcl")
    if args.model == "simgcl" and args.layer_cl is not None:
        parser.error("--layer-cl applies to xsimgcl only")
    model = build_model(args.model, data.n_users, data.n_items, feature_dim, norm_adj,
                        overrides=overrides)
    if hasattr(model, "hparams"):
        logger.info(f"Model hyperparameters: {model.hparams}")

    # Set features for models that accept side features
    if args.model in ("lightgcn_sf",) and feature_names:
        item_features = feature_loader.get_combined_tensor(feature_names, device=device)
        model.set_features(item_features)

    # Train
    results = train_model(
        model, data,
        device=device,
        lr=args.lr,
        weight_decay=args.wd,
        num_epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        dense_eval_until=args.dense_eval_until,
        evaluate_test=args.evaluate_test,
        batch_size=args.batch_size,
        experiment_name=experiment_name,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        seed=args.seed,
        resume=args.resume,
        ignore_early_stop=args.ignore_early_stop,
    )

    logger.info(f"═══ Done: {experiment_name} ═══")
    return results


if __name__ == "__main__":
    main()
