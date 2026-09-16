#!/usr/bin/env python3
"""cross-LLM closure — single-cell retrain harness for M4/M7 with Claude OR GPT-4o-mini features.

One process = one (config, llm, seed) cell. Resumable via train_model's existing
checkpoint logic (`training_state.pt`). Outputs `best_model.pt`, `training_state.pt`,
`results.json` per cell.

Usage:
    python3 scripts/run_cross_llm_retrain.py --config m7 --llm gpt --seed 42 --device cuda
    python3 scripts/run_cross_llm_retrain.py --config m4 --llm claude --seed 123 --device mps

Outputs:
    code/benchmark/checkpoints_cross_llm/<config>_<llm>/seed-<N>/best_model.pt
    code/benchmark/checkpoints_cross_llm/<config>_<llm>/seed-<N>/training_state.pt
    code/benchmark/results_ml20m_gpt4omini/<config>_<llm>/seed-<N>/results.json
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
BENCH = code("benchmark")
sys.path.insert(0, str(BENCH))

import numpy as np
import torch

from config import EMBED_DIM, LIGHTGCN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.lightgcn import LightGCNSF
from train import train_model

# Embedding dirs per LLM (both must contain bge-large-en-v1.5 subdir w/ identical 10,381 IDs)
LLM_EMB_DIRS = {
    "claude": code("embedding_generator", "output", "bge-large-en-v1.5"),
    "gpt":    code("embedding_generator", "output_ml20m_gpt4omini", "bge-large-en-v1.5"),
}

# M4 = profile only (1024-d); M7 = profile + mood (1034-d)
CONFIG_FEATURES = {
    "m4": ["profile"],
    "m7": ["profile", "mood"],
}

DATA_DIR = BENCH / "data" / "processed"  # ML-20M


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIG_FEATURES), required=True)
    p.add_argument("--llm",    choices=list(LLM_EMB_DIRS), required=True)
    p.add_argument("--seed",   type=int, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--patience",   type=int, default=PATIENCE)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--no_resume", action="store_true",
                   help="ignore existing training_state.pt and start from scratch")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[cross_llm] config={args.config}, llm={args.llm}, seed={args.seed}, device={device}")

    emb_dir = LLM_EMB_DIRS[args.llm]
    feat_names = CONFIG_FEATURES[args.config]
    if not (emb_dir / "profile_embeddings.npy").exists():
        print(f"ERROR: missing {emb_dir}/profile_embeddings.npy")
        sys.exit(2)

    data = InteractionData(data_dir=DATA_DIR)
    fl   = FeatureLoader(data_dir=DATA_DIR, embedding_dir=emb_dir)
    feat = fl.get_combined_tensor(feat_names, device=device)
    print(f"[cross_llm] n_users={data.n_users}, n_items={data.n_items}, feat_dim={feat.shape[1]}")

    model = LightGCNSF(data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS,
                       feature_dim=feat.shape[1]).to(device)
    model.set_adj(data.get_norm_adj().to(device))
    model.set_features(feat)

    exp_name = f"{args.config}_{args.llm}/seed-{args.seed}"
    ckpt_root = code("benchmark", "checkpoints_cross_llm")
    res_root  = code("benchmark", "results_ml20m_gpt4omini")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    # Skip-if-done short-circuit (idempotent)
    out_results = res_root / exp_name / "results.json"
    if out_results.exists() and not args.no_resume:
        existing = json.loads(out_results.read_text())
        if existing.get("test_metrics"):
            print(f"[cross_llm] {exp_name} already complete: NDCG@10={existing['test_metrics'].get('NDCG@10', '?')}")
            print(f"[cross_llm] re-run with --no_resume to overwrite")
            return

    t0 = time.time()
    result = train_model(
        model=model,
        interaction_data=data,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=args.num_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        experiment_name=exp_name,
        resume=(not args.no_resume),
        checkpoint_dir=ckpt_root,
        results_dir=res_root,
        seed=args.seed,
    )
    wall_s = time.time() - t0

    print(f"\n[cross_llm] {exp_name} DONE in {wall_s/60:.1f} min")
    if result.get("test_metrics"):
        tm = result["test_metrics"]
        print(f"[cross_llm] test NDCG@10={tm.get('NDCG@10'):.4f}  Recall@10={tm.get('Recall@10'):.4f}  MRR={tm.get('MRR'):.4f}")


if __name__ == "__main__":
    main()
