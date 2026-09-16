#!/usr/bin/env python3
"""Run cold-start evaluation from saved checkpoints for M1, M2b, M4, M7.

For each configuration, this script:
  1. Locates the checkpoint under the new layout
     checkpoints/{config}/{encoder}/seed-{seed}/best_model.pt
  2. Reconstructs the model (LightGCN for M1, LightGCN-SF for M2b/M4/M7)
  3. Loads the matching side-feature tensor (raw genome for M2b, bge profile
     for M4, bge profile+mood for M7)
  4. Calls evaluate_cold_start() which stratifies the 9,906 items into
     cold (<10 train interactions), medium (10-49), warm (>=50) and reports
     NDCG@K / Recall@K / HR@K / MRR per bucket.
  5. Writes per-config JSON results next to the checkpoints and a combined
     summary to results/analysis/cold_start_inputs/{config}/{encoder}/seed-{seed}/cold_start.json and
     results/analysis/cold_start_summary.json.

Usage:
    python scripts/run_cold_start_eval.py
    python scripts/run_cold_start_eval.py --device cpu         # force CPU
    python scripts/run_cold_start_eval.py --configs M4 M7      # subset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from _paths import code  # dual-layout paths: dev tree or published release

# Repo layout: this file lives at <repo>/scripts/; benchmark code is at
# <repo>/code/benchmark/. Add the benchmark dir to sys.path so we can reuse
# the existing modules (InteractionData, models, evaluate_cold_start).
REPO = Path(__file__).resolve().parent.parent
BENCH = code("benchmark")
sys.path.insert(0, str(BENCH))

from config import (  # noqa: E402
    EMBED_DIM, LIGHTGCN_LAYERS, DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    EMBEDDING_DIR, COLD_START_BUCKETS,
)
from data.dataset import InteractionData  # noqa: E402
from features.loader import FeatureLoader  # noqa: E402
from models.lightgcn import LightGCN, LightGCNSF  # noqa: E402
from evaluate import evaluate_cold_start  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cold_start_eval")


# ---------------------------------------------------------------------------
# Experiment definitions — one (config, seed) per row.
# Only these four are requested; add rows here to evaluate more configs.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # (config_label, seed,  model_name,    features_key)
    ("M1",  789,  "lightgcn",    "none"),
    ("M2b", 42,   "lightgcn_sf", "genome_raw"),
    ("M4",  2026, "lightgcn_sf", "llm_profile"),
    ("M7",  123,  "lightgcn_sf", "llm_prof_mood"),
]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _genome_raw_features(item_map: dict, device: str) -> torch.Tensor:
    """Reconstruct the 1128-dim raw genome tag-relevance matrix, aligned to
    benchmark item IDs. Items without genome scores get zeros (there shouldn't
    be any since the benchmark already restricts to genome-covered movies)."""
    import pandas as pd

    genome_scores_csv = code("profile_generator") / \
        "llm-movie-profiler-v1-20260402" / "data" / "ml-20m" / "genome-scores.csv"
    genome_tags_csv = code("profile_generator") / \
        "llm-movie-profiler-v1-20260402" / "data" / "ml-20m" / "genome-tags.csv"

    logger.info(f"Loading raw genome scores from {genome_scores_csv.name}")
    scores = pd.read_csv(genome_scores_csv)
    tags = pd.read_csv(genome_tags_csv)
    n_tags = len(tags)
    tag_id_to_idx = {tid: i for i, tid in enumerate(sorted(tags["tagId"].unique()))}

    n_items = len(item_map)
    feat = np.zeros((n_items, n_tags), dtype=np.float32)

    # item_map keys are original movieIds (as int). Group scores by movieId.
    logger.info(f"Building {n_items}x{n_tags} genome_raw matrix...")
    grouped = scores.groupby("movieId")
    filled = 0
    for mid, bench_id in item_map.items():
        if mid in grouped.groups:
            sub = grouped.get_group(mid)
            for _, row in sub.iterrows():
                feat[bench_id, tag_id_to_idx[row["tagId"]]] = row["relevance"]
            filled += 1
    logger.info(f"  filled rows: {filled}/{n_items}")

    return torch.from_numpy(feat).to(device)


def _build_model(model_name: str, n_users: int, n_items: int, feature_dim: int,
                 norm_adj, device: str):
    if model_name == "lightgcn":
        m = LightGCN(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS).to(device)
    elif model_name == "lightgcn_sf":
        m = LightGCNSF(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, feature_dim).to(device)
    else:
        raise ValueError(f"unsupported model for cold-start eval: {model_name}")
    m.set_adj(norm_adj)
    return m


def _load_features(model_name: str, features_key: str, item_map: dict,
                   feature_loader: FeatureLoader, device: str):
    """Return (feature_tensor_or_None, feature_dim)."""
    if model_name == "lightgcn":
        return None, 0
    if features_key == "genome_raw":
        t = _genome_raw_features(item_map, device)
        return t, t.shape[1]
    feat_parts = {
        "llm_profile":   ["profile"],
        "llm_prof_mood": ["profile", "mood"],
    }[features_key]
    t = feature_loader.get_combined_tensor(feat_parts, device=device)
    return t, t.shape[1]


def _load_item_map(data_dir: Path) -> dict:
    with open(data_dir / "item_map.json") as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


def run(configs_wanted: list[str], device: str, top_k: list[int]):
    logger.info(f"Device: {device}")
    logger.info(f"EMBEDDING_DIR: {EMBEDDING_DIR}")
    logger.info(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")

    logger.info("Loading interactions …")
    data = InteractionData(data_dir=DATA_DIR)
    item_map = _load_item_map(DATA_DIR)
    feature_loader = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMBEDDING_DIR)

    logger.info("Building normalized adjacency …")
    norm_adj = data.get_norm_adj().to(device)

    encoder = Path(EMBEDDING_DIR).name  # e.g. "bge-large-en-v1.5"
    summary = {}

    for cfg, seed, model_name, features_key in EXPERIMENTS:
        if cfg not in configs_wanted:
            continue

        ck_path = CHECKPOINT_DIR / cfg.lower() / encoder / f"seed-{seed}" / "best_model.pt"
        if not ck_path.exists():
            logger.warning(f"[{cfg}] checkpoint missing: {ck_path} — skipping")
            continue

        logger.info(f"[{cfg} | seed {seed} | {model_name}/{features_key}] loading {ck_path.relative_to(REPO)}")
        feat_tensor, feat_dim = _load_features(model_name, features_key, item_map,
                                               feature_loader, device)
        model = _build_model(model_name, data.n_users, data.n_items, feat_dim,
                             norm_adj, device)
        state = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        if feat_tensor is not None:
            model.set_features(feat_tensor)
        model.eval()

        logger.info(f"[{cfg}] running evaluate_cold_start(top_k={top_k}) …")
        buckets = evaluate_cold_start(model, data, device=device, top_k=top_k)

        # Pretty log
        for bname in COLD_START_BUCKETS:
            m = buckets.get(bname, {})
            line = f"  {bname:<6}"
            for k in top_k:
                for metric in ("NDCG", "Recall", "HR"):
                    key = f"{metric}@{k}"
                    if key in m:
                        line += f"  {key}={m[key]:.4f}"
            if "MRR" in m:
                line += f"  MRR={m['MRR']:.4f}"
            if "n_eval" in m:
                line += f"  (n_eval={int(m['n_eval'])})"
            logger.info(line)

        # Persist per-config result
        out_dir = RESULTS_DIR / "analysis" / "cold_start_inputs" / cfg.lower() / encoder / f"seed-{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cold_start.json"
        with open(out_path, "w") as f:
            json.dump({
                "config": cfg,
                "seed": seed,
                "model": model_name,
                "features": features_key,
                "encoder": encoder,
                "top_k": top_k,
                "buckets": {bn: buckets.get(bn, {}) for bn in COLD_START_BUCKETS},
            }, f, indent=2, default=str)
        logger.info(f"  → saved {out_path.relative_to(REPO)}")
        summary[cfg] = {
            "seed": seed,
            "encoder": encoder,
            "buckets": {bn: buckets.get(bn, {}) for bn in COLD_START_BUCKETS},
        }

        # Free memory before loading the next model.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Combined summary at results root
    summary_path = RESULTS_DIR / "analysis" / "cold_start_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "encoder": encoder,
            "top_k": top_k,
            "buckets_definition": {k: list(v) for k, v in COLD_START_BUCKETS.items()},
            "configs": summary,
        }, f, indent=2, default=str)
    logger.info(f"Combined summary → {summary_path.relative_to(REPO)}")

    # Tabular final print
    _print_table(summary, top_k)


def _print_table(summary: dict, top_k: list[int]):
    if not summary:
        return
    print()
    print("=" * 100)
    print(f"{'Config':<6} {'Seed':<6} {'Bucket':<8} " +
          "  ".join(f"{'NDCG@'+str(k):<10}" for k in top_k) +
          "  " + "  ".join(f"{'Recall@'+str(k):<10}" for k in top_k) +
          "  " + f"{'MRR':<8}  {'n_eval':<8}")
    print("=" * 100)
    for cfg, info in summary.items():
        for bname, m in info["buckets"].items():
            cells = [f"{cfg:<6}", f"{info['seed']:<6}", f"{bname:<8}"]
            for k in top_k:
                cells.append(f"{m.get(f'NDCG@{k}', float('nan')):<10.4f}")
            for k in top_k:
                cells.append(f"{m.get(f'Recall@{k}', float('nan')):<10.4f}")
            cells.append(f"{m.get('MRR', float('nan')):<8.4f}")
            cells.append(f"{int(m.get('n_eval', 0)):<8}")
            print("  ".join(cells))
    print("=" * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["M1", "M2b", "M4", "M7"])
    p.add_argument("--device", default="auto")
    p.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    args = p.parse_args()
    run(args.configs, _pick_device(args.device), args.top_k)


if __name__ == "__main__":
    main()
