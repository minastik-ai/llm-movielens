#!/usr/bin/env python3
"""5-seed cold-start evaluation for M1, M4, M7 (bge-large-en-v1.5).

Loads the matched-architecture trio (LightGCN / LightGCN-SF + LLM profile /
LightGCN-SF + profile+mood) at each of seeds {42, 123, 456, 789, 2026}
and reports per-bucket metrics: rank_median, MRR_full, Recall@K
for K in {10, 50, 100, 500, 1000}, NDCG@K likewise.

Aggregates across seeds: mean ± std per bucket per metric.
Saves JSON to results/analysis/cold_start_5seeds.json for the paper.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
BENCH = code("benchmark")
sys.path.insert(0, str(BENCH))

from config import (  # noqa: E402
    EMBED_DIM, LIGHTGCN_LAYERS, DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    EMBEDDING_DIR,
)
from data.dataset import InteractionData  # noqa: E402
from features.loader import FeatureLoader  # noqa: E402
from models.lightgcn import LightGCN, LightGCNSF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("cold_start_5seeds")

CONFIGS = [
    # (label, model_name,   features_key)
    ("M1", "lightgcn",     "none"),
    ("M4", "lightgcn_sf",  "llm_profile"),
    ("M7", "lightgcn_sf",  "llm_prof_mood"),
]
SEEDS = [42, 123, 456, 789, 2026]
K_VALUES = [10, 50, 100, 500, 1000]

BUCKETS = {  # half-open intervals on training-interaction count
    "cold":   (0, 10),
    "medium": (10, 50),
    "warm":   (50, float("inf")),
}


def _pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(name: str, n_users: int, n_items: int, feat_dim: int, norm_adj, device: str):
    if name == "lightgcn":
        m = LightGCN(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS).to(device)
    else:
        m = LightGCNSF(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, feat_dim).to(device)
    m.set_adj(norm_adj)
    return m


def _features(name: str, key: str, fl: FeatureLoader, device: str):
    if name == "lightgcn":
        return None, 0
    parts = {"llm_profile": ["profile"], "llm_prof_mood": ["profile", "mood"]}[key]
    t = fl.get_combined_tensor(parts, device=device)
    return t, t.shape[1]


def _bucket_map(data) -> dict:
    counts = defaultdict(int)
    for items in data.train_user_items.values():
        for it in items:
            counts[it] += 1
    out = {}
    for it in range(data.n_items):
        c = counts.get(it, 0)
        for name, (lo, hi) in BUCKETS.items():
            if lo <= c < hi:
                out[it] = name
                break
    return out


@torch.no_grad()
def _collect_gt_ranks(model, data, device, batch=256):
    model.eval()
    triples = []
    eval_users = sorted(data.test_user_items.keys())
    n_items = data.n_items
    for start in range(0, len(eval_users), batch):
        bu = eval_users[start:start + batch]
        scores = model.predict(torch.LongTensor(bu).to(device)).cpu().numpy()
        for i, uid in enumerate(bu):
            gt = data.test_user_items.get(uid, set())
            if not gt:
                continue
            s = scores[i].copy()
            for t in data.train_user_items.get(uid, set()):
                s[t] = -np.inf
            order = np.argsort(-s)
            rank_of = np.empty(n_items, dtype=np.int32)
            rank_of[order] = np.arange(n_items)
            for g in gt:
                triples.append((uid, g, int(rank_of[g])))
    return triples


def _bucket_metrics(triples, bucket_map, k_values):
    """Return per-bucket dict of metric -> per-user mean (one number per bucket
    per metric, averaged across users having ≥1 GT item in that bucket)."""
    user_gt = defaultdict(list)
    for uid, it, r in triples:
        user_gt[uid].append((it, r))

    out = {}
    for bname in BUCKETS:
        per_user = defaultdict(list)
        ranks_in_bucket = []
        for uid, pairs in user_gt.items():
            gt_b = [(it, r) for it, r in pairs if bucket_map.get(it) == bname]
            if not gt_b:
                continue
            ranks_b = [r for _, r in gt_b]
            ranks_in_bucket.extend(ranks_b)
            for K in k_values:
                hits = [r for r in ranks_b if r < K]
                per_user[f"Recall@{K}"].append(len(hits) / len(gt_b))
                dcg = sum(1.0 / np.log2(r + 2) for r in hits)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_b), K)))
                per_user[f"NDCG@{K}"].append(dcg / idcg if idcg > 0 else 0.0)
            per_user["MRR_full"].append(1.0 / (min(ranks_b) + 1))
        per_bucket = {k: float(np.mean(v)) for k, v in per_user.items()}
        per_bucket["n_users"] = len(per_user["Recall@10"]) if "Recall@10" in per_user else 0
        per_bucket["n_gt_items"] = len(ranks_in_bucket)
        if ranks_in_bucket:
            arr = np.asarray(ranks_in_bucket) + 1  # 1-indexed
            per_bucket["rank_median"] = float(np.median(arr))
            per_bucket["rank_mean"] = float(arr.mean())
        out[bname] = per_bucket
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = _pick_device(args.device)
    logger.info(f"Device: {device}")
    encoder = Path(EMBEDDING_DIR).name

    data = InteractionData(data_dir=DATA_DIR)
    fl = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMBEDDING_DIR)
    norm_adj = data.get_norm_adj().to(device)
    bucket_map = _bucket_map(data)

    # Build per-seed metrics: results[cfg_label][seed][bucket][metric]
    all_results = {cfg: {} for cfg, _, _ in CONFIGS}

    for seed in SEEDS:
        for label, model_name, feat_key in CONFIGS:
            ckpt = CHECKPOINT_DIR / label.lower() / encoder / f"seed-{seed}" / "best_model.pt"
            if not ckpt.exists():
                logger.warning(f"[{label} seed={seed}] missing {ckpt}")
                continue
            logger.info(f"[{label} seed={seed}] loading {ckpt.relative_to(REPO)}")
            feat, fdim = _features(model_name, feat_key, fl, device)
            model = _build_model(model_name, data.n_users, data.n_items, fdim, norm_adj, device)
            try:
                state = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state)
            except RuntimeError as e:
                logger.warning(f"[{label} seed={seed}] checkpoint mismatch (likely mislabeled file); skipping. Detail: {str(e)[:100]}")
                del model
                continue
            if feat is not None:
                model.set_features(feat)
            triples = _collect_gt_ranks(model, data, device)
            metrics = _bucket_metrics(triples, bucket_map, K_VALUES)
            all_results[label][seed] = metrics
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Aggregate across seeds: mean ± std per (config, bucket, metric)
    summary = {}
    for label in [c for c, _, _ in CONFIGS]:
        per_seed = all_results[label]
        if not per_seed:
            continue
        seeds_done = sorted(per_seed.keys())
        summary[label] = {"seeds": seeds_done, "buckets": {}}
        for bname in BUCKETS:
            metric_names = sorted(set(per_seed[seeds_done[0]][bname].keys()))
            agg = {}
            for met in metric_names:
                vals = [per_seed[s][bname][met] for s in seeds_done]
                if met in ("n_users", "n_gt_items"):
                    agg[met] = vals[0]  # constant across seeds
                else:
                    agg[met] = {"mean": float(np.mean(vals)),
                                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                "values": [float(v) for v in vals]}
            summary[label]["buckets"][bname] = agg

    out_path = RESULTS_DIR / "analysis" / "cold_start_5seeds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "encoder": encoder,
            "seeds": SEEDS,
            "k_values": K_VALUES,
            "buckets": {k: list(v) for k, v in BUCKETS.items()},
            "configs": summary,
        }, f, indent=2, default=str)
    logger.info(f"Saved {out_path.relative_to(REPO)}")

    # Pretty print
    print()
    print("=" * 110)
    print(f"5-seed cold-start summary  (encoder: {encoder})")
    print("=" * 110)
    header = f"{'Config':<5} {'Bucket':<7} {'rank_med':>10} {'MRR_full':>11} {'R@10':>11} {'R@1000':>11} {'N@10':>11} {'N@1000':>11} {'n_gt':>5}"
    print(header)
    print("-" * 110)
    for label in summary:
        for bname in BUCKETS:
            agg = summary[label]["buckets"][bname]
            def fmt(metric, fmt_str=".4f"):
                d = agg.get(metric)
                if not isinstance(d, dict):
                    return "n/a"
                return f"{d['mean']:{fmt_str}}\u00b1{d['std']:{fmt_str}}"
            n_gt = agg.get("n_gt_items", "?")
            print(f"{label:<5} {bname:<7} "
                  f"{fmt('rank_median', '.0f'):>10} "
                  f"{fmt('MRR_full'):>11} "
                  f"{fmt('Recall@10'):>11} "
                  f"{fmt('Recall@1000'):>11} "
                  f"{fmt('NDCG@10'):>11} "
                  f"{fmt('NDCG@1000'):>11} "
                  f"{n_gt:>5}")
    print("=" * 110)


if __name__ == "__main__":
    main()
