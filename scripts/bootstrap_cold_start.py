#!/usr/bin/env python3
"""User-level bootstrap 95% CI on cold-start metrics for M1, M4, M7.

Combines all 5 seeds: for each (config, bucket, metric), each user's per-user
metric is averaged across seeds, then 1000 bootstrap resamples (with
replacement) over the user pool produce a 95% confidence interval on the
bucket-level mean. This captures user-level uncertainty that the 5-seed std
(which captures seed-level uncertainty) does not.

Saves: code/benchmark/results/analysis/cold_start_bootstrap.json
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
logger = logging.getLogger("bootstrap")

CONFIGS = [
    ("M1", "lightgcn",     "none"),
    ("M4", "lightgcn_sf",  "llm_profile"),
    ("M7", "lightgcn_sf",  "llm_prof_mood"),
]
SEEDS = [42, 123, 456, 789, 2026]
K_VALUES = [10, 50, 100, 500, 1000]
N_BOOT = 1000

BUCKETS = {
    "cold":   (0, 10),
    "medium": (10, 50),
    "warm":   (50, float("inf")),
}


def _pick_device(req: str) -> str:
    if req != "auto": return req
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(name, n_users, n_items, feat_dim, norm_adj, device):
    if name == "lightgcn":
        m = LightGCN(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS).to(device)
    else:
        m = LightGCNSF(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, feat_dim).to(device)
    m.set_adj(norm_adj)
    return m


def _features(name, key, fl, device):
    if name == "lightgcn":
        return None, 0
    parts = {"llm_profile": ["profile"], "llm_prof_mood": ["profile", "mood"]}[key]
    t = fl.get_combined_tensor(parts, device=device)
    return t, t.shape[1]


def _bucket_map(data):
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
def _per_user_bucket_metrics(model, data, bucket_map, device, batch=256):
    """Return {bucket: {user_id: {metric: value}}} for one checkpoint."""
    model.eval()
    n_items = data.n_items
    out = {b: {} for b in BUCKETS}
    eval_users = sorted(data.test_user_items.keys())

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

            # Per-user, per-bucket metrics
            for bname in BUCKETS:
                gt_b = [g for g in gt if bucket_map.get(g) == bname]
                if not gt_b:
                    continue
                ranks_b = [int(rank_of[g]) for g in gt_b]
                user_metrics = {}
                user_metrics["MRR_full"] = 1.0 / (min(ranks_b) + 1)
                user_metrics["rank_min"] = min(ranks_b) + 1   # 1-indexed
                for K in K_VALUES:
                    hits = [r for r in ranks_b if r < K]
                    user_metrics[f"Recall@{K}"] = len(hits) / len(gt_b)
                    dcg = sum(1.0 / np.log2(r + 2) for r in hits)
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_b), K)))
                    user_metrics[f"NDCG@{K}"] = dcg / idcg if idcg > 0 else 0.0
                out[bname][uid] = user_metrics
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    device = _pick_device(args.device)
    logger.info(f"Device: {device}, N_boot: {args.n_boot}")
    encoder = Path(EMBEDDING_DIR).name

    data = InteractionData(data_dir=DATA_DIR)
    fl = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMBEDDING_DIR)
    norm_adj = data.get_norm_adj().to(device)
    bucket_map = _bucket_map(data)

    # Step 1: collect per-user per-bucket metrics for every (config, seed)
    # all_data[label][seed][bucket][user_id][metric] = value
    all_data = {label: {} for label, _, _ in CONFIGS}
    for seed in SEEDS:
        for label, model_name, feat_key in CONFIGS:
            ckpt = CHECKPOINT_DIR / label.lower() / encoder / f"seed-{seed}" / "best_model.pt"
            if not ckpt.exists():
                logger.warning(f"[{label} seed={seed}] missing ckpt")
                continue
            logger.info(f"[{label} seed={seed}] inference …")
            feat, fdim = _features(model_name, feat_key, fl, device)
            model = _build_model(model_name, data.n_users, data.n_items, fdim, norm_adj, device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
            if feat is not None:
                model.set_features(feat)
            per_user = _per_user_bucket_metrics(model, data, bucket_map, device)
            all_data[label][seed] = per_user
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Step 2: average per-user metrics across seeds, then bootstrap over users
    rng = np.random.default_rng(42)
    summary = {}
    for label in [c for c, _, _ in CONFIGS]:
        per_seed = all_data[label]
        if not per_seed:
            continue
        seeds_done = sorted(per_seed.keys())
        summary[label] = {"n_seeds": len(seeds_done), "buckets": {}}

        for bname in BUCKETS:
            # User pool = users present in bucket for ≥1 seed
            user_pool = set()
            for s in seeds_done:
                user_pool.update(per_seed[s][bname].keys())
            user_pool = sorted(user_pool)

            # Determine metric names from any sample user
            first_seed = seeds_done[0]
            any_user = next(iter(per_seed[first_seed][bname]), None)
            if any_user is None:
                continue
            metric_names = sorted(per_seed[first_seed][bname][any_user].keys())

            # Per-user, seed-averaged metrics
            seed_avg = {}
            for uid in user_pool:
                vals = {met: [] for met in metric_names}
                for s in seeds_done:
                    user_metrics = per_seed[s][bname].get(uid)
                    if user_metrics is None:
                        continue
                    for met in metric_names:
                        vals[met].append(user_metrics[met])
                seed_avg[uid] = {met: float(np.mean(v)) if v else float('nan') for met, v in vals.items()}

            # Build flat arrays per metric
            metric_arrays = {met: np.array([seed_avg[uid][met] for uid in user_pool])
                             for met in metric_names}

            # Bootstrap
            n_users = len(user_pool)
            results = {"n_users": n_users}
            for met in metric_names:
                arr = metric_arrays[met]
                # Mean of original sample
                point_est = float(arr.mean())
                # Bootstrap
                boot_means = np.empty(args.n_boot)
                for b in range(args.n_boot):
                    idx = rng.integers(0, n_users, size=n_users)
                    boot_means[b] = arr[idx].mean()
                ci_low = float(np.percentile(boot_means, 2.5))
                ci_high = float(np.percentile(boot_means, 97.5))
                # Median rank: also report the median of seed-averaged ranks
                if met == "rank_min":
                    median = float(np.median(arr))
                    median_boot = np.empty(args.n_boot)
                    for b in range(args.n_boot):
                        idx = rng.integers(0, n_users, size=n_users)
                        median_boot[b] = np.median(arr[idx])
                    results["rank_median"] = {
                        "value": median,
                        "ci_low": float(np.percentile(median_boot, 2.5)),
                        "ci_high": float(np.percentile(median_boot, 97.5)),
                    }
                results[met] = {"mean": point_est, "ci_low": ci_low, "ci_high": ci_high}

            summary[label]["buckets"][bname] = results

    out = RESULTS_DIR / "analysis" / "cold_start_bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "encoder": encoder,
            "seeds": SEEDS,
            "n_bootstrap": args.n_boot,
            "buckets": {k: list(v) for k, v in BUCKETS.items()},
            "configs": summary,
        }, f, indent=2, default=str)
    logger.info(f"Saved {out.relative_to(REPO)}")

    # Pretty-print
    print()
    print("=" * 130)
    print(f"User-level bootstrap 95% CI (N={args.n_boot}, encoder={encoder})")
    print("=" * 130)
    print(f"{'Cfg':<4} {'Bucket':<7} {'rank_med [95% CI]':<22} {'MRR_full [95% CI]':<28} {'R@1000 [95% CI]':<28} {'NDCG@10 [95% CI]':<28}  {'n_users':>7}")
    print("-" * 130)
    for label in summary:
        for bname in BUCKETS:
            d = summary[label]["buckets"].get(bname, {})
            if not d:
                continue
            n = d.get("n_users", 0)
            rm = d.get("rank_median", {})
            mrr = d.get("MRR_full", {})
            r1000 = d.get("Recall@1000", {})
            n10 = d.get("NDCG@10", {})
            def fmt(x, fmt_str=".4f"):
                if not x: return "n/a"
                v = x.get("value", x.get("mean"))
                return f"{v:{fmt_str}} [{x['ci_low']:{fmt_str}}, {x['ci_high']:{fmt_str}}]"
            print(f"{label:<4} {bname:<7} {fmt(rm, '.0f'):<22} {fmt(mrr):<28} {fmt(r1000):<28} {fmt(n10):<28}  {n:>7}")
    print("=" * 130)


if __name__ == "__main__":
    main()
