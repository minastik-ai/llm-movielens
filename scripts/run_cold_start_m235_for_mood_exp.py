#!/usr/bin/env python3
"""Experiment B for the §3.5 mood reframe: cold-start eval for M2 (genome PCA-128d),
M3 (BERT-title 1024d), M5 (mood-only 10d) across 5 seeds, mirroring the existing
M1/M4/M7 protocol in scripts/run_cold_start_5seeds.py.

Output: code/benchmark/results/analysis/mood_analysis/exp_b_cold_start_m235.json
        — per-bucket {mean, std} for NDCG@10, Recall@1000 across 5 seeds

The mood-relevant comparison: does M5 (mood, 10-d) match M2 (genome PCA, 128-d)
or M3 (BERT-title, 1024-d) on the cold bucket? If yes at far smaller dim, mood
has a compactness contribution worth a paper subsection.
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
BENCH = code("benchmark")
sys.path.insert(0, str(BENCH))

from config import EMBED_DIM, LIGHTGCN_LAYERS  # noqa: E402
from data.dataset import InteractionData  # noqa: E402
from features.loader import FeatureLoader  # noqa: E402
from models.lightgcn import LightGCN, LightGCNSF  # noqa: E402

DATA_DIR  = BENCH / "data" / "processed"
EMB_DIR   = code("embedding_generator", "output", "bge-large-en-v1.5")
CK_DIR    = BENCH / "checkpoints"
OUT_DIR   = BENCH / "results" / "analysis" / "mood_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # label,  ckpt_subdir,  feature_keys (None = pure CF), feat_dim_for_log
    ("M2", "m2/bge-large-en-v1.5",  ["genome"],     128),
    ("M3", "m3/bge-large-en-v1.5",   ["bert_title"], 1024),  # NB: dir name spelled bge-large-en-v1.5
    ("M5", "m5/bge-large-en-v1.5",  ["mood"],       10),
]
SEEDS    = [42, 123, 456, 789, 2026]
K_VALUES = [10, 1000]
BUCKETS  = {"cold": (0, 10), "medium": (10, 50), "warm": (50, float("inf"))}


def pick_device(req="auto"):
    if req != "auto": return req
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def bucket_map(data):
    counts = defaultdict(int)
    for items in data.train_user_items.values():
        for it in items: counts[it] += 1
    out = {}
    for it in range(data.n_items):
        c = counts.get(it, 0)
        for name, (lo, hi) in BUCKETS.items():
            if lo <= c < hi:
                out[it] = name; break
    return out


@torch.no_grad()
def gt_ranks(model, data, device, batch=256):
    model.eval()
    triples = []
    eval_users = sorted(data.test_user_items.keys())
    n_items = data.n_items
    for start in range(0, len(eval_users), batch):
        bu = eval_users[start:start+batch]
        scores = model.predict(torch.LongTensor(bu).to(device)).cpu().numpy()
        for i, uid in enumerate(bu):
            gt = data.test_user_items.get(uid, set())
            if not gt: continue
            s = scores[i].copy()
            for t in data.train_user_items.get(uid, set()): s[t] = -np.inf
            order = np.argsort(-s)
            rank_of = np.empty(n_items, dtype=np.int32)
            rank_of[order] = np.arange(n_items)
            for g in gt:
                triples.append((uid, g, int(rank_of[g])))
    return triples


def bucket_metrics(triples, bmap):
    user_gt = defaultdict(list)
    for uid, it, r in triples:
        user_gt[uid].append((it, r))
    out = {}
    for bname in BUCKETS:
        per_user = defaultdict(list)
        ranks_in_bucket = []
        for uid, pairs in user_gt.items():
            gt_b = [(it, r) for it, r in pairs if bmap.get(it) == bname]
            if not gt_b: continue
            ranks_b = [r for _, r in gt_b]
            ranks_in_bucket.extend(ranks_b)
            for K in K_VALUES:
                hits = [r for r in ranks_b if r < K]
                per_user[f"Recall@{K}"].append(len(hits) / len(gt_b))
                dcg  = sum(1.0 / np.log2(r + 2) for r in hits)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_b), K)))
                per_user[f"NDCG@{K}"].append(dcg / idcg if idcg > 0 else 0.0)
        per_bucket = {k: float(np.mean(v)) for k, v in per_user.items()}
        per_bucket["n_users"] = len(per_user.get("Recall@10", []))
        if ranks_in_bucket:
            arr = np.asarray(ranks_in_bucket) + 1
            per_bucket["rank_median"] = float(np.median(arr))
        out[bname] = per_bucket
    return out


def main():
    device = pick_device()
    print(f"Device: {device}")

    data = InteractionData(data_dir=DATA_DIR)
    fl   = FeatureLoader(data_dir=DATA_DIR, embedding_dir=EMB_DIR)
    norm_adj = data.get_norm_adj().to(device)
    bmap = bucket_map(data)
    counts = {b: sum(1 for v in bmap.values() if v == b) for b in BUCKETS}
    print(f"Item buckets: {counts}")

    all_results = {}
    for label, ck_subdir, feat_keys, feat_dim in CONFIGS:
        all_results[label] = {"feat_dim": feat_dim, "by_seed": {}}
        for seed in SEEDS:
            ck = CK_DIR / ck_subdir / f"seed-{seed}" / "best_model.pt"
            if not ck.exists():
                print(f"  [{label} seed={seed}] MISSING {ck}"); continue
            feat = fl.get_combined_tensor(feat_keys, device=device)
            model = LightGCNSF(data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS, feat.shape[1]).to(device)
            model.set_adj(norm_adj)
            try:
                state = torch.load(ck, map_location=device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
            except RuntimeError as e:
                print(f"  [{label} seed={seed}] skip — {str(e)[:100]}"); continue
            model.set_features(feat)
            triples = gt_ranks(model, data, device)
            metrics = bucket_metrics(triples, bmap)
            all_results[label]["by_seed"][seed] = metrics
            cold_r1k = metrics.get("cold", {}).get("Recall@1000", "N/A")
            cold_n10 = metrics.get("cold", {}).get("NDCG@10", "N/A")
            print(f"  [{label} seed={seed}] cold: NDCG@10={cold_n10:.5f}  R@1000={cold_r1k:.5f}")

    # Aggregate mean ± std per bucket per metric
    summary = {"feat_dims": {l: f for l, _, _, f in CONFIGS}, "configs": {}}
    for label in [l for l, _, _, _ in CONFIGS]:
        per_seed = all_results[label]["by_seed"]
        if not per_seed: continue
        seeds_done = sorted(per_seed.keys())
        agg = {"seeds": seeds_done, "feat_dim": all_results[label]["feat_dim"], "buckets": {}}
        for bname in BUCKETS:
            mname = sorted(set(per_seed[seeds_done[0]][bname].keys()))
            ag = {}
            for met in mname:
                vals = [per_seed[s][bname][met] for s in seeds_done]
                if met == "n_users":
                    ag[met] = vals[0]
                else:
                    ag[met] = {"mean": float(np.mean(vals)),
                               "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                               "values": [float(v) for v in vals]}
            agg["buckets"][bname] = ag
        summary["configs"][label] = agg

    out_path = OUT_DIR / "exp_b_cold_start_m235.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved → {out_path}")

    # Pretty print headline
    print()
    print("=" * 90)
    print(f"Experiment B: cold-start by feature dimension (5-seed mean ± std)")
    print("=" * 90)
    print(f"{'Label':<5} {'Feature':<28} {'Dim':>5} {'Cold NDCG@10':>14} {'Cold R@1000':>13} {'Med R@1000':>12}")
    for label, _, feat_keys, feat_dim in CONFIGS:
        if label not in summary["configs"]: continue
        c = summary["configs"][label]["buckets"]
        feat_name = "+".join(feat_keys)
        cn = c["cold"].get("NDCG@10", {})
        cr = c["cold"].get("Recall@1000", {})
        mr = c["medium"].get("Recall@1000", {})
        if isinstance(cn, dict):
            print(f"{label:<5} {feat_name:<28} {feat_dim:>5} "
                  f"{cn['mean']:>10.5f}±{cn['std']:.5f} "
                  f"{cr['mean']:>9.4f}±{cr['std']:.4f} "
                  f"{mr['mean']:>8.4f}±{mr['std']:.4f}")


if __name__ == "__main__":
    main()
