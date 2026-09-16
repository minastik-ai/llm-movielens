"""
Full-ranking evaluation for recommendation models.

Computes NDCG@K, Recall@K, HR@K, MRR over ALL items (no sampling).
Per Krichene & Rendle (KDD 2020), sampled metrics are inconsistent.
"""

import numpy as np
import torch
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import TOP_K, EVAL_BATCH_SIZE, COLD_START_BUCKETS


def compute_metrics(
    scores: np.ndarray,
    ground_truth: Set[int],
    train_items: Set[int],
    n_items: int,
    top_k: List[int] = TOP_K,
) -> Dict[str, float]:
    """
    Compute ranking metrics for a single user.

    Args:
        scores: (n_items,) predicted scores for all items
        ground_truth: set of relevant item IDs in test/val
        train_items: set of item IDs to exclude (already seen in training)
        n_items: total number of items
        top_k: list of K values

    Returns:
        dict of metric_name → value
    """
    if not ground_truth:
        return {}

    # Mask training items (set to -inf so they rank last)
    scores = scores.copy()
    for item in train_items:
        scores[item] = -np.inf

    # Get top-K item indices
    max_k = max(top_k)
    top_indices = np.argpartition(scores, -max_k)[-max_k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    metrics = {}
    for k in top_k:
        top_k_items = set(top_indices[:k].tolist())
        hits = top_k_items & ground_truth

        # HR@K (Hit Rate): 1 if any relevant item in top-K
        metrics[f"HR@{k}"] = 1.0 if hits else 0.0

        # Recall@K: fraction of relevant items found in top-K
        metrics[f"Recall@{k}"] = len(hits) / len(ground_truth)

        # NDCG@K
        dcg = 0.0
        for rank, item in enumerate(top_indices[:k].tolist()):
            if item in ground_truth:
                dcg += 1.0 / np.log2(rank + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        metrics[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0

    # MRR (Mean Reciprocal Rank) — position of first relevant item
    mrr = 0.0
    for rank, item in enumerate(top_indices[:max(top_k)].tolist()):
        if item in ground_truth:
            mrr = 1.0 / (rank + 1)
            break
    metrics["MRR"] = mrr

    return metrics


@torch.no_grad()
def evaluate_model(
    model,
    interaction_data,
    split: str = "test",
    device: str = "cpu",
    top_k: List[int] = TOP_K,
    batch_size: int = EVAL_BATCH_SIZE,
) -> Dict[str, float]:
    """
    Full-ranking evaluation: score ALL items for each user, compute metrics.

    Args:
        model: recommendation model with .predict(user_ids) → (batch, n_items) scores
        interaction_data: InteractionData instance
        split: "val" or "test"
        device: torch device
        top_k: K values for metrics

    Returns:
        dict of averaged metrics
    """
    model.eval()

    if split == "val":
        eval_user_items = interaction_data.val_user_items
    else:
        eval_user_items = interaction_data.test_user_items

    train_user_items = interaction_data.train_user_items
    n_items = interaction_data.n_items

    # Collect per-user metrics
    all_metrics = defaultdict(list)
    eval_users = sorted(eval_user_items.keys())

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        user_tensor = torch.LongTensor(batch_users).to(device)

        # Get scores for all items: (batch_size, n_items)
        scores = model.predict(user_tensor).cpu().numpy()

        for i, uid in enumerate(batch_users):
            ground_truth = eval_user_items.get(uid, set())
            if not ground_truth:
                continue
            train_items = train_user_items.get(uid, set())
            user_metrics = compute_metrics(scores[i], ground_truth, train_items, n_items, top_k)
            for k, v in user_metrics.items():
                all_metrics[k].append(v)

    # Average across users
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics["n_eval_users"] = len(eval_users)
    return avg_metrics


@torch.no_grad()
def evaluate_cold_start(
    model,
    interaction_data,
    device: str = "cpu",
    top_k: List[int] = [10, 20],
    batch_size: int = EVAL_BATCH_SIZE,
) -> Dict[str, Dict[str, float]]:
    """
    Cold-start breakdown: compute metrics separately for cold/medium/warm items.

    Returns: {bucket_name: {metric: value}}
    """
    model.eval()
    train_user_items = interaction_data.train_user_items
    test_user_items = interaction_data.test_user_items
    n_items = interaction_data.n_items

    # Compute item interaction counts in training
    item_counts = defaultdict(int)
    for items in train_user_items.values():
        for item in items:
            item_counts[item] += 1

    # Assign items to buckets
    item_buckets = {}
    for item in range(n_items):
        count = item_counts.get(item, 0)
        for bucket_name, (lo, hi) in COLD_START_BUCKETS.items():
            if lo <= count < hi:
                item_buckets[item] = bucket_name
                break

    # Evaluate per bucket
    bucket_metrics = {name: defaultdict(list) for name in COLD_START_BUCKETS}
    eval_users = sorted(test_user_items.keys())

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start + batch_size]
        user_tensor = torch.LongTensor(batch_users).to(device)
        scores = model.predict(user_tensor).cpu().numpy()

        for i, uid in enumerate(batch_users):
            train_items = train_user_items.get(uid, set())
            ground_truth = test_user_items.get(uid, set())
            if not ground_truth:
                continue

            # Split ground truth by item bucket
            for bucket_name in COLD_START_BUCKETS:
                bucket_gt = {item for item in ground_truth if item_buckets.get(item) == bucket_name}
                if not bucket_gt:
                    continue
                user_metrics = compute_metrics(scores[i], bucket_gt, train_items, n_items, top_k)
                for k, v in user_metrics.items():
                    bucket_metrics[bucket_name][k].append(v)

    # Average
    result = {}
    for bucket_name, metrics in bucket_metrics.items():
        result[bucket_name] = {k: np.mean(v) for k, v in metrics.items()}
        result[bucket_name]["n_eval"] = len(metrics.get(f"NDCG@{top_k[0]}", []))
    return result
