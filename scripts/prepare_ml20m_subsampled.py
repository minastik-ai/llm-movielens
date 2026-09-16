#!/usr/bin/env python3
"""
Subsample ML-20M to a same-domain density gradient datapoint between ML-20M
and ML-1M. Preserves long-tail item distribution; loaders see contiguous IDs.

Output: code/benchmark/data/processed_ml20m_sub163/
        train.csv, val.csv, test.csv  (userId, movieId, timestamp — contiguous IDs)
        stats.json                    (main-pipeline schema: n_users / n_items / ...)
        item_map.json, user_map.json  (str(orig_processed_id) -> new_contiguous_id)
        subsampled_to_ml20m_movieId.json  (str(new_id) -> ml20m_orig_movieId)
                                      (used by FeatureLoader to fetch correct
                                       embedding rows for subsampled items)

Reproducibility: subsample seed = 42 (PROTOCOL seed, distinct from the 5
evaluation seeds {42, 123, 456, 789, 2026}). Re-running yields identical splits.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _paths import code  # dual-layout paths: dev tree or published release

ROOT = code("benchmark", "data", "processed")
OUT  = code("benchmark", "data", "processed_ml20m_sub163")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_INT_PER_ITEM = 163
SUBSAMPLE_SEED = 42


def kcore(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    while True:
        u_count = df.groupby("userId").size()
        i_count = df.groupby("movieId").size()
        u_keep = u_count[u_count >= k].index
        i_keep = i_count[i_count >= k].index
        new = df[df.userId.isin(u_keep) & df.movieId.isin(i_keep)]
        if len(new) == len(df):
            return new
        df = new


def main() -> None:
    # 1) Load source (already contiguous IDs in [0, n_users), [0, n_items))
    train = pd.read_csv(ROOT / "train.csv")
    val   = pd.read_csv(ROOT / "val.csv")
    test  = pd.read_csv(ROOT / "test.csv")
    print(f"Source ML-20M: {len(train):,} train / {train.userId.nunique():,} u / "
          f"{train.movieId.nunique():,} i / {len(train)/train.movieId.nunique():.0f} int/item")

    # 2) User-uniform subsample to hit target total interactions
    target_total = TARGET_INT_PER_ITEM * train.movieId.nunique()
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    u_int = train.groupby("userId").size()
    users = u_int.index.values.copy()
    rng.shuffle(users)
    cum = u_int.loc[users].cumsum().values
    n_keep = int(np.searchsorted(cum, target_total) + 1)
    kept = set(users[:n_keep].tolist())
    print(f"Sampled {n_keep:,} users  →  ~{cum[n_keep-1]:,} train interactions")

    # 3) Filter all 3 splits + 10-core on train
    train = train[train.userId.isin(kept)].reset_index(drop=True)
    val   = val[val.userId.isin(kept)].reset_index(drop=True)
    test  = test[test.userId.isin(kept)].reset_index(drop=True)
    train = kcore(train, k=10).reset_index(drop=True)

    items_kept = set(train.movieId.unique().tolist())
    users_kept = set(train.userId.unique().tolist())
    val   = val[val.userId.isin(users_kept) & val.movieId.isin(items_kept)].reset_index(drop=True)
    test  = test[test.userId.isin(users_kept) & test.movieId.isin(items_kept)].reset_index(drop=True)

    # 4) Build contiguous remappings (orig processed id → new sub-contiguous id)
    item_old_to_new = {int(o): i for i, o in enumerate(sorted(items_kept))}
    user_old_to_new = {int(o): i for i, o in enumerate(sorted(users_kept))}
    for df in (train, val, test):
        df["userId"]  = df["userId"].map(user_old_to_new).astype("int64")
        df["movieId"] = df["movieId"].map(item_old_to_new).astype("int64")

    # 5) For embedding loader: subsampled-contiguous → ML-20M-original-movieId
    with open(ROOT / "item_map.json") as f:
        ml20m_main_map = json.load(f)  # {str(ml20m_orig_id): processed_contiguous_id}
    inv_main = {int(v): int(k) for k, v in ml20m_main_map.items()}
    sub_to_ml20m = {str(new): inv_main[old] for old, new in item_old_to_new.items() if old in inv_main}

    # 6) Write outputs
    train.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    with open(OUT / "item_map.json", "w") as f:
        json.dump({str(k): v for k, v in item_old_to_new.items()}, f)
    with open(OUT / "user_map.json", "w") as f:
        json.dump({str(k): v for k, v in user_old_to_new.items()}, f)
    with open(OUT / "subsampled_to_ml20m_movieId.json", "w") as f:
        json.dump(sub_to_ml20m, f)

    n_u = int(train.userId.max() + 1)
    n_i = int(train.movieId.max() + 1)
    counts = train.groupby("movieId").size()
    stats = {
        "n_users": n_u,
        "n_items": n_i,
        "n_train": int(len(train)),
        "n_val":   int(len(val)),
        "n_test":  int(len(test)),
        "density": float(len(train) / (n_u * n_i)),
        "n_zero_shot_items": 0,
        "n_cold_items":   int((counts < 10).sum()),
        "n_medium_items": int(((counts >= 10) & (counts < 50)).sum()),
        "n_warm_items":   int((counts >= 50).sum()),
        "subsample_strategy": "user-uniform random + 10-core",
        "subsample_seed": SUBSAMPLE_SEED,
        "density_int_per_item":  float(len(train) / n_i),
        "density_items_per_user": float(len(train) / n_u),
    }
    with open(OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== FINAL ===")
    print(f"  users / items / train: {n_u:,} / {n_i:,} / {len(train):,}")
    print(f"  per-item density: {stats['density_int_per_item']:.1f}  (target ~{TARGET_INT_PER_ITEM})")
    print(f"  per-user density: {stats['density_items_per_user']:.1f}")
    print(f"  cold/medium/warm items: {stats['n_cold_items']} / {stats['n_medium_items']} / {stats['n_warm_items']}")
    print(f"  output dir: {OUT}/")


if __name__ == "__main__":
    main()
