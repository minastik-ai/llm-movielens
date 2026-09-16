#!/usr/bin/env python3
"""
Data preprocessing for ML-20M benchmark.

Pipeline:
  1. Load ratings.csv
  2. Filter to genome-covered movies (10,381)
  3. Convert to implicit feedback (rating >= 3.5)
  4. Apply iterative k-core filtering (users & items with >= K interactions)
  5. Temporal split into train / val / test
  6. Remap user/item IDs to contiguous 0-indexed integers
  7. Save processed splits + mappings

Output files (in data/processed/):
  - train.csv, val.csv, test.csv  (userId, itemId, timestamp)
  - user_map.json   {original_userId: contiguous_id}
  - item_map.json   {original_movieId: contiguous_id}
  - stats.json      {n_users, n_items, n_train, n_val, n_test, ...}
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RATINGS_CSV, MOVIE_ID_INDEX,
    POSITIVE_THRESHOLD, K_CORE,
    TRAIN_END_TS, VAL_END_TS, DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_filter_ratings() -> pd.DataFrame:
    """Load ratings, filter to genome-covered movies, convert to implicit."""
    logger.info("Loading ratings.csv...")
    df = pd.read_csv(
        RATINGS_CSV,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    logger.info(f"  Raw: {len(df):,} ratings, {df['userId'].nunique():,} users, {df['movieId'].nunique():,} items")

    # Filter to genome-covered movies
    with open(MOVIE_ID_INDEX) as f:
        genome_movie_ids = set(json.load(f))
    df = df[df["movieId"].isin(genome_movie_ids)]
    logger.info(f"  After genome filter: {len(df):,} ratings, {df['movieId'].nunique():,} items")

    # Convert to implicit feedback
    df = df[df["rating"] >= POSITIVE_THRESHOLD].drop(columns=["rating"])
    logger.info(f"  After implicit (>= {POSITIVE_THRESHOLD}): {len(df):,} interactions")

    # Deduplicate (keep first interaction per user-item pair)
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="first")
    logger.info(f"  After dedup: {len(df):,} interactions")

    return df


def k_core_filter(df: pd.DataFrame, k: int = K_CORE) -> pd.DataFrame:
    """Iterative k-core filtering: keep users and items with >= k interactions."""
    logger.info(f"Applying {k}-core filtering...")
    prev_len = 0
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        # Filter users
        user_counts = df["userId"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["userId"].isin(valid_users)]

        # Filter items
        item_counts = df["movieId"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["movieId"].isin(valid_items)]

        logger.info(f"  Iteration {iteration}: {len(df):,} interactions, "
                     f"{df['userId'].nunique():,} users, {df['movieId'].nunique():,} items")

    logger.info(f"  K-core done after {iteration} iterations")
    return df


def temporal_split(df: pd.DataFrame):
    """Split by timestamp: train / val / test."""
    train = df[df["timestamp"] < TRAIN_END_TS]
    val = df[(df["timestamp"] >= TRAIN_END_TS) & (df["timestamp"] < VAL_END_TS)]
    test = df[df["timestamp"] >= VAL_END_TS]

    logger.info(f"Temporal split:")
    logger.info(f"  Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    logger.info(f"  Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    logger.info(f"  Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")

    return train, val, test


def remap_ids(train, val, test):
    """Remap user and item IDs to contiguous 0-indexed integers."""
    # Build mappings from train set (users/items not in train are dropped from val/test)
    all_users = sorted(train["userId"].unique())
    all_items = sorted(train["movieId"].unique())

    user_map = {uid: i for i, uid in enumerate(all_users)}
    item_map = {mid: i for i, mid in enumerate(all_items)}

    def remap(df):
        df = df.copy()
        df["userId"] = df["userId"].map(user_map)
        df["movieId"] = df["movieId"].map(item_map)
        # Drop rows with unmapped IDs (users/items not in train)
        df = df.dropna(subset=["userId", "movieId"])
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        return df

    train_r = remap(train)
    val_r = remap(val)
    test_r = remap(test)

    # Filter val/test to users with at least 1 positive
    val_r = val_r[val_r["userId"].isin(train_r["userId"].unique())]
    test_r = test_r[test_r["userId"].isin(train_r["userId"].unique())]

    logger.info(f"After remapping:")
    logger.info(f"  Users: {len(user_map):,}, Items: {len(item_map):,}")
    logger.info(f"  Train: {len(train_r):,}, Val: {len(val_r):,}, Test: {len(test_r):,}")

    return train_r, val_r, test_r, user_map, item_map


def save_splits(train, val, test, user_map, item_map):
    """Save processed data to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save splits
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Saved {path} ({len(df):,} rows)")

    # Save ID mappings (original → contiguous)
    with open(DATA_DIR / "user_map.json", "w") as f:
        json.dump({str(k): v for k, v in user_map.items()}, f)

    with open(DATA_DIR / "item_map.json", "w") as f:
        json.dump({str(k): v for k, v in item_map.items()}, f)

    # Compute and save stats
    train_items_set = set(train["movieId"].unique())
    val_items_set = set(val["movieId"].unique())
    test_items_set = set(test["movieId"].unique())

    # Cold items: in test but not in train
    zero_shot_items = test_items_set - train_items_set

    # Item interaction counts in train
    item_counts = train.groupby("movieId").size()
    cold_items = set(item_counts[item_counts < 10].index)
    medium_items = set(item_counts[(item_counts >= 10) & (item_counts < 50)].index)
    warm_items = set(item_counts[item_counts >= 50].index)

    stats = {
        "n_users": len(user_map),
        "n_items": len(item_map),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "density": len(train) / (len(user_map) * len(item_map)),
        "n_zero_shot_items": len(zero_shot_items),
        "n_cold_items": len(cold_items),
        "n_medium_items": len(medium_items),
        "n_warm_items": len(warm_items),
    }
    with open(DATA_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Stats: {stats}")


def main():
    logger.info("=" * 60)
    logger.info("ML-20M Benchmark Data Preprocessing")
    logger.info("=" * 60)

    df = load_and_filter_ratings()
    df = k_core_filter(df)
    train, val, test = temporal_split(df)
    train, val, test, user_map, item_map = remap_ids(train, val, test)
    save_splits(train, val, test, user_map, item_map)

    logger.info("Done.")


if __name__ == "__main__":
    main()
