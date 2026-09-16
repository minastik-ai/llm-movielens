#!/usr/bin/env python3
"""ML-1M preprocessing for the LLM-MovieLens benchmark.

Mirrors `code/benchmark/data/preprocess.py` (which targets ML-20M) with
ML-1M-specific paths, file format (`::`-separated `.dat`), and split
timestamps proportional to ML-1M's 2000-04 → 2003-03 timespan.

Pipeline:
  1. Load ratings.dat
  2. Convert to implicit feedback (rating >= 3.5)
  3. Apply iterative k-core filtering (default k=10)
  4. Temporal split: train < 2002-12-01, val < 2003-01-01, test >= 2003-01-01
  5. Remap user/item IDs to contiguous 0-indexed integers
  6. Save processed splits + mappings + stats

Output files (under code/benchmark/data/processed_ml1m/):
  - train.csv, val.csv, test.csv  (userId, movieId, timestamp)
  - user_map.json   {original_userId: contiguous_id}
  - item_map.json   {original_movieId: contiguous_id}
  - stats.json      {n_users, n_items, density, bucket counts, ...}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = code("profile_generator", "llm-movie-profiler-v1-20260402", "data", "ml-1m")
OUT_DIR = code("benchmark", "data", "processed_ml1m")

POSITIVE_THRESHOLD = 3.5
K_CORE = 10
# Percentile-based temporal split: train=earliest 80%, val=next 10%, test=last 10%.
# This is the dominant convention in the ML-1M literature (RLMRec, KAR, LightGCL)
# and gives a cleanly-sized eval set; fixed-date cutoffs on ML-1M's 3-year span
# leave too few users in val/test under k-core+positive filtering.
TRAIN_PCT = 0.80
VAL_PCT   = 0.10   # test = remaining 10%

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prep_ml1m")


def load_and_filter() -> pd.DataFrame:
    """Load ratings.dat, threshold to implicit, dedup."""
    logger.info("Loading ratings.dat...")
    df = pd.read_csv(
        RAW_DIR / "ratings.dat", sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    logger.info(f"  Raw: {len(df):,} ratings, {df['userId'].nunique():,} users, "
                f"{df['movieId'].nunique():,} items")

    df = df[df["rating"] >= POSITIVE_THRESHOLD].drop(columns=["rating"])
    logger.info(f"  After implicit (>= {POSITIVE_THRESHOLD}): {len(df):,} interactions")

    df = df.drop_duplicates(subset=["userId", "movieId"], keep="first")
    logger.info(f"  After dedup: {len(df):,} interactions")
    return df


def k_core_filter(df: pd.DataFrame, k: int = K_CORE) -> pd.DataFrame:
    logger.info(f"Applying {k}-core filtering...")
    prev_len, it = 0, 0
    while len(df) != prev_len:
        prev_len = len(df)
        it += 1
        u = df["userId"].value_counts()
        df = df[df["userId"].isin(u[u >= k].index)]
        i = df["movieId"].value_counts()
        df = df[df["movieId"].isin(i[i >= k].index)]
        logger.info(f"  iter {it}: {len(df):,} interactions, "
                    f"{df['userId'].nunique():,} users, {df['movieId'].nunique():,} items")
    logger.info(f"  k-core converged after {it} iterations")
    return df


def temporal_split(df: pd.DataFrame):
    """80/10/10 percentile temporal split (conventional for ML-1M).

    Sort by timestamp; first ``TRAIN_PCT`` of interactions = train, next
    ``VAL_PCT`` = val, remainder = test. Cuts are computed on raw row indices
    after sort, so each interaction is assigned to exactly one bucket
    deterministically.
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_PCT)
    val_end = int(n * (TRAIN_PCT + VAL_PCT))
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    logger.info(f"Percentile temporal split (train={TRAIN_PCT*100:.0f}%, "
                f"val={VAL_PCT*100:.0f}%, test={(1-TRAIN_PCT-VAL_PCT)*100:.0f}%):")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        logger.info(f"  {name}: {len(d):,} ({len(d)/n*100:.1f}%)")
    return train, val, test


def remap_ids(train, val, test):
    all_users = sorted(train["userId"].unique())
    all_items = sorted(train["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(all_users)}
    item_map = {mid: i for i, mid in enumerate(all_items)}

    def remap(df):
        df = df.copy()
        df["userId"] = df["userId"].map(user_map)
        df["movieId"] = df["movieId"].map(item_map)
        df = df.dropna(subset=["userId", "movieId"])
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        return df

    train_r = remap(train)
    val_r = remap(val)
    test_r = remap(test)
    val_r = val_r[val_r["userId"].isin(train_r["userId"].unique())]
    test_r = test_r[test_r["userId"].isin(train_r["userId"].unique())]
    logger.info(f"After remap: {len(user_map):,} users, {len(item_map):,} items, "
                f"train {len(train_r):,}, val {len(val_r):,}, test {len(test_r):,}")
    return train_r, val_r, test_r, user_map, item_map


def save_splits(train, val, test, user_map, item_map):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"  saved {path.relative_to(REPO)}: {len(df):,} rows")

    (OUT_DIR / "user_map.json").write_text(
        json.dumps({str(k): v for k, v in user_map.items()})
    )
    (OUT_DIR / "item_map.json").write_text(
        json.dumps({str(k): v for k, v in item_map.items()})
    )

    item_counts = train.groupby("movieId").size()
    cold = int((item_counts < 10).sum())
    medium = int(((item_counts >= 10) & (item_counts < 50)).sum())
    warm = int((item_counts >= 50).sum())
    train_items = set(train["movieId"].unique())
    test_items = set(test["movieId"].unique())
    n_zero_shot = len(test_items - train_items)

    stats = {
        "n_users": int(len(user_map)),
        "n_items": int(len(item_map)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "density": float(len(train) / (len(user_map) * len(item_map))),
        "n_zero_shot_items": int(n_zero_shot),
        "n_cold_items": cold,
        "n_medium_items": medium,
        "n_warm_items": warm,
        "split_method": "80/10/10 percentile temporal",
        "split_train_pct": TRAIN_PCT,
        "split_val_pct": VAL_PCT,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"  stats: {stats}")


def main():
    logger.info("=" * 60)
    logger.info("ML-1M Preprocessing")
    logger.info("=" * 60)
    df = load_and_filter()
    df = k_core_filter(df)
    tr, va, te = temporal_split(df)
    tr, va, te, um, im = remap_ids(tr, va, te)
    save_splits(tr, va, te, um, im)
    logger.info(f"\nDone. Output in {OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
