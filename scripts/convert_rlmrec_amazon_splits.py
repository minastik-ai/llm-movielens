#!/usr/bin/env python3
"""Convert RLMRec Amazon-Books pickled sparse matrices to the benchmark CSV
format expected by code/benchmark/run_ablation.py.

RLMRec uses scipy.sparse coo_matrix files (trn_mat.pkl, val_mat.pkl,
tst_mat.pkl) with shape (n_users, n_items). The benchmark expects
processed/train.csv, val.csv, test.csv with columns
(userId, movieId, timestamp), plus item_map.json and stats.json.

Since the RLMRec splits are not temporal (they are random non-overlapping
splits of the implicit interaction matrix), we synthesize timestamps so the
ordering inside each split is deterministic, but we keep the splits exactly as
RLMRec defined them. This preserves direct numerical comparability with
RLMRec's published numbers, and the benchmark's evaluation does not require
true temporal ordering — it only uses train items to mask user histories at
eval time.

Output (under code/benchmark/data/processed_amazon/):
    train.csv, val.csv, test.csv      columns: userId, movieId, timestamp
    item_map.json                     {rlmrec_iid: bench_id}, the identity: RLMRec's item ids ARE the
                                      bench ids, and the released book profiles are keyed by them
    iid_to_asin.json                  {rlmrec_iid: asin}, for joining back to Amazon metadata
    user_map.json                     {rlmrec_uid: bench_user_id}
    stats.json                        n_users, n_items, density, bucket counts
"""
from __future__ import annotations
import argparse

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from _paths import code, require  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
RLMREC_DIR = code("benchmark", "external", "RLMRec", "data", "amazon")
MAPPER_DIR = code("benchmark", "external", "RLMRec", "data", "mapper")
OUT_DIR = code("benchmark", "data", "processed_amazon")
# Amazon-Books is RLMRec's published split, which this release does not
# redistribute: the reader fetches RLMRec's data folder, and this turns it into the
# harness's CSVs. Every Amazon runner reads what this writes.
FETCH = ("Amazon-Books is RLMRec's published split, which this release does not\n"
         "redistribute. Two parts, both from the RLMRec project:\n"
         "  1. Download https://archive.org/download/rlmrec_data/data.zip (about 475 MB),\n"
         "     unzip it, and copy its amazon/ folder (trn_mat.pkl, val_mat.pkl, tst_mat.pkl)\n"
         f"     to {RLMREC_DIR}/\n"
         "  2. From https://github.com/HKUDS/RLMRec take data/mapper/amazon_item.json\n"
         "     (and amazon_user.json) and place them in\n"
         f"     {MAPPER_DIR}/")


def load_mapper():
    """Return iid_to_asin (dict) for items."""
    iid_to_asin = {}
    with open(MAPPER_DIR / "amazon_item.json") as f:
        for line in f:
            obj = json.loads(line)
            iid_to_asin[int(obj["iid"])] = obj["asin"]
    return iid_to_asin


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    require(MAPPER_DIR / "amazon_item.json", "RLMRec's Amazon item mapper", FETCH)
    for fname in ("trn_mat.pkl", "val_mat.pkl", "tst_mat.pkl"):
        require(RLMREC_DIR / fname, "an RLMRec Amazon-Books split", FETCH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iid_to_asin = load_mapper()
    print(f"Loaded {len(iid_to_asin)} item ASINs from mapper")

    # Read sparse matrices
    splits = {}
    for split, fname in [("train", "trn_mat.pkl"),
                         ("val", "val_mat.pkl"),
                         ("test", "tst_mat.pkl")]:
        with open(RLMREC_DIR / fname, "rb") as f:
            m = pickle.load(f)
        splits[split] = m
        print(f"  {split}: shape={m.shape}, nnz={m.nnz}")

    n_users, n_items = splits["train"].shape

    # Item map: keep RLMRec's iid as the contiguous bench id (0..n_items-1).
    # The benchmark code treats `movieId` as the original-domain id; for
    # consistency with the ML-20M pipeline we use the iid directly as both
    # keys. (For Amazon-Books there is no separate "original id" concept.)
    item_map = {str(iid): iid for iid in range(n_items)}

    # User map: identity (RLMRec uid = bench uid)
    user_map = {str(uid): uid for uid in range(n_users)}

    # Convert each split to (userId, movieId, timestamp) CSV rows
    base_ts = 1_000_000_000  # synthetic monotonic timestamp; ordering is arbitrary
    for split, mat in splits.items():
        coo = mat.tocoo()
        rows = list(zip(coo.row.tolist(), coo.col.tolist()))
        rows.sort()  # deterministic ordering by (user, item)
        out = OUT_DIR / f"{split}.csv"
        with open(out, "w") as f:
            f.write("userId,movieId,timestamp\n")
            for i, (u, it) in enumerate(rows):
                f.write(f"{u},{it},{base_ts + i}\n")
        print(f"  wrote {out.relative_to(REPO)}: {len(rows)} rows")

    # Compute item bucket statistics (training-interaction count)
    item_train_counts = defaultdict(int)
    coo = splits["train"].tocoo()
    for it in coo.col.tolist():
        item_train_counts[it] += 1

    cold = sum(1 for c in item_train_counts.values() if c < 10)
    medium = sum(1 for c in item_train_counts.values() if 10 <= c < 50)
    warm = sum(1 for c in item_train_counts.values() if c >= 50)
    # items with zero training interactions (zero-shot)
    n_zero_shot = n_items - len(item_train_counts)
    cold_inc_zero = cold + n_zero_shot

    stats = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_train": int(splits["train"].nnz),
        "n_val": int(splits["val"].nnz),
        "n_test": int(splits["test"].nnz),
        "density": float(splits["train"].nnz / (n_users * n_items)),
        "n_zero_shot_items": int(n_zero_shot),
        "n_cold_items": int(cold_inc_zero),    # <10 train interactions (incl. zero-shot)
        "n_medium_items": int(medium),         # 10-49
        "n_warm_items": int(warm),             # >=50
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {(OUT_DIR / 'stats.json').relative_to(REPO)}")
    print(f"  stats: {stats}")

    with open(OUT_DIR / "item_map.json", "w") as f:
        json.dump(item_map, f)
    with open(OUT_DIR / "user_map.json", "w") as f:
        json.dump(user_map, f)

    # Cross-reference: also write a separate iid-to-asin lookup (NOT used by
    # the benchmark, but keeps the chain auditable).
    with open(OUT_DIR / "iid_to_asin.json", "w") as f:
        json.dump({str(k): v for k, v in iid_to_asin.items()}, f)

    print()
    print("Phase 0e complete: RLMRec splits converted.")
    print(f"  Output: {OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
