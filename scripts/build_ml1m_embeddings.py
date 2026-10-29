#!/usr/bin/env python3
"""Build ML-1M profile + mood + themes embeddings by subsetting the existing
ML-20M outputs (no new LLM generation needed since ML-1M items are a subset).

Strategy:
  1. Read ML-1M item_map.json -> {original_movieId: contiguous_id 0..n-1}
  2. For each ML-1M item, look up its movieId in the ML-20M
     `movie_id_index.json` (which maps row-index -> original_movieId for the
     10,381 genome-covered ML-20M movies).
  3. For ML-1M items present in ML-20M genome set: copy the row of
     profile_embeddings, mood_vectors, theme_matrix, combined_features.
  4. For ML-1M items NOT in the genome set: leave the row as zeros (consistent
     with the Amazon-Books missing-items policy; FeatureLoader handles
     transparently).
  5. Save under code/embedding_generator/output_ml1m/bge-large-en-v1.5/

Output files (row i = the ML-1M item with contiguous id i):
  profile_embeddings.npy   (n, 1024)
  mood_vectors.npy         (n, 10)
  theme_matrix.npy         (n, n_themes)
  combined_features.npy    (n, 1034)   [profile + mood — M7 input]
  combined_full.npy        (n, ?)      [profile + mood + themes]
  movie_id_index.json      the MovieLens movieId of each row, in row order
  embedding_metadata.json

The index holds movieIds, as the ML-20M index does, because that is what the
FeatureLoader looks up: it maps each item_map.json key (an original movieId) through
this file. Row i of every array is the ML-20M row of the movieId at position i of the
index; the arrays are slices of the ML-20M pack, so each row can be checked against
the ML-20M row for that movieId.

    python3 scripts/build_ml1m_embeddings.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
ML1M_DATA_DIR = code("benchmark", "data", "processed_ml1m")
ML20M_EMB_DIR = code("embedding_generator", "output", "bge-large-en-v1.5")
OUT_DIR = code("embedding_generator", "output_ml1m", "bge-large-en-v1.5")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_ml1m_emb")


def ml20m_file(name: str) -> Path:
    """A file of the ML-20M pack: beside its arrays, else one directory up.

    scripts/download_artifacts.py writes the whole pack into the encoder directory,
    and that is where an index describing these arrays belongs. The development tree
    may also keep copies one level up, so that location is tried second.
    """
    for p in (ML20M_EMB_DIR / name, ML20M_EMB_DIR.parent / name):
        if p.is_file():
            return p
    sys.exit(f"missing {ML20M_EMB_DIR / name}\n"
             "Fetch the ML-20M pack first:  python3 scripts/download_artifacts.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where to write the pack (default: %(default)s)")
    out_dir = ap.parse_args().out_dir
    if not (ML1M_DATA_DIR / "item_map.json").is_file():
        sys.exit(f"missing {ML1M_DATA_DIR / 'item_map.json'}\n"
                 "Build the ML-1M splits first:  python3 scripts/prepare_ml1m.py")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ML-1M item map: {orig_movieId_str: contiguous_id_int}
    item_map = json.loads((ML1M_DATA_DIR / "item_map.json").read_text())
    n_ml1m = len(item_map)
    logger.info(f"ML-1M items: {n_ml1m}")

    # Inverse: contiguous_id -> orig_movieId_int
    cid_to_orig = {int(v): int(k) for k, v in item_map.items()}

    # 2. Load ML-20M profile-row index: list of orig movieIds in profile_embeddings row order
    ml20m_movie_ids = json.loads(ml20m_file("movie_id_index.json").read_text())
    orig_to_row = {int(mid): row for row, mid in enumerate(ml20m_movie_ids)}
    logger.info(f"ML-20M genome-covered items in profile pack: {len(ml20m_movie_ids)}")

    # 3. Determine intersection
    present, missing = [], []
    alignment = {}  # ml1m_contiguous_id -> ml20m_row_index
    for cid in range(n_ml1m):
        orig = cid_to_orig[cid]
        if orig in orig_to_row:
            alignment[cid] = orig_to_row[orig]
            present.append(cid)
        else:
            missing.append(cid)
    logger.info(f"  present in genome: {len(present)} ({100*len(present)/n_ml1m:.1f}%)")
    logger.info(f"  missing (zero-vector content features): {len(missing)} "
                f"({100*len(missing)/n_ml1m:.1f}%)")
    if missing[:5]:
        logger.info(f"  first missing orig IDs: {[cid_to_orig[c] for c in missing[:5]]}...")

    # 4. Subset the .npy files
    def subset(name: str):
        arr = np.load(ml20m_file(f"{name}.npy"))
        logger.info(f"  loaded {name}.npy: shape {arr.shape}")
        out = np.zeros((n_ml1m, arr.shape[1]), dtype=arr.dtype)
        for cid, row in alignment.items():
            out[cid] = arr[row]
        np.save(out_dir / f"{name}.npy", out)
        logger.info(f"    -> {name}.npy: {out.shape}, {100*(out.any(axis=1)).sum()/n_ml1m:.1f}% non-zero rows")
        return out

    profile = subset("profile_embeddings")
    mood = subset("mood_vectors")
    themes = subset("theme_matrix")
    combo = subset("combined_features")
    combo_full = subset("combined_full")

    # 5. Movie ID index: row i -> the MovieLens movieId of the item in row i. The
    #    FeatureLoader maps item_map.json's keys (movieIds) through this list.
    index = [cid_to_orig[cid] for cid in range(n_ml1m)]
    assert all(int(ml20m_movie_ids[row]) == index[cid] for cid, row in alignment.items())
    (out_dir / "movie_id_index.json").write_text(json.dumps(index))

    # Theme vocabulary (copy verbatim from ML-20M — same vocabulary applies)
    (out_dir / "theme_vocabulary.json").write_text(
        ml20m_file("theme_vocabulary.json").read_text())
    src_meta = json.loads(ml20m_file("embedding_metadata.json").read_text())

    # Metadata
    meta = {
        "source": "subset of ML-20M LLM profile pack (no new LLM generation)",
        # where the source pack ships in the data release, not where it sat on the
        # machine that ran this: a local path here is a development path in a
        # published file, and the release copy had to be hand-edited to remove it
        "ml20m_source_dir": f"embeddings/ml20m/{ML20M_EMB_DIR.name}",
        "n_ml1m_items": n_ml1m,
        "n_with_profile": len(present),
        "n_missing": len(missing),
        "missing_orig_ids": [cid_to_orig[c] for c in missing],
        "encoder": "bge-large-en-v1.5 (inherited)",
        "shapes": {
            "profile_embeddings": list(profile.shape),
            "mood_vectors": list(mood.shape),
            "theme_matrix": list(themes.shape),
            "combined_features": list(combo.shape),
            "combined_full": list(combo_full.shape),
        },
        # the composites are cut from the source's, so its layout is theirs
        "feature_layout": src_meta["feature_layout"],
    }
    (out_dir / "embedding_metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"\nDone. Output in {out_dir}/")
    logger.info(f"  combined_features (M7 input): {combo.shape}, "
                f"non-zero rows: {(combo.any(axis=1)).sum()}/{n_ml1m}")


if __name__ == "__main__":
    main()
