#!/usr/bin/env python3
"""
Generate profile embeddings using e5-large-v2 for encoder sensitivity analysis.

This produces the same profile embeddings as bge-large-en-v1.5 but with a different
encoder, to verify that downstream results are encoder-agnostic (cross-encoder validation).

Usage:
    python scripts/generate_e5_embeddings.py

Inputs: the released profiles (scripts/download_artifacts.py), the bge-large-en-v1.5
feature directory (for the row order and the encoder-independent features), and
MovieLens 20M's movies.csv (scripts/download_ml20m.sh) for the titles.

Output, in src/embedding_generator/output/e5-large-v2/:
    profile_embeddings.npy, bert_title_embeddings.npy  (10381, 1024)
    combined_features.npy, combined_full.npy           recomputed with the e5 profiles
    movie_id_index.json, theme_vocabulary.json and the encoder-independent arrays,
    copied from the bge-large-en-v1.5 directory
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os
from _paths import REPO, code, require  # dual-layout paths: dev tree or published release

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/e5-large-v2"
OUTPUT_DIR = code("embedding_generator", "output", "e5-large-v2")
BGE_DIR = code("embedding_generator", "output", "bge-large-en-v1.5")
PROFILES_PATH = code("profile_generator", "llm-movie-profiler-v1-20260402", "output", "movie_profiles.json")
INDEX_PATH = BGE_DIR / "movie_id_index.json"
# MovieLens 20M: ML20M_DIR if set, else where scripts/download_ml20m.sh puts it,
# else the generation layout.
_ML20M = [Path(os.environ["ML20M_DIR"])] if os.environ.get("ML20M_DIR") else [
    REPO / "data" / "raw" / "ml-20m",
    code("profile_generator", "llm-movie-profiler-v1-20260402", "data", "ml-20m")]
MOVIES_CSV = next((d / "movies.csv" for d in _ML20M if (d / "movies.csv").exists()),
                  _ML20M[0] / "movies.csv")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import pandas as pd

    require(PROFILES_PATH, "the released profiles", "python3 scripts/download_artifacts.py")
    require(INDEX_PATH, "the bge-large-en-v1.5 features", "python3 scripts/download_artifacts.py")
    require(MOVIES_CSV, "MovieLens 20M", "bash scripts/download_ml20m.sh")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load movie ID index
    with open(INDEX_PATH) as f:
        movie_ids = json.load(f)
    logger.info(f"Movie ID index: {len(movie_ids)} movies")

    # Load profiles
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)

    # Load movies for title+genre
    movies = pd.read_csv(MOVIES_CSV)
    movies["genres_clean"] = movies["genres"].str.replace("|", ", ")
    mid_to_title_genre = {
        row["movieId"]: f"{row['title']} | {row['genres_clean']}"
        for _, row in movies.iterrows()
    }

    # Prepare texts
    profile_texts = []
    title_texts = []
    for mid in movie_ids:
        p = profiles.get(str(mid), {})
        # e5 models require "query: " or "passage: " prefix
        profile_text = p.get("profile", "Unknown movie")
        profile_texts.append(f"passage: {profile_text}")
        title_texts.append(f"passage: {mid_to_title_genre.get(mid, 'Unknown')}")

    logger.info(f"Prepared {len(profile_texts)} profile texts and {len(title_texts)} title texts")

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode profiles
    logger.info("Encoding profiles...")
    profile_emb = model.encode(profile_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Profile embeddings: {profile_emb.shape}")
    np.save(OUTPUT_DIR / "profile_embeddings.npy", profile_emb.astype(np.float32))

    # Encode titles
    logger.info("Encoding titles...")
    title_emb = model.encode(title_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Title embeddings: {title_emb.shape}")
    np.save(OUTPUT_DIR / "bert_title_embeddings.npy", title_emb.astype(np.float32))

    # The encoder-independent files are the bge directory's. genome_embeddings.npy is
    # copied only if it has been built (it is not in the data release).
    import shutil
    for fname in ["movie_id_index.json", "theme_vocabulary.json", "mood_vectors.npy",
                  "theme_matrix.npy", "genome_embeddings.npy"]:
        if (BGE_DIR / fname).exists():
            shutil.copy2(BGE_DIR / fname, OUTPUT_DIR / fname)
            logger.info(f"Copied {fname}")

    # Recompute combined features with new profile embeddings
    mood = np.load(BGE_DIR / "mood_vectors.npy")
    combined = np.concatenate([profile_emb, mood], axis=1)
    np.save(OUTPUT_DIR / "combined_features.npy", combined.astype(np.float32))
    logger.info(f"Combined features: {combined.shape}")

    theme = np.load(BGE_DIR / "theme_matrix.npy")
    combined_full = np.concatenate([profile_emb, mood, theme], axis=1)
    np.save(OUTPUT_DIR / "combined_full.npy", combined_full.astype(np.float32))
    logger.info(f"Combined full: {combined_full.shape}")

    logger.info(f"\nDone! All embeddings saved to {OUTPUT_DIR}/")
    logger.info(f"\nTo run M4 with e5-large-v2:")
    logger.info(f"  cd src/benchmark && EMBEDDING_DIR={OUTPUT_DIR} python3 run_experiment.py --config M4 --seed 42")


if __name__ == "__main__":
    main()
