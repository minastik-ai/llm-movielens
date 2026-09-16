#!/usr/bin/env python3
"""
Generate profile embeddings using e5-large-v2 for encoder sensitivity analysis.

This produces the same profile embeddings as bge-large-en-v1.5 but with a different
encoder, to verify that downstream results are encoder-agnostic (cross-encoder validation).

Usage:
    python scripts/generate_e5_embeddings.py

Output:
    code/embedding_generator/output/e5-large-v2/profile_embeddings.npy  (10381, 1024)
    code/embedding_generator/output/e5-large-v2/bert_title_embeddings.npy  (10381, 1024)
"""

import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from _paths import code  # dual-layout paths: dev tree or published release

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/e5-large-v2"
OUTPUT_DIR = code("embedding_generator", "output", "e5-large-v2")
PROFILES_PATH = code("profile_generator", "llm-movie-profiler-v1-20260402", "output", "movie_profiles.json")
MOVIES_CSV = code("profile_generator", "llm-movie-profiler-v1-20260402", "data", "ml-20m", "movies.csv")
INDEX_PATH = code("embedding_generator", "output", "movie_id_index.json")


def main():
    import pandas as pd

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

    # Copy genome, mood, theme files from bge-large directory (they don't depend on encoder)
    import shutil
    bge_dir = code("embedding_generator", "output", "bge-large-en-v1.5")
    for fname in ["genome_embeddings.npy", "mood_vectors.npy", "theme_matrix.npy",
                  "combined_features.npy", "combined_full.npy"]:
        src = bge_dir / fname
        if src.exists():
            # For combined features, we need to recompute with new profile embeddings
            if "combined" in fname:
                continue
            shutil.copy2(src, OUTPUT_DIR / fname)
            logger.info(f"Copied {fname}")

    # Recompute combined features with new profile embeddings
    mood = np.load(bge_dir / "mood_vectors.npy")
    combined = np.concatenate([profile_emb, mood], axis=1)
    np.save(OUTPUT_DIR / "combined_features.npy", combined.astype(np.float32))
    logger.info(f"Combined features: {combined.shape}")

    theme = np.load(bge_dir / "theme_matrix.npy")
    combined_full = np.concatenate([profile_emb, mood, theme], axis=1)
    np.save(OUTPUT_DIR / "combined_full.npy", combined_full.astype(np.float32))
    logger.info(f"Combined full: {combined_full.shape}")

    logger.info(f"\nDone! All embeddings saved to {OUTPUT_DIR}/")
    logger.info(f"\nTo run M4 with e5-large-v2:")
    logger.info(f"  EMBEDDING_DIR={OUTPUT_DIR} python -m llm_movielens.benchmark.run_experiment --config M4 --seed 42")


if __name__ == "__main__":
    main()
