#!/usr/bin/env python3
"""
Generate BERT(title+genre) baseline embeddings.

Encodes "Title | Genre1, Genre2, ..." using the same sentence-transformer
as the profile embeddings (default: bge-large-en-v1.5, 1024-dim, no PCA).
This controls for the embedding method — isolating the value of LLM-generated
profile content vs. naive title text.
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MOVIES_CSV, MOVIE_ID_INDEX, EMBED_DIM, BERT_TITLE_EMB_NPY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import the path instead of restating it. This used to spell out
# output/bge-large-en-v1.5/bert_title_embeddings.npy while the benchmark reads
# config.BERT_TITLE_EMB_NPY, which is the PARENT output/ dir on purpose -- the
# title embeddings do not depend on which encoder produced the profiles. So a
# reader who ran this exactly as documented still could not run M3: the
# generator wrote one path and the loader read another.
OUTPUT_PATH = BERT_TITLE_EMB_NPY


def main():
    # Checked before the heavy imports and before pandas opens anything, so a
    # reader who has not fetched MovieLens yet gets the command to run instead of
    # a FileNotFoundError three frames down. The path named is the one
    # scripts/download_ml20m.sh creates and docs/REPRODUCIBILITY.md documents.
    if not MOVIES_CSV.exists():
        raise SystemExit(
            f"MovieLens 20M is not present: {MOVIES_CSV} does not exist.\n"
            f"Fetch it first (it is not redistributed with this release):\n"
            f"    bash scripts/download_ml20m.sh\n"
            f"or point this script at an existing copy:\n"
            f"    ML20M_DIR=/path/to/ml-20m python3 features/bert_baseline.py")

    import pandas as pd
    from sentence_transformers import SentenceTransformer

    # Load movie metadata
    logger.info("Loading movies.csv...")
    movies = pd.read_csv(MOVIES_CSV)
    movies["genres_clean"] = movies["genres"].str.replace("|", ", ")

    # Load genome movie ID index (same order as other embeddings)
    with open(MOVIE_ID_INDEX) as f:
        movie_ids = json.load(f)

    # Build title+genre text for each movie
    mid_to_text = {}
    for _, row in movies.iterrows():
        mid = row["movieId"]
        title = row["title"]
        genres = row["genres_clean"]
        mid_to_text[mid] = f"{title} | {genres}"

    texts = [mid_to_text.get(mid, "Unknown") for mid in movie_ids]
    logger.info(f"Prepared {len(texts)} texts for encoding")

    # Encode with same model as profile embeddings
    model_name = "BAAI/bge-large-en-v1.5"
    logger.info(f"Loading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info("Encoding titles...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Raw shape: {embeddings.shape}")

    # No PCA — keep full 1024-dim to match LLM profile embeddings for fair comparison.
    # The side-feature projection MLP handles dimensionality reduction.
    logger.info(f"No PCA applied. Keeping full {embeddings.shape[1]}-dim for fair comparison with LLM profile.")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)
    logger.info(f"Saved: {OUTPUT_PATH} {embeddings.shape}")


if __name__ == "__main__":
    main()
