"""
Movie Embedding Generator.

Produces multiple embedding types from LLM-generated movie profiles:
  A. Profile text  → sentence-transformer → dense vector (384 or 768-dim) → optional PCA
  B. Mood vector   → direct 10-dim numeric feature (no encoding needed)
  C. Key themes    → multi-hot encoding over vocabulary
  D. Genome tags   → 1128-dim relevance vector → optional PCA (baseline comparison)
  E. Combined      → concatenation of A + B + C for unified item features
"""

import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from config import (
    PROFILES_JSON, MOOD_AXES, ENCODE_BATCH_SIZE,
    GENOME_SCORES_CSV, GENOME_TAGS_CSV, THEME_MIN_COUNT,
)

logger = logging.getLogger(__name__)


def load_profiles(path: Path = PROFILES_JSON) -> Tuple[List[int], Dict]:
    """Load profiles and return (sorted_movie_ids, profiles_dict)."""
    with open(path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    movie_ids = sorted(int(k) for k in profiles.keys())
    logger.info(f"Loaded {len(movie_ids):,} profiles from {path}")
    return movie_ids, profiles


# Models with custom implementations that are incompatible with MPS
_CPU_ONLY_MODELS = {
    "Alibaba-NLP/gte-large-en-v1.5",  # custom RoPE embedding fails on MPS
}


def _needs_cpu_fallback(model_name: str) -> bool:
    """Check if a model requires CPU due to known MPS incompatibilities."""
    import platform
    if platform.processor() != "arm" and not platform.processor().startswith("Apple"):
        # Not Apple Silicon — no MPS concern
        return False
    return model_name in _CPU_ONLY_MODELS


# ──────────────────────────────────────────────────────────────────────────────
# A. PROFILE TEXT EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────
def encode_profiles(
    movie_ids: List[int],
    profiles: Dict,
    model_name: str,
    batch_size: int = ENCODE_BATCH_SIZE,
    pca_dims: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Encode profile text using a sentence-transformer model.
    Returns (embeddings, pca_model_or_None).
    """
    from sentence_transformers import SentenceTransformer

    # Some models (e.g. gte-large with custom RoPE) fail on MPS; use CPU for those
    device = "cpu" if _needs_cpu_fallback(model_name) else None
    logger.info(f"Loading sentence-transformer: {model_name}" +
                (f" (forced device={device})" if device else ""))
    model = SentenceTransformer(model_name, trust_remote_code=True,
                                **({"device": device} if device else {}))

    texts = [profiles[str(mid)]["profile"] for mid in movie_ids]
    logger.info(f"Encoding {len(texts):,} profiles (batch_size={batch_size}, device={model.device})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"  Raw embedding shape: {embeddings.shape}")

    pca_model = None
    if pca_dims and pca_dims < embeddings.shape[1]:
        logger.info(f"  Applying PCA: {embeddings.shape[1]} → {pca_dims} dims")
        pca_model = PCA(n_components=pca_dims, random_state=42)
        embeddings = pca_model.fit_transform(embeddings)
        embeddings = normalize(embeddings)  # re-normalize after PCA
        variance = sum(pca_model.explained_variance_ratio_)
        logger.info(f"  PCA explained variance: {variance:.4f}")

    logger.info(f"  Final profile embedding shape: {embeddings.shape}")
    return embeddings, pca_model


# ──────────────────────────────────────────────────────────────────────────────
# B. MOOD VECTORS
# ──────────────────────────────────────────────────────────────────────────────
def extract_mood_vectors(
    movie_ids: List[int],
    profiles: Dict,
) -> np.ndarray:
    """Extract 10-dim mood vectors. Shape: (n_movies, 10)."""
    mood_matrix = np.array([
        [profiles[str(mid)]["mood_vector"][axis] for axis in MOOD_AXES]
        for mid in movie_ids
    ], dtype=np.float32)
    logger.info(f"Mood vectors shape: {mood_matrix.shape}")
    return mood_matrix


# ──────────────────────────────────────────────────────────────────────────────
# C. KEY THEMES (MULTI-HOT)
# ──────────────────────────────────────────────────────────────────────────────
def encode_key_themes(
    movie_ids: List[int],
    profiles: Dict,
    min_count: int = THEME_MIN_COUNT,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build multi-hot encoding over filtered theme vocabulary.
    Only keeps themes appearing in >= min_count movies.
    Returns (theme_matrix, theme_vocabulary).
    """
    from collections import Counter

    # Count theme frequencies across all movies
    theme_counts = Counter()
    for mid in movie_ids:
        themes = profiles[str(mid)].get("key_themes", [])
        theme_counts.update(t.lower().strip() for t in themes)

    # Filter to themes with sufficient frequency
    vocab = sorted(t for t, c in theme_counts.items() if c >= min_count)
    theme_to_idx = {t: i for i, t in enumerate(vocab)}

    total_themes = len(theme_counts)
    logger.info(f"Theme filtering: {total_themes} total → {len(vocab)} with count >= {min_count}")

    # Encode
    theme_matrix = np.zeros((len(movie_ids), len(vocab)), dtype=np.float32)
    for i, mid in enumerate(movie_ids):
        themes = profiles[str(mid)].get("key_themes", [])
        for t in themes:
            idx = theme_to_idx.get(t.lower().strip())
            if idx is not None:
                theme_matrix[i, idx] = 1.0

    logger.info(f"Key themes: {len(vocab)} themes, matrix shape: {theme_matrix.shape}")
    return theme_matrix, vocab


# ──────────────────────────────────────────────────────────────────────────────
# D. GENOME TAG VECTORS (BASELINE)
# ──────────────────────────────────────────────────────────────────────────────
def load_genome_vectors(
    movie_ids: List[int],
    pca_dims: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Load raw 1128-dim genome relevance vectors for each movie.
    Optionally apply PCA. This is the baseline to compare against LLM embeddings.
    """
    logger.info("Loading genome-scores.csv...")
    genome_scores = pd.read_csv(GENOME_SCORES_CSV)
    genome_tags = pd.read_csv(GENOME_TAGS_CSV)

    n_tags = len(genome_tags)
    mid_set = set(movie_ids)
    mid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # Build dense matrix: (n_movies, n_tags)
    genome_matrix = np.zeros((len(movie_ids), n_tags), dtype=np.float32)

    logger.info(f"Building {len(movie_ids)}x{n_tags} genome matrix...")
    for _, row in genome_scores.iterrows():
        mid = int(row["movieId"])
        if mid in mid_set:
            tag_idx = int(row["tagId"]) - 1  # tagId is 1-indexed
            genome_matrix[mid_to_idx[mid], tag_idx] = row["relevance"]

    logger.info(f"  Raw genome matrix shape: {genome_matrix.shape}")

    pca_model = None
    if pca_dims and pca_dims < n_tags:
        logger.info(f"  Applying PCA: {n_tags} → {pca_dims} dims")
        pca_model = PCA(n_components=pca_dims, random_state=42)
        genome_matrix = pca_model.fit_transform(genome_matrix)
        genome_matrix = normalize(genome_matrix)
        variance = sum(pca_model.explained_variance_ratio_)
        logger.info(f"  PCA explained variance: {variance:.4f}")

    logger.info(f"  Final genome embedding shape: {genome_matrix.shape}")
    return genome_matrix, pca_model


# ──────────────────────────────────────────────────────────────────────────────
# E. COMBINED FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def build_combined_features(
    profile_emb: np.ndarray,
    mood_vectors: np.ndarray,
    theme_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Concatenate profile embeddings + mood vectors (+ optional themes).
    This is the recommended item feature for recommendation models.
    """
    parts = [profile_emb, mood_vectors]
    if theme_matrix is not None:
        parts.append(theme_matrix)
    combined = np.concatenate(parts, axis=1).astype(np.float32)
    logger.info(f"Combined feature shape: {combined.shape} "
                f"(profile={profile_emb.shape[1]} + mood={mood_vectors.shape[1]}"
                f"{f' + themes={theme_matrix.shape[1]}' if theme_matrix is not None else ''})")
    return combined
