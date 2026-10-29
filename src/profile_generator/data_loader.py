"""
Data loader for MovieLens 20M.
Loads genome-covered movies, extracts top-K genome tags, aggregates user tags and ratings.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from config.settings import (
    MOVIES_CSV, RATINGS_CSV, TAGS_CSV, LINKS_CSV,
    GENOME_SCORES_CSV, GENOME_TAGS_CSV, TOP_K_GENOME_TAGS,
)

logger = logging.getLogger(__name__)


def load_genome_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load genome scores and genome tag names."""
    logger.info("Loading genome-tags.csv...")
    genome_tags = pd.read_csv(GENOME_TAGS_CSV)
    logger.info(f"  → {len(genome_tags)} genome tags loaded")

    logger.info("Loading genome-scores.csv (this may take ~30 seconds)...")
    genome_scores = pd.read_csv(GENOME_SCORES_CSV)
    logger.info(f"  → {len(genome_scores):,} genome score entries loaded")

    return genome_scores, genome_tags


def get_genome_covered_movie_ids(genome_scores: pd.DataFrame) -> List[int]:
    """Return sorted list of movieIds that have genome tag coverage."""
    movie_ids = sorted(genome_scores["movieId"].unique().tolist())
    logger.info(f"  → {len(movie_ids):,} movies with genome tag coverage")
    return movie_ids


def extract_top_k_genome_tags(
    movie_id: int,
    genome_scores: pd.DataFrame,
    genome_tags: pd.DataFrame,
    k: int = TOP_K_GENOME_TAGS,
) -> List[Dict[str, Any]]:
    """
    For a given movie, return top-K genome tags sorted by relevance score descending.
    Returns list of {"tag": str, "relevance": float}.
    """
    movie_scores = genome_scores[genome_scores["movieId"] == movie_id]
    top_k = movie_scores.nlargest(k, "relevance")
    merged = top_k.merge(genome_tags, on="tagId", how="left")
    return [
        {"tag": row["tag"], "relevance": round(float(row["relevance"]), 4)}
        for _, row in merged.iterrows()
    ]


def precompute_all_top_k_tags(
    genome_scores: pd.DataFrame,
    genome_tags: pd.DataFrame,
    movie_ids: List[int],
    k: int = TOP_K_GENOME_TAGS,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Efficiently compute top-K genome tags for all movies at once.
    Uses vectorized operations instead of per-movie filtering.
    """
    logger.info(f"Precomputing top-{k} genome tags for {len(movie_ids):,} movies...")

    # Merge tag names into scores
    scores_with_names = genome_scores.merge(genome_tags, on="tagId", how="left")

    # Filter to genome-covered movies only
    scores_with_names = scores_with_names[
        scores_with_names["movieId"].isin(set(movie_ids))
    ]

    # For each movie, get top-K by relevance
    result = {}
    grouped = scores_with_names.groupby("movieId")
    for mid, group in grouped:
        top_k = group.nlargest(k, "relevance")
        result[int(mid)] = [
            {"tag": row["tag"], "relevance": round(float(row["relevance"]), 4)}
            for _, row in top_k.iterrows()
        ]

    logger.info(f"  → top-{k} tags computed for {len(result):,} movies")
    return result


def load_movies() -> pd.DataFrame:
    """Load movies.csv with title and genres."""
    logger.info("Loading movies.csv...")
    movies = pd.read_csv(MOVIES_CSV)
    # Extract year from title like "Toy Story (1995)"
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    logger.info(f"  → {len(movies):,} movies loaded")
    return movies


def load_links() -> pd.DataFrame:
    """Load links.csv with imdbId and tmdbId."""
    logger.info("Loading links.csv...")
    links = pd.read_csv(LINKS_CSV)
    # imdbId needs to be zero-padded to 7 digits for URL construction
    links["imdbId"] = links["imdbId"].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
    )
    links["tmdbId"] = links["tmdbId"].apply(
        lambda x: int(x) if pd.notna(x) else None
    )
    logger.info(f"  → {len(links):,} links loaded")
    return links


def compute_rating_stats(movie_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """Compute average rating and count per movie from ratings.csv."""
    logger.info("Loading ratings.csv (this may take ~60 seconds)...")
    # Read only needed columns to save memory
    ratings = pd.read_csv(
        RATINGS_CSV,
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"},
    )
    logger.info(f"  → {len(ratings):,} ratings loaded")

    # Filter to genome-covered movies
    ratings = ratings[ratings["movieId"].isin(set(movie_ids))]

    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    result = {}
    for mid, row in stats.iterrows():
        result[int(mid)] = {
            "avg_rating": round(float(row["mean"]), 2),
            "rating_count": int(row["count"]),
        }

    logger.info(f"  → rating stats computed for {len(result):,} movies")
    return result


def aggregate_user_tags(movie_ids: List[int], max_tags_per_movie: int = 15) -> Dict[int, str]:
    """
    Aggregate user-generated free-text tags per movie.
    Returns top tags by frequency as a comma-separated string.
    """
    logger.info("Loading tags.csv...")
    tags = pd.read_csv(TAGS_CSV, usecols=["movieId", "tag"])
    tags = tags[tags["movieId"].isin(set(movie_ids))]
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()

    result = {}
    grouped = tags.groupby("movieId")["tag"]
    for mid, tag_series in grouped:
        tag_counts = tag_series.value_counts().head(max_tags_per_movie)
        result[int(mid)] = ", ".join(tag_counts.index.tolist())

    logger.info(f"  → user tags aggregated for {len(result):,} movies")
    return result


def format_genome_tags_for_prompt(tags: List[Dict[str, Any]]) -> str:
    """Format top-K genome tags into a readable string for the LLM prompt."""
    lines = []
    for i, t in enumerate(tags, 1):
        lines.append(f"  {i:2d}. {t['tag']}: {t['relevance']:.4f}")
    return "\n".join(lines)


def load_all_data() -> Dict[str, Any]:
    """
    Master loader: loads all ML-20M data needed for profile generation.
    Returns a dict with all preprocessed data structures.
    """
    genome_scores, genome_tags = load_genome_data()
    movie_ids = get_genome_covered_movie_ids(genome_scores)
    movies = load_movies()
    links = load_links()
    rating_stats = compute_rating_stats(movie_ids)
    user_tags = aggregate_user_tags(movie_ids)
    top_k_tags = precompute_all_top_k_tags(genome_scores, genome_tags, movie_ids)

    # Free large DataFrames from memory
    del genome_scores

    return {
        "movie_ids": movie_ids,
        "movies": movies,
        "links": links,
        "rating_stats": rating_stats,
        "user_tags": user_tags,
        "top_k_tags": top_k_tags,
        "genome_tags": genome_tags,
    }
