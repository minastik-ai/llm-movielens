#!/usr/bin/env python3
"""
main.py — LLM Movie Profile Generator for MovieLens 20M

End-to-end pipeline:
  1. Load ML-20M data (genome tags, movies, links, ratings, user tags)
  2. Crawl TMDb metadata (overview, cast, crew, keywords)
  3. Generate 80-120 word movie profiles using Claude Haiku 4.5
  4. Output as JSON with full prompt logging

Usage:
  # Full run (all 10,381 genome-covered movies):
  python main.py

  # Dry run (first 5 movies, no API calls to Claude):
  python main.py --dry-run --limit 5

  # Resume from checkpoint:
  python main.py --resume

  # Skip TMDb crawl (use cache only):
  python main.py --skip-tmdb

  # Generate for specific movies:
  python main.py --movie-ids 1 2 3 50 100

Environment variables (loaded automatically from .env):
  ANTHROPIC_API_KEY  — Your Anthropic API key
  TMDB_API_KEY       — Your TMDb API key (free at https://www.themoviedb.org/settings/api)
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root or parent directories
load_dotenv()
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")  # repo-root .env

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    DATA_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR,
    PROFILES_OUTPUT, PROFILES_PARTIAL, PROMPT_LOG,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS,
    TOP_K_GENOME_TAGS, PROFILE_MIN_WORDS, PROFILE_MAX_WORDS,
    MOOD_AXES, USE_PROMPT_CACHING,
)
from data_loader import load_all_data, format_genome_tags_for_prompt
from tmdb_crawler import load_cache, crawl_tmdb_batch, get_metadata_for_movie
from profile_generator import (
    ProfileGenerator, build_user_prompt, load_existing_profiles,
)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path = LOG_DIR):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def prepare_movie_data(
    data: Dict[str, Any],
    tmdb_metadata: Dict[int, Dict[str, Any]],
    movie_ids: List[int],
) -> List[Dict[str, Any]]:
    """
    Prepare the list of (movie_id, user_prompt) dicts for the profile generator.
    """
    movies_df = data["movies"]
    rating_stats = data["rating_stats"]
    user_tags = data["user_tags"]
    top_k_tags = data["top_k_tags"]

    movie_data = []
    for mid in movie_ids:
        # Movie info
        movie_row = movies_df[movies_df["movieId"] == mid]
        if movie_row.empty:
            continue
        movie_row = movie_row.iloc[0]
        title = movie_row["clean_title"]
        year = movie_row.get("year", "Unknown")
        genres = movie_row["genres"]

        # Genome tags
        genome_tags = top_k_tags.get(mid, [])
        if not genome_tags:
            continue  # skip movies without genome coverage

        # TMDb metadata
        tmdb_meta = get_metadata_for_movie(mid, tmdb_metadata)

        # Rating stats
        stats = rating_stats.get(mid, {"avg_rating": "N/A", "rating_count": 0})

        # User tags
        user_tags_summary = user_tags.get(mid, "No user tags available")

        # Build prompt
        user_prompt = build_user_prompt(
            movie_id=mid,
            title=title,
            year=year,
            genres=genres,
            genome_tags=genome_tags,
            tmdb_meta=tmdb_meta,
            rating_stats=stats,
            user_tags_summary=user_tags_summary,
        )

        movie_data.append({
            "movie_id": mid,
            "title": f"{title} ({year})",
            "user_prompt": user_prompt,
        })

    return movie_data


def print_prompt_example(movie_data: List[Dict[str, Any]], n: int = 1):
    """Print example prompt(s) for verification."""
    print("\n" + "=" * 80)
    print("PROMPT EXAMPLE (for verification)")
    print("=" * 80)
    print(f"\n{'─' * 40} SYSTEM PROMPT {'─' * 40}\n")
    print(SYSTEM_PROMPT)
    for i, movie in enumerate(movie_data[:n]):
        print(f"\n{'─' * 40} USER PROMPT [{movie['title']}] {'─' * 40}\n")
        print(movie["user_prompt"])
    print("\n" + "=" * 80 + "\n")


def print_summary(movie_data: List[Dict[str, Any]]):
    """Print pipeline summary before execution."""
    avg_prompt_len = sum(len(m["user_prompt"]) for m in movie_data) / len(movie_data)
    est_user_tokens = avg_prompt_len / 4  # rough char-to-token
    est_system_tokens = len(SYSTEM_PROMPT) / 4  # charged in full on every call:
        # the system prompt is below the model's minimum cacheable prefix, so the
        # cache_control marker creates no entry (see README, Cost).
    est_total_output = 250 * len(movie_data)  # ~250 tokens output per 80-120 word profile + JSON

    # With prompt caching: system prompt charged at cache-read rate (0.1×) after first call
    est_cost_first_call = (est_system_tokens + est_user_tokens) / 1e6 * 1.0
    est_cost_cached_calls = (est_user_tokens / 1e6 * 1.0 + est_system_tokens / 1e6 * 0.1) * (len(movie_data) - 1)
    est_cost_cache_write = est_system_tokens / 1e6 * 1.25  # one-time cache write
    est_cost_output = est_total_output / 1e6 * 5.0
    est_cost = est_cost_first_call + est_cost_cached_calls + est_cost_cache_write + est_cost_output

    print(f"\n{'═' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Movies to process:     {len(movie_data):,}")
    print(f"  Model:                 {CLAUDE_MODEL}")
    print(f"  Temperature:           {CLAUDE_TEMPERATURE}")
    print(f"  Max tokens/response:   {CLAUDE_MAX_TOKENS}")
    print(f"  Top-K genome tags:     {TOP_K_GENOME_TAGS}")
    print(f"  Profile word range:    {PROFILE_MIN_WORDS}–{PROFILE_MAX_WORDS} words")
    print(f"  Mood vector axes:      {len(MOOD_AXES)}")
    print(f"  Prompt caching:        {'ON' if USE_PROMPT_CACHING else 'OFF'}")
    print(f"  System prompt:         ~{est_system_tokens:,.0f} tokens (cached after 1st call)")
    print(f"  Avg user prompt:       {avg_prompt_len:,.0f} chars (~{est_user_tokens:,.0f} tokens)")
    print(f"  Est. total output:     {est_total_output:,.0f} tokens")
    print(f"  Est. cost:             ${est_cost:.2f}")
    print(f"  Output file:           {PROFILES_OUTPUT}")
    print(f"  Prompt log:            {PROMPT_LOG}")
    print(f"{'═' * 60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate LLM movie profiles for ML-20M")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling LLM")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-tmdb", action="store_true", help="Skip TMDb crawl, use cache only")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N movies (0=all)")
    parser.add_argument("--movie-ids", nargs="+", type=int, help="Process specific movie IDs")
    parser.add_argument("--example-prompts", type=int, default=2, help="Number of example prompts to print")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("LLM Movie Profile Generator — MovieLens 20M")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # ── Step 0: Verify data exists ──
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        logger.error("Please download ML-20M from https://grouplens.org/datasets/movielens/20m/")
        logger.error(f"Extract to: {DATA_DIR}")
        sys.exit(1)

    # ── Step 1: Load ML-20M data ──
    logger.info("STEP 1: Loading MovieLens 20M data...")
    data = load_all_data()
    movie_ids = data["movie_ids"]
    logger.info(f"  → {len(movie_ids):,} genome-covered movies identified")

    # ── Step 2: Filter movie IDs if requested ──
    if args.movie_ids:
        movie_ids = [mid for mid in args.movie_ids if mid in data["top_k_tags"]]
        logger.info(f"  → Filtered to {len(movie_ids)} specific movies")
    elif args.limit > 0:
        movie_ids = movie_ids[:args.limit]
        logger.info(f"  → Limited to first {len(movie_ids)} movies")

    # ── Step 3: Crawl TMDb metadata ──
    logger.info("STEP 2: TMDb metadata crawl...")
    tmdb_api_key = os.environ.get("TMDB_API_KEY", "")

    if args.skip_tmdb or not tmdb_api_key:
        if not tmdb_api_key:
            logger.warning("TMDB_API_KEY not set — using cache only (profiles will have limited metadata)")
        tmdb_cache = load_cache()
        tmdb_metadata = {int(k): v for k, v in tmdb_cache.items() if int(k) in set(movie_ids)}
    else:
        # Build movieId → tmdbId mapping
        links_df = data["links"]
        tmdb_id_map = {}
        for mid in movie_ids:
            link_row = links_df[links_df["movieId"] == mid]
            if not link_row.empty and pd.notna(link_row.iloc[0]["tmdbId"]):
                tmdb_id_map[mid] = int(link_row.iloc[0]["tmdbId"])

        logger.info(f"  → {len(tmdb_id_map):,} movies with TMDb IDs")
        tmdb_cache = load_cache()
        tmdb_metadata = asyncio.run(
            crawl_tmdb_batch(tmdb_id_map, tmdb_api_key, tmdb_cache)
        )

    logger.info(f"  → TMDb metadata available for {len(tmdb_metadata):,} movies")

    # ── Step 4: Prepare prompts ──
    logger.info("STEP 3: Preparing prompts...")
    movie_data = prepare_movie_data(data, tmdb_metadata, movie_ids)
    logger.info(f"  → {len(movie_data):,} prompts prepared")

    # Print example prompts
    print_prompt_example(movie_data, n=args.example_prompts)
    print_summary(movie_data)

    if args.dry_run:
        logger.info("DRY RUN — exiting without calling Claude API")
        # Save example prompts to file for inspection
        example_output = OUTPUT_DIR / "example_prompts.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(example_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt_template": USER_PROMPT_TEMPLATE,
                    "examples": [
                        {"movie_id": m["movie_id"], "title": m["title"], "user_prompt": m["user_prompt"]}
                        for m in movie_data[:5]
                    ],
                },
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"Example prompts saved to: {example_output}")
        return

    # ── Step 5: Generate profiles ──
    logger.info("STEP 4: Generating profiles with Claude Haiku 4.5...")
    existing = load_existing_profiles() if args.resume else {}

    generator = ProfileGenerator()
    profiles = generator.generate_all_profiles(movie_data, existing_profiles=existing)

    logger.info(f"DONE. {len(profiles):,} profiles generated.")
    logger.info(f"Output: {PROFILES_OUTPUT}")


if __name__ == "__main__":
    main()
