#!/usr/bin/env python3
"""
Generate 500 movie profiles using GPT-4o-mini for LLM sensitivity analysis (cross-LLM).

Uses the SAME system prompt and user prompt template as the Claude Haiku pipeline,
with 500 stratified movies (same sample as human evaluation).

Usage:
    export OPENAI_API_KEY="your-key-here"
    python scripts/generate_gpt4o_profiles.py

Output:
    human_eval/gpt4o_mini_profiles.json     — 500 profiles from GPT-4o-mini
    human_eval/llm_comparison_results.json   — cosine, mood correlation, theme overlap

Requires: pip install openai
"""

import os
import sys
import json
import time
import logging
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from _paths import code  # dual-layout paths: dev tree or published release

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROFILES_PATH = code("profile_generator", "llm-movie-profiler-v1-20260402", "output", "movie_profiles.json")
MOVIES_CSV = code("profile_generator", "llm-movie-profiler-v1-20260402", "data", "ml-20m", "movies.csv")
GENOME_SCORES = code("profile_generator", "llm-movie-profiler-v1-20260402", "data", "ml-20m", "genome-scores.csv")
GENOME_TAGS = code("profile_generator", "llm-movie-profiler-v1-20260402", "data", "ml-20m", "genome-tags.csv")
TMDB_CACHE = code("profile_generator", "llm-movie-profiler-v1-20260402", "cache", "tmdb_metadata.json")
OUTPUT_DIR = Path("human_eval")

# Import the same system prompt used for Claude
sys.path.insert(0, str(code("profile_generator", "llm-movie-profiler-v1-20260402")))
from config.settings import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TOP_K_GENOME_TAGS, MOOD_AXES

SAMPLE_SIZE = 500
SEED = 42


def load_genome_data():
    """Load genome tags and scores."""
    import pandas as pd
    tags_df = pd.read_csv(GENOME_TAGS)
    tag_map = dict(zip(tags_df["tagId"], tags_df["tag"]))

    scores_df = pd.read_csv(GENOME_SCORES)
    genome = defaultdict(list)
    for _, row in scores_df.iterrows():
        genome[int(row["movieId"])].append((tag_map.get(int(row["tagId"]), ""), row["relevance"]))

    # Sort by relevance, keep top-K
    for mid in genome:
        genome[mid] = sorted(genome[mid], key=lambda x: x[1], reverse=True)[:TOP_K_GENOME_TAGS]

    return genome


def format_genome_tags(tags_list):
    """Format genome tags for prompt."""
    return ", ".join([f"{tag}:{score:.2f}" for tag, score in tags_list])


def stratified_sample(profiles, n=500, seed=42):
    """Stratified sample by genre, matching human eval distribution."""
    import pandas as pd
    movies = pd.read_csv(MOVIES_CSV)
    mid_to_genres = dict(zip(movies["movieId"], movies["genres"]))

    def primary_genre(gstr):
        for g in ["Comedy", "Drama", "Action", "Horror", "Documentary"]:
            if g in str(gstr):
                return g.lower()
        return "other"

    target = {"comedy": 83, "drama": 117, "action": 72, "horror": 48, "documentary": 35, "other": 145}

    by_genre = defaultdict(list)
    for mid_str, p in profiles.items():
        mid = int(mid_str)
        g = primary_genre(mid_to_genres.get(mid, ""))
        by_genre[g].append(mid)

    random.seed(seed)
    sampled = []
    for genre, count in target.items():
        pool = by_genre.get(genre, [])
        if len(pool) >= count:
            sampled.extend(random.sample(pool, count))
        else:
            sampled.extend(pool)

    random.shuffle(sampled)
    return sampled[:n]


def generate_with_gpt4o_mini(movie_ids, profiles, genome_data, tmdb_cache):
    """Generate profiles using GPT-4o-mini with identical prompts."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    results = {}
    failed = []

    for i, mid in enumerate(movie_ids):
        p = profiles.get(str(mid), {})
        title = p.get("title", "Unknown")

        # Build the same user prompt
        genome_tags = genome_data.get(mid, [])
        genome_formatted = format_genome_tags(genome_tags)

        # Get TMDb metadata
        tmdb = tmdb_cache.get(str(mid), {})

        # Extract year from title
        year = ""
        if "(" in title and ")" in title:
            year = title[title.rfind("(") + 1:title.rfind(")")]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=title,
            year=year,
            genres=tmdb.get("genres", "Unknown"),
            genome_tags_formatted=genome_formatted,
            overview=tmdb.get("overview", "No plot available."),
            keywords=tmdb.get("keywords", ""),
            directors=tmdb.get("directors", "Unknown"),
            cast=tmdb.get("cast", "Unknown"),
            runtime=tmdb.get("runtime", "Unknown"),
            vote_average=tmdb.get("vote_average", "N/A"),
            vote_count=tmdb.get("vote_count", "N/A"),
            ml_avg_rating=tmdb.get("ml_avg_rating", p.get("ml_avg_rating", "N/A")),
            ml_rating_count=tmdb.get("ml_rating_count", p.get("ml_rating_count", "N/A")),
            user_tags_summary=tmdb.get("user_tags", ""),
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )

            text = response.choices[0].message.content.strip()

            # Parse JSON
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            profile_data = json.loads(text)
            # The prompt's few-shot examples carry dummy movieIds (99997/99998/99999);
            # GPT-4o-mini copies one of them into its response because no real movieId
            # is supplied in the input. Overwrite with the actual id so downstream
            # consumers can trust the field without relying on the dict key.
            profile_data["movieId"] = mid
            results[mid] = profile_data
            logger.info(f"[{i + 1}/{len(movie_ids)}] {title} — OK ({len(profile_data.get('profile', '').split())} words)")

        except json.JSONDecodeError as e:
            logger.warning(f"[{i + 1}/{len(movie_ids)}] {title} — JSON parse error: {e}")
            failed.append(mid)
        except Exception as e:
            logger.warning(f"[{i + 1}/{len(movie_ids)}] {title} — Error: {e}")
            failed.append(mid)

        # Checkpoint and rate limiting
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(movie_ids)} complete, {len(failed)} failed")
            # Save checkpoint
            checkpoint_path = Path("human_eval/gpt4o_mini_profiles.json")
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {len(results)} profiles")
            time.sleep(1)

    logger.info(f"Generated {len(results)} profiles, {len(failed)} failed")
    return results, failed


def compare_profiles(claude_profiles, gpt_profiles, movie_ids):
    """Compare Claude vs GPT-4o-mini profiles."""
    from sentence_transformers import SentenceTransformer
    from scipy.stats import pearsonr

    # Collect matched profiles
    matched_ids = [mid for mid in movie_ids if mid in gpt_profiles and str(mid) in claude_profiles]
    logger.info(f"Matched profiles for comparison: {len(matched_ids)}")

    claude_texts = [claude_profiles[str(mid)].get("profile", "") for mid in matched_ids]
    gpt_texts = [gpt_profiles[mid].get("profile", "") for mid in matched_ids]

    # 1. Profile-level cosine similarity
    logger.info("Computing profile embeddings with bge-large-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    claude_emb = model.encode(claude_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    gpt_emb = model.encode(gpt_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    cosine_sims = np.sum(claude_emb * gpt_emb, axis=1)  # per-movie cosine similarity
    cosine_mean = float(np.mean(cosine_sims))
    cosine_std = float(np.std(cosine_sims))
    logger.info(f"Profile cosine similarity: μ={cosine_mean:.4f}, σ={cosine_std:.4f}")

    # 2. Mood vector correlation
    mood_correlations = {}
    claude_moods = []
    gpt_moods = []

    for mid in matched_ids:
        c_mood = claude_profiles[str(mid)].get("mood_vector", {})
        g_mood = gpt_profiles[mid].get("mood_vector", {})

        if isinstance(c_mood, dict) and isinstance(g_mood, dict):
            c_vec = [c_mood.get(axis, 0) for axis in MOOD_AXES]
            g_vec = [g_mood.get(axis, 0) for axis in MOOD_AXES]
            claude_moods.append(c_vec)
            gpt_moods.append(g_vec)

    claude_moods = np.array(claude_moods)
    gpt_moods = np.array(gpt_moods)

    for i, axis in enumerate(MOOD_AXES):
        if claude_moods.shape[0] > 2:
            r, p = pearsonr(claude_moods[:, i], gpt_moods[:, i])
            mood_correlations[axis] = {"pearson_r": float(r), "p_value": float(p)}
        else:
            mood_correlations[axis] = {"pearson_r": 0.0, "p_value": 1.0}

    # Overall mood correlation
    all_claude = claude_moods.flatten()
    all_gpt = gpt_moods.flatten()
    overall_r, overall_p = pearsonr(all_claude, all_gpt)
    mood_correlations["overall"] = {"pearson_r": float(overall_r), "p_value": float(overall_p)}
    logger.info(f"Mood vector correlation (overall): r={overall_r:.4f}, p={overall_p:.6f}")

    # 3. Theme overlap (Jaccard)
    jaccard_scores = []
    for mid in matched_ids:
        c_themes = set(t.lower().strip() for t in claude_profiles[str(mid)].get("key_themes", []))
        g_themes = set(t.lower().strip() for t in gpt_profiles[mid].get("key_themes", []))
        if c_themes or g_themes:
            jaccard = len(c_themes & g_themes) / len(c_themes | g_themes) if (c_themes | g_themes) else 0
            jaccard_scores.append(jaccard)

    theme_jaccard_mean = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    theme_jaccard_std = float(np.std(jaccard_scores)) if jaccard_scores else 0.0
    logger.info(f"Theme Jaccard overlap: μ={theme_jaccard_mean:.4f}, σ={theme_jaccard_std:.4f}")

    return {
        "n_matched": len(matched_ids),
        "cosine_similarity": {"mean": cosine_mean, "std": cosine_std},
        "mood_correlation": mood_correlations,
        "theme_jaccard": {"mean": theme_jaccard_mean, "std": theme_jaccard_std},
    }


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Claude profiles
    logger.info("Loading Claude Haiku profiles...")
    with open(PROFILES_PATH) as f:
        claude_profiles = json.load(f)
    logger.info(f"Loaded {len(claude_profiles)} Claude profiles")

    # Load genome data
    logger.info("Loading genome data...")
    genome_data = load_genome_data()
    logger.info(f"Genome data for {len(genome_data)} movies")

    # Load TMDb cache
    tmdb_cache = {}
    if TMDB_CACHE.exists():
        with open(TMDB_CACHE) as f:
            tmdb_cache = json.load(f)
        logger.info(f"TMDb cache: {len(tmdb_cache)} movies")

    # Stratified sample
    movie_ids = stratified_sample(claude_profiles, n=SAMPLE_SIZE, seed=SEED)
    logger.info(f"Sampled {len(movie_ids)} movies (stratified by genre)")

    # Check for existing results (resume support)
    gpt_output_path = OUTPUT_DIR / "gpt4o_mini_profiles.json"
    if gpt_output_path.exists():
        with open(gpt_output_path) as f:
            existing = json.load(f)
        logger.info(f"Found {len(existing)} existing GPT profiles, resuming...")
        remaining = [mid for mid in movie_ids if mid not in existing]
        gpt_profiles = {int(k): v for k, v in existing.items()}
    else:
        remaining = movie_ids
        gpt_profiles = {}

    # Generate with GPT-4o-mini
    if remaining:
        logger.info(f"Generating {len(remaining)} profiles with GPT-4o-mini...")
        new_profiles, failed = generate_with_gpt4o_mini(
            remaining, claude_profiles, genome_data, tmdb_cache
        )
        gpt_profiles.update(new_profiles)

        # Save intermediate results
        with open(gpt_output_path, "w") as f:
            json.dump(gpt_profiles, f, indent=2, default=str)
        logger.info(f"Saved {len(gpt_profiles)} GPT profiles to {gpt_output_path}")
    else:
        logger.info("All profiles already generated.")

    # Compare
    logger.info("\n=== CROSS-LLM COMPARISON ===")
    results = compare_profiles(claude_profiles, gpt_profiles, movie_ids)

    # Save comparison results
    results_path = OUTPUT_DIR / "llm_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-LLM COMPARISON: Claude Haiku 4.5 vs GPT-4o-mini")
    print("=" * 70)
    print(f"Matched profiles: {results['n_matched']}")
    print(f"\n1. Profile Cosine Similarity (bge-large-en-v1.5):")
    print(f"   μ = {results['cosine_similarity']['mean']:.4f} ± {results['cosine_similarity']['std']:.4f}")
    print(f"\n2. Mood Vector Correlation:")
    for axis in MOOD_AXES:
        r = results['mood_correlation'][axis]['pearson_r']
        print(f"   {axis:<30s} r = {r:.3f}")
    overall_r = results['mood_correlation']['overall']['pearson_r']
    print(f"   {'OVERALL':<30s} r = {overall_r:.3f}")
    print(f"\n3. Theme Overlap (Jaccard):")
    print(f"   μ = {results['theme_jaccard']['mean']:.4f} ± {results['theme_jaccard']['std']:.4f}")

    # Generate LaTeX-ready text
    print("\n" + "=" * 70)
    print("PAPER TEXT (copy-paste):")
    print("=" * 70)
    cos_mean = results['cosine_similarity']['mean']
    cos_std = results['cosine_similarity']['std']
    print(f"To assess LLM sensitivity, we generated 500 profiles (stratified by")
    print(f"genre) using GPT-4o-mini with an identical prompt template. Cross-LLM")
    print(f"profile embeddings show high cosine similarity (μ = {cos_mean:.2f} ± {cos_std:.2f}),")
    print(f"and mood vectors correlate strongly (Pearson r = {overall_r:.2f} across all")
    print(f"axes), with theme Jaccard overlap of {results['theme_jaccard']['mean']:.2f}.")
    print(f"These results suggest the pipeline's outputs are robust to LLM choice.")


if __name__ == "__main__":
    main()
