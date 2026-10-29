#!/usr/bin/env python3
"""
Prepare human evaluation materials.

Generates:
  - human_eval/sample_500_movie_ids.json       — the 500 films of the main round, in order;
                                                 byte-identical to the released file of the
                                                 same name, which is what the annotators rated
  - human_eval/pilot_30.csv                    — 30 profiles for calibration
  - human_eval/main_500.csv                    — 500 stratified profiles for main round
  - human_eval/mood_pairs_100.csv              — 100 movie pairs for mood validation
  - human_eval/mood_pairs_answer_key.csv       — signed key for those pairs
  - human_eval/pilot_annotation_template.csv   — blank annotation sheet, pilot round
  - human_eval/main_annotation_template.csv    — blank annotation sheet, main round

Usage:
    python scripts/prepare_human_eval.py
    python scripts/prepare_human_eval.py --profiles-path code/profile_generator/.../output/movie_profiles.json
"""

import argparse
import json
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

sys.path.insert(0, str(code("benchmark")))
from config import MOVIES_CSV  # noqa: E402  -- the reader's own MovieLens 20M download

# One genre per group, checked in this order; a film matching none is "other". This
# is the rule the released sample was drawn under: it reproduces all 500 released
# `primary_genre` labels, and the draw below reproduces the 500 films in order. A
# later edit had widened "action" to Adventure and "horror" to Thriller, which
# relabels 82 of the 500 and draws a different sample.
GENRE_GROUPS = {
    "comedy": ["Comedy"],
    "drama": ["Drama"],
    "action": ["Action"],
    "horror": ["Horror"],
    "documentary": ["Documentary"],
}

# Target counts per genre group (proportional to dataset)
TARGET_COUNTS = {
    "comedy": 83,
    "drama": 117,
    "action": 72,
    "horror": 48,
    "documentary": 35,
    "other": 145,
}

# The released records store mood_vector as a dict keyed by these names, and the
# answer key is read BY NAME.
# The order sets the column order of the sheets. Taken verbatim from the released
# human_eval/mood_pairwise_key.json "axes" field.
MOOD_AXES = [
    "dark_light",
    "serious_playful",
    "slow_fast",
    "cerebral_visceral",
    "realistic_fantastical",
    "intimate_epic",
    "conventional_experimental",
    "emotional_detached",
    "nostalgic_contemporary",
    "predictable_subversive",
]

PILOT_SIZE = 30
MAIN_SIZE = 500
MOOD_PAIRS = 100
SEED = 42


def classify_genre(genres_str: str) -> str:
    """Classify a movie into one of 6 genre groups."""
    genres = [g.strip() for g in genres_str.split("|")]
    for group_name, group_genres in GENRE_GROUPS.items():
        if any(g in genres for g in group_genres):
            return group_name
    return "other"


def load_profiles(profiles_path: Path) -> list:
    """Load profiles, with MovieLens genres joined on movieId.

    The released records carry no `genres` field -- genres are MovieLens's, and the
    release does not redistribute MovieLens -- so a stratification reading the record
    alone put all 10,381 films in "other", drew 145, and failed its size assertion on
    every run.
    """
    if not profiles_path.exists():
        sys.exit(f"The released profiles are not here:\n    {profiles_path}\n"
                 "Fetch them first:  python3 scripts/download_artifacts.py")
    with open(profiles_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        profiles = list(data.values())
    else:
        profiles = data

    if not MOVIES_CSV.exists():
        sys.exit(f"Genres come from MovieLens, which we do not redistribute:\n"
                 f"    {MOVIES_CSV}\nFetch it first:  bash scripts/download_ml20m.sh")
    with open(MOVIES_CSV, encoding="utf-8") as f:
        genres = {int(r["movieId"]): r["genres"] for r in csv.DictReader(f)}
    for p in profiles:
        p.setdefault("genres", genres.get(int(p["movieId"]), ""))

    print(f"Loaded {len(profiles)} profiles from {profiles_path}")
    return profiles


def load_tmdb_overviews() -> dict:
    """The TMDb reference summaries, from the profile generator's own TMDb cache.

    TMDb text is not redistributed, so the cache exists only once a reader has run
    src/profile_generator/tmdb_crawler.py with their own API key. Without it the
    reference column is left empty and annotators look the film up themselves.
    """
    cache = code("profile_generator", "llm-movie-profiler-v1-20260402", "cache",
                 "tmdb_metadata.json")
    if not cache.exists():
        print(f"  note: no TMDb cache at {cache}; the tmdb_overview column will be empty")
        return {}
    with open(cache, encoding="utf-8") as f:
        return {str(k): v.get("overview", "") for k, v in json.load(f).items()}


def stratified_sample(profiles: list, target_counts: dict, seed: int) -> list:
    """Stratified random sample proportional to genre distribution."""
    random.seed(seed)

    # Group by genre
    by_genre = defaultdict(list)
    for p in profiles:
        genre_group = classify_genre(p.get("genres", ""))
        by_genre[genre_group].append(p)

    print("\nGenre distribution in full dataset:")
    for g, ps in sorted(by_genre.items()):
        print(f"  {g}: {len(ps)}")

    # Sample
    sampled = []
    for genre_group, count in target_counts.items():
        pool = by_genre.get(genre_group, [])
        if len(pool) < count:
            print(f"  Warning: {genre_group} has only {len(pool)} profiles, sampling all")
            sampled.extend(pool)
        else:
            sampled.extend(random.sample(pool, count))

    # A FRESH generator for the presentation order, not the global one: that is how
    # the released order was produced, and it reproduces it exactly.
    random.Random(seed).shuffle(sampled)
    return sampled


def generate_mood_pairs(profiles: list, n_pairs: int, seed: int) -> list:
    """Generate random movie pairs for mood pairwise comparison."""
    random.seed(seed + 1)  # Different seed from main sampling

    pairs = []
    indices = list(range(len(profiles)))

    for _ in range(n_pairs):
        i, j = random.sample(indices, 2)
        p_a = profiles[i]
        p_b = profiles[j]
        pair = {
            "pair_id": len(pairs) + 1,
            "movie_a_id": p_a["movieId"],
            "movie_a_title": p_a["title"],
            "movie_b_id": p_b["movieId"],
            "movie_b_title": p_b["title"],
        }
        # Include LLM-assigned mood values (for later comparison, NOT shown to annotators)
        for axis in MOOD_AXES:
            pair[f"a_{axis}"] = p_a["mood_vector"][axis]
            pair[f"b_{axis}"] = p_b["mood_vector"][axis]
        pairs.append(pair)

    return pairs


def write_profile_csv(profiles: list, output_path: Path, overviews: dict):
    """Write profiles to CSV for annotation."""
    fieldnames = [
        "eval_id", "movieId", "title", "genres", "profile_text",
        "tmdb_overview", "mood_vector",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(profiles):
            writer.writerow({
                "eval_id": i + 1,
                "movieId": p["movieId"],
                "title": p["title"],
                "genres": p.get("genres", ""),
                "profile_text": p["profile"],
                "tmdb_overview": overviews.get(str(p["movieId"]), ""),
                "mood_vector": json.dumps(p["mood_vector"]),
            })

    print(f"Wrote {len(profiles)} profiles to {output_path}")


def write_annotation_template(profiles: list, output_path: Path):
    """Write blank annotation spreadsheet."""
    fieldnames = [
        "eval_id", "movieId", "title",
        "thematic_accuracy", "discriminativeness", "rule_compliance",
        "factual_consistency", "coherence_fluency", "notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, p in enumerate(profiles):
            writer.writerow({
                "eval_id": i + 1,
                "movieId": p["movieId"],
                "title": p["title"],
                "thematic_accuracy": "",
                "discriminativeness": "",
                "rule_compliance": "",
                "factual_consistency": "",
                "coherence_fluency": "",
                "notes": "",
            })

    print(f"Wrote annotation template to {output_path}")


def write_mood_pairs_csv(pairs: list, output_path: Path):
    """Write mood pairs for pairwise comparison task."""
    # Annotator version: no LLM values shown
    fieldnames = ["pair_id", "movie_a_id", "movie_a_title", "movie_b_id", "movie_b_title"]
    for axis in MOOD_AXES:
        fieldnames.append(f"higher_{axis}")  # Annotator fills: A, B, or tie

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            row = {k: pair[k] for k in ["pair_id", "movie_a_id", "movie_a_title", "movie_b_id", "movie_b_title"]}
            for axis in MOOD_AXES:
                row[f"higher_{axis}"] = ""  # Annotator fills this
            writer.writerow(row)

    # Also write answer key (with LLM values, for analysis)
    answer_key_path = output_path.parent / "mood_pairs_answer_key.csv"
    all_fields = list(pairs[0].keys())
    with open(answer_key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Wrote {len(pairs)} mood pairs to {output_path}")
    print(f"Wrote answer key to {answer_key_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare human evaluation materials")
    parser.add_argument(
        "--profiles-path",
        type=str,
        default=str(code("profile_generator", "llm-movie-profiler-v1-20260402", "output", "movie_profiles.json")),
        help="Path to movie_profiles.json",
    )
    parser.add_argument("--output-dir", type=str, default="human_eval")
    args = parser.parse_args()

    profiles_path = Path(args.profiles_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    profiles = load_profiles(profiles_path)

    # Stratified sample for main round
    main_sample = stratified_sample(profiles, TARGET_COUNTS, SEED)
    assert len(main_sample) == MAIN_SIZE, f"Expected {MAIN_SIZE}, got {len(main_sample)}"

    # Pilot: first 30 from a separate random draw
    random.seed(SEED + 100)
    pilot_pool = [p for p in profiles if p not in main_sample]
    pilot_sample = random.sample(pilot_pool, PILOT_SIZE)

    # Write outputs
    (output_dir / "sample_500_movie_ids.json").write_text(
        json.dumps([p["movieId"] for p in main_sample], indent=2))
    overviews = load_tmdb_overviews()
    write_profile_csv(pilot_sample, output_dir / "pilot_30.csv", overviews)
    write_profile_csv(main_sample, output_dir / "main_500.csv", overviews)
    write_annotation_template(pilot_sample, output_dir / "pilot_annotation_template.csv")
    write_annotation_template(main_sample, output_dir / "main_annotation_template.csv")

    # Mood pairs from full dataset
    mood_pairs = generate_mood_pairs(profiles, MOOD_PAIRS, SEED)
    write_mood_pairs_csv(mood_pairs, output_dir / "mood_pairs_100.csv")

    print(f"\n=== All materials ready in {output_dir}/ ===")
    print(f"  Pilot: {PILOT_SIZE} profiles")
    print(f"  Main:  {MAIN_SIZE} profiles (stratified by genre)")
    print(f"  Mood:  {MOOD_PAIRS} movie pairs × {len(MOOD_AXES)} axes")
    print(f"\nNext steps:")
    print(f"  1. Give each annotator: pilot_30.csv + pilot_annotation_template.csv")
    print(f"  2. After pilot: run calibration meeting")
    print(f"  3. Give each annotator: main_500.csv + main_annotation_template.csv")
    print(f"  4. After main: Fleiss' kappa per axis over the three filled sheets, and\n"
          f"     tools/verify_mood_agreement.py for the pairwise task "
          f"(docs/HUMAN_EVAL_PROTOCOL.md, Phase 3)")


if __name__ == "__main__":
    main()
