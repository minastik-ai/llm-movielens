#!/usr/bin/env python3
"""
Prepare human evaluation materials.

Generates:
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
from collections import defaultdict
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release


GENRE_GROUPS = {
    "comedy": ["Comedy"],
    "drama": ["Drama"],
    "action": ["Action", "Adventure"],
    "horror": ["Horror", "Thriller"],
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

# ORDER IS LOAD-BEARING: this list is zipped POSITIONALLY against
# mood_vector, so a wrong name at position k mislabels dimension k in the
# answer key the mood task is scored against. It previously read
# bleak_hopeful / grounded_fantastical / tense_relaxed at positions 5-7,
# which shifted six of the ten axes. Taken verbatim from the released
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
    """Load profiles from JSON file."""
    with open(profiles_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        profiles = list(data.values())
    else:
        profiles = data

    print(f"Loaded {len(profiles)} profiles from {profiles_path}")
    return profiles


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

    random.shuffle(sampled)
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
        for k, axis in enumerate(MOOD_AXES):
            pair[f"a_{axis}"] = p_a["mood_vector"][k]
            pair[f"b_{axis}"] = p_b["mood_vector"][k]
        pairs.append(pair)

    return pairs


def write_profile_csv(profiles: list, output_path: Path):
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
                "profile_text": p["profile_text"],
                "tmdb_overview": p.get("tmdb_overview", p.get("overview", "")),
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
    write_profile_csv(pilot_sample, output_dir / "pilot_30.csv")
    write_profile_csv(main_sample, output_dir / "main_500.csv")
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
    print(f"  4. After main: run scripts/analyze_human_eval.py")


if __name__ == "__main__":
    main()
