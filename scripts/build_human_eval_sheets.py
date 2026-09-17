#!/usr/bin/env python3
"""Build two annotator-facing sheets for the human evaluation study described
in Appendix D of the paper:

  1. annotator_sheet.csv
     500 rows, one per sample movie. Each row shows both the Claude Haiku 4.5
     profile and the GPT-4o-mini profile side-by-side, with blank cells for the
     annotator's 5-point Likert scores on each of the 5 quality axes
     (thematic accuracy, discriminativeness, rule compliance, factual
     consistency, coherence & fluency). Two profile blocks per row allows a
     within-subjects design where one annotator rates both LLMs on the same
     movie in a single sitting.

  2. mood_pairwise_sheet.csv
     100 rows, one per randomly selected movie pair. Annotators rate which of
     the two movies should score higher on each of the 10 mood axes
     (A / B / T for "too close to call"). The pair sample is drawn with
     seed=42 from the same 500-movie pool so it's reproducible and shares the
     genre stratification of the profile-quality sheet.

Usage:
    python scripts/build_human_eval_sheets.py
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
HUMAN_EVAL = REPO / "human_eval"

SAMPLE_IDS = HUMAN_EVAL / "sample_500_movie_ids.json"
SAMPLE_CSV = HUMAN_EVAL / "sample_500_movies.csv"
CLAUDE_PROFILES = code("profile_generator") \
    / "llm-movie-profiler-v1-20260402" / "output" / "movie_profiles.json"
GPT_PROFILES = HUMAN_EVAL / "gpt4o_mini_profiles.json"

OUT_PROFILES = HUMAN_EVAL / "annotator_sheet.csv"
OUT_MOOD = HUMAN_EVAL / "mood_pairwise_sheet.csv"
OUT_MOOD_KEY = HUMAN_EVAL / "mood_pairwise_key.json"

# Five quality axes from paper Appendix D.2
QUALITY_AXES = [
    "thematic_accuracy",
    "discriminativeness",
    "rule_compliance",
    "factual_consistency",
    "coherence_fluency",
]

# Ten mood axes from paper Appendix C (Table 1)
MOOD_AXES = [
    ("dark_light",                "bleak/grim (-1)", "bright/uplifting (+1)"),
    ("serious_playful",           "grave/solemn (-1)", "comedic/absurd (+1)"),
    ("slow_fast",                 "contemplative (-1)", "frenetic (+1)"),
    ("cerebral_visceral",         "intellectual (-1)", "sensory/action (+1)"),
    ("realistic_fantastical",     "gritty naturalism (-1)", "fantasy/surreal (+1)"),
    ("intimate_epic",             "small-scale (-1)", "grand/sweeping (+1)"),
    ("conventional_experimental", "mainstream (-1)", "avant-garde (+1)"),
    ("emotional_detached",        "cold/analytical (-1)", "intensely emotional (+1)"),
    ("nostalgic_contemporary",    "retro (-1)", "modern zeitgeist (+1)"),
    ("predictable_subversive",    "formulaic (-1)", "expectation-defying (+1)"),
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load() -> tuple[list[int], dict, dict, dict]:
    ids = json.loads(SAMPLE_IDS.read_text())
    print(f"Sample size: {len(ids)} movie IDs")

    with open(CLAUDE_PROFILES) as f:
        claude = json.load(f)
    with open(GPT_PROFILES) as f:
        gpt = json.load(f)

    # Build movieId -> metadata row map from sample_500_movies.csv
    meta = {}
    with open(SAMPLE_CSV) as f:
        for row in csv.DictReader(f):
            meta[int(row["movieId"])] = row
    missing_claude = [mid for mid in ids if str(mid) not in claude]
    missing_gpt = [mid for mid in ids if str(mid) not in gpt]
    missing_meta = [mid for mid in ids if mid not in meta]
    if missing_claude or missing_gpt or missing_meta:
        raise RuntimeError(
            f"Missing coverage: {len(missing_claude)} Claude, "
            f"{len(missing_gpt)} GPT, {len(missing_meta)} metadata."
        )
    print("All 500 sample movies have Claude + GPT-4o-mini profiles and metadata.")
    return ids, claude, gpt, meta


# ---------------------------------------------------------------------------
# Sheet 1: profile-quality annotator sheet
# ---------------------------------------------------------------------------
def build_annotator_sheet(ids, claude, gpt, meta) -> None:
    # Columns: context + two profile blocks + per-profile per-axis rating cells
    # + notes.
    # BLINDED COLUMN NAMES. Annotators must not be able to tell which model wrote
    # which profile, so the sheet names the two columns by letter and never by
    # model. The mapping is written to a separate key file that annotators do not
    # receive. It was previously applied by hand between generating this sheet and
    # issuing it, which left no trace in the artifact and made the shipped sheet
    # look unblinded to anyone auditing it.
    #
    # The mapping is FIXED across rows: profile_a is the same model on every row.
    # That blinds model identity, not column position -- see ORDER below.
    header = [
        "eval_id", "movieId", "title", "genres", "primary_genre",
        "profile_a_text",
        "profile_b_text",
    ]
    for prefix in ("profile_a", "profile_b"):
        for ax in QUALITY_AXES:
            header.append(f"{prefix}_{ax}")  # 1-5 Likert, blank for annotator
    header.append("notes")

    with open(OUT_PROFILES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        for i, mid in enumerate(ids, start=1):
            m = meta[mid]
            c = claude[str(mid)]
            g = gpt[str(mid)]
            row = [
                i, mid, m["title"], m["genres"], m["primary_genre"],
                c["profile"],
                g["profile"],
            ]
            row.extend([""] * (len(QUALITY_AXES) * 2))  # Likert cells blank
            row.append("")  # notes
            w.writerow(row)
    # The decode, kept beside the sheet and withheld from annotators.
    key_path = OUT_PROFILES.with_name("annotator_sheet_key.json")
    key_path.write_text(json.dumps({
        "_what": "Which generating model each blinded column holds.",
        "_why": ("Annotators receive the sheet with columns named profile_a / "
                 "profile_b and never see a model name. This key is the decode, "
                 "published separately so the sheet as issued and the mapping "
                 "needed to analyse it are both available."),
        "_order": ("The mapping is FIXED across all rows: profile_a is the same "
                   "model on every row. Model identity is therefore blinded; "
                   "column position is not randomised."),
        "mapping": {"profile_a": "claude-haiku-4-5", "profile_b": "gpt-4o-mini"},
        "rows": len(ids),
        "axes": list(QUALITY_AXES),
    }, indent=2) + "\n")
    print(f"Wrote {OUT_PROFILES.relative_to(REPO)}  ({len(ids)} rows, "
          f"{len(header)} columns, blinded) and {key_path.name}")


# ---------------------------------------------------------------------------
# Sheet 2: mood pairwise sheet + private key
# ---------------------------------------------------------------------------
def build_mood_pairwise_sheet(ids, claude, meta) -> None:
    rng = random.Random(42)
    # Draw 100 unordered, unique pairs (i < j indices into the 500-ID list)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_tries = 100_000
    while len(pairs) < 100 and max_tries > 0:
        a, b = rng.sample(range(len(ids)), 2)
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
        max_tries -= 1
    if len(pairs) < 100:
        raise RuntimeError("Could not draw 100 unique pairs.")

    # Annotator sheet (no LLM mood values shown → unbiased ranking)
    ann_header = [
        "pair_id",
        "movie_a_id", "movie_a_title", "movie_a_genres",
        "movie_b_id", "movie_b_title", "movie_b_genres",
    ]
    ann_header += [ax for ax, _, _ in MOOD_AXES]  # 10 blank cells per row
    ann_header.append("notes")

    legend_row = [
        "",  # pair_id
        "", "", "",
        "", "", "",
    ]
    for _, neg, pos in MOOD_AXES:
        legend_row.append(f"A higher / B higher / T (-1={neg}, +1={pos})")
    legend_row.append("Write 'A', 'B', or 'T' (too close) in each axis cell")

    with open(OUT_MOOD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(ann_header)
        w.writerow(legend_row)
        for pid, (i, j) in enumerate(pairs, start=1):
            mid_a, mid_b = ids[i], ids[j]
            ma, mb = meta[mid_a], meta[mid_b]
            row = [
                pid,
                mid_a, ma["title"], ma["genres"],
                mid_b, mb["title"], mb["genres"],
            ]
            row.extend([""] * len(MOOD_AXES))  # blank axis cells
            row.append("")  # notes
            w.writerow(row)
    print(f"Wrote {OUT_MOOD.relative_to(REPO)}  ({len(pairs)} pairs, "
          f"{len(MOOD_AXES)} axes each; row 2 = legend)")

    # Private key — the LLM's mood values for each pair, used after annotation
    # to compute agreement (majority vote vs LLM ordering, per paper §D.3).
    key = {
        "seed": 42,
        "n_pairs": len(pairs),
        "axes": [ax for ax, _, _ in MOOD_AXES],
        "pairs": [],
    }
    for pid, (i, j) in enumerate(pairs, start=1):
        mid_a, mid_b = ids[i], ids[j]
        ca = claude[str(mid_a)]["mood_vector"]
        cb = claude[str(mid_b)]["mood_vector"]
        key["pairs"].append({
            "pair_id": pid,
            "movie_a_id": mid_a,
            "movie_b_id": mid_b,
            "mood_a": ca,
            "mood_b": cb,
            # Precompute LLM's ordering per axis: "A" if a > b, "B" if b > a,
            # "T" if |diff| < 0.1 (too close to call).
            "llm_ordering": {
                ax: ("T" if abs(ca[ax] - cb[ax]) < 0.1
                     else ("A" if ca[ax] > cb[ax] else "B"))
                for ax, _, _ in MOOD_AXES
            },
        })
    OUT_MOOD_KEY.write_text(json.dumps(key, indent=2))
    print(f"Wrote {OUT_MOOD_KEY.relative_to(REPO)}  (private key; keep "
          f"separate from annotator sheet)")


if __name__ == "__main__":
    ids, claude, gpt, meta = load()
    build_annotator_sheet(ids, claude, gpt, meta)
    build_mood_pairwise_sheet(ids, claude, meta)
