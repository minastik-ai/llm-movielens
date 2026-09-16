#!/usr/bin/env python3
"""cross-LLM fix: generate the 82 missing GPT-4o-mini profiles using Claude's
TMDb-fallback convention so both providers end up with 10,381-item
embeddings (fair paired comparison).

When TMDb is missing, Claude's pipeline (tmdb_crawler.get_metadata_for_movie)
substitutes safe defaults: overview='No overview available.', cast='Unknown',
directors='Unknown', etc. We replicate that exactly.

Cost: ~82 standard-API calls × ~2200 tokens × $0.150/M input + $0.600/M output
       ≈ $0.05 total.
Wall time: ~3 min sequential, ~30s if we batch.

Usage:
    python3 scripts/gpt4omini_fill_82.py
    # then re-run the embedding step (main.py) over the now-complete profiles
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
PROFILER_DIR = code("profile_generator", "llm-movie-profiler-v1-20260402")
OUT_DIR = code("profile_generator", "output_ml20m_gpt4omini")
PROFILES_FILE = OUT_DIR / "movie_profiles.json"

sys.path.insert(0, str(PROFILER_DIR))
from config.settings import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE,
    PROFILE_MIN_WORDS, PROFILE_MAX_WORDS,
)
from data_loader import load_all_data, format_genome_tags_for_prompt
from tmdb_crawler import get_metadata_for_movie  # provides Claude's safe defaults

MODEL = "gpt-4o-mini-2024-07-18"


def build_user_prompt_with_fallback(mid: int, data, tmdb_cache_int_keys):
    """Same as before, but uses Claude's safe-default fallback when TMDb missing."""
    movies_df = data["movies"]
    row = movies_df[movies_df["movieId"] == mid]
    if row.empty:
        return None
    title_full = row.iloc[0]["title"]
    if title_full.endswith(")") and "(" in title_full:
        title = title_full[:title_full.rfind("(")].strip()
        year = title_full[title_full.rfind("(")+1:-1]
    else:
        title, year = title_full, ""
    genres = row.iloc[0]["genres"]

    top30 = data["top_k_tags"].get(mid)
    if top30 is None:
        return None  # without genome tags we genuinely can't proceed

    # *** FIX *** use Claude's safe-default fallback for missing TMDb
    tmdb = get_metadata_for_movie(mid, tmdb_cache_int_keys)
    rs = data["rating_stats"].get(mid, {"avg_rating": "N/A", "rating_count": 0})
    user_tags = data["user_tags"].get(mid, "No user tags available")

    return USER_PROMPT_TEMPLATE.format(
        title=title, year=year or "Unknown", genres=genres,
        genome_tags_formatted=format_genome_tags_for_prompt(top30),
        overview=tmdb["overview"], directors=tmdb["directors"],
        cast=tmdb["cast"], runtime=tmdb["runtime"],
        vote_average=tmdb["vote_average"], vote_count=tmdb["vote_count"],
        keywords=tmdb["keywords"],
        ml_avg_rating=rs.get("avg_rating", "N/A"),
        ml_rating_count=rs.get("rating_count", 0),
        user_tags_summary=user_tags or "No user tags available",
    )


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    client = OpenAI()

    # Load data + identify the 82 missing IDs
    print("[fill] Loading ML-20M data + TMDb cache...")
    data = load_all_data()
    with open(PROFILER_DIR / "cache" / "tmdb_metadata.json") as f:
        tmdb_str_keys = json.load(f)
    tmdb_int_keys = {int(k): v for k, v in tmdb_str_keys.items()}

    profiles = json.loads(PROFILES_FILE.read_text())
    have_ids = set(int(k) for k in profiles.keys())
    all_genome_ids = set(data["movie_ids"])
    missing_ids = sorted(all_genome_ids - have_ids)
    print(f"[fill] Have {len(have_ids)} profiles, missing {len(missing_ids)} (genome-covered total {len(all_genome_ids)})")

    if not missing_ids:
        print("[fill] Nothing to do."); return

    # Generate via standard API (small enough that batch isn't worth the overhead)
    n_done, n_failed = 0, []
    in_toks_total, out_toks_total = 0, 0
    t0 = time.time()
    for mid in missing_ids:
        user_prompt = build_user_prompt_with_fallback(mid, data, tmdb_int_keys)
        if user_prompt is None:
            n_failed.append((mid, "no genome tags or movie row"))
            continue
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}],
                temperature=CLAUDE_TEMPERATURE,
                max_tokens=CLAUDE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            n_failed.append((mid, f"API error: {type(e).__name__}: {str(e)[:120]}"))
            continue

        in_toks_total += resp.usage.prompt_tokens
        out_toks_total += resp.usage.completion_tokens
        text = resp.choices[0].message.content
        try:
            parsed = json.loads(text)
            assert all(k in parsed for k in ("profile", "mood_vector", "key_themes"))
            wc = parsed.get("word_count") or len(parsed.get("profile", "").split())
            assert PROFILE_MIN_WORDS <= wc <= PROFILE_MAX_WORDS
        except Exception as e:
            n_failed.append((mid, f"validation: {type(e).__name__}: {str(e)[:120]}"))
            continue

        profiles[str(mid)] = parsed
        n_done += 1
        if n_done % 20 == 0:
            print(f"[fill] {n_done}/{len(missing_ids)} done ({time.time()-t0:.0f}s)")

    PROFILES_FILE.write_text(json.dumps(profiles, indent=2))
    cost_in = in_toks_total * 0.150 / 1e6
    cost_out = out_toks_total * 0.600 / 1e6
    print(f"\n[fill] Wrote {len(profiles)} total profiles to {PROFILES_FILE}")
    print(f"[fill] Generated {n_done} new, {len(n_failed)} failed")
    print(f"[fill] Tokens: in={in_toks_total}, out={out_toks_total}")
    print(f"[fill] Cost: in=${cost_in:.4f} + out=${cost_out:.4f} = ${cost_in+cost_out:.4f}")
    print(f"[fill] Wall time: {time.time()-t0:.0f}s")
    if n_failed:
        print(f"\n[fill] Failed samples (first 5): {n_failed[:5]}")

    # Update usage.json
    usage_path = OUT_DIR / "usage.json"
    if usage_path.exists():
        u = json.loads(usage_path.read_text())
        u["fill_82_missing"] = {
            "n_done": n_done, "n_failed": len(n_failed),
            "input_tokens": in_toks_total, "output_tokens": out_toks_total,
            "cost_USD": round(cost_in + cost_out, 4),
        }
        u["cost_total_USD"] = round(u["cost_total_USD"] + cost_in + cost_out, 4)
        u["n_validated_profiles"] = len(profiles)
        usage_path.write_text(json.dumps(u, indent=2))


if __name__ == "__main__":
    main()
