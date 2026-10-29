#!/usr/bin/env python3
"""cross-LLM Option B — Step 1: GPT-4o-mini token calibration on N movies.

Builds the IDENTICAL user prompt the Claude pipeline used (same SYSTEM_PROMPT,
same USER_PROMPT_TEMPLATE, same data_loader.load_all_data, same temperature)
and calls the OpenAI standard API for N randomly-sampled movies. Measures
actual token usage to project the full-10K batch cost before committing.

Reads OPENAI_API_KEY from the environment. Estimated cost: ~$0.05–0.10 for N=100.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 scripts/gpt4omini_calibrate.py --n 100 --seed 42
    # outputs: /tmp/gpt4omini_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
PROFILER_DIR = code("profile_generator", "llm-movie-profiler-v1-20260402")
sys.path.insert(0, str(PROFILER_DIR))

from config.settings import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE,
    PROFILE_MIN_WORDS, PROFILE_MAX_WORDS,
)
from data_loader import load_all_data, format_genome_tags_for_prompt

# GPT-4o-mini pricing per 1M tokens (verify on platform.openai.com/docs/pricing)
PRICE_INPUT_STD = 0.150
PRICE_INPUT_CACHED = 0.075
PRICE_OUTPUT_STD = 0.600
PRICE_INPUT_BATCH = 0.075
PRICE_OUTPUT_BATCH = 0.300


def build_user_prompt_for(mid: int, data, tmdb_cache):
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

    tmdb = tmdb_cache.get(str(mid))
    if tmdb is None:
        return None
    top30 = data["top_k_tags"].get(mid)
    if top30 is None:
        return None
    rs = data["rating_stats"].get(mid, {"avg_rating": "N/A", "rating_count": 0})
    user_tags = data["user_tags"].get(mid, "No user tags available")

    return USER_PROMPT_TEMPLATE.format(
        title=title, year=year or "Unknown", genres=genres,
        genome_tags_formatted=format_genome_tags_for_prompt(top30),
        overview=tmdb["overview"],
        directors=tmdb["directors"],
        cast=tmdb["cast"],
        runtime=tmdb["runtime"],
        vote_average=tmdb["vote_average"],
        vote_count=tmdb["vote_count"],
        keywords=tmdb["keywords"],
        ml_avg_rating=rs.get("avg_rating", "N/A"),
        ml_rating_count=rs.get("rating_count", 0),
        user_tags_summary=user_tags or "No user tags available",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18",
                   help="exact GPT-4o-mini snapshot for reproducibility")
    p.add_argument("--out", default="/tmp/gpt4omini_calibration.json")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    print(f"[calib] Loading ML-20M data + TMDb cache (this takes ~30–60s)...")
    t_load = time.time()
    data = load_all_data()
    with open(PROFILER_DIR / "cache" / "tmdb_metadata.json") as f:
        tmdb_cache = json.load(f)
    print(f"[calib] Loaded in {time.time()-t_load:.1f}s. "
          f"Genome-covered movies: {len(data['movie_ids'])}")

    rng = random.Random(args.seed)
    sample_ids = rng.sample(data["movie_ids"], args.n)
    print(f"[calib] Sampling {args.n} movies (seed={args.seed})")
    print(f"[calib] Model: {args.model}, temp={CLAUDE_TEMPERATURE}, max_tokens={CLAUDE_MAX_TOKENS}")

    results, failed = [], []
    t0 = time.time()
    for i, mid in enumerate(sample_ids, 1):
        user_prompt = build_user_prompt_for(mid, data, tmdb_cache)
        if user_prompt is None:
            failed.append((mid, "missing inputs"))
            continue
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}],
                temperature=CLAUDE_TEMPERATURE,
                max_tokens=CLAUDE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            failed.append((mid, f"API error: {type(e).__name__}: {str(e)[:120]}"))
            continue

        usage = resp.usage
        text = resp.choices[0].message.content
        try:
            parsed = json.loads(text)
            valid = all(k in parsed for k in ("profile", "mood_vector", "key_themes"))
            wc = parsed.get("word_count") or len(parsed.get("profile", "").split())
            in_range = PROFILE_MIN_WORDS <= wc <= PROFILE_MAX_WORDS
        except Exception:
            valid, in_range, wc = False, False, 0

        cached = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = usage.prompt_tokens_details.cached_tokens or 0

        results.append({
            "movie_id": mid, "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens, "cached_tokens": cached,
            "valid_json": valid, "in_word_range": in_range, "word_count": wc,
        })
        if i % 10 == 0:
            print(f"[calib] {i}/{args.n} ({time.time()-t0:.0f}s) — last: "
                  f"in={usage.prompt_tokens} (cached={cached}), out={usage.completion_tokens}, "
                  f"valid={valid}, wc={wc}")

    n_ok = len(results)
    if n_ok == 0:
        print("ERROR: no successful calls. Check OPENAI_API_KEY + sample data.")
        sys.exit(1)

    in_mean = sum(r["input_tokens"] for r in results) / n_ok
    in_max = max(r["input_tokens"] for r in results)
    out_mean = sum(r["output_tokens"] for r in results) / n_ok
    out_max = max(r["output_tokens"] for r in results)
    cached_mean = sum(r["cached_tokens"] for r in results) / n_ok
    valid_rate = sum(1 for r in results if r["valid_json"]) / n_ok
    in_range_rate = sum(1 for r in results if r["in_word_range"]) / n_ok

    N = 10_381
    proj_in_total = in_mean * N
    proj_in_uncached = (in_mean - cached_mean) * N
    proj_in_cached = cached_mean * N
    proj_out_total = out_mean * N

    cost_std = (proj_in_total * PRICE_INPUT_STD + proj_out_total * PRICE_OUTPUT_STD) / 1e6
    cost_std_cached = ((proj_in_uncached * PRICE_INPUT_STD
                        + proj_in_cached * PRICE_INPUT_CACHED
                        + proj_out_total * PRICE_OUTPUT_STD) / 1e6)
    cost_batch = (proj_in_total * PRICE_INPUT_BATCH + proj_out_total * PRICE_OUTPUT_BATCH) / 1e6

    cal_cost = ((sum(r["input_tokens"] for r in results) * PRICE_INPUT_STD)
                + (sum(r["output_tokens"] for r in results) * PRICE_OUTPUT_STD)) / 1e6

    summary = {
        "n_calibrated": n_ok, "n_failed": len(failed),
        "model": args.model, "seed": args.seed,
        "input_tokens": {"mean": round(in_mean, 1), "max": in_max},
        "cached_input_tokens": {"mean": round(cached_mean, 1)},
        "output_tokens": {"mean": round(out_mean, 1), "max": out_max},
        "valid_json_rate": round(valid_rate, 3),
        "in_word_range_rate": round(in_range_rate, 3),
        "wall_time_s": round(time.time() - t0, 1),
        "calibration_cost_USD": round(cal_cost, 4),
        "projected_full_10381": {
            "input_tokens_total": int(proj_in_total),
            "input_tokens_cached": int(proj_in_cached),
            "input_tokens_uncached": int(proj_in_uncached),
            "output_tokens_total": int(proj_out_total),
            "cost_standard_no_cache_USD": round(cost_std, 2),
            "cost_standard_with_cache_USD": round(cost_std_cached, 2),
            "cost_batch_USD": round(cost_batch, 2),
        },
        "failed_samples": failed[:10],
    }

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 70 + "\nCALIBRATION SUMMARY\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
