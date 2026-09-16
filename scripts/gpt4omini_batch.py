#!/usr/bin/env python3
"""cross-LLM Option B — Steps 2/3: Submit OpenAI Batch API request for full ML-20M
GPT-4o-mini profile generation, then parse + validate the response.

Three modes (set via --mode):
    build  — build the JSONL request file from ML-20M data + TMDb cache
    submit — upload JSONL + create batch + print batch ID
    poll   — poll batch status until completed (or failed)
    parse  — fetch result file + validate + write profiles JSON
    full   — run all four sequentially (build → submit → poll → parse)

Outputs:
    /tmp/gpt4omini_requests.jsonl                       — batch input
    code/profile_generator/output_ml20m_gpt4omini/      — batch metadata + parsed results
        batch_info.json                                  — batch ID + status
        movie_profiles.json                              — validated profiles
        failed_iids.json                                 — movies that failed
        usage.json                                       — token+cost accounting

Usage:
    export OPENAI_API_KEY=sk-...
    python3 scripts/gpt4omini_batch.py --mode build
    python3 scripts/gpt4omini_batch.py --mode submit
    # ... wait, then ...
    python3 scripts/gpt4omini_batch.py --mode poll
    python3 scripts/gpt4omini_batch.py --mode parse
    # OR end-to-end:
    python3 scripts/gpt4omini_batch.py --mode full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
PROFILER_DIR = code("profile_generator", "llm-movie-profiler-v1-20260402")
OUT_DIR = code("profile_generator", "output_ml20m_gpt4omini")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REQ_FILE = Path("/tmp/gpt4omini_requests.jsonl")
INFO_FILE = OUT_DIR / "batch_info.json"
PROFILES_FILE = OUT_DIR / "movie_profiles.json"
FAILED_FILE = OUT_DIR / "failed_iids.json"
USAGE_FILE = OUT_DIR / "usage.json"

sys.path.insert(0, str(PROFILER_DIR))
from config.settings import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE,
    PROFILE_MIN_WORDS, PROFILE_MAX_WORDS,
)
from data_loader import load_all_data, format_genome_tags_for_prompt

MODEL = "gpt-4o-mini-2024-07-18"
PRICE_INPUT_BATCH = 0.075   # $/M tokens
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
        overview=tmdb["overview"], directors=tmdb["directors"],
        cast=tmdb["cast"], runtime=tmdb["runtime"],
        vote_average=tmdb["vote_average"], vote_count=tmdb["vote_count"],
        keywords=tmdb["keywords"],
        ml_avg_rating=rs.get("avg_rating", "N/A"),
        ml_rating_count=rs.get("rating_count", 0),
        user_tags_summary=user_tags or "No user tags available",
    )


def cmd_build():
    """Build the JSONL request file for all genome-covered movies."""
    print(f"[build] Loading ML-20M data + TMDb cache...")
    t0 = time.time()
    data = load_all_data()
    with open(PROFILER_DIR / "cache" / "tmdb_metadata.json") as f:
        tmdb_cache = json.load(f)
    print(f"[build] Loaded in {time.time()-t0:.1f}s. Movies: {len(data['movie_ids'])}")

    n_built, n_skipped = 0, 0
    skipped_ids = []
    with open(REQ_FILE, "w") as f:
        for mid in sorted(data["movie_ids"]):
            user_prompt = build_user_prompt_for(mid, data, tmdb_cache)
            if user_prompt is None:
                n_skipped += 1
                skipped_ids.append(mid)
                continue
            req = {
                "custom_id": f"movie-{mid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": CLAUDE_TEMPERATURE,
                    "max_tokens": CLAUDE_MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(req) + "\n")
            n_built += 1

    sz_mb = REQ_FILE.stat().st_size / 1e6
    print(f"[build] {n_built} requests written, {n_skipped} skipped (no inputs)")
    print(f"[build] File: {REQ_FILE} ({sz_mb:.1f} MB; OpenAI batch limit 200 MB)")
    if skipped_ids:
        FAILED_FILE.write_text(json.dumps(
            {"missing_inputs_at_build": skipped_ids}, indent=2))
        print(f"[build] Skipped IDs saved to {FAILED_FILE}")
    return n_built


def cmd_submit():
    """Upload JSONL and create batch."""
    if not REQ_FILE.exists():
        print(f"ERROR: {REQ_FILE} not found. Run --mode build first.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI()
    print(f"[submit] Uploading {REQ_FILE} ({REQ_FILE.stat().st_size/1e6:.1f} MB)...")
    f = client.files.create(file=open(REQ_FILE, "rb"), purpose="batch")
    print(f"[submit] file_id: {f.id}")
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "llm-movielens GPT-4o-mini ML-20M profiles (cross-LLM)"},
    )
    info = {
        "batch_id": batch.id,
        "input_file_id": f.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
        "model": MODEL,
        "n_requests": batch.request_counts.total if batch.request_counts else None,
    }
    INFO_FILE.write_text(json.dumps(info, indent=2))
    print(f"[submit] Batch ID: {batch.id}")
    print(f"[submit] Status: {batch.status}")
    print(f"[submit] Saved to {INFO_FILE}")


def cmd_poll(poll_interval=60):
    """Poll batch until completed or failed."""
    if not INFO_FILE.exists():
        print(f"ERROR: {INFO_FILE} not found. Run --mode submit first.")
        sys.exit(1)
    info = json.loads(INFO_FILE.read_text())
    from openai import OpenAI
    client = OpenAI()
    print(f"[poll] Watching batch {info['batch_id']} (poll every {poll_interval}s)")
    t0 = time.time()
    while True:
        b = client.batches.retrieve(info["batch_id"])
        rc = b.request_counts
        elapsed = time.time() - t0
        print(f"[poll] {time.strftime('%H:%M:%S')} status={b.status} "
              f"completed={rc.completed if rc else 0}/{rc.total if rc else 0} "
              f"failed={rc.failed if rc else 0} elapsed={elapsed:.0f}s")
        info["status"] = b.status
        if rc:
            info["request_counts"] = {"total": rc.total, "completed": rc.completed,
                                       "failed": rc.failed}
        info["output_file_id"] = b.output_file_id
        info["error_file_id"] = b.error_file_id
        INFO_FILE.write_text(json.dumps(info, indent=2))
        if b.status in ("completed", "failed", "expired", "cancelled"):
            print(f"[poll] Terminal status: {b.status}")
            break
        time.sleep(poll_interval)


def cmd_parse():
    """Fetch output file + validate."""
    if not INFO_FILE.exists():
        print(f"ERROR: {INFO_FILE} not found.")
        sys.exit(1)
    info = json.loads(INFO_FILE.read_text())
    if info["status"] != "completed":
        print(f"ERROR: batch status is {info['status']}, not completed.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI()
    out_id = info["output_file_id"]
    print(f"[parse] Downloading output file {out_id}...")
    raw = client.files.content(out_id).text

    profiles = {}
    failed_parse = []
    total_in_toks, total_out_toks, total_cached = 0, 0, 0
    n_total = 0
    for line in raw.strip().split("\n"):
        if not line:
            continue
        n_total += 1
        rec = json.loads(line)
        cid = rec["custom_id"]
        mid = int(cid.replace("movie-", ""))
        if rec.get("error") or rec["response"]["status_code"] != 200:
            failed_parse.append({"movie_id": mid,
                                 "reason": "API error",
                                 "detail": str(rec.get("error") or rec["response"])[:200]})
            continue
        body = rec["response"]["body"]
        usage = body["usage"]
        total_in_toks += usage["prompt_tokens"]
        total_out_toks += usage["completion_tokens"]
        ptd = usage.get("prompt_tokens_details") or {}
        total_cached += ptd.get("cached_tokens", 0)
        text = body["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(text)
        except Exception as e:
            failed_parse.append({"movie_id": mid, "reason": "JSON parse error",
                                 "detail": str(e)[:200], "raw": text[:300]})
            continue
        if not all(k in parsed for k in ("profile", "mood_vector", "key_themes")):
            failed_parse.append({"movie_id": mid, "reason": "schema fields missing",
                                 "detail": str(parsed)[:200]})
            continue
        wc = parsed.get("word_count") or len(parsed.get("profile", "").split())
        if not (PROFILE_MIN_WORDS <= wc <= PROFILE_MAX_WORDS):
            failed_parse.append({"movie_id": mid, "reason": "word_count out of range",
                                 "detail": f"wc={wc}", "raw": text[:300]})
            continue
        profiles[str(mid)] = parsed

    PROFILES_FILE.write_text(json.dumps(profiles, indent=2))
    # Merge with build-time skips
    existing_failed = {}
    if FAILED_FILE.exists():
        existing_failed = json.loads(FAILED_FILE.read_text())
    existing_failed["api_or_validation_failed"] = failed_parse
    FAILED_FILE.write_text(json.dumps(existing_failed, indent=2))

    cost_in = total_in_toks * PRICE_INPUT_BATCH / 1e6
    cost_out = total_out_toks * PRICE_OUTPUT_BATCH / 1e6
    usage_summary = {
        "n_records_total": n_total,
        "n_validated_profiles": len(profiles),
        "n_failed_validation": len(failed_parse),
        "input_tokens_total": total_in_toks,
        "cached_input_tokens_total": total_cached,
        "output_tokens_total": total_out_toks,
        "cost_input_USD": round(cost_in, 4),
        "cost_output_USD": round(cost_out, 4),
        "cost_total_USD": round(cost_in + cost_out, 4),
        "model": MODEL,
    }
    USAGE_FILE.write_text(json.dumps(usage_summary, indent=2))

    print(f"[parse] {len(profiles)} valid profiles, {len(failed_parse)} failed validation")
    print(f"[parse] Profiles: {PROFILES_FILE}")
    print(f"[parse] Failed:   {FAILED_FILE}")
    print(f"[parse] Usage:    {USAGE_FILE}")
    print(json.dumps(usage_summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "submit", "poll", "parse", "full"], required=True)
    p.add_argument("--poll_interval", type=int, default=60, help="seconds between batch status polls")
    args = p.parse_args()

    if args.mode in ("submit", "poll", "parse", "full") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    if args.mode in ("build", "full"):
        cmd_build()
    if args.mode in ("submit", "full"):
        cmd_submit()
    if args.mode in ("poll", "full"):
        cmd_poll(poll_interval=args.poll_interval)
    if args.mode in ("parse", "full"):
        cmd_parse()


if __name__ == "__main__":
    main()
