#!/usr/bin/env python3
"""Stage-1 LLM profile generation for Amazon-Books (cross-domain experiment).

For each of 9,332 books in the RLMRec Amazon-Books subset, generate an
embedding-optimized profile + 10-axis mood vector + 3-5 key themes using
Claude Haiku 4.5 with the books-specific prompt in
`code/profile_generator/llm-movie-profiler-v1-20260402/config/settings_amazon.py`.

Inputs (must exist before running):
    code/profile_generator/output_amazon/book_metadata.json
        produced by scripts/fetch_amazon_books_metadata.py
    code/benchmark/data/processed_amazon/item_map.json
        produced by scripts/convert_rlmrec_amazon_splits.py

Outputs:
    code/profile_generator/output_amazon/book_profiles.json
        keyed by iid (string), one entry per book with
            profile (str, 80-120 words), word_count (int),
            key_themes (list[str], 3-5), mood_vector (dict of 10 floats)
    code/profile_generator/output_amazon/logs/prompts.jsonl
        full prompt + response log for reproducibility

Cost / runtime: ~9,332 calls × ~$0.0013/call ≈ $12, ~50 minutes with
Anthropic prompt caching enabled for the system prompt.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python scripts/generate_book_profiles.py
    python scripts/generate_book_profiles.py --limit 5    # smoke test
    python scripts/generate_book_profiles.py --resume     # continue partial
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
SETTINGS_DIR = code("profile_generator", "llm-movie-profiler-v1-20260402")
sys.path.insert(0, str(SETTINGS_DIR))
from config.settings_amazon import (  # noqa: E402
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, MOOD_AXES,
    ANTHROPIC_MODEL, ANTHROPIC_TEMPERATURE, ANTHROPIC_MAX_TOKENS,
    PROFILE_WORD_MIN, PROFILE_WORD_MAX,
    KEY_THEMES_MIN, KEY_THEMES_MAX, MAX_RETRIES,
    BOOK_METADATA_JSON, PROFILES_OUTPUT, PROFILES_PARTIAL, PROMPT_LOG,
    OUTPUT_DIR, LOG_DIR, ID_FIELD_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOG_DIR / "run.log", mode="a")],
)
logger = logging.getLogger("gen_books")


def format_user_prompt(meta: dict) -> str:
    """Fill in the books USER_PROMPT_TEMPLATE from a metadata entry."""
    title = meta.get("title") or f"Untitled (ASIN {meta['asin']})"
    if meta.get("subtitle"):
        title = f"{title}: {meta['subtitle']}"
    subjects = ", ".join(meta.get("subjects") or []) or "(no subjects available)"
    description = (meta.get("rlmrec_profile") or "").strip() or "(no description available)"
    authors = ", ".join(meta.get("authors") or []) or "Unknown"
    publishers = ", ".join(meta.get("publishers") or []) or "Unknown"
    pages = meta.get("number_of_pages") or "?"
    publish_date = meta.get("publish_date") or "n/a"
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        subjects=subjects,
        description=description[:1500],     # cap to keep input tokens bounded
        authors=authors,
        publishers=publishers,
        pages=pages,
        publish_date=publish_date,
    )


def validate_profile(obj: dict) -> tuple[bool, str]:
    """Return (ok, reason). Mirrors the movie pipeline's validation."""
    if "profile" not in obj or not isinstance(obj["profile"], str):
        return False, "missing or non-string profile"
    wc = len(obj["profile"].split())
    if not (PROFILE_WORD_MIN - 5 <= wc <= PROFILE_WORD_MAX):
        return False, f"word_count {wc} out of [{PROFILE_WORD_MIN},{PROFILE_WORD_MAX}]"
    themes = obj.get("key_themes")
    if not (isinstance(themes, list) and KEY_THEMES_MIN <= len(themes) <= KEY_THEMES_MAX):
        return False, f"key_themes count {len(themes) if isinstance(themes, list) else 0}"
    mv = obj.get("mood_vector")
    if not isinstance(mv, dict):
        return False, "mood_vector not dict"
    expected_axes = {a[0] for a in MOOD_AXES}
    if set(mv.keys()) != expected_axes:
        missing = expected_axes - set(mv.keys())
        return False, f"mood_vector axes mismatch (missing {missing})"
    for ax, v in mv.items():
        if not isinstance(v, (int, float)) or not -1.0 <= v <= 1.0:
            return False, f"mood_vector[{ax}]={v} out of [-1,1]"
    return True, ""


def call_claude(client, system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic API with prompt caching on the system prompt."""
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=ANTHROPIC_TEMPERATURE,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


def parse_json(text: str) -> dict | None:
    """Strip markdown fencing and parse JSON. Returns None on failure."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N items (smoke test).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from PROFILES_PARTIAL if present.")
    p.add_argument("--shuffle", action="store_true",
                   help="Process items in shuffled order (helps spread API errors).")
    p.add_argument("--workers", type=int, default=15,
                   help="Concurrent API workers (default 15; tier-1 limit 200 RPM).")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set in environment.")
        sys.exit(1)

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("Install: pip install anthropic")
        sys.exit(1)

    client = Anthropic()

    if not BOOK_METADATA_JSON.exists():
        logger.error(f"{BOOK_METADATA_JSON} not found. "
                     f"Run scripts/fetch_amazon_books_metadata.py first.")
        sys.exit(1)
    metadata = json.loads(BOOK_METADATA_JSON.read_text())
    iids = sorted(metadata.keys(), key=int)
    if args.shuffle:
        import random
        random.Random(42).shuffle(iids)
    if args.limit:
        iids = iids[: args.limit]

    profiles = {}
    if args.resume and PROFILES_PARTIAL.exists():
        profiles = json.loads(PROFILES_PARTIAL.read_text())
        logger.info(f"Resumed: {len(profiles)} profiles already generated")
    iids = [i for i in iids if i not in profiles]
    logger.info(f"Will generate {len(iids)} new profiles "
                f"(total target: {len(metadata)})")

    failed = []
    t_start = time.time()
    log_handle = open(PROMPT_LOG, "a")
    lock = threading.Lock()
    counter = {"done": 0}

    def process_one(iid):
        meta = metadata[iid]
        user_prompt = format_user_prompt(meta)
        obj = None
        for attempt in range(MAX_RETRIES):
            try:
                text = call_claude(client, SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                logger.warning(f"[iid={iid}] API error attempt {attempt+1}: {e}")
                time.sleep(2.0 * (attempt + 1))
                continue
            parsed = parse_json(text)
            if parsed is None:
                logger.warning(f"[iid={iid}] JSON parse failed (attempt {attempt+1})")
                continue
            ok, reason = validate_profile(parsed)
            if ok:
                obj = parsed
                break
            logger.warning(f"[iid={iid}] validation failed: {reason}")
        if obj is None:
            return iid, None, user_prompt
        # Override the model's self-reported count with the real one, as the
        # primary ML-20M pipeline does (generator.py ~line 130). Models
        # overstate it by a median of 4-8 words, and a released field that
        # disagrees with its own text is a defect a reviewer can find by counting.
        obj["word_count"] = len(obj["profile"].split())
        obj[ID_FIELD_NAME] = int(iid)
        obj["asin"] = meta["asin"]
        obj["title"] = meta.get("title")
        return iid, obj, user_prompt

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, iid): iid for iid in iids}
            for fut in as_completed(futures):
                iid, obj, user_prompt = fut.result()
                with lock:
                    counter["done"] += 1
                    n_done = counter["done"]
                    if obj is None:
                        failed.append(iid)
                        logger.error(f"[iid={iid}] giving up after {MAX_RETRIES} retries")
                    else:
                        profiles[iid] = obj
                        log_handle.write(json.dumps({
                            "iid": iid, "asin": metadata[iid]["asin"],
                            "user_prompt": user_prompt, "response_obj": obj,
                        }) + "\n")
                        log_handle.flush()
                    if n_done % 100 == 0:
                        PROFILES_PARTIAL.write_text(
                            json.dumps(profiles, indent=2, default=str))
                        elapsed = time.time() - t_start
                        eta_min = (len(iids) - n_done) * elapsed / n_done / 60
                        logger.info(f"  {n_done}/{len(iids)} done, "
                                    f"validated {len(profiles) - (len(profiles)-n_done+len(failed))}, "
                                    f"failed {len(failed)}, "
                                    f"elapsed {elapsed:.0f}s, eta ~{eta_min:.1f} min")
    finally:
        log_handle.close()

    PROFILES_OUTPUT.write_text(json.dumps(profiles, indent=2, default=str))
    logger.info(f"Wrote {len(profiles)} profiles to "
                f"{PROFILES_OUTPUT.relative_to(REPO)}")
    if failed:
        (OUTPUT_DIR / "failed_iids.json").write_text(json.dumps(failed))
        logger.warning(f"{len(failed)} items failed; iids written to "
                       f"output_amazon/failed_iids.json")


if __name__ == "__main__":
    main()
