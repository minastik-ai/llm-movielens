#!/usr/bin/env python3
"""Stage-1 LLM profile generation for Steam (cross-domain, third catalogue).

For each game surviving the benchmark's 10-core, generate an embedding-optimized
profile + 10-axis mood vector + 3-5 key themes using Claude Haiku 4.5 with the
games prompt in
`code/profile_generator/llm-movie-profiler-v1-20260402/config/settings_steam.py`.

Structurally identical to scripts/generate_book_profiles.py -- same validation,
same retry policy, same prompt-caching call, same partial/resume handling -- so
the three catalogues differ only in their domain wording.

READ BEFORE INTERPRETING THE OUTPUT. Steam ships no free-text description, so the
median prompt payload is ~61 tokens against ML-20M's ~2,012. An 80-120 word
profile is therefore about twice the length of its own input, and the surplus
comes from the model's pretrained knowledge of the game. Run BOTH arms:

    python scripts/generate_steam_profiles.py                    # ~$1
    python scripts/generate_steam_profiles.py --mask-identity    # ~$1

The masked arm withholds title, developer and publisher, leaving only tags,
genres and modes. The gap between the two profile sets is how much of a Steam
profile is recall rather than synthesis. Do not report the unmasked arm as
evidence about the reasoning step on its own.

Inputs (must exist before running):
    code/profile_generator/output_steam/game_metadata.json
        produced by ecir-.../src/prepare_steam_metadata.py

Outputs:
    code/profile_generator/output_steam/game_profiles.json          (or _masked)
    code/profile_generator/output_steam/logs/prompts.jsonl

Cost / runtime: ~9,346 calls x ~$0.0001/call ~= $1 per arm, ~20 minutes at 15
workers. The per-call cost is far below the movie pipeline's because the input
is ~33x shorter.

Usage:
    python scripts/generate_steam_profiles.py --dry-run --limit 3   # no API calls
    export ANTHROPIC_API_KEY=sk-...
    python scripts/generate_steam_profiles.py --limit 20            # smoke test, ~$0.01
    python scripts/generate_steam_profiles.py                       # full run
    python scripts/generate_steam_profiles.py --resume              # continue partial
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
SETTINGS_DIR = code("profile_generator", "llm-movie-profiler-v1-20260402")
sys.path.insert(0, str(SETTINGS_DIR))
from config.settings_steam import (  # noqa: E402
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE_MASKED, MOOD_AXES,
    ANTHROPIC_MODEL, ANTHROPIC_TEMPERATURE, ANTHROPIC_MAX_TOKENS,
    PROFILE_WORD_MIN, PROFILE_WORD_MAX,
    KEY_THEMES_MIN, KEY_THEMES_MAX, MAX_RETRIES,
    GAME_METADATA_JSON, PROFILES_OUTPUT, PROFILES_PARTIAL, PROFILES_MASKED,
    PROMPT_LOG, LOG_DIR, ID_FIELD_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOG_DIR / "run.log", mode="a")],
)
logger = logging.getLogger("gen_steam")


def format_user_prompt(meta: dict, masked: bool = False) -> str:
    """Fill the games template. The masked variant withholds item identity."""
    genres = ", ".join(meta.get("genres") or []) or "(none listed)"
    tags = ", ".join(meta.get("tags") or []) or "(none listed)"
    specs = ", ".join(meta.get("specs") or []) or "(none listed)"
    if masked:
        return USER_PROMPT_TEMPLATE_MASKED.format(genres=genres, tags=tags, specs=specs)
    return USER_PROMPT_TEMPLATE.format(
        title=meta.get("title") or "Untitled",
        genres=genres, tags=tags, specs=specs,
        developer=meta.get("developer") or "Unknown",
        publisher=meta.get("publisher") or "Unknown",
    )


def validate_profile(obj: dict) -> tuple[bool, str]:
    """Return (ok, reason). Identical to the movie and book pipelines."""
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
    expected = {a[0] for a in MOOD_AXES}
    if set(mv.keys()) != expected:
        return False, f"mood_vector axes mismatch (missing {expected - set(mv.keys())})"
    for ax, v in mv.items():
        if not isinstance(v, (int, float)) or not -1.0 <= v <= 1.0:
            return False, f"mood_vector[{ax}]={v} out of [-1,1]"
    return True, ""


# Billing is MEASURED, not modelled. A cost projection for a new catalogue was
# wrong by ~3x twice running -- once by scaling on input length alone, once by
# omitting the (uncacheable) system prompt and assuming a 3x-too-low output
# length. The API reports exactly what it charged; record it and stop guessing.
USAGE = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "calls": 0}
_usage_lock = threading.Lock()


def call_claude(client, system_prompt: str, user_prompt: str) -> str:
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=ANTHROPIC_TEMPERATURE,
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )
    u = resp.usage
    with _usage_lock:
        USAGE["in"] += u.input_tokens
        USAGE["out"] += u.output_tokens
        USAGE["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        USAGE["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        USAGE["calls"] += 1
    return resp.content[0].text.strip()


def report_usage(logger, n_items_full_catalogue: int):
    """Print measured per-profile usage and the honest full-catalogue projection."""
    c = USAGE["calls"]
    if not c:
        return
    pin_s, pout_s = 1.0e-6, 5.0e-6          # Haiku 4.5 list (synchronous)
    sync = USAGE["in"] * pin_s + USAGE["out"] * pout_s
    logger.info("MEASURED USAGE ---------------------------------------------")
    logger.info(f"  calls {c:,} | input {USAGE['in']:,} ({USAGE['in']/c:,.0f}/call) "
                f"| output {USAGE['out']:,} ({USAGE['out']/c:,.0f}/call)")
    logger.info(f"  cache read {USAGE['cache_read']:,} | cache write {USAGE['cache_write']:,}"
                f"   <- 0/0 means the cache_control marker created nothing")
    logger.info(f"  this run: ${sync:.4f} synchronous (${sync/2:.4f} at batch price)")
    logger.info(f"  full {n_items_full_catalogue:,}-item arm: "
                f"${sync/c*n_items_full_catalogue:.2f} sync, "
                f"${sync/c*n_items_full_catalogue/2:.2f} batch")
    logger.info("------------------------------------------------------------")


def parse_json(text: str, item_id: int | str | None = None) -> dict | None:
    """Parse the model's JSON, repairing the one field it does not reliably emit.

    The model builds an identifier out of the title rather than echoing the one it
    was given, and when the title starts with or contains digits the result is an
    unquoted token that is not valid JSON: `"itemId": 40k_dow2_cr,` for
    *Warhammer 40,000: Dawn of War II Chaos Rising*, exactly as `34street1994`
    appears for *Miracle on 34th Street* in the movie catalogue. The movie pipeline
    overwrites the identifier with the catalogue's own before parsing; this path
    must do the same or it silently loses those items. Identifiers in the output
    therefore come from the catalogue, never from the model.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if item_id is not None:
        s = re.sub(rf'"{ID_FIELD_NAME}"\s*:\s*[^,}}\]]+', f'"{ID_FIELD_NAME}": {int(item_id)}',
                   s, count=1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="stop after N items")
    p.add_argument("--resume", action="store_true", help="resume from the partial file")
    p.add_argument("--shuffle", action="store_true", help="shuffle order (spreads API errors)")
    p.add_argument("--workers", type=int, default=15, help="concurrent workers")
    p.add_argument("--mask-identity", action="store_true",
                   help="withhold title/developer/publisher — the recall-vs-synthesis control")
    p.add_argument("--dry-run", action="store_true",
                   help="print the prompts that WOULD be sent and exit; no API calls, no cost")
    args = p.parse_args()

    if not GAME_METADATA_JSON.exists():
        logger.error(f"{GAME_METADATA_JSON} not found. "
                     f"Run ecir-.../src/prepare_steam_metadata.py first.")
        sys.exit(1)
    metadata = json.loads(GAME_METADATA_JSON.read_text())
    iids = sorted(metadata.keys(), key=int)
    if args.shuffle:
        import random
        random.Random(42).shuffle(iids)
    if args.limit:
        iids = iids[: args.limit]

    if args.dry_run:
        logger.info(f"DRY RUN — {len(metadata):,} games available, showing {len(iids)}")
        tot = 0
        for iid in iids:
            up = format_user_prompt(metadata[iid], args.mask_identity)
            tot += len(up)
            print(f"\n----- app {iid} ({len(up)} chars, ~{len(up)/3.8:.0f} tokens) -----\n{up}")
        print(f"\n  mean user-prompt {tot/max(len(iids),1):.0f} chars "
              f"(~{tot/max(len(iids),1)/3.8:.0f} tokens); system prompt "
              f"{len(SYSTEM_PROMPT)} chars, cached across calls")
        print("  no API calls made, no cost incurred")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("Install: pip install anthropic")
        sys.exit(1)
    client = Anthropic()

    out_path = PROFILES_MASKED if args.mask_identity else PROFILES_OUTPUT
    partial = PROFILES_PARTIAL.with_name(
        PROFILES_PARTIAL.stem + ("_masked" if args.mask_identity else "") + ".json")

    profiles = {}
    if args.resume and partial.exists():
        profiles = json.loads(partial.read_text())
        logger.info(f"Resumed: {len(profiles)} profiles already generated")
    iids = [i for i in iids if i not in profiles]
    logger.info(f"Will generate {len(iids)} new profiles "
                f"(target {len(metadata)}, arm={'masked' if args.mask_identity else 'full'})")

    failed, t_start = [], time.time()
    log_handle = open(PROMPT_LOG, "a")
    lock, counter = threading.Lock(), {"done": 0}

    def process_one(iid):
        meta = metadata[iid]
        user_prompt = format_user_prompt(meta, args.mask_identity)
        obj = None
        for attempt in range(MAX_RETRIES):
            try:
                text = call_claude(client, SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                logger.warning(f"[app={iid}] API error attempt {attempt+1}: {e}")
                time.sleep(2.0 * (attempt + 1))
                continue
            parsed = parse_json(text, iid)
            if parsed is None:
                logger.warning(f"[app={iid}] JSON parse failed (attempt {attempt+1})")
                continue
            ok, reason = validate_profile(parsed)
            if ok:
                obj = parsed
                break
            logger.warning(f"[app={iid}] validation failed: {reason}")
        if obj is None:
            return iid, None, user_prompt
        # Override the model's self-reported count with the real one, as the
        # primary ML-20M pipeline does (profile_generator.py line ~130). Models
        # overstate it by ~8 words, and a released field that disagrees with its
        # own text is a defect a reviewer can find by counting.
        obj["word_count"] = len(obj["profile"].split())
        obj[ID_FIELD_NAME] = int(iid)
        obj["app_id"] = iid
        obj["title"] = meta.get("title")
        obj["arm"] = "masked" if args.mask_identity else "full"
        return iid, obj, user_prompt

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, i): i for i in iids}
            for fut in as_completed(futures):
                iid, obj, user_prompt = fut.result()
                with lock:
                    counter["done"] += 1
                    n = counter["done"]
                    if obj is None:
                        failed.append(iid)
                        logger.error(f"[app={iid}] giving up after {MAX_RETRIES} retries")
                    else:
                        profiles[iid] = obj
                        log_handle.write(json.dumps({
                            "app_id": iid, "arm": obj["arm"],
                            "user_prompt": user_prompt, "response_obj": obj,
                        }) + "\n")
                    if n % 200 == 0:
                        json.dump(profiles, open(partial, "w"))
                        rate = n / max(time.time() - t_start, 1e-9)
                        logger.info(f"{n}/{len(iids)} done ({rate*60:.0f}/min, "
                                    f"{len(failed)} failed)")
    finally:
        log_handle.close()
        json.dump(profiles, open(partial, "w"))

    json.dump(profiles, open(out_path, "w"), indent=1)
    if failed:
        fp = out_path.with_name("failed_app_ids" +
                                ("_masked" if args.mask_identity else "") + ".json")
        json.dump(sorted(failed), open(fp, "w"), indent=1)
        logger.warning(f"{len(failed)} failed -> {fp.name} (release this list)")
    logger.info(f"Wrote {len(profiles)} profiles -> {out_path} "
                f"in {(time.time()-t_start)/60:.1f} min")
    report_usage(logger, len(metadata))


if __name__ == "__main__":
    main()
