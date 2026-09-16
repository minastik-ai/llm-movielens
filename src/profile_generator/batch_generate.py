#!/usr/bin/env python3
"""
batch_generate.py — profile generation through the Anthropic Message Batches API.

THIS IS THE PATH THE PAPER'S COST FIGURE ASSUMES, and the one to use for a full
catalogue. The reported generation cost is the Batches-API price at a 50%
discount. main.py submits the same requests synchronously and therefore costs
about twice as much; use it for small tests, where the batch queue's start-up
latency dominates (5 requests took 43m43s, 10,000 took 13m12s).

The two paths differ ONLY in transport. Every scientific unit is imported from
the same modules rather than duplicated here:
  - config.settings          SYSTEM_PROMPT / model / temperature / max_tokens
  - profile_generator        build_user_prompt(), validate_profile_json()
  - main.prepare_movie_data  the prompt payload assembled per movie
So the records the two paths emit are identical, and a change to the prompt or
the schema cannot reach one path without reaching the other.

Batch results return in ARBITRARY order -> always key by custom_id.

Usage:
  python batch_generate.py --dry-run          # assemble prompts, submit nothing
  python batch_generate.py --limit 5          # small live batch
  python batch_generate.py                    # all 10,381
  python batch_generate.py --resume           # only movies not already in output
"""
import argparse, json, logging, os, sys, time
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import dotenv_values, load_dotenv

# Resolve .env relative to THIS file, never an absolute home path: a hardcoded
# /Users/... both leaks the author's machine layout and breaks on anyone else's.
# code/profile_generator/<pkg>/ -> repo root is three levels up.
ENV_FILES = [Path(__file__).resolve().parent / ".env",
             Path(__file__).resolve().parents[3] / ".env"]
for _f in ENV_FILES:
    load_dotenv(_f)                     # non-key settings; nearest file wins


def resolve_api_key():
    """Return (key, where_it_came_from).

    NOT just os.environ: load_dotenv() never overrides an already-set value, so a
    stale placeholder in a nearer .env silently masks a real key in a further one
    and the script dies claiming the key is "not set" while the reader is looking
    right at it. Pick the first source that yields a plausible key, and report
    which one, so a misconfiguration is a five-second fix rather than a hunt.
    """
    env = os.environ.get("ANTHROPIC_API_KEY", "")
    if env.startswith("sk-ant"):
        return env, "environment"
    for f in ENV_FILES:
        if f.exists():
            v = (dotenv_values(f).get("ANTHROPIC_API_KEY") or "").strip()
            if v.startswith("sk-ant"):
                return v, str(f)
    return "", ""

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import (  # noqa: E402
    SYSTEM_PROMPT, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS,
    USE_PROMPT_CACHING, PROFILES_OUTPUT, LOG_DIR, GENOME_TAGS_CSV,
)
from data_loader import load_all_data  # noqa: E402
from tmdb_crawler import load_cache  # noqa: E402
from profile_generator import validate_profile_json, load_existing_profiles  # noqa: E402
from main import prepare_movie_data  # noqa: E402

CHUNK = 2_500           # ~20MB/submit; 10k in one POST caused APIConnectionError
POLL_SECONDS = 30       # matches the original run's polling cadence
MAX_RETRY_ROUNDS = 3    # original needed a 2nd pass for ~3% of movies

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"batch_{datetime.now():%Y%m%d_%H%M%S}.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def system_block():
    if USE_PROMPT_CACHING:
        return [{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}]
    return SYSTEM_PROMPT


def submit_batch(client, movies):
    """Submit one chunk; return (batch_id, movies)."""
    requests = [
        Request(
            custom_id=f"movie_{m['movie_id']}",
            params=MessageCreateParamsNonStreaming(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                temperature=CLAUDE_TEMPERATURE,
                system=system_block(),
                messages=[{"role": "user", "content": m["user_prompt"]}],
            ),
        )
        for m in movies
    ]
    for attempt in range(1, 6):
        try:
            b = client.messages.batches.create(requests=requests)
            log.info(f"  submitted {len(requests):,} requests | batch {b.id}")
            return b.id, movies
        except (anthropic.BadRequestError, anthropic.AuthenticationError,
                anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            # 400/401/403/404 are permanent — retrying cannot help. Surface the
            # message (credit exhaustion arrives here as a 400) and stop.
            log.error(f"  NON-RETRYABLE {type(e).__name__}: "
                      f"{getattr(e, 'message', str(e))[:300]}")
            raise
        except Exception as e:
            wait = 15 * attempt
            log.warning(f"  submit attempt {attempt}/5 failed ({type(e).__name__}: "
                        f"{str(e)[:200]}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("batch submission failed after 5 attempts")


def collect(client, jobs):
    """Wait for every submitted batch, then retrieve. Returns {movie_id: raw_text}."""
    raw = {}
    for bid, _ in jobs:
        while True:
            b = client.messages.batches.retrieve(bid)
            c = b.request_counts
            if b.processing_status == "ended":
                log.info(f"  {bid} ended | ok={c.succeeded} err={c.errored} exp={c.expired}")
                break
            log.info(f"  {bid} {b.processing_status} | ok={c.succeeded} "
                     f"err={c.errored} proc={c.processing}")
            time.sleep(POLL_SECONDS)

        n_err = 0
        for r in client.messages.batches.results(bid):
            mid = int(r.custom_id.removeprefix("movie_"))   # key by custom_id
            if r.result.type == "succeeded":
                raw[mid] = r.result.message.content[0].text
            else:
                n_err += 1
        log.info(f"  {bid} retrieved | running total {len(raw):,} ok, {n_err:,} errored")
    return raw


def save(profiles, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(k): v for k, v in sorted(profiles.items())},
                              ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N movies (0=all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be submitted and exit; no API calls, no cost")
    ap.add_argument("--resume", action="store_true", help="skip movies already in output")
    ap.add_argument("--out", type=Path, default=PROFILES_OUTPUT)
    args = ap.parse_args()

    log.info("=" * 70)
    log.info("BATCH PROFILE GENERATION")
    log.info(f"  model={CLAUDE_MODEL} temp={CLAUDE_TEMPERATURE} "
             f"max_tokens={CLAUDE_MAX_TOKENS} caching={USE_PROMPT_CACHING}")
    log.info("=" * 70)

    # Ask the resolved location, do not rebuild it -- a second construction of
    # the same path is a second thing that can disagree with the download script.
    # Ask the resolved location, do not rebuild it -- a second construction of the
    # same path is a second thing that can disagree with the download script.
    _need = Path(GENOME_TAGS_CSV)
    if not _need.exists():
        sys.exit(
            "Source data not found. We do not redistribute MovieLens; fetch it first:\n"
            "    bash scripts/download_ml20m.sh\n"
            f"expected at: {_need}\n"
            "This applies to --dry-run too: the dry run assembles the real prompts, which\n"
            "is what makes it worth running before you spend anything.")

    data = load_all_data()
    movie_ids = data["movie_ids"]
    # main.py applies this transform before prepare_movie_data — match it exactly
    tmdb_cache = load_cache()
    tmdb_metadata = {int(k): v for k, v in tmdb_cache.items() if int(k) in set(movie_ids)}
    log.info(f"TMDb metadata available for {len(tmdb_metadata):,} movies")
    movies = prepare_movie_data(data, tmdb_metadata, movie_ids)
    log.info(f"prepared {len(movies):,} prompts")

    profiles = {}
    if args.resume and args.out.exists():
        profiles = {int(k): v for k, v in json.loads(args.out.read_text()).items()}
        movies = [m for m in movies if m["movie_id"] not in profiles]
        log.info(f"resume: {len(profiles):,} existing, {len(movies):,} remaining")

    if args.limit:
        movies = movies[: args.limit]
        log.info(f"LIMIT: {len(movies)} movies")

    if args.dry_run:
        n = len(movies)
        log.info(f"DRY RUN: would submit {n:,} requests in "
                 f"{(n + CHUNK - 1)//CHUNK} chunk(s) of up to {CHUNK:,}")
        if movies:
            log.info(f"  first custom_id would be movie_{movies[0]["movie_id"]}")
        log.info("  no API calls made, no cost incurred")
        return

    pending = movies
    try:
      for rnd in range(1, MAX_RETRY_ROUNDS + 1):
          if not pending:
              break
          log.info(f"── round {rnd}: {len(pending):,} movies ──")
          jobs = []
          for i in range(0, len(pending), CHUNK):
              chunk = pending[i : i + CHUNK]
              log.info(f"  chunk {i//CHUNK + 1} ({len(chunk):,} requests)")
              jobs.append(submit_batch(client, chunk))
          raw = collect(client, jobs)
          for m in pending:
              mid = m["movie_id"]
              if mid not in raw:
                  continue
              profile, err = validate_profile_json(raw[mid], mid)
              if profile is not None:
                  profiles[mid] = profile
              else:
                  log.warning(f"  movie {mid}: validation failed — {err}")

          pending = [m for m in movies if m["movie_id"] not in profiles]
          save(profiles, args.out)          # checkpoint: never lose completed rounds
          log.info(f"round {rnd} done: {len(profiles):,} valid, {len(pending):,} outstanding")

    except Exception:
        save(profiles, args.out)          # persist before propagating
        log.error(f"aborted — {len(profiles):,} profiles saved to {args.out}")
        raise

    save(profiles, args.out)
    log.info("=" * 70)
    log.info(f"COMPLETE: {len(profiles):,} profiles -> {args.out}")
    if pending:
        log.warning(f"{len(pending):,} movies still missing after "
                    f"{MAX_RETRY_ROUNDS} rounds")
    log.info("=" * 70)


if __name__ == "__main__":
    # --help and --dry-run must work WITHOUT credentials: a reproduction script a
    # reader cannot even inspect without an API key is a poor artifact.
    _argv = sys.argv[1:]
    if not ({"-h", "--help", "--dry-run"} & set(_argv)):
        key, source = resolve_api_key()
        if not key:
            sys.exit("No usable ANTHROPIC_API_KEY (expected to start with 'sk-ant').\n"
                     "Searched: the environment, then " +
                     ", ".join(str(f) for f in ENV_FILES))
        log.info(f"API key from {source}")
        client = anthropic.Anthropic(api_key=key, timeout=900.0, max_retries=5)
    main()
