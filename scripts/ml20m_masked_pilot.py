#!/usr/bin/env python3
"""ML-20M identity-handle ablation — the other half of the Steam comparison.

THE DESIGN, AND WHY IT IS NOT SYMMETRIC WITH STEAM.

The reviewer question behind both arms is "is the model just remembering the
item?". The clean test withholds the IDENTITY HANDLE -- the fields whose only job
is to name the thing -- while keeping every field that genuinely describes it.

  withheld : title, year, director, cast
  kept     : genome tags, genres, plot overview, keywords, scores, user tags

The plot stays. A synopsis will often let a model recognise the film anyway, and
that is FINE: the plot is supplied metadata, so using it is synthesis, not recall.
What we are isolating is knowledge the model brings that is NOT in the input.

This is a STRUCTURED-handle ablation, and the distinction matters. The free-text
fields still leak identity: the plot names Woody and Buzz Lightyear, and USER TAGS
carries "tom hanks" even though the CAST field is withheld. That is not a bug to
patch -- stripping names out of user-contributed tags would be arbitrary and would
destroy metadata the pipeline legitimately supplies -- but it does mean the result
must be reported as "the structured identity handle was removed", never as "the
model could not know which film this was". Expect a HIGH masked-vs-full similarity
on ML-20M for exactly this reason, and read it as "the title is redundant when the
plot is present", not as "the model is not using recall".

On Steam the identical ablation removes proportionally far more, because Steam
ships no plot at all -- 61 tokens of tags and specs, minus the title, leaves almost
nothing. That asymmetry is the finding, not a flaw in the comparison: it is exactly
what "metadata-poor catalogue" means, quantified.

settings.py is NOT modified. Its SYSTEM_PROMPT and USER_PROMPT_TEMPLATE hashes are
published in the release manifest; this script derives a masked template at runtime
and leaves the released prompts untouched.

Usage:  python ml20m_masked_pilot.py --limit 1000 [--dry-run]
"""
from __future__ import annotations
import argparse, json, logging, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

GEN = code("profile_generator", "llm-movie-profiler-v1-20260402")
sys.path.insert(0, str(GEN))
os.chdir(GEN)

from config.settings import (SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, CLAUDE_MODEL,
                             CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS)
from data_loader import load_all_data
from tmdb_crawler import load_cache
from main import prepare_movie_data
from profile_generator import validate_profile_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml20m-masked")
OUT = GEN / "output" / "movie_profiles_masked_pilot.json"

# The released template's identity-bearing lines, rewritten. Everything else is
# left byte-identical so the only difference between arms is the handle.
def mask_prompt(p: str) -> str:
    p = re.sub(r"^MOVIE: .*$", "MOVIE: (title, year and cast withheld)",
               p, count=1, flags=re.M)
    p = re.sub(r"^DIRECTOR: .*?\| CAST: .*?\| RUNTIME: (\S+)$",
               r"RUNTIME: \1", p, count=1, flags=re.M)
    return p

USAGE = {"in": 0, "out": 0, "cache_read": 0, "calls": 0}
_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    log.info("loading ML-20M + TMDb cache ...")
    data = load_all_data()
    tmdb = {int(k): v for k, v in load_cache().items()}
    rows = prepare_movie_data(data, tmdb, sorted(data["movie_ids"]))[: a.limit]
    log.info(f"prepared {len(rows):,} movies")

    genome_kept = sum(1 for r in rows if "GENOME TAGS" in mask_prompt(r["user_prompt"]))
    plot_kept = sum(1 for r in rows if "PLOT:" in mask_prompt(r["user_prompt"]))
    leaked = [r["movie_id"] for r in rows
              if re.search(r"^MOVIE: (?!\(title)", mask_prompt(r["user_prompt"]), re.M)]
    log.info(f"masking check: genome tags kept {genome_kept}/{len(rows)}, "
             f"plot kept {plot_kept}/{len(rows)}, MOVIE line unmasked in {len(leaked)}")
    if leaked:
        sys.exit(f"masking failed on {len(leaked)} rows, e.g. {leaked[:3]} -- fix before spending")

    if a.dry_run:
        ex = mask_prompt(rows[0]["user_prompt"])
        print("\n--- example masked prompt ---\n" + ex[:900])
        print(f"\n  {len(rows)} prompts ready; no API calls made, no cost incurred")
        return

    from anthropic import Anthropic
    from dotenv import dotenv_values
    # Take the first PLAUSIBLE key, not the first non-empty one. `config.settings`
    # calls load_dotenv() at import, which picks up a package-local .env holding a
    # 23-char placeholder; an `or` chain then short-circuits on that and never
    # reaches the real key one directory up. Same defect as batch_generate.py had.
    def usable(v): return (v or "").strip().startswith("sk-ant")
    key, src = "", ""
    for cand, label in ((os.environ.get("ANTHROPIC_API_KEY", ""), "environment"),
                        (dotenv_values(GEN / ".env").get("ANTHROPIC_API_KEY"), str(GEN / ".env")),
                        (dotenv_values(GEN.parents[2] / ".env").get("ANTHROPIC_API_KEY"),
                         str(GEN.parents[2] / ".env"))):
        if usable(cand):
            key, src = cand.strip(), label
            break
    if not key:
        sys.exit("No usable ANTHROPIC_API_KEY (expected 'sk-ant...'). Searched: environment, "
                 f"{GEN / '.env'}, {GEN.parents[2] / '.env'}")
    log.info(f"API key from {src}")
    client = Anthropic(api_key=key)

    profiles, failed = {}, []
    plock = threading.Lock()

    def one(r):
        mid = r["movie_id"]
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
                    temperature=CLAUDE_TEMPERATURE, system=SYSTEM_PROMPT,
                    messages=[{"role": "user",
                               "content": mask_prompt(r["user_prompt"])}])
            except Exception as e:
                log.warning(f"[{mid}] {type(e).__name__}: {str(e)[:120]}")
                time.sleep(2 * (attempt + 1)); continue
            u = resp.usage
            with _lock:
                USAGE["in"] += u.input_tokens; USAGE["out"] += u.output_tokens
                USAGE["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                USAGE["calls"] += 1
            obj, err = validate_profile_json(resp.content[0].text.strip(), mid)
            if obj:
                obj["arm"] = "masked"
                with plock:
                    profiles[str(mid)] = obj
                return
            log.warning(f"[{mid}] validation: {err}")
        failed.append(mid)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, rows))
    OUT.write_text(json.dumps(profiles, ensure_ascii=False, indent=1))

    c = max(USAGE["calls"], 1)
    sync = USAGE["in"] * 1e-6 + USAGE["out"] * 5e-6
    log.info(f"wrote {len(profiles):,} profiles -> {OUT.name} "
             f"({len(failed)} failed) in {(time.time()-t0)/60:.1f} min")
    log.info("MEASURED USAGE ---------------------------------------------")
    log.info(f"  calls {c:,} | input {USAGE['in']/c:,.0f}/call | output {USAGE['out']/c:,.0f}/call")
    log.info(f"  cache read {USAGE['cache_read']:,}")
    log.info(f"  this run ${sync:.4f} sync (${sync/2:.4f} batch); "
             f"full 10,381 arm ${sync/c*10381:.2f} sync, ${sync/c*10381/2:.2f} batch")
    log.info("------------------------------------------------------------")


if __name__ == "__main__":
    main()
