#!/usr/bin/env python3
"""Build the prompt-provenance manifest for the ML-20M profile pack.

WHY THIS EXISTS. A resource paper needs the reader to be able to verify what the
model was actually shown. Publishing the rendered prompts would do that, but they
embed TMDb synopses/keywords and MovieLens genome scores — third-party content we
have no right to redistribute. So we publish a SHA-256 per rendered prompt
instead: full coverage of all 10,381 items, byte-verifiable, and zero
redistribution. Anyone holding their own copy of ML-20M plus TMDb access can
re-render a prompt with this same code and confirm the hash matches.

This REPLACES generation_logs/*/prompts_*.jsonl.gz, which covered only 721
distinct movies (6.9%), mixed in 149 failed retry attempts, and truncated every
record at ~500 characters — a run trace, not a provenance record.

Renders nothing to any API. Usage:
    python build_prompt_manifest.py --out manifest/            # build + self-verify
"""
import argparse, hashlib, json, sys
from pathlib import Path

def _find_up(start, *patterns):
    """First ancestor of `start` containing any of `patterns` (globs allowed).
    Names no sibling project, so the same code runs from a published clone."""
    for base in [start, *start.parents]:
        for pat in patterns:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


GEN = (_find_up(Path(__file__).resolve().parent, "src/profile_generator",
                "*/code/profile_generator/llm-movie-profiler-*")
       or Path(__file__).resolve().parents[1] / "src" / "profile_generator")
sys.path.insert(0, str(GEN))


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build(out_dir: Path):
    import os, types
    os.chdir(GEN)                       # settings.py uses paths relative to the package
    # main.py imports API-client machinery we neither need nor want to invoke.
    # Stub it rather than reimplementing prepare_movie_data, so the prompt is
    # rendered by the EXACT code path that produced the released profiles.
    for name, attrs in [("dotenv", {"load_dotenv": lambda *a, **k: None,
                   "dotenv_values": lambda *a, **k: {}}),
                        ("anthropic", {"Anthropic": object, "AsyncAnthropic": object,
                                       "APIError": type("APIError", (Exception,), {}),
                                       "RateLimitError": type("RateLimitError", (Exception,), {}),
                                       "APIStatusError": type("APIStatusError", (Exception,), {}),
                                       "APIConnectionError": type("APIConnectionError", (Exception,), {})}),
                        ("aiohttp", {"ClientSession": object, "ClientTimeout": object})]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
    from config import settings
    from data_loader import load_all_data
    from main import prepare_movie_data
    import json as _json

    print("loading ML-20M + TMDb cache (ratings.csv is ~530 MB, this is slow) ...", flush=True)
    data = load_all_data()
    tmdb = {int(k): v for k, v in _json.load(open(settings.METADATA_CACHE)).items()}
    # Use the genome-covered set the real run used, NOT the TMDb-cache keys.
    # get_metadata_for_movie() supplies documented placeholders ("No overview
    # available.", "Unknown") for movies absent from the cache, so those items
    # were still generated — with no TMDb content in their prompt at all.
    ids = sorted(data["movie_ids"])
    n_no_tmdb = sum(1 for m in ids if m not in tmdb)
    print(f"  genome-covered movies: {len(ids)}  (without TMDb metadata: {n_no_tmdb})", flush=True)

    rows = prepare_movie_data(data, tmdb, ids)
    print(f"  rendered prompts: {len(rows)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "prompt_manifest.jsonl"
    with open(man, "w") as f:
        f.write(_json.dumps({
            "record": "header",
            "purpose": "SHA-256 provenance for every rendered generation prompt; "
                       "prompts themselves are not redistributable (TMDb + GroupLens terms)",
            "model": settings.MODEL_NAME if hasattr(settings, "MODEL_NAME") else "claude-haiku-4-5-20251001",
            "temperature": getattr(settings, "TEMPERATURE", 0.3),
            "max_tokens": getattr(settings, "MAX_TOKENS", 600),
            "top_k_genome_tags": getattr(settings, "TOP_K_GENOME_TAGS", 30),
            "system_prompt_sha256": sha(settings.SYSTEM_PROMPT),
            "user_prompt_template_sha256": sha(settings.USER_PROMPT_TEMPLATE),
            "prompt_fields": ["title", "year", "genres", "genome_tags(top-30)", "overview",
                              "keywords", "directors", "cast", "runtime", "vote_average",
                              "vote_count", "ml_avg_rating", "ml_rating_count", "user_tags"],
            "n_prompts": len(rows),
            "n_without_tmdb_metadata": n_no_tmdb,
        }) + "\n")
        for r in rows:
            p = r["user_prompt"]
            f.write(_json.dumps({"movie_id": r["movie_id"], "prompt_sha256": sha(p),
                                 "prompt_chars": len(p),
                                 "tmdb_metadata": r["movie_id"] in tmdb}) + "\n")
    print(f"  wrote {man}")
    return {r["movie_id"]: r["user_prompt"] for r in rows}


def verify(rendered: dict):
    """Ground-truth check: every logged (truncated) prompt must be a PREFIX of the
    prompt we re-render. 721 distinct movies are covered by the old run trace."""
    import gzip
    log = (_find_up(Path(__file__).resolve().parent,
                "*/submission/hf_dataset/generation_logs/ml20m/prompts_claude-haiku-4-5.jsonl.gz")
           or Path("prompts_claude-haiku-4-5.jsonl.gz"))
    if not log.exists():
        print(f"  verify: skipped, run trace not present at {log}")
        return
    ok = bad = miss = 0
    for line in gzip.open(log, "rt"):
        r = json.loads(line)
        if r.get("type") != "request" or not r.get("success"):
            continue
        mid, logged = r["movie_id"], r["user_prompt"]
        if mid not in rendered:
            miss += 1; continue
        if rendered[mid].startswith(logged.rstrip(".")[:400]):
            ok += 1
        else:
            bad += 1
            if bad <= 2:
                print(f"    MISMATCH movie {mid}\n      logged   : {logged[:110]!r}\n"
                      f"      rendered : {rendered[mid][:110]!r}")
    print(f"\n  VERIFY vs the old run trace: {ok} match, {bad} mismatch, {miss} not rendered")
    return bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifest")
    a = ap.parse_args()
    rendered = build(Path(a.out).resolve())
    sys.exit(0 if verify(rendered) else 1)
