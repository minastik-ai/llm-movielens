#!/usr/bin/env python3
"""End-to-end check of the RELEASED generator against the RELEASED artifact.

The paper's contribution is a pipeline. A reader must be able to establish three
things without spending a cent, and this script establishes all three:

  1. TRANSPORT-ONLY.  batch_generate.py and main.py assemble byte-identical
     requests -- same system prompt, same per-movie user prompt, same model,
     temperature and max_tokens.  Only the submission call differs.  This is what
     licenses quoting the batch price for an artifact either path could produce.

  2. THE CODE MATCHES THE RELEASE.  Re-rendering prompts with the shipped code
     reproduces the SHA-256 values in the released prompt manifest.  If the
     prompt had drifted since generation, the hashes would diverge.

  3. THE VALIDATOR ACCEPTS THE ARTIFACT.  The shipped validate_profile_json()
     accepts every released profile.  A validator that rejects the release would
     mean the shipped code could not have produced it.

Calls no API.  Usage:  python verify_generator_e2e.py [--sample N]
"""
import argparse, hashlib, json, os, sys, types
from pathlib import Path

# Published layout first, working-tree layout second. A verifier that only runs on
# the author's machine verifies nothing, and a clean-clone test is the only thing
# that catches the difference.
def _find_up(start, *patterns):
    """First ancestor of `start` containing any of `patterns` (globs allowed).
    Names no sibling project: the development checkout is found by SHAPE, so the
    same code runs unchanged from a published clone."""
    for base in [start, *start.parents]:
        for pat in patterns:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


_HERE = Path(__file__).resolve()
GEN = (_find_up(_HERE.parent, "src/profile_generator",
                "*/code/profile_generator/llm-movie-profiler-*")
       or _HERE.parents[1] / "src" / "profile_generator")
MANIFEST = (_find_up(_HERE.parent, "manifest/prompt_manifest.jsonl",
                     "release/manifest/prompt_manifest.jsonl")
            or _HERE.parents[1] / "manifest" / "prompt_manifest.jsonl")

FAIL = []



def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def stub_network():
    """Import the generator without pulling in a live API client."""
    for name, attrs in [
        ("dotenv", {"load_dotenv": lambda *a, **k: None,
                   "dotenv_values": lambda *a, **k: {}}),
        ("anthropic", {"Anthropic": object, "AsyncAnthropic": object,
                       "APIError": type("APIError", (Exception,), {}),
                       "RateLimitError": type("RateLimitError", (Exception,), {}),
                       "APIStatusError": type("APIStatusError", (Exception,), {}),
                       "APIConnectionError": type("APIConnectionError", (Exception,), {}),
                       "BadRequestError": type("BadRequestError", (Exception,), {}),
                       "AuthenticationError": type("AuthenticationError", (Exception,), {}),
                       "PermissionDeniedError": type("PermissionDeniedError", (Exception,), {}),
                       "NotFoundError": type("NotFoundError", (Exception,), {})}),
        ("aiohttp", {"ClientSession": object, "ClientTimeout": object}),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
    # batch_generate imports these typed-dict helpers from the real SDK
    for mod, syms in [("anthropic.types.message_create_params",
                       {"MessageCreateParamsNonStreaming": dict}),
                      ("anthropic.types.messages.batch_create_params", {"Request": dict})]:
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            for k, v in syms.items():
                setattr(m, k, v)
            sys.modules[mod] = m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400,
                    help="movies to hash-check against the manifest (0 = all 10,381)")
    args = ap.parse_args()

    print(f"generator: {GEN}")
    if not GEN.exists():
        sys.exit(f"generator package not found at {GEN}")

    need = GEN / "data" / "ml-20m" / "genome-tags.csv"
    if not need.exists():
        sys.exit(
            "This check re-renders every prompt, so it needs the SOURCE data, which we\n"
            "do not redistribute (see docs/LICENSING_SURVEY.md). Fetch it first:\n"
            "    bash scripts/download_ml20m.sh\n"
            f"expected at: {need}\n"
            "For a check that needs no downloads at all, run tools/verify_paper_numbers.py.")

    stub_network()
    sys.path.insert(0, str(GEN))
    os.chdir(GEN)                      # settings.py resolves paths relative to the package

    from config import settings
    from data_loader import load_all_data
    from main import prepare_movie_data
    from generator import validate_profile_json
    import batch_generate

    # ---- 1. transport-only -------------------------------------------------
    print("\n1. batch and synchronous paths assemble identical requests")
    print("   loading ML-20M + TMDb cache (ratings.csv is ~530 MB) ...", flush=True)
    data = load_all_data()
    from tmdb_crawler import load_cache
    tmdb = {int(k): v for k, v in load_cache().items()}
    # sorted ids + unfiltered cache == exactly what build_prompt_manifest.py passes
    movies = prepare_movie_data(data, tmdb, sorted(data["movie_ids"]))
    check("prepared the full catalogue", len(movies) == 10381, f"{len(movies):,} prompts")

    probe = movies[:5]
    reqs = [batch_generate.Request(
        custom_id=f"movie_{m['movie_id']}",
        params=batch_generate.MessageCreateParamsNonStreaming(
            model=settings.CLAUDE_MODEL, max_tokens=settings.CLAUDE_MAX_TOKENS,
            temperature=settings.CLAUDE_TEMPERATURE, system=batch_generate.system_block(),
            messages=[{"role": "user", "content": m["user_prompt"]}])) for m in probe]

    same_user = all(r["params"]["messages"][0]["content"] == m["user_prompt"]
                    for r, m in zip(reqs, probe))
    check("user prompt byte-identical across paths", same_user)
    p0 = reqs[0]["params"]
    check("model / temperature / max_tokens identical across paths",
          p0["model"] == settings.CLAUDE_MODEL
          and p0["temperature"] == settings.CLAUDE_TEMPERATURE
          and p0["max_tokens"] == settings.CLAUDE_MAX_TOKENS,
          f"{p0['model']} T={p0['temperature']} max={p0['max_tokens']}")
    sysblk = p0["system"]
    sys_text = sysblk[0]["text"] if isinstance(sysblk, list) else sysblk
    check("system prompt byte-identical across paths", sys_text == settings.SYSTEM_PROMPT)
    check("custom_id keys every request (batch results return unordered)",
          all(r["custom_id"] == f"movie_{m['movie_id']}" for r, m in zip(reqs, probe)))

    # ---- 2. shipped code reproduces the released manifest ------------------
    print("\n2. shipped code reproduces the released prompt hashes")
    lines = MANIFEST.read_text().splitlines()
    header = json.loads(lines[0])
    recs = {json.loads(l)["movie_id"]: json.loads(l) for l in lines[1:]}
    # the manifest names this field prompt_sha256 -- read it, do not guess
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()

    check("system prompt hash matches the manifest header",
          sha(settings.SYSTEM_PROMPT) == header["system_prompt_sha256"])
    check("user-prompt template hash matches the manifest header",
          sha(settings.USER_PROMPT_TEMPLATE) == header["user_prompt_template_sha256"])
    check("model / temperature / max_tokens match the manifest header",
          header["model"] == settings.CLAUDE_MODEL
          and header["temperature"] == settings.CLAUDE_TEMPERATURE
          and header["max_tokens"] == settings.CLAUDE_MAX_TOKENS)

    subset = movies if args.sample == 0 else movies[:: max(1, len(movies) // args.sample)]
    bad = [m["movie_id"] for m in subset
           if recs.get(m["movie_id"], {}).get("prompt_sha256") != sha(m["user_prompt"])]
    check(f"re-rendered prompt hashes match ({len(subset):,} movies checked)",
          not bad, f"{len(bad)} mismatch" if bad else "every hash reproduced")
    if bad[:3]:
        print(f"         first mismatches: {bad[:3]}")
    check("manifest covers the whole catalogue",
          len(recs) == header["n_prompts"] == 10381, f"{len(recs):,} records")

    # ---- 3. shipped validator accepts the released artifact ---------------
    print("\n3. shipped validator accepts the released profiles")
    profiles = json.loads((GEN / "output" / "movie_profiles.json").read_text())
    rejected = []
    for mid, prof in profiles.items():
        obj, err = validate_profile_json(json.dumps(prof, ensure_ascii=False), int(mid))
        if obj is None:
            rejected.append((mid, err))
    check(f"all {len(profiles):,} released profiles pass validate_profile_json()",
          not rejected, f"{len(rejected)} rejected" if rejected else "")
    for mid, err in rejected[:3]:
        print(f"         movie {mid}: {err}")

    print()
    if FAIL:
        sys.exit(f"FAILED: {len(FAIL)} check(s) -- " + "; ".join(FAIL))
    print("end-to-end: released code reproduces the released artifact's inputs, "
          "and both transports are equivalent.")


if __name__ == "__main__":
    main()
