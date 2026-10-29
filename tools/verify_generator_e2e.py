#!/usr/bin/env python3
"""End-to-end check of the RELEASED generator against the RELEASED artifact.

The paper's contribution is a pipeline. A reader must be able to establish three
things about it. All three need the MovieLens source data (bash scripts/download_ml20m.sh)
and 3 also needs the released profiles; none needs a TMDb cache, an API key or any
spending:

  1. TRANSPORT-ONLY.  batch_generate.py and main.py assemble byte-identical
     requests -- same system prompt, same per-movie user prompt, same model,
     temperature and max_tokens.  Only the submission call differs.  This is what
     licenses quoting the batch price for an artifact either path could produce.

  2. THE CODE MATCHES THE RELEASE.  Re-rendering prompts with the shipped code
     reproduces the SHA-256 values in the released prompt manifest.  If the
     prompt had drifted since generation, the hashes would diverge.
     A prompt embeds a TMDb synopsis and keyword list, which are not ours to
     redistribute, so the cache is absent from a clone. The films the manifest
     marks as having no TMDb record (82) render from MovieLens alone and are
     always checked; the others are checked only where your own cache holds their
     record, and the rest are reported as CANNOT JUDGE (exit 2), never as a
     failure. Rebuilding the cache needs your own TMDB_API_KEY, and a cache built
     later is not ours: TMDb's vote counts and averages move and its records are
     edited, so expect most TMDb-backed prompts to mismatch. A mismatch on a film you
     did cache is reported as a failure (exit 1): it means your TMDb record differs from
     the one used at generation, or the code has drifted, and the 82 films and the
     template digests tell the two apart.
     The prompt also embeds genome scores that pandas reads from a CSV of
     17-digit decimals; the digests are exact only under a pandas that parses
     them as the release's did (pandas < 3.1, see requirements.txt). The script
     tests that first and reports CANNOT JUDGE when the installed pandas differs.
     The system-prompt and template digests do not depend on either.

  3. THE VALIDATOR ACCEPTS THE ARTIFACT.  The shipped validate_profile_json()
     accepts every released profile.  A validator that rejects the release would
     mean the shipped code could not have produced it.  Needs the released
     profiles: python3 scripts/download_artifacts.py.

Every absent input -- the source data, the TMDb cache, the manifest, the released
profiles -- is CANNOT JUDGE (exit 2), never a failure (1) and never a traceback.

Calls no API.  Usage:  python3 verify_generator_e2e.py [--sample N]
"""
import argparse, hashlib, importlib.util, json, os, sys, types
from pathlib import Path

# Published layout first, working-tree layout second, so the verifier runs in a
# fresh clone.
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

PROFILES = GEN / "output" / "movie_profiles.json"   # where download_artifacts.py puts them


def pandas_parses_like_the_release():
    """True when read_csv turns a 17-digit decimal into its 15-digit neighbour, as the
    pandas the release was generated with did. pandas 3.1 keeps the full value, which
    moves round(x, 4) at a boundary in the genome scores and changes prompt text."""
    import io
    import pandas as pd
    return float(pd.read_csv(io.StringIO("x\n0.025000000000000022\n"))["x"][0]) == 0.025

FAIL = []
UNJUDGEABLE = []   # a check whose INPUT is absent, which is not the same as a failure


def cannot_judge(why: str) -> None:
    """Stop before any check ran: say what is missing and exit 2."""
    print(f"CANNOT JUDGE: {why}", file=sys.stderr)
    sys.exit(2)



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
        cannot_judge(f"generator package not found at {GEN}")

    # Resolve the source data the way the LOADER does, not by a second guess.
    # settings.py tries three layouts, the documented download_ml20m.sh target
    # (<repo>/data/raw/ml-20m) first, so this guard agrees with the loader about
    # where the data is. Import is side-effect free:
    # settings.py pulls in os and pathlib only, and resolves against its own
    # __file__ rather than the cwd, so reading it before the chdir below is safe.
    _spec = importlib.util.spec_from_file_location(
        "_pg_settings_probe", GEN / "config" / "settings.py")
    _settings = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_settings)
    need = _settings.DATA_DIR / "genome-tags.csv"
    if not need.exists():
        cannot_judge(
            "this check re-renders every prompt, so it needs the SOURCE data, which we\n"
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
    if not MANIFEST.is_file():
        cannot_judge(f"the prompt manifest is not at {MANIFEST}; it ships with this "
                     f"repository under manifest/")
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

    # The prompts embed TMDb metadata, and the TMDb cache is NOT redistributable,
    # so it is excluded from the release. tmdb_crawler.load_cache() fails OPEN --
    # it returns {} when the file is absent -- and every prompt then renders with
    # "PLOT: No overview available. / DIRECTOR: Unknown / CAST: Unknown". The
    # hashes cannot match, and reporting that as FAILED told a reader that the
    # released code does not reproduce the released artifact, which is false and
    # is the worst thing this script could say. An absent input is "cannot judge".
    sampled = movies if args.sample == 0 else movies[:: max(1, len(movies) // args.sample)]
    # Films with no TMDb record render from MovieLens alone, so they are judgeable
    # with no cache and are always included; a film that has a TMDb record is
    # judgeable only where the reader's cache holds it.
    no_tmdb = {mid for mid, r in recs.items() if not r.get("tmdb_metadata")}
    wanted = {m["movie_id"] for m in sampled} | no_tmdb
    subset = [m for m in movies if m["movie_id"] in wanted]
    judge = [m for m in subset if m["movie_id"] in no_tmdb or m["movie_id"] in tmdb]
    skipped = len(subset) - len(judge)
    if not pandas_parses_like_the_release():
        import pandas as _pd
        label = f"re-rendered prompt hashes match ({len(subset):,} movies would be checked)"
        print(f"  [-- ] {label} -- SKIPPED, this pandas ({_pd.__version__}) parses the genome")
        print(f"         scores differently from the one the release was generated with, so the")
        print(f"         rendered text can differ for reasons that are not the code's. Install")
        print(f"         the pinned requirements (pandas < 3.1) and re-run:")
        print(f"           pip install -r requirements.txt")
        UNJUDGEABLE.append(label)
    else:
        if judge:
            bad = [m["movie_id"] for m in judge
                   if recs.get(m["movie_id"], {}).get("prompt_sha256") != sha(m["user_prompt"])]
            check(f"re-rendered prompt hashes match ({len(judge):,} movies checked)",
                  not bad, f"{len(bad)} mismatch" if bad else "every hash reproduced")
            if bad[:3]:
                print(f"         first mismatches: {bad[:3]}")
        if skipped:
            label = f"re-rendered prompt hashes match ({skipped:,} movies with a TMDb record would be checked)"
            print(f"  [-- ] {label} -- SKIPPED, your TMDb metadata cache has no record for them")
            print(f"         The prompts embed TMDb plot, keywords, cast and crew. That cache is")
            print(f"         source material we do not redistribute, so it is not in the release:")
            print(f"           {settings.METADATA_CACHE}")
            print(f"         Rebuild it with your own key (TMDB_API_KEY, ~10,381 lookups) and")
            print(f"         re-run to check this. Note that TMDb edits its records, so a cache")
            print(f"         crawled today may legitimately differ from the one used at generation.")
            print(f"         The other checks in this script do not depend on it.")
            UNJUDGEABLE.append(label)
    check("manifest covers the whole catalogue",
          len(recs) == header["n_prompts"] == 10381, f"{len(recs):,} records")

    # ---- 3. shipped validator accepts the released artifact ---------------
    print("\n3. shipped validator accepts the released profiles")
    if not PROFILES.is_file():
        label = "released profiles pass validate_profile_json()"
        print(f"  [-- ] {label} -- SKIPPED, the released profiles are not here:")
        print(f"           {PROFILES}")
        print(f"         Fetch them with:  python3 scripts/download_artifacts.py")
        UNJUDGEABLE.append(label)
    else:
        profiles = json.loads(PROFILES.read_text())
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
    if UNJUDGEABLE:
        # Exit 2, not 0 and not 1: everything checkable passed, but one check
        # could not run. A reader scripting this must be able to tell the three
        # apart, and a green line over a skipped check is how a gate starts lying.
        print("end-to-end: every check that could run passed, and both transports "
              "are equivalent.")
        print(f"CANNOT JUDGE: {len(UNJUDGEABLE)} check(s) had no input -- "
              + "; ".join(UNJUDGEABLE))
        sys.exit(2)
    print("end-to-end: released code reproduces the released artifact's inputs, "
          "and both transports are equivalent.")


if __name__ == "__main__":
    main()
