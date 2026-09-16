#!/usr/bin/env python3
"""Build the Steam item-metadata file the profile generator consumes.

Runs the same protocol as the recon -- positive signal, iterative k-core -- to
decide WHICH games are in the benchmark, then joins the surviving ids against
steam_games.json.gz and writes one record per game. Generating profiles for the
full 15,474-game catalogue would be wasted spend: only the k-core survivors are
ever scored, so only they get a profile.

    python3 prepare_steam_metadata.py                 # writes game_metadata.json
    python3 prepare_steam_metadata.py --limit 50      # small file for a smoke test

Output schema, one entry per game, keyed by Steam app id (string):
    title, genres[], tags[], specs[], developer, publisher

Deliberately EXCLUDED from the record: price, release_date and the store's
sentiment label. The system prompt forbids the model from mentioning them, and
sentiment in particular is a popularity signal -- feeding it in would recreate,
on a new dataset, exactly the leakage the paper spends a section controlling.
"""
import argparse
import ast
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recon_steam import GAMES_URL, REVIEWS_URL, load_reviews, verify_fast_path  # noqa: E402
from recon_amazon import fetch, k_core_filter  # noqa: E402

# Fields the generator is shown. See the docstring for what is withheld and why.
KEEP = ("title", "app_name", "genres", "tags", "specs", "developer", "publisher")


def log(m):
    print(m, flush=True)


def _find_up(start, *patterns):
    """First ancestor of `start` containing any of `patterns` (globs allowed).
    Names no sibling project, so the same code runs from a published clone."""
    for base in [start, *start.parents]:
        for pat in patterns:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-core", type=int, default=10)
    ap.add_argument("--min-hours", type=float, default=0.0)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/steam_recon"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: code/profile_generator/output_steam/game_metadata.json")
    ap.add_argument("--limit", type=int, default=0, help="first N games only, for a smoke test")
    a = ap.parse_args()

    out = a.out or (_find_up(Path(__file__).resolve().parent,
                            "output_steam", "*/code/profile_generator/output_steam")
                    or Path("output_steam")) / "game_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    rev = fetch(REVIEWS_URL, a.cache / "steam_reviews.json.gz")
    games = fetch(GAMES_URL, a.cache / "steam_games.json.gz")

    verify_fast_path(rev)
    df = load_reviews(rev, a.min_hours)
    df, _ = k_core_filter(df, a.k_core)
    keep = set(df["parent_asin"].unique())
    log(f"  {a.k_core}-core leaves {len(keep):,} games")

    recs, seen, thin = {}, set(), 0
    with gzip.open(games, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = ast.literal_eval(line)
            except Exception:
                continue
            i = str(r.get("id"))
            if i not in keep or i in seen:
                continue
            seen.add(i)
            rec = {
                "title": r.get("title") or r.get("app_name") or "",
                "genres": r.get("genres") or [],
                "tags": r.get("tags") or [],
                "specs": r.get("specs") or [],
                "developer": r.get("developer") or "",
                "publisher": r.get("publisher") or "",
            }
            # A game with no title and no tags gives the generator nothing at all.
            if not rec["title"] or not (rec["tags"] or rec["genres"]):
                thin += 1
                continue
            recs[i] = rec

    missing = sorted(keep - set(recs))
    log(f"  wrote metadata for {len(recs):,} games "
        f"({thin:,} dropped as too thin, {len(missing):,} with no metadata record)")
    if a.limit:
        recs = dict(list(recs.items())[:a.limit])
        log(f"  --limit {a.limit}: truncated to {len(recs)} games (SMOKE TEST, not a real run)")

    json.dump(recs, open(out, "w"), indent=1)
    if missing:
        mp = out.parent / "missing_metadata_ids.json"
        json.dump(missing, open(mp, "w"), indent=1)
        log(f"  unmatched ids -> {mp.name} (release this list; do not quietly shrink the denominator)")
    log(f"  -> {out}")

    tags = [len(v["tags"]) for v in recs.values()]
    if tags:
        tags.sort()
        log(f"  tags per game: median {tags[len(tags)//2]}, "
            f"min {tags[0]}, max {tags[-1]}")


if __name__ == "__main__":
    main()
