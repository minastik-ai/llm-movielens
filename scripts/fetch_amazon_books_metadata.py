#!/usr/bin/env python3
"""Fetch Amazon-Books metadata via OpenLibrary API.

For each ASIN (= ISBN-10) in the RLMRec Amazon-Books subset, query OpenLibrary
for title / authors / subjects / publish_date, and combine with RLMRec's
pre-generated item profile as a "description" surrogate (since we don't have
the original Amazon Books metadata file locally).

Output: code/profile_generator/output_amazon/book_metadata.json
   keyed by RLMRec item id (0..9331), one entry per book with:
     - asin, title, authors, subjects, publish_date,
       rlmrec_profile (text), word_count, etc.
   Items not found in OpenLibrary still get an entry with title=null and the
   RLMRec profile as a fallback content cue.

Usage:  python scripts/fetch_amazon_books_metadata.py
        python scripts/fetch_amazon_books_metadata.py --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import requests
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
RLMREC = code("benchmark", "external", "RLMRec", "data")
OUT_DIR = code("profile_generator", "output_amazon")
OUT_FILE = OUT_DIR / "book_metadata.json"
CACHE_FILE = OUT_DIR / "openlibrary_cache.json"

BATCH_SIZE = 50
SLEEP_SEC = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_metadata")


def load_rlmrec_profiles() -> dict:
    """Return {iid: profile_text} from RLMRec's itm_prf.pkl."""
    with open(RLMREC / "amazon" / "itm_prf.pkl", "rb") as f:
        d = pickle.load(f)
    return {int(k): v.get("profile", "") for k, v in d.items()}


def load_iid_to_asin() -> dict:
    """Return {iid: asin} from amazon_item.json (one JSON object per line)."""
    out = {}
    with open(RLMREC / "mapper" / "amazon_item.json") as f:
        for line in f:
            obj = json.loads(line)
            out[int(obj["iid"])] = obj["asin"]
    return out


def query_openlibrary_batch(isbns: list[str], session: requests.Session) -> dict:
    """Batch-fetch metadata for up to ~50 ISBNs in one call."""
    if not isbns:
        return {}
    keys = ",".join(f"ISBN:{i}" for i in isbns)
    url = f"https://openlibrary.org/api/books?bibkeys={keys}&format=json&jscmd=data"
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Batch lookup failed for {len(isbns)} ISBNs: {e}")
        return {}


def extract_fields(ol_entry: dict) -> dict:
    """Pull the fields we'll feed to the prompt template."""
    if not ol_entry:
        return {}
    return {
        "title": ol_entry.get("title"),
        "subtitle": ol_entry.get("subtitle"),
        "authors": [a["name"] for a in ol_entry.get("authors", []) if "name" in a],
        "subjects": [s["name"] for s in ol_entry.get("subjects", []) if "name" in s][:10],
        "publish_date": ol_entry.get("publish_date"),
        "publishers": [p["name"] for p in ol_entry.get("publishers", []) if "name" in p][:2],
        "number_of_pages": ol_entry.get("number_of_pages"),
        "openlibrary_url": ol_entry.get("url"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N items (smoke test).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from cached partial results if present.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    iid_to_asin = load_iid_to_asin()
    rlmrec_profiles = load_rlmrec_profiles()
    n_total = len(iid_to_asin)
    logger.info(f"Loaded {n_total} (iid, asin) pairs and {len(rlmrec_profiles)} RLMRec profiles")

    # Optional resume cache
    cache = {}
    if args.resume and CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        logger.info(f"Resuming from cache: {len(cache)} ASINs already fetched")

    asins_to_fetch = [a for a in iid_to_asin.values() if a not in cache]
    if args.limit:
        asins_to_fetch = asins_to_fetch[: args.limit]
    logger.info(f"Fetching metadata for {len(asins_to_fetch)} new ASINs from OpenLibrary")

    session = requests.Session()
    t0 = time.time()
    n_hits = 0
    for i in range(0, len(asins_to_fetch), BATCH_SIZE):
        batch = asins_to_fetch[i:i + BATCH_SIZE]
        result = query_openlibrary_batch(batch, session)
        for isbn in batch:
            key = f"ISBN:{isbn}"
            cache[isbn] = result.get(key, {})
            if cache[isbn]:
                n_hits += 1
        # Throttle (OpenLibrary recommends ~100 req/s ceiling; we go gentler)
        time.sleep(SLEEP_SEC)
        # Periodic checkpoint every 1000 items
        if (i + BATCH_SIZE) % 1000 < BATCH_SIZE:
            CACHE_FILE.write_text(json.dumps(cache))
            elapsed = time.time() - t0
            done = i + BATCH_SIZE
            eta_min = (len(asins_to_fetch) - done) * elapsed / max(done, 1) / 60
            logger.info(f"  {done}/{len(asins_to_fetch)} fetched, "
                        f"hits so far: {n_hits} ({100*n_hits/max(done,1):.1f}%), "
                        f"elapsed {elapsed:.0f}s, eta ~{eta_min:.1f} min")
    CACHE_FILE.write_text(json.dumps(cache))

    # Build the final book_metadata.json keyed by iid
    out = {}
    for iid, asin in iid_to_asin.items():
        ol = cache.get(asin, {})
        fields = extract_fields(ol)
        prof = rlmrec_profiles.get(iid, "")
        out[str(iid)] = {
            "iid": iid,
            "asin": asin,
            "title": fields.get("title"),
            "subtitle": fields.get("subtitle"),
            "authors": fields.get("authors", []),
            "subjects": fields.get("subjects", []),
            "publish_date": fields.get("publish_date"),
            "publishers": fields.get("publishers", []),
            "number_of_pages": fields.get("number_of_pages"),
            "openlibrary_url": fields.get("openlibrary_url"),
            "rlmrec_profile": prof,  # used as a "description" surrogate
        }

    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    n_with_title = sum(1 for v in out.values() if v["title"])
    logger.info(f"Wrote {OUT_FILE.relative_to(REPO)}: {len(out)} books, "
                f"{n_with_title} ({100*n_with_title/len(out):.1f}%) with OpenLibrary title.")


if __name__ == "__main__":
    main()
