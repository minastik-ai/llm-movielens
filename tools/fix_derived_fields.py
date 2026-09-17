#!/usr/bin/env python3
"""Repair derived fields in the released profile files.

Two fields in a profile record describe the record rather than adding to it, so
both must agree with it, and in three of the four released files they do not:

  word_count  the model reports its own length and overstates it (median +4 to
              +8 words). The PRIMARY ML-20M pipeline overrides it with
              len(profile.split()) at generation time (generator.py
              ~line 130); the book runner and the GPT-4o-mini run do not.
  movieId /   should equal the JSON key. GPT-4o-mini echoed the few-shot
  itemId      example's id (99997) into all 10,381 of its records.

Neither defect touches a paper claim -- validation and the human audit both read
the profile text, never these fields -- but a released artifact whose
self-describing fields contradict its own content is what a reviewer who
downloads the data finds. Both repairs are lossless: the truth is already in the
record (the text, and the key).

SAFETY. The script first proves its serializer is faithful by re-encoding each
file UNCHANGED and requiring byte-identity with the original. Only then does it
edit. Afterwards it re-parses and asserts that no field other than the two named
above differs. Run with --check to audit without writing.

    python3 fix_derived_fields.py --check      # report only
    python3 fix_derived_fields.py              # repair, then re-verify
"""
import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release" / "hf"

# (path relative to the release root, id field name)
TARGETS = [
    ("profiles/amazon_books_2018/claude-haiku-4-5/book_profiles.json", "itemId"),
    ("profiles/ml20m/gpt-4o-mini/movie_profiles.json", "movieId"),
    ("profiles/ml20m/claude-haiku-4-5/movie_profiles.json", "movieId"),
    ("human_eval/gpt4o_mini_profiles.json", "movieId"),
]
FIXABLE = ("word_count", "__id__")


def log(m):
    print(m, flush=True)


def load(p):
    return json.loads(p.read_text(), object_pairs_hook=OrderedDict)


def dumps(d):
    """Match the files' existing encoding: 2-space indent, ASCII, no trailing space."""
    return json.dumps(d, indent=2, ensure_ascii=True) + "\n"


def faithful(p, d):
    """Prove the serializer reproduces the file byte-for-byte before we trust it."""
    orig = p.read_bytes()
    for text in (dumps(d), json.dumps(d, indent=2, ensure_ascii=True)):
        if text.encode() == orig:
            return text[len(text):] or text  # marker: this encoding is the faithful one
    return None


def audit(d, idf):
    wc = [k for k, v in d.items()
          if isinstance(v.get("word_count"), int) and isinstance(v.get("profile"), str)
          and v["word_count"] != len(v["profile"].split())]
    ids = [k for k, v in d.items() if idf in v and str(v[idf]) != str(k)]
    return wc, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    total_wc = total_id = 0
    for rel, idf in TARGETS:
        p = RELEASE / rel
        if not p.exists():
            log(f"  MISSING {rel}")
            continue
        d = load(p)
        wc, ids = audit(d, idf)
        total_wc += len(wc)
        total_id += len(ids)
        log(f"  {rel}")
        log(f"      {len(d):>6,} records | word_count wrong {len(wc):>6,} | {idf} wrong {len(ids):>6,}")
        if a.check or not (wc or ids):
            continue

        # 1. prove the serializer is faithful on THIS file before editing it
        enc = None
        orig_bytes = p.read_bytes()
        for cand in (json.dumps(d, indent=2, ensure_ascii=True),
                     json.dumps(d, indent=2, ensure_ascii=True) + "\n"):
            if cand.encode() == orig_bytes:
                enc = cand
                break
        if enc is None:
            log("      ABORT: cannot reproduce this file byte-for-byte; refusing to rewrite")
            sys.exit(1)
        trailing = "\n" if enc.endswith("\n") else ""

        before = {k: dict(v) for k, v in d.items()}
        for k in wc:
            d[k]["word_count"] = len(d[k]["profile"].split())
        for k in ids:
            d[k][idf] = int(k) if str(k).lstrip("-").isdigit() else k

        # 2. assert nothing but the two named fields moved
        for k, v in d.items():
            b = before[k]
            if list(v.keys()) != list(b.keys()):
                sys.exit(f"      ABORT: key order changed at {k}")
            for f in v:
                if f in ("word_count", idf):
                    continue
                if v[f] != b[f]:
                    sys.exit(f"      ABORT: unrelated field {f!r} changed at {k}")

        out = json.dumps(d, indent=2, ensure_ascii=True) + trailing
        p.write_text(out)
        wc2, ids2 = audit(load(p), idf)
        log(f"      repaired -> word_count wrong {len(wc2)}, {idf} wrong {len(ids2)}, "
            f"sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}…")
        if wc2 or ids2:
            sys.exit("      ABORT: repair did not converge")

    log(f"\n  totals: {total_wc:,} word_count and {total_id:,} id fields "
        f"{'would be' if a.check else ''} repaired")
    if not a.check and (total_wc or total_id):
        log("  NEXT: regenerate SHA256SUMS in release/hf (the file hashes have changed)")


if __name__ == "__main__":
    main()
