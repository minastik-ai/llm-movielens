#!/usr/bin/env python3
"""Recompute the identity-handle ablation's retrieval figures from its two arms.

    python3 tools/identity_retrieval.py FULL.json MASKED.json [--encoder NAME]

FULL and MASKED are profile files keyed by item id, in the release's format
({id: {"profile": ...}, ...}): the same items generated once normally and once with
their structured identity withheld. The script pairs the items present in both,
encodes both arms with the paper's encoder, and ranks each masked profile against
the full profiles of the paired items. It then prints the figures that
src/identity_ablation.json records: recall@1, recall@10, median rank, MRR, the
median paired cosine, and the background cosine, which is the mean cosine between
a masked profile and the OTHER items' full profiles.

The record's figures came from a one-off computation that no script kept, so a
reader could not recompute them even after regenerating both arms. This script is
that computation. On the development arms it reproduces the record: MovieLens 1,000
pairs, Steam 998.

Needs sentence-transformers. Exit 0; 2 when an input is missing or fewer than ten
items pair, which is a missing input and not a finding.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    if isinstance(d, list):
        d = {str(r.get("movieId") or r.get("itemId") or r.get("app_id")): r for r in d}
    return {str(k): v["profile"] for k, v in d.items() if isinstance(v, dict) and v.get("profile")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("full", type=Path)
    ap.add_argument("masked", type=Path)
    ap.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--json", action="store_true", help="print the figures as JSON")
    a = ap.parse_args(argv)
    for f in (a.full, a.masked):
        if not f.is_file():
            print(f"CANNOT JUDGE: {f} is not here", file=sys.stderr)
            return 2
    full, masked = load(a.full), load(a.masked)
    ids = sorted(set(full) & set(masked), key=lambda k: (len(k), k))
    if len(ids) < 10:
        print(f"CANNOT JUDGE: only {len(ids)} items appear in both arms", file=sys.stderr)
        return 2
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(a.encoder)
    F = model.encode([full[i] for i in ids], normalize_embeddings=True, batch_size=32)
    M = model.encode([masked[i] for i in ids], normalize_embeddings=True, batch_size=32)
    S = M @ F.T                                   # masked i against every full j
    own = np.diag(S)
    ranks = 1 + (S > own[:, None]).sum(axis=1)    # rank of i's own full profile
    n = len(ids)
    off = (S.sum() - own.sum()) / (n * n - n)
    out = {"paired_items": n,
           "recall_at_1": round(100 * float((ranks == 1).mean()), 1),
           "recall_at_10": round(100 * float((ranks <= 10).mean()), 1),
           "median_rank": int(np.median(ranks)),
           "mrr": round(float((1 / ranks).mean()), 3),
           "paired_cosine_median": round(float(np.median(own)), 3),
           "background_cosine": round(float(off), 3)}
    if a.json:
        print(json.dumps(out))
    else:
        for k, v in out.items():
            print(f"  {k:<22} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
