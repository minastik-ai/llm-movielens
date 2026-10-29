#!/usr/bin/env python3
"""How much popularity is linearly decodable from each content feature set?

The probe behind the paper's section-6 R^2 table. No recommender is involved: a ridge
regression predicts each aggregate the generation prompt carried from each feature set,
scored by out-of-fold R^2 (5-fold CV, alpha chosen inside each fold).

Feature sets (the 10,381 genome-covered ML-20M movies, in movie_id_index.json order):
  M4 profile embedding (1024-d), with and without aggregates in the prompt
  M2 genome-PCA (128-d), which never saw popularity
  M3 title embedding (1024-d), same encoder as M4
Targets: log1p(ML rating count), ML mean rating, and -- given a TMDb metadata cache --
log1p(TMDb vote count) and TMDb vote average.
Controls: every feature set reduced to 128 dimensions by PCA, so M4's width cannot
inflate its R^2; and a permuted-target floor, which must be about 0.

Inputs:
  --emb-dir        the released embeddings (scripts/download_artifacts.py puts them here)
  --noagg-emb-dir  embeddings of profiles regenerated with PROFILE_PROMPT=no-aggregates
                   (src/profile_generator) and encoded by src/embedding_generator; optional
  --ratings        MovieLens 20M ratings.csv (scripts/download_ml20m.sh)
  --tmdb-cache     src/profile_generator/cache/tmdb_metadata.json from a TMDb crawl; optional

Usage:
    python3 scripts/probe_popularity.py --noagg-emb-dir <dir> --tmdb-cache <file> --out probe.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO, code  # noqa: E402

warnings.filterwarnings("ignore")
SEED, FOLDS = 42, 5
ALPHAS = np.logspace(-1, 5, 25)


def probe(X, y, matched_dim=None, permute=False):
    m = np.isfinite(y)
    X, y = X[m], y[m]
    if permute:
        y = np.random.default_rng(SEED).permutation(y)
    steps = [StandardScaler()]
    if matched_dim is not None and X.shape[1] > matched_dim:
        steps.append(PCA(n_components=matched_dim, random_state=SEED))
    steps.append(RidgeCV(alphas=ALPHAS))
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    s = cross_val_score(make_pipeline(*steps), X, y, cv=cv, scoring="r2", n_jobs=-1)
    return s.mean(), s.std()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emb-dir", type=Path,
                    default=code("embedding_generator", "output", "bge-large-en-v1.5"))
    ap.add_argument("--noagg-emb-dir", type=Path, default=None)
    ap.add_argument("--ratings", type=Path, default=REPO / "data" / "raw" / "ml-20m" / "ratings.csv")
    ap.add_argument("--tmdb-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    needed = [(a.emb_dir / f, "the ML-20M embeddings (python3 scripts/download_artifacts.py; the genome and "
                              "title baselines come from Stage 2, src/embedding_generator/main.py)")
              for f in ("movie_id_index.json", "profile_embeddings.npy",
                        "genome_embeddings.npy", "bert_title_embeddings.npy")]
    needed.append((a.ratings, "the MovieLens 20M ratings (bash scripts/download_ml20m.sh)"))
    if a.noagg_emb_dir:
        needed += [(a.noagg_emb_dir / f, "the embeddings of the no-aggregate profiles (encode them with Stage 2, "
                                         "src/embedding_generator/main.py --profiles ... --output-dir <dir>)")
                   for f in ("movie_id_index.json", "profile_embeddings.npy")]
    if a.tmdb_cache:
        needed.append((a.tmdb_cache, "your own TMDb metadata cache (not redistributed)"))
    missing = [(p, how) for p, how in needed if not p.exists()]
    if missing:
        raise SystemExit("missing input:\n" + "\n".join(f"  {p}\n      needs {how}" for p, how in missing))

    movie_ids = [int(m) for m in json.loads((a.emb_dir / "movie_id_index.json").read_text())]
    n = len(movie_ids)
    feats = {"M4 profile (agg-fed)": np.load(a.emb_dir / "profile_embeddings.npy"),
             "M2 genome-PCA": np.load(a.emb_dir / "genome_embeddings.npy"),
             "M3 title-emb": np.load(a.emb_dir / "bert_title_embeddings.npy")}
    if a.noagg_emb_dir:
        if (a.noagg_emb_dir / "movie_id_index.json").read_text() != \
                (a.emb_dir / "movie_id_index.json").read_text():
            raise SystemExit("the no-aggregate embeddings use a different row order")
        feats["M4 profile (no-agg)"] = np.load(a.noagg_emb_dir / "profile_embeddings.npy")
    for k, v in feats.items():
        assert v.shape[0] == n, f"{k}: {v.shape[0]} rows, expected {n}"
        print(f"  {k:24s} {v.shape}")

    r = pd.read_csv(a.ratings, usecols=["movieId", "rating"],
                    dtype={"movieId": "int32", "rating": "float32"})
    g = r.groupby("movieId")["rating"].agg(["mean", "count"])
    targets = {"log1p(ml_rating_count)": np.log1p(g["count"].reindex(movie_ids).to_numpy(float)),
               "ml_avg_rating": g["mean"].reindex(movie_ids).to_numpy(float)}
    if a.tmdb_cache:
        tmdb = json.loads(a.tmdb_cache.read_text())

        def field(name):
            out = np.full(n, np.nan)
            for i, mid in enumerate(movie_ids):
                rec = tmdb.get(str(mid))
                if isinstance(rec, dict) and rec.get(name) is not None:
                    try:
                        out[i] = float(rec[name])
                    except (TypeError, ValueError):
                        pass
            return out
        targets["log1p(tmdb_vote_count)"] = np.log1p(field("vote_count"))
        targets["tmdb_vote_average"] = field("vote_average")

    rows = []
    for mode, dim in [("native dim", None), ("matched 128-d", 128)]:
        print(f"\nout-of-fold R^2 ({mode}, {FOLDS}-fold)")
        for tname, y in targets.items():
            cells = []
            for fname, X in feats.items():
                mu, sd = probe(X, y, matched_dim=dim)
                cells.append(f"{mu:.3f}+-{sd:.3f}")
                rows.append({"mode": mode, "target": tname, "features": fname,
                             "r2_mean": round(float(mu), 4), "r2_std": round(float(sd), 4)})
            print(f"  {tname:26s}" + "".join(f"{c:>18s}" for c in cells))
    print("\npermuted-target floor (must be about 0):")
    for fname, X in feats.items():
        mu, sd = probe(X, targets["log1p(ml_rating_count)"], permute=True)
        print(f"  {fname:24s} R^2 = {mu:+.4f} +- {sd:.4f}")
        rows.append({"mode": "permuted floor", "target": "log1p(ml_rating_count)",
                     "features": fname, "r2_mean": round(float(mu), 4), "r2_std": round(float(sd), 4)})

    if a.out:
        a.out.write_text(json.dumps({
            "note": ("Popularity-decodability probe (section 6), both arms: the aggregate-fed "
                     "profiles and the no-aggregate regeneration."),
            "n_items": n, "folds": FOLDS, "seed": SEED, "results": rows}, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
