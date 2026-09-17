#!/usr/bin/env python3
"""Rebuild the LLM-MovieLens benchmark splits from your own copy of MovieLens 20M.

WHY YOU HAVE TO RUN THIS. We release the profiles, embeddings and evaluation
harness we created, but not MovieLens interaction data: GroupLens permits use
and forbids redistribution without separate permission. So the splits are not
shipped — they are rebuilt here, deterministically, from the ratings file you
download yourself. This is the same arrangement GroupLens use in `links.csv`
(ship an identifier, defer to the provider's terms) and the one TensorFlow
Datasets uses for MovieLens (fetch at runtime, host nothing).

The rebuild is byte-exact. Every output file is checked against a SHA-256
recorded when the paper's results were produced, so you can confirm you are
evaluating on precisely the data we evaluated on — a stronger guarantee than
downloading a CSV from us and trusting it.

IT NEEDS THE PROFILES TOO. The item universe is read off the released profiles,
so this runs AFTER `scripts/download_artifacts.py` (or after you generate your own),
not before it. Both defaults below resolve against this repository, not against your
working directory, so the splits land where the benchmark reads them.

    1. Get ML-20M from https://grouplens.org/datasets/movielens/20m/
    2. python3 scripts/download_artifacts.py          # or generate your own profiles
    3. python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m

Requires: pandas. No network access, no API keys, no GPU. Runs in ~2 minutes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# --- Where things are. Found by SHAPE, never by a sibling path: a literal path to
# --- the author's machine is dead code in a clone, and the defaults here are the
# --- ones a reader gets when they follow the README without thinking about paths.
HERE = Path(__file__).resolve()
_LAYOUTS = ("src", "code")          # released tree, development tree


def _repo_root() -> Path:
    for base in HERE.parents:
        if any((base / lay / "benchmark").is_dir() for lay in _LAYOUTS):
            return base
    return HERE.parents[1]


ROOT = _repo_root()
_LAY = next((lay for lay in _LAYOUTS if (ROOT / lay / "benchmark").is_dir()), "src")

# The benchmark reads its splits from inside its own package (benchmark/config.py
# resolves DATA_DIR against that directory, not against the repository root). The
# default used to be `data/processed` relative to the CURRENT DIRECTORY, so anyone
# who ran the README's command from the repository root wrote the splits to a
# directory nothing reads, and reproduce_all.sh then reported them missing.
DEFAULT_OUT = ROOT / _LAY / "benchmark" / "data" / "processed"

# The profiles define the item universe. The default used to be the path they have
# INSIDE THE DATA REPOSITORY, which exists in neither a fresh clone nor after
# download_artifacts.py -- so the second command in the README died on a bare
# FileNotFoundError with nothing to act on.
PROFILE_CANDIDATES = [
    ROOT / _LAY / "profile_generator" / "output" / "movie_profiles.json",
    Path("profiles/ml20m/claude-haiku-4-5/movie_profiles.json"),
]


def resolve_profiles(given):
    if given is not None:
        return given
    for c in PROFILE_CANDIDATES:
        if c.exists():
            return c
    sys.exit(
        "ERROR: no profiles file found, and the item universe is read from it.\n"
        "Looked in:\n" + "".join(f"  {c}\n" for c in PROFILE_CANDIDATES) +
        "\nFetch the ones we released (188 MB, no API key):\n"
        "  python3 scripts/download_artifacts.py\n"
        "or point at your own with --profiles.")


# --- Protocol constants. These define the benchmark; do not change them if you
# --- intend your numbers to be comparable with the paper's. -----------------
POSITIVE_THRESHOLD = 3.5          # rating >= this becomes a positive interaction
K_CORE = 10                       # iterative k-core over users AND items
TRAIN_END_TS = 1388534400         # 2014-01-01 00:00:00 UTC
VAL_END_TS = 1404172800           # 2014-07-01 00:00:00 UTC

# --- Expected output, recorded from the run that produced the paper. --------
EXPECTED_SHA256 = {
    "train.csv":     "05c394b54d9f7f34d28d8d5d16df3e41e2ee07928c24bd09fa47105c1dfc7b62",
    "val.csv":       "63bb7abbe8febb0217c7ff4c0192f5c3ca4a10a77774540cf11bd3edbf0d74a4",
    "test.csv":      "e687999e90fbadd6e6b2bd1c4032ae8839929667d5c9bc342d4467f5ef605fc6",
    "user_map.json": "d7ece2478bdf3666c6834683284d908c96d40600925f0c5964774c4fbcb6bacb",
    "item_map.json": "c48c9e98d77a230967b3b43938c6fa8be805375a73fdcf37c7e526e06df14ad1",
    "stats.json":    "cf7645060224d3626e6d2222dd17ca78e89e7e7b3412a3a9db5ed6a66eb40528",
}
EXPECTED_STATS = {"n_users": 127371, "n_items": 9906, "n_train": 11499778,
                  "n_val": 49668, "n_test": 67466}


def log(msg):
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def item_universe(profiles_path: Path) -> set:
    """The 10,381 genome-covered films, taken from the profiles we release.

    Deriving the item set from our own artifact rather than shipping a separate
    ID list keeps the two in step by construction: the benchmark covers exactly
    the items we generated a profile for.
    """
    with open(profiles_path) as f:
        profiles = json.load(f)
    ids = {int(k) for k in profiles}
    log(f"  item universe from released profiles: {len(ids):,} films")
    return ids


def k_core_filter(df: pd.DataFrame, k: int = K_CORE) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions.

    Iterative because removing a sparse user can push an item below threshold
    and vice versa; a single pass over each would leave the condition unmet.
    """
    it = 0
    while True:
        it += 1
        n0 = len(df)
        uc = df["userId"].value_counts()
        df = df[df["userId"].isin(uc[uc >= k].index)]
        ic = df["movieId"].value_counts()
        df = df[df["movieId"].isin(ic[ic >= k].index)]
        if len(df) == n0:
            break
    log(f"  {k}-core converged after {it} iterations: {len(df):,} interactions")
    return df


def remap_ids(train, val, test):
    """Contiguous 0-indexed ids, built from TRAIN only.

    Anything appearing first in val/test is dropped rather than assigned an id:
    a model cannot be asked about a user or item it never saw during training.
    """
    users = sorted(train["userId"].unique())
    items = sorted(train["movieId"].unique())
    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(m): i for i, m in enumerate(items)}

    def remap(df):
        df = df.copy()
        df["userId"] = df["userId"].map(user_map)
        df["movieId"] = df["movieId"].map(item_map)
        df = df.dropna(subset=["userId", "movieId"])
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        return df

    tr, va, te = remap(train), remap(val), remap(test)
    seen = set(tr["userId"].unique())
    va = va[va["userId"].isin(seen)]
    te = te[te["userId"].isin(seen)]
    return tr, va, te, user_map, item_map


def build(ml20m_dir: Path, profiles: Path, out: Path):
    ratings = ml20m_dir / "ratings.csv"
    if not ratings.exists():
        sys.exit(f"ERROR: {ratings} not found.\n"
                 f"Download ML-20M from https://grouplens.org/datasets/movielens/20m/ "
                 f"and pass its directory with --ml20m-dir.")

    keep = item_universe(profiles)

    log("  loading ratings.csv (~530 MB) ...")
    df = pd.read_csv(ratings, usecols=["userId", "movieId", "rating", "timestamp"])
    log(f"  ratings: {len(df):,}")

    df = df[df["movieId"].isin(keep)]
    log(f"  after item filter: {len(df):,}")

    df = df[df["rating"] >= POSITIVE_THRESHOLD].drop(columns=["rating"])
    log(f"  after implicit threshold (>= {POSITIVE_THRESHOLD}): {len(df):,}")

    df = k_core_filter(df)

    train = df[df["timestamp"] < TRAIN_END_TS]
    val = df[(df["timestamp"] >= TRAIN_END_TS) & (df["timestamp"] < VAL_END_TS)]
    test = df[df["timestamp"] >= VAL_END_TS]
    log(f"  temporal split: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    train, val, test, user_map, item_map = remap_ids(train, val, test)
    log(f"  after remap: {len(user_map):,} users, {len(item_map):,} items, "
        f"train {len(train):,}")

    out.mkdir(parents=True, exist_ok=True)
    for name, d in [("train.csv", train), ("val.csv", val), ("test.csv", test)]:
        d.to_csv(out / name, index=False)
    json.dump(user_map, open(out / "user_map.json", "w"))
    json.dump(item_map, open(out / "item_map.json", "w"))
    counts = train["movieId"].value_counts()
    n_items = len(item_map)
    n_cold = int(((counts > 0) & (counts < 10)).sum())
    n_medium = int(((counts >= 10) & (counts < 50)).sum())
    n_warm = int((counts >= 50).sum())
    stats = {
        "n_users": len(user_map), "n_items": n_items,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "density": len(train) / (len(user_map) * n_items),
        "n_zero_shot_items": n_items - (n_cold + n_medium + n_warm),
        "n_cold_items": n_cold, "n_medium_items": n_medium, "n_warm_items": n_warm,
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    return stats


def verify(out: Path, stats: dict) -> bool:
    log("\n  verifying against the paper's recorded output")
    ok = True
    for name, want in EXPECTED_SHA256.items():
        got = sha256(out / name)
        mark = "OK  " if got == want else "FAIL"
        if got != want:
            ok = False
        log(f"    [{mark}] {name:<15} {got[:16]}…")
    for k, want in EXPECTED_STATS.items():
        got = stats[k]
        if got != want:
            ok = False
            log(f"    [FAIL] {k}: got {got:,}, expected {want:,}")
    if ok:
        log("\n  ✓ byte-exact match — you are evaluating on the paper's data")
    else:
        log("\n  ✗ MISMATCH. Most likely causes, in order:\n"
            "      1. a different ML-20M release (this targets ml-20m, updated 2016-10-17)\n"
            "      2. a modified profiles file, changing the item universe\n"
            "      3. a pandas version that orders value_counts ties differently\n"
            "    Please open an issue with your pandas version and the stats.json produced.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ml20m-dir", required=True, type=Path,
                    help="directory of your own extracted ml-20m download")
    ap.add_argument("--profiles", type=Path, default=None,
                    help="released profiles, used to define the item universe "
                         f"(default: {PROFILE_CANDIDATES[0]})")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"where to write the splits (default: {DEFAULT_OUT} -- "
                         "the directory the benchmark reads)")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    log("rebuilding LLM-MovieLens benchmark splits")
    out = a.out if a.out is not None else DEFAULT_OUT
    stats = build(a.ml20m_dir, resolve_profiles(a.profiles), out)
    if a.no_verify:
        sys.exit(0)
    sys.exit(0 if verify(out, stats) else 1)
