#!/usr/bin/env python3
"""Fetch the released profiles and embeddings so a reader can enter at stage 2 or 3.

The released artifacts let a reader skip generation and encoding. This fetches them
from the data repository and puts each file where the harness reads it.
`download_ml20m.sh` does the same for the source data.

Idempotent: a file already present is left alone, so re-running costs nothing.

About 190 MB over the wire, and roughly twice that on disk: `hf_hub_download` keeps
its own copy under ~/.cache/huggingface and this copies from there into place.

    python3 scripts/download_artifacts.py              # profiles + bge embeddings
    python3 scripts/download_artifacts.py --dry-run    # show the plan, fetch nothing
    python3 scripts/download_artifacts.py --encoder e5-large-v2
    python3 scripts/download_artifacts.py --human-eval   # + the filled annotation sheets
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "minastik-ai/llm-movielens"   # a repo id (org/name), not a URL
ROOT = Path(__file__).resolve().parents[1]

# The filled sheets behind the human-evaluation numbers, for the tools that check them
# (tools/verify_mood_agreement.py reads the pairwise four).
HUMAN_EVAL = ["annotator_sheet_filled_A1.csv", "annotator_sheet_filled_A2.csv",
              "annotator_sheet_filled_A3.csv", "annotator_sheet_key.json",
              "mood_pairwise_sheet_filled_A1.csv", "mood_pairwise_sheet_filled_A2.csv",
              "mood_pairwise_sheet_filled_A3.csv", "mood_pairwise_key.json"]


NO_METADATA = {"e5-large-v2"}


# (path in the data repository, path the harness reads)
def plan(encoder: str, human_eval: bool = False) -> list[tuple[str, Path]]:
    emb = ROOT / "src" / "embedding_generator" / "output" / encoder
    items = [("profiles/ml20m/claude-haiku-4-5/movie_profiles.json",
              ROOT / "src" / "profile_generator" / "output" / "movie_profiles.json")]
    files = ["profile_embeddings.npy", "mood_vectors.npy", "theme_matrix.npy",
             "combined_features.npy", "combined_full.npy",
             "movie_id_index.json", "theme_vocabulary.json", "embedding_metadata.json"]
    # The e5 directory ships no embedding_metadata.json (its generator never wrote
    # one; the data repository's RELEASE_MANIFEST says so), so it is not requested.
    if encoder in NO_METADATA:
        files.remove("embedding_metadata.json")
    for f in files:
        items.append((f"embeddings/ml20m/{encoder}/{f}", emb / f))
    if human_eval:
        items += [(f"human_eval/{f}", ROOT / "data" / "human_eval" / f) for f in HUMAN_EVAL]
    return items


def sizes(repo_id: str) -> dict:
    """Byte size per file, from the hub. Empty when offline or unavailable.

    Printed rather than assumed: the one thing a reader wants before starting a
    download is how big it is, and a number hard-coded here would drift the first
    time an artifact is regenerated.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
        return {s.rfilename: s.size for s in info.siblings if s.size}
    except Exception:
        return {}


def human(n: int) -> str:
    return f"{n/1e6:,.1f} MB" if n >= 1e6 else f"{n/1e3:,.0f} KB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="bge-large-en-v1.5",
                    help="which encoder's embeddings to fetch (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched and where it would go")
    ap.add_argument("--repo-id", default=REPO_ID,
                    help="data repository (default: the released one)")
    ap.add_argument("--human-eval", action="store_true",
                    help="also fetch the filled annotation sheets into data/human_eval/")
    a = ap.parse_args()

    items = plan(a.encoder, a.human_eval)
    have = [(s, d) for s, d in items if d.exists()]
    need = [(s, d) for s, d in items if not d.exists()]
    print(f"  repository: {a.repo_id}")
    print(f"  {len(items)} file(s) in the plan; {len(have)} already present, "
          f"{len(need)} to fetch")
    size = sizes(a.repo_id) if not a.repo_id.startswith("[") else {}
    for s, d in items:
        tag = f"  {human(size[s])}" if s in size else ""
        print(f"   {'have' if d.exists() else 'FETCH':>5}  {s}{tag}")
        print(f"          -> {d.relative_to(ROOT)}")
    if size:
        n_bytes = sum(size.get(s, 0) for s, _ in need)
        print(f"\n  {human(n_bytes)} to fetch, about twice that on disk: "
              f"huggingface_hub caches its own copy under ~/.cache/huggingface")
    else:
        print("\n  (file sizes unavailable -- expect about 190 MB, twice that on disk)")
    if a.dry_run:
        print("\n  dry run: nothing fetched.")
        return 0
    if not need:
        print("\n  everything is already in place.")
        return 0
    if a.repo_id.startswith("["):
        print("\n  The data repository has not been minted yet, so this script cannot "
              "fetch.\n  Download the files by hand from the link in README.md, or pass "
              "--repo-id.", file=sys.stderr)
        return 1
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\n  huggingface_hub is required: pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    for src, dst in need:
        dst.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(repo_id=a.repo_id, filename=src, repo_type="dataset")
        shutil.copyfile(got, dst)
        print(f"   fetched {src}")
    print(f"\n  {len(need)} file(s) fetched. Two steps still stand between this and a\n"
          f"  benchmark run -- the source data is not ours to redistribute:\n"
          f"    bash scripts/download_ml20m.sh\n"
          f"    python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m\n"
          f"    bash scripts/reproduce_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
