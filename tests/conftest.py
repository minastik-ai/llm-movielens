"""Locate the code and the artifacts in BOTH layouts, and skip when data is absent.

WHY THIS EXISTS. The development checkout keeps generators under
`<repo>/code/<component>/` and the published release flattens them to
`<repo>/src/<component>/`. Every test here spelled the development layout
literally, so the suite could not even be COLLECTED from a release checkout --
`test_resume_guard.py` died looking for `code/benchmark/train.py`. `_paths.py`
already solved this for the shipped scripts; the tests never adopted it.

The second job is honest skipping. A reviewer who clones the repository and runs
pytest has the code but not the 566 MB of artifacts. Those tests must SKIP, not
fail: a red suite on a fresh clone says "this is broken" when it means "you have
not downloaded the data", and that is the first thing a reviewer sees.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import REPO, component, _first  # noqa: E402

@pytest.fixture(scope="session")
def benchmark_dir() -> Path:
    d = component("benchmark")
    if d is None:
        pytest.skip("no benchmark package in either layout")
    return d


@pytest.fixture(scope="session")
def embeddings_dir() -> Path:
    """The 1024-d per-encoder directory.

    NOT the generator's top-level output/: that holds a PCA-128 variant of the
    same names, and reading it is why four shape tests asserted (10381, 1024) and
    got (10381, 128). Same filenames, different arrays, one directory apart.
    """
    gen = component("embedding_generator")
    cands = [REPO / "data" / "embeddings" / "bge-large-en-v1.5"]
    if gen:
        cands += [gen / "output" / "bge-large-en-v1.5"]
    cands += [REPO / "release" / "hf" / "embeddings" / "ml20m" / "bge-large-en-v1.5"]
    d = _first(*cands)
    if d is None:
        pytest.skip("no embeddings directory; fetch them with "
                    "scripts/download_artifacts.py")
    return d


@pytest.fixture(scope="session")
def profiles() -> dict:
    gen = component("profile_generator")
    cands = [REPO / "data" / "profiles" / "movie_profiles.json"]
    if gen:
        # The development tree nests the profiler one level deeper, in a dated
        # directory the release flattens away, so look through it as well as at it.
        cands += [gen / "output" / "movie_profiles.json"]
        cands += sorted(gen.glob("*/output/movie_profiles.json"))
    cands += [REPO / "release" / "hf" / "profiles" / "ml20m" / "claude-haiku-4-5"
              / "movie_profiles.json"]
    p = _first(*cands)
    if p is None:
        pytest.skip("no movie_profiles.json; fetch it with "
                    "scripts/download_artifacts.py")
    d = json.loads(p.read_text())
    return d if isinstance(d, dict) else {str(r["movieId"]): r for r in d}


@pytest.fixture(scope="session")
def movie_id_index(embeddings_dir) -> list:
    p = embeddings_dir / "movie_id_index.json"
    if not p.exists():
        pytest.skip("no movie_id_index.json beside the embeddings")
    return json.loads(p.read_text())
