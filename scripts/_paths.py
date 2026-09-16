#!/usr/bin/env python3
"""Resolve the project's trees in BOTH the development and the published layout.

WHY THIS EXISTS. The development checkout keeps generators under
`<repo>/code/<component>/`, with the profiler in a dated directory. The published
release flattens that to `<repo>/src/<component>/` and lifts the benchmark's
per-seed results to `<repo>/results/`. A script that spells either layout
literally runs in one of the two places and dies in the other -- and the one it
died in was the release, which is the only one a stranger has. An audit ran every
shipped script: 21 of them raised ImportError or FileNotFoundError on the first
line that touched a path, because `<repo>/code/` does not exist in a release.

`code(*parts)` takes the DEVELOPMENT spelling -- the one already written
throughout these scripts -- and returns wherever that tree actually is. The
development layout is tried first, so behaviour in this repository is unchanged.

`require()` is the other half. Sixteen of those scripts read trees the release
deliberately does not redistribute (vendored baselines, checkpoints, processed
splits, generated embeddings). A path fix alone moves their failure from the
import line to a bare traceback deeper in. `require()` makes them say what is
missing and how to obtain it, which is the difference between a broken script and
a script with a precondition.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# development prefix -> published prefix. Longest match wins, so the results tree
# is lifted to the repository root before the benchmark rule can claim it.
_LAYOUT: dict[tuple[str, ...], tuple[str, ...]] = {
    ("benchmark", "results"): ("results",),
    ("benchmark",): ("src", "benchmark"),
    ("profile_generator", "llm-movie-profiler-v1-20260402"): ("src", "profile_generator"),
    ("profile_generator",): ("src", "profile_generator"),
    ("embedding_generator",): ("src", "embedding_generator"),
}


# Decide the LAYOUT once, from the checkout, not per path. Testing whether each
# individual leaf exists looks equivalent and is not: a path that is CREATED at
# run time (an output directory, a log tree) does not exist yet on the first run,
# so a per-path test sends it down the published branch and the script writes into
# a tree this checkout does not use. Ten of these expressions had that shape.
#
# The marker is a FILE inside the tree, not the `code/` directory itself. A bare
# directory test is not sound: several of these scripts create output directories
# at import time, so a single stray run leaves an empty `code/` behind in a
# published checkout -- and the next run then reads that as a development tree and
# resolves everything to somewhere nothing ships. The marker has to be something
# only a real source tree contains.
_DEV_MARKER = REPO / "code" / "benchmark" / "config.py"
_PUB_MARKER = REPO / "src" / "benchmark" / "config.py"
_DEVELOPMENT = _DEV_MARKER.is_file() or not _PUB_MARKER.is_file()


def code(*parts: str) -> Path:
    """Path to the development `code/<parts>` tree, wherever it lives here."""
    if _DEVELOPMENT:
        return REPO.joinpath("code", *parts)
    for prefix in sorted(_LAYOUT, key=len, reverse=True):
        if tuple(parts[: len(prefix)]) == prefix:
            return REPO.joinpath(*_LAYOUT[prefix], *parts[len(prefix):])
    # Unmapped in a published checkout: nothing ships under it, so return the
    # development spelling and let require() name what is missing.
    return REPO.joinpath("code", *parts)


def require(path: Path, what: str, how: str) -> Path:
    """Exit with a sentence naming the missing input, not a traceback.

    Exits 1 (an artifact problem, not a defect in this script) and writes to
    stderr, so a caller that checks the status sees a failure rather than a
    message on stdout that reads like output.
    """
    if not Path(path).exists():
        print(f"Missing {what}: {path}", file=sys.stderr)
        print("", file=sys.stderr)
        print(how, file=sys.stderr)
        raise SystemExit(1)
    return Path(path)
