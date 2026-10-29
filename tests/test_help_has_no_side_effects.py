"""`--help` and plain import must not touch the checkout.

A script asked for `--help` should print usage and leave the tree as it found it. A script
that creates output directories or log files at import time leaves new directories in a
reader's clone when they only wanted a usage line, and a path resolver that tests "does
this directory exist" then misjudges the layout.

The test runs every command-line script with `--help` in a subprocess (no API keys, no
network proxy that answers) and compares the file and directory listing of the checkout
before and after. A script whose own imports are missing (torch, sentence-transformers)
is skipped with the module name, never failed.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import REPO

SKIP_DIRS = {"__pycache__", ".pytest_cache", "external", "tests", ".git", "data", "cache", "logs"}


def _scripts() -> list[Path]:
    found: list[Path] = []
    for top in ("scripts", "tools", "src", "code"):
        base = REPO / top
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if SKIP_DIRS & set(p.relative_to(REPO).parts):
                continue
            text = p.read_text(errors="ignore")
            if "argparse" in text and "__main__" in text:
                found.append(p)
    return found


def _listing() -> set[str]:
    out: set[str] = set()
    for dp, dn, fn in os.walk(REPO):
        dn[:] = [d for d in dn if d not in {"__pycache__", ".pytest_cache", ".git"}]
        rel = os.path.relpath(dp, REPO)
        out.add(rel + "/")
        out.update(os.path.join(rel, f) for f in fn)
    return out


SCRIPTS = _scripts()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_help_leaves_the_tree_unchanged(script: Path):
    env = {k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET", k)}
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    before = _listing()
    r = subprocess.run([sys.executable, str(script), "--help"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
    after = _listing()
    m = re.search(r"No module named '([\w.]+)'", r.stderr)
    if r.returncode != 0 and m:
        pytest.skip(f"{script.name} needs {m.group(1)!r}")
    assert r.returncode == 0, f"{script.name} --help exited {r.returncode}: {r.stderr[-300:]}"
    assert "usage" in (r.stdout + r.stderr).lower(), f"{script.name} --help printed no usage"
    created = sorted(after - before)
    for rel in reversed(created):             # leave the tree as it was found (children first)
        p = REPO / rel
        try:
            p.rmdir() if rel.endswith("/") else p.unlink()
        except OSError:
            pass
    assert not created, f"{script.name} --help created: {created[:5]}"
