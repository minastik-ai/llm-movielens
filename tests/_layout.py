"""Where the code lives, in both the development and the published layout.

A module rather than a conftest fixture because the test modules need it at
IMPORT time -- test_resume_guard.py resolves the benchmark directory while
pytest is still collecting, and a conftest is not importable that early.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def component(name: str) -> Path | None:
    """`benchmark`, `embedding_generator`, `profile_generator` in either layout."""
    for base in (REPO / "src", REPO / "code"):
        d = base / name
        if d.is_dir():
            return d
        if base.is_dir():                      # the profiler sits in a dated dir
            hit = sorted(p for p in base.glob(f"{name}*") if p.is_dir())
            if hit:
                return hit[0]
    return None


def load_module(component_name: str, filename: str):
    """Import a shipped file by path, or skip with the reason it could not load.

    Three ways this fails and none of them is a defect in the test: the component
    is missing (a layout neither branch of `component` knows), the file is missing,
    or the module's OWN imports are missing -- evaluate.py and train.py both import
    torch, which a reviewer running `pip install -r tests/requirements.txt` does not
    have. That last one used to surface as an ERROR at collection, and pytest stops
    collecting after the first: one traceback, zero tests, for a suite whose other
    47 need no torch. Every one of the three must SKIP and say what is absent.
    """
    import importlib.util
    import sys

    import pytest

    d = component(component_name)
    if d is None:
        pytest.skip(f"no {component_name} package in either layout",
                    allow_module_level=True)
    src = d / filename
    if not src.exists():
        pytest.skip(f"{component_name}/{filename} not present",
                    allow_module_level=True)
    if str(d) not in sys.path:                 # the shipped files import siblings
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"_{component_name}_{src.stem}", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        pytest.skip(f"{component_name}/{filename} needs {e.name!r}: "
                    f"pip install -r requirements.txt", allow_module_level=True)
    return mod


def _first(*candidates: Path) -> Path | None:
    return next((p for p in candidates if p.exists()), None)


