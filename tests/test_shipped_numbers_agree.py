"""Numbers and names that live in code must agree with what ships beside them.

A fact kept in code must agree with the file that holds it. Three such facts are
compared here, because a code copy that drifts from its file gives a wrong verdict:

  * the e5-large-v2 NDCG@10 anchors in `eval_checkpoints.py` equal the means of the
    shipped e5 per-seed files;
  * `build_prompt_manifest.py` reads settings names that exist, so the manifest header
    carries the real settings rather than hard-coded fallbacks;
  * `run_ablation.py` prints the sample standard deviation, as the paper and the
    verifier do.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import REPO, component


def _results_root() -> Path | None:
    for p in (REPO / "results", REPO / "code" / "benchmark" / "results"):
        if p.is_dir():
            return p
    return None


def test_e5_anchors_are_the_shipped_means():
    bench = component("benchmark")
    src = (bench / "eval_checkpoints.py").read_text() if bench else ""
    m = re.search(r"E5_ANCHORS = \{([^}]*)\}", src)
    root = _results_root()
    if not m or root is None:
        pytest.skip("eval_checkpoints.py or the per-seed results are not present")
    anchors = {k: float(v) for k, v in re.findall(r'"(m\d)": ([\d.]+)', m.group(1))}
    assert anchors, "no e5 anchors parsed"
    for cfg, anchor in anchors.items():
        vals = [json.loads(f.read_text())["test_metrics"]["NDCG@10"]
                for f in sorted((root / cfg / "e5-large-v2").glob("seed-*/results.json"))]
        if not vals:
            pytest.skip(f"no shipped e5 results for {cfg}")
        assert abs(sum(vals) / len(vals) - anchor) < 5.1e-5, \
            f"{cfg}: anchor {anchor} vs shipped mean {sum(vals) / len(vals):.4f}"


def test_manifest_builder_reads_settings_names_that_exist():
    builder = REPO / "tools" / "build_prompt_manifest.py"
    prof = component("profile_generator")
    if not builder.exists() or prof is None or not (prof / "config" / "settings.py").exists():
        pytest.skip("builder or settings not present")
    used = set(re.findall(r"settings\.([A-Z_]+)", builder.read_text()))
    used |= set(re.findall(r'getattr\(settings, "([A-Z_]+)"', builder.read_text()))
    defined = set(re.findall(r"^([A-Z_]+)\s*=", (prof / "config" / "settings.py").read_text(), re.M))
    assert not (used - defined), f"names read from settings but not defined there: {sorted(used - defined)}"


def test_ablation_driver_reports_the_sample_standard_deviation():
    bench = component("benchmark")
    src = (bench / "run_ablation.py").read_text() if bench else ""
    if not src:
        pytest.skip("run_ablation.py not present")
    assert re.search(r"np\.std\(vals, ddof=1\)", src), "run_ablation.py must use ddof=1"
    assert not re.search(r"np\.std\(vals\)", src)
