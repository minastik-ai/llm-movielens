#!/usr/bin/env python3
"""Run M6 (LightGCN-SF + LLM themes only) on Amazon-Books with a single seed.

Note: book themes (518-dim multi-hot) differ from movie themes (528-dim);
the dimension is set automatically by the FeatureLoader at runtime.

Examples:
    python scripts/run_amazon_m6.py
    python scripts/run_amazon_m6.py --seed 123 --gpu 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _amazon_run_helper import run_amazon_config


if __name__ == "__main__":
    run_amazon_config(model="lightgcn_sf", features="llm_themes", label="M6")
