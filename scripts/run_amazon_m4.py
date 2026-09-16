#!/usr/bin/env python3
"""Run M4 (LightGCN-SF + LLM profile) on Amazon-Books with a single seed.

Examples:
    python scripts/run_amazon_m4.py
    python scripts/run_amazon_m4.py --seed 123
    python scripts/run_amazon_m4.py --seed 456 --gpu 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _amazon_run_helper import run_amazon_config


if __name__ == "__main__":
    run_amazon_config(model="lightgcn_sf", features="llm_profile", label="M4")
