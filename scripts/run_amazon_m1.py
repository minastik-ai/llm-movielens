#!/usr/bin/env python3
"""Run M1 (LightGCN, ID-only) on Amazon-Books with a single seed.

Examples:
    python scripts/run_amazon_m1.py
    python scripts/run_amazon_m1.py --seed 123
    python scripts/run_amazon_m1.py --seed 456 --gpu 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _amazon_run_helper import run_amazon_config


if __name__ == "__main__":
    run_amazon_config(model="lightgcn", features="none", label="M1")
