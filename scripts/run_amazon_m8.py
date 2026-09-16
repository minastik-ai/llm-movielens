#!/usr/bin/env python3
"""Run M8 (LightGCN-SF + LLM profile + mood + themes, full feature set) on Amazon-Books.

Feature dim: profile(1024) + mood(10) + themes(518) = 1552

Examples:
    python scripts/run_amazon_m8.py
    python scripts/run_amazon_m8.py --seed 123 --gpu 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _amazon_run_helper import run_amazon_config


if __name__ == "__main__":
    run_amazon_config(model="lightgcn_sf", features="llm_all", label="M8")
