#!/usr/bin/env python3
"""Run M7 (LightGCN-SF + LLM profile + mood) on Amazon-Books with a single seed.

Examples:
    python scripts/run_amazon_m7.py
    python scripts/run_amazon_m7.py --seed 123
    python scripts/run_amazon_m7.py --seed 456 --gpu 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _amazon_run_helper import run_amazon_config


if __name__ == "__main__":
    run_amazon_config(model="lightgcn_sf", features="llm_prof_mood", label="M7")
