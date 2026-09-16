#!/usr/bin/env python3
"""Run M7 (LightGCN-SF + LLM profile + mood) on ML-1M with a single seed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ml1m_run_helper import run_ml1m_config


if __name__ == "__main__":
    run_ml1m_config(model="lightgcn_sf", features="llm_prof_mood", label="M7")
