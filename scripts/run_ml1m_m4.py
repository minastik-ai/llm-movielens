#!/usr/bin/env python3
"""Run M4 (LightGCN-SF + LLM profile) on ML-1M with a single seed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ml1m_run_helper import run_ml1m_config


if __name__ == "__main__":
    run_ml1m_config(model="lightgcn_sf", features="llm_profile", label="M4")
