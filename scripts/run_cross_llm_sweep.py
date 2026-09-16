#!/usr/bin/env python3
"""cross-LLM sweep orchestrator: 2 configs × 2 LLMs × 5 seeds = 20 cells.

- Idempotent: skips cells whose results.json exists.
- Mid-cell resumable: train_model auto-loads training_state.pt if present.
- Killable: SIGTERM/SIGINT mid-cell preserves state for resume on next run.

Usage:
    # Run unattended (Colab Pro / lab GPU):
    nohup python3 scripts/run_cross_llm_sweep.py --device cuda > /tmp/cross_llm_sweep.log 2>&1 &

    # Run only specific cells (re-run a single failure):
    python3 scripts/run_cross_llm_sweep.py --device cuda --only m7_gpt seed=42

    # Skip Claude cells (since they may already exist from the original paper run):
    python3 scripts/run_cross_llm_sweep.py --device cuda --skip_llm claude
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "scripts" / "run_cross_llm_retrain.py"
RES_ROOT = code("benchmark", "results_ml20m_gpt4omini")

CONFIGS = ["m4", "m7"]
LLMS    = ["claude", "gpt"]
SEEDS   = [42, 123, 456, 789, 2026]


def cell_done(config: str, llm: str, seed: int) -> bool:
    p = RES_ROOT / f"{config}_{llm}" / f"seed-{seed}" / "results.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        return bool(d.get("test_metrics"))
    except Exception:
        return False


def run_one(config: str, llm: str, seed: int, device: str) -> bool:
    exp = f"{config}_{llm}/seed-{seed}"
    if cell_done(config, llm, seed):
        print(f"[sweep] {exp:30s} SKIP (already complete)")
        return True
    print(f"\n[sweep] {exp:30s} START")
    t0 = time.time()
    cmd = [sys.executable, str(HARNESS),
           "--config", config, "--llm", llm,
           "--seed", str(seed), "--device", device]
    rc = subprocess.run(cmd).returncode
    dt = (time.time() - t0) / 60
    if rc != 0:
        print(f"[sweep] {exp:30s} FAILED (rc={rc}, {dt:.1f} min)")
        return False
    if not cell_done(config, llm, seed):
        print(f"[sweep] {exp:30s} EXIT 0 BUT NO results.json (interrupted? rerun to resume)")
        return False
    d = json.loads((RES_ROOT / f"{config}_{llm}" / f"seed-{seed}" / "results.json").read_text())
    tm = d["test_metrics"]
    print(f"[sweep] {exp:30s} DONE in {dt:.1f} min — NDCG@10={tm['NDCG@10']:.4f}")
    return True


def parse_only(only):
    """`--only m7_gpt seed=42 m4_claude` → list of (config, llm, seed)."""
    cells = []
    last_cl = None
    for tok in only or []:
        if "_" in tok and "=" not in tok:
            cfg, llm = tok.split("_")
            last_cl = (cfg, llm)
            for s in SEEDS:
                cells.append((cfg, llm, s))
        elif tok.startswith("seed=") and last_cl:
            s = int(tok.split("=")[1])
            cells = [c for c in cells if not (c[0] == last_cl[0] and c[1] == last_cl[1])]
            cells.append((*last_cl, s))
    return cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_llm", choices=LLMS, default="claude",
                   help="skip an entire LLM. Default: 'claude' (re-uses the paper's "
                        "existing M4/M7 Claude results — no scientific reason to retrain). "
                        "Pass --skip_llm none to actually run both.")
    p.add_argument("--only", nargs="+", default=None,
                   help="run a subset, e.g. 'm7_gpt seed=42'")
    args = p.parse_args()

    if args.only:
        cells = parse_only(args.only)
    else:
        cells = [(c, l, s) for c in CONFIGS for l in LLMS for s in SEEDS
                 if l != args.skip_llm]

    print(f"[sweep] {len(cells)} cells: configs={CONFIGS}, llms={LLMS}, seeds={SEEDS}")
    print(f"[sweep] device={args.device}, skip_llm={args.skip_llm}")
    print(f"[sweep] results root: {RES_ROOT}\n")

    n_done, n_failed = 0, 0
    t_start = time.time()
    for config, llm, seed in cells:
        ok = run_one(config, llm, seed, args.device)
        if ok: n_done += 1
        else:  n_failed += 1

    elapsed = (time.time() - t_start) / 60
    print(f"\n[sweep] === FINISHED ===")
    print(f"[sweep] {n_done} cells ok, {n_failed} failed, total {elapsed:.1f} min")


if __name__ == "__main__":
    main()
