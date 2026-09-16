#!/usr/bin/env python3
"""cross-LLM single-cell retrain for Colab Pro — resume-friendly across disconnects.

One invocation = one (config, llm, seed) cell. Resumable via train_model's
existing training_state.pt (saved after every epoch). Re-running the same
command after a Colab disconnect picks up at epoch+1 from the last save.

Output convention matches the existing repo layout (parallel to
checkpoints_ml1m/<config>/<encoder>/seed-<N>/, etc.):

    <ckpt_root>/<config>_<llm>/<encoder>/seed-<N>/best_model.pt
    <ckpt_root>/<config>_<llm>/<encoder>/seed-<N>/training_state.pt
    <res_root>/<config>_<llm>/<encoder>/seed-<N>/results.json

Defaults:
    ckpt_root = <repo>/code/benchmark/checkpoints   (ML-20M base)
    res_root  = <repo>/code/benchmark/results
    encoder   = bge-large-en-v1.5

So cross-LLM cells land at e.g. checkpoints/m4_gpt/bge-large-en-v1.5/seed-42/, parallel
to the existing checkpoints/m4/bge-large-en-v1.5/seed-42/ (Claude-side).

Colab Pro usage:
    !python3 scripts/run_cross_llm_colab.py --config m7 --seed 42 \
        --ckpt_root /content/drive/MyDrive/llm-movielens/code/benchmark/checkpoints \
        --res_root  /content/drive/MyDrive/llm-movielens/code/benchmark/results

    # Inspect state without launching training:
    !python3 scripts/run_cross_llm_colab.py --config m7 --seed 42 \
        --ckpt_root ... --res_root ... --status

    # Force a fresh run (delete any prior state, retrain from scratch):
    !python3 scripts/run_cross_llm_colab.py --config m7 --seed 42 \
        --ckpt_root ... --res_root ... --no_resume
"""
from __future__ import annotations

import argparse, json, os, random, shutil, sys, time
from pathlib import Path
from _paths import code  # dual-layout paths: dev tree or published release

REPO = Path(__file__).resolve().parent.parent
BENCH = code("benchmark")
sys.path.insert(0, str(BENCH))

import numpy as np
import torch

from config import EMBED_DIM, LIGHTGCN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.lightgcn import LightGCNSF
from train import train_model

LLM_EMB_DIRS = {
    "claude": code("embedding_generator", "output", "bge-large-en-v1.5"),
    "gpt":    code("embedding_generator", "output_ml20m_gpt4omini", "bge-large-en-v1.5"),
}
CONFIG_FEATURES = {"m4": ["profile"], "m7": ["profile", "mood"]}
DATA_DIR = BENCH / "data" / "processed"


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ENCODER = "bge-large-en-v1.5"

# Repo convention (matches existing checkpoints_ml1m/<config>/<encoder>/seed-<N>/):
#   <ckpt_root>/<config>_<llm>/<encoder>/seed-<N>/best_model.pt
#   <ckpt_root>/<config>_<llm>/<encoder>/seed-<N>/training_state.pt
#   <res_root>/<config>_<llm>/<encoder>/seed-<N>/results.json
def experiment_subpath(config: str, llm: str, seed: int) -> str:
    return f"{config}_{llm}/{ENCODER}/seed-{seed}"


def cell_paths(ckpt_root: Path, res_root: Path, config: str, llm: str, seed: int):
    sub = experiment_subpath(config, llm, seed)
    return {
        "ckpt_dir":      ckpt_root / sub,
        "res_dir":       res_root / sub,
        "best_model":    ckpt_root / sub / "best_model.pt",
        "train_state":   ckpt_root / sub / "training_state.pt",
        "results":       res_root / sub / "results.json",
    }


def cmd_status(paths):
    ckpt_dir, res_dir = paths["ckpt_dir"], paths["res_dir"]
    if not ckpt_dir.exists() and not res_dir.exists():
        print(f"[status] no prior runs at {ckpt_dir} or {res_dir}")
        return
    print(f"[status] ckpt_dir = {ckpt_dir}")
    print(f"[status] res_dir  = {res_dir}")
    files_present = []
    for k, p in paths.items():
        if k.endswith("_dir"):
            continue
        if p.exists():
            files_present.append(f"  ✓ {k}: {p.name} ({p.stat().st_size/1024:.1f} KB)")
        else:
            files_present.append(f"  ✗ {k}: missing")
    print("\n".join(files_present))

    if paths["results"].exists():
        try:
            d = json.loads(paths["results"].read_text())
            tm = d.get("test_metrics") or {}
            be = d.get("best_epoch", "?")
            print(f"[status] results.json — best_epoch={be}")
            print(f"  test NDCG@10 = {tm.get('NDCG@10'):.4f}" if tm.get('NDCG@10') is not None else "  test metrics: missing")
            print(f"  test Recall@10 = {tm.get('Recall@10'):.4f}" if tm.get('Recall@10') is not None else "")
            print(f"  test MRR = {tm.get('MRR'):.4f}" if tm.get('MRR') is not None else "")
            return "complete"
        except Exception as e:
            print(f"[status] results.json present but unreadable: {e}")

    if paths["train_state"].exists():
        try:
            s = torch.load(paths["train_state"], map_location="cpu", weights_only=False)
            print(f"[status] training_state.pt — at epoch {s.get('epoch', '?')}, "
                  f"best_val={s.get('best_val_ndcg', '?')}, "
                  f"patience_counter={s.get('patience_counter', '?')}")
            return "in_progress"
        except Exception as e:
            print(f"[status] training_state.pt present but unreadable: {e}")
    print("[status] no prior state found — fresh run on next invocation")
    return "fresh"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIG_FEATURES), required=True,
                   help="m4 = LightGCN-SF + profile (1024-d); m7 = + mood (1034-d)")
    p.add_argument("--seed",   type=int, required=True,
                   help="paper seeds: 42, 123, 456, 789, 2026")
    p.add_argument("--llm",    choices=list(LLM_EMB_DIRS), default="gpt",
                   help="default 'gpt' for cross-LLM closure; use 'claude' to also retrain Claude")
    p.add_argument("--device", default="auto",
                   help="auto / cuda / mps / cpu")
    p.add_argument("--ckpt_root", type=Path, default=None,
                   help="Root for checkpoints (best_model.pt, training_state.pt). "
                        "Default: <repo>/code/benchmark/checkpoints (ML-20M base convention). "
                        "On Colab Pro pass an absolute Drive path "
                        "e.g., /content/drive/MyDrive/llm-movielens/code/benchmark/checkpoints "
                        "so disconnects don't lose progress.")
    p.add_argument("--res_root", type=Path, default=None,
                   help="Root for results.json. Default: <repo>/code/benchmark/results.")
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
                   help=f"max training epochs (default {NUM_EPOCHS}, with patience early-stop)")
    p.add_argument("--patience",   type=int, default=PATIENCE,
                   help=f"early-stop patience in epochs (default {PATIENCE})")
    p.add_argument("--eval_every", type=int, default=5,
                   help="evaluate val every N epochs (also the checkpoint cadence)")
    p.add_argument("--no_resume", action="store_true",
                   help="ignore existing checkpoints and start fresh")
    p.add_argument("--status", action="store_true",
                   help="print state only, do not train")
    args = p.parse_args()

    # Default to ML-20M base convention: code/benchmark/{checkpoints,results}
    if args.ckpt_root is None:
        args.ckpt_root = code("benchmark", "checkpoints")
        print(f"[cross_llm] --ckpt_root not set; using repo-local {args.ckpt_root}")
        print(f"     On Colab Pro, pass --ckpt_root /content/drive/MyDrive/llm-movielens/code/benchmark/checkpoints")
    if args.res_root is None:
        args.res_root = code("benchmark", "results")
        print(f"[cross_llm] --res_root not set; using repo-local {args.res_root}")

    paths = cell_paths(args.ckpt_root, args.res_root, args.config, args.llm, args.seed)

    if args.status:
        cmd_status(paths); return

    if args.no_resume:
        for d in (paths["ckpt_dir"], paths["res_dir"]):
            if d.exists():
                print(f"[cross_llm] --no_resume: removing prior state at {d}")
                shutil.rmtree(d)

    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    paths["res_dir"].mkdir(parents=True, exist_ok=True)

    # Idempotent short-circuit
    if paths["results"].exists() and not args.no_resume:
        try:
            d = json.loads(paths["results"].read_text())
            if d.get("test_metrics"):
                tm = d["test_metrics"]
                print(f"[cross_llm] {experiment_subpath(args.config, args.llm, args.seed)} ALREADY COMPLETE")
                print(f"     test NDCG@10={tm.get('NDCG@10'):.4f}  "
                      f"Recall@10={tm.get('Recall@10'):.4f}  MRR={tm.get('MRR'):.4f}")
                print(f"[cross_llm] To force re-run, pass --no_resume")
                return
        except Exception:
            pass  # proceed to retrain

    # Status print before launching
    print("=" * 70)
    print(f"cross-LLM retrain | config={args.config} | llm={args.llm} | seed={args.seed}")
    print(f"  ckpt_dir = {paths['ckpt_dir']}")
    print(f"  res_dir  = {paths['res_dir']}")
    cmd_status(paths)
    print("=" * 70)

    # Setup
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[cross_llm] device = {device}")

    emb_dir = LLM_EMB_DIRS[args.llm]
    if not (emb_dir / "profile_embeddings.npy").exists():
        print(f"ERROR: missing {emb_dir}/profile_embeddings.npy"); sys.exit(2)

    print(f"[cross_llm] Loading ML-20M data + features...")
    data = InteractionData(data_dir=DATA_DIR)
    fl   = FeatureLoader(data_dir=DATA_DIR, embedding_dir=emb_dir)
    feat = fl.get_combined_tensor(CONFIG_FEATURES[args.config], device=device)
    print(f"[cross_llm] n_users={data.n_users}, n_items={data.n_items}, feat_dim={feat.shape[1]}")

    model = LightGCNSF(data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS,
                       feature_dim=feat.shape[1]).to(device)
    model.set_adj(data.get_norm_adj().to(device))
    model.set_features(feat)

    # train_model joins <root>/<experiment_name>/. With experiment_name including
    # encoder, final paths are:
    #   <ckpt_root>/<config>_<llm>/<encoder>/seed-<N>/{best_model,training_state}.pt
    #   <res_root>/<config>_<llm>/<encoder>/seed-<N>/results.json
    exp_name = experiment_subpath(args.config, args.llm, args.seed)

    t0 = time.time()
    result = train_model(
        model=model,
        interaction_data=data,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=args.num_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        experiment_name=exp_name,
        resume=(not args.no_resume),
        checkpoint_dir=args.ckpt_root,
        results_dir=args.res_root,
        seed=args.seed,
    )
    wall_min = (time.time() - t0) / 60
    tm = (result or {}).get("test_metrics") or {}
    print(f"\n[cross_llm] {exp_name} DONE in {wall_min:.1f} min")
    if tm:
        print(f"     test NDCG@10={tm.get('NDCG@10'):.4f}  "
              f"Recall@10={tm.get('Recall@10'):.4f}  MRR={tm.get('MRR'):.4f}")
    cmd_status(paths)


if __name__ == "__main__":
    main()
