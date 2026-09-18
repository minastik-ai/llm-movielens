#!/usr/bin/env python3
"""
Eval-only re-evaluation of released checkpoints (no training).

`run_experiment.py` trains-then-evals; this script only loads an existing
`best_model.pt` state_dict and runs the full-ranking test evaluation, writing the
per-seed leaf `results<suffix>/<config>/<encoder>/seed-<s>/results.json` (same
schema train.py emits). Use it to (re)generate result files for configs whose
checkpoints exist but whose per-seed eval JSONs were never persisted.

Eval is deterministic given the weights (model.eval() disables contrastive noise),
so the 5-seed mean reproduces the paper exactly.

    python3 eval_checkpoints.py                              # base ML-20M, all 14 M-configs
    python3 eval_checkpoints.py --dataset amazon             # Amazon m4/m5/m6/m7/m8
    python3 eval_checkpoints.py --configs m5 m6              # subset
    python3 eval_checkpoints.py --dataset amazon --device cpu
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURE_CONFIGS, SEEDS, EMBEDDING_DIR, EMBED_DIM, LIGHTGCN_LAYERS, get_config_name
from data.dataset import InteractionData
from features.loader import FeatureLoader
from evaluate import evaluate_model
from run_experiment import build_model, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

B = Path(__file__).resolve().parent          # code/benchmark
CODE_ROOT = B.parent                          # code/
ENC = "bge-large-en-v1.5"
GNN = {"lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl", "kar", "hypernet_replacer"}
FEAT_MODELS = {"lightgcn_sf", "kar", "hypernet_replacer"}  # models that take item side-features
RLABEL = {"r2": "R2", "r3": "R3"}                          # leaf "config" label override

# config dir -> (model, features key). Superset; per-dataset subset chosen below.
SPEC = {
    "m0":  ("bpr_mf", "none"),       "m1":  ("lightgcn", "none"),
    "m1b": ("simgcl", "none"),       "m1c": ("xsimgcl", "none"),  "m1d": ("lightgcl", "none"),
    "m2":  ("lightgcn_sf", "genome"),     "m2b": ("lightgcn_sf", "genome_raw"),
    "m3":  ("lightgcn_sf", "bert_title"), "m4":  ("lightgcn_sf", "llm_profile"),
    "m5":  ("lightgcn_sf", "llm_mood"),   "m6":  ("lightgcn_sf", "llm_themes"),
    "m7":  ("lightgcn_sf", "llm_prof_mood"), "m8": ("lightgcn_sf", "llm_all"),
    "m9":  ("lightgcn_sf", "genome_llm"),
    # Tier-3 replacers (project models, d=128, profile+mood; same evaluate_model path as M7):
    "r2":  ("kar", "llm_prof_mood"),  "r3": ("hypernet_replacer", "llm_prof_mood"),
    "r2_retuned": ("kar", "llm_prof_mood"),  # Amazon per-dataset-retuned R2 (wd=1e-4); ckpt regenerated 2026-06
}

# Per-dataset: data dir, embedding dir, configs to eval, paper NDCG@10 anchors.
DATASETS = {
    "": dict(  # base ML-20M
        data=B / "data" / "processed",
        emb=EMBEDDING_DIR,
        configs=["m0", "m1", "m1b", "m1c", "m1d", "m2", "m2b", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "r2", "r3"],
        anchors={"m0": 0.1137, "m1": 0.1139, "m1b": 0.1132, "m1c": 0.1144, "m1d": 0.1136,
                 "m2": 0.1144, "m2b": 0.1140, "m3": 0.1136, "m4": 0.1173, "m5": 0.1149,
                 "m6": 0.1102, "m7": 0.1175, "m8": 0.1156, "m9": 0.1150, "r2": 0.1145, "r3": 0.1099},
    ),
    "amazon": dict(  # Amazon-Books (paper Table amazon_tier1 + r2/r3 retune apps)
        data=B / "data" / "processed_amazon",
        emb=CODE_ROOT / "embedding_generator" / "output_amazon" / ENC,
        configs=["m4", "m5", "m6", "m7", "m8", "r2", "r2_retuned", "r3"],
        anchors={"m1": 0.0334, "m4": 0.0561, "m5": 0.0543, "m6": 0.0525, "m7": 0.0563, "m8": 0.0536,
                 # r2 = original-protocol (paper headline Table amazon_tier1); r2_retuned = per-dataset
                 # retune (App. r2_retune), checkpoint regenerated 2026-06 (5-seed mean 0.0440 ≈ paper 0.0439).
                 "r2": 0.0401, "r2_retuned": 0.0440, "r3": 0.0497},
    ),
    "ml1m": dict(  # ML-1M cross-density (paper Table ml1m_cross_density + r2/r3 retune)
        data=B / "data" / "processed_ml1m",
        emb=CODE_ROOT / "embedding_generator" / "output_ml1m" / ENC,
        configs=["r2", "r2_retuned", "r3"],
        # r2 = original-protocol (paper headline Table ml1m_cross_density, 0.1677); r2_retuned =
        # per-dataset retune (App. r2_retune, 0.1680; no-op +0.2% n.s.), checkpoint regenerated 2026-06.
        anchors={"r2": 0.1677, "r2_retuned": 0.1680, "r3": 0.1669},
    ),
    "ml20m_sub163": dict(  # subsampled-ML-20M control; reuses the base ML-20M content features
        data=B / "data" / "processed_ml20m_sub163",
        emb=EMBEDDING_DIR,
        configs=["r2", "r3"],
        anchors={"r2": 0.1021, "r3": 0.1034},
    ),
    "ml20m_gpt4omini": dict(  # cross-LLM GPT side (paper Table cross_llm_downstream / Table 8)
        data=B / "data" / "processed",  # SAME ML-20M interactions; only the profile FEATURES are GPT-4o-mini's
        emb=CODE_ROOT / "embedding_generator" / "output_ml20m_gpt4omini" / ENC,
        configs=["m4", "m7"],
        anchors={"m4": 0.1165, "m7": 0.1160},
    ),
}


def genome_raw_features(item_map: dict, device: str) -> torch.Tensor:
    """Reconstruct the 1128-dim raw genome tag-relevance matrix aligned to benchmark
    item ids (M2b). Mirrors scripts/run_cold_start_eval.py:_genome_raw_features."""
    import pandas as pd
    # config resolves ML-20M through its candidate list, which puts the
    # documented download_ml20m.sh target first. This used to hardcode the
    # generation-time path, which does not exist in a clone, so M2b's eval path
    # was as unreachable for a reader as its training path was.
    from config import GENOME_SCORES_CSV, GENOME_TAGS_CSV
    if not GENOME_SCORES_CSV.exists():
        raise FileNotFoundError(
            "M2b's raw genome features are built from the MovieLens genome scores,\n"
            f"which we do not redistribute:\n    {GENOME_SCORES_CSV}\n"
            "Fetch MovieLens first:\n        bash scripts/download_ml20m.sh")
    scores = pd.read_csv(GENOME_SCORES_CSV)
    tags = pd.read_csv(GENOME_TAGS_CSV)
    tag_idx = {tid: i for i, tid in enumerate(sorted(tags["tagId"].unique()))}
    feat = np.zeros((len(item_map), len(tags)), dtype=np.float32)
    # vectorized fill (equivalent to the per-row loop in run_cold_start_eval, ~100x faster)
    s = scores.copy()
    s["bench"] = s["movieId"].map(item_map)
    s["tcol"] = s["tagId"].map(tag_idx)
    s = s.dropna(subset=["bench"])
    feat[s["bench"].to_numpy(dtype=int), s["tcol"].to_numpy(dtype=int)] = s["relevance"].to_numpy(dtype=np.float32)
    return torch.from_numpy(feat).to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="", choices=list(DATASETS.keys()),
                    help="'' = base ML-20M; 'amazon' = Amazon-Books")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--encoder", default=ENC,
                    help="encoder subdir for ckpt/leaf paths + embedding dir (default bge-large-en-v1.5; "
                         "use e5-large-v2 for the cross-encoder sensitivity cell — base dataset, M3/M4/M7/M8)")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    sfx = args.dataset
    enc = args.encoder
    if enc != ENC:
        # Non-default encoder (e.g., e5-large-v2 cross-encoder sensitivity): ckpt / leaf
        # paths use `enc`, and features come from the sibling <encoder> output dir.
        # Anchors below are plot_sensitivity.py's e5_vals (verifies the figure numbers).
        E5_ANCHORS = {"m3": 0.1144, "m4": 0.1154, "m7": 0.1173, "m8": 0.1140}
        ds = {**ds, "emb": ds["emb"].parent / enc,
              "anchors": {**ds["anchors"], **(E5_ANCHORS if enc == "e5-large-v2" else {})}}
    configs = args.configs or ds["configs"]
    ckpt_root = B / ("checkpoints" if sfx == "" else f"checkpoints_{sfx}")
    res_root = B / ("results" if sfx == "" else f"results_{sfx}")

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cpu")
    logger.info(f"dataset={sfx or 'ML-20M'}  device={device}  data={ds['data'].name}  emb={ds['emb'].name}")

    # Seed-independent: load interaction data + graph ONCE.
    data = InteractionData(data_dir=ds["data"])
    norm_adj = data.get_norm_adj().to(device)
    loader = FeatureLoader(data_dir=ds["data"], embedding_dir=ds["emb"])
    feat_cache: dict[str, torch.Tensor] = {}
    item_map = {int(k): int(v) for k, v in json.load(open(ds["data"] / "item_map.json")).items()}

    overall_ok = True
    for cfg in configs:
        model_name, fkey = SPEC[cfg]
        feat_names = FEATURE_CONFIGS[fkey]
        ndcgs = []
        for seed in args.seeds:
            set_seed(seed)
            ckpt = ckpt_root / cfg / enc / f"seed-{seed}" / "best_model.pt"
            if not ckpt.exists():
                logger.error(f"  MISSING checkpoint: {ckpt}"); overall_ok = False; continue

            # Resolve item features (genome_raw is built from genome-scores.csv).
            feat_tensor, feat_dim = None, 0
            if model_name in FEAT_MODELS:
                if fkey == "genome_raw":
                    if "genome_raw" not in feat_cache:
                        feat_cache["genome_raw"] = genome_raw_features(item_map, device)
                    feat_tensor = feat_cache["genome_raw"]
                elif feat_names:
                    if fkey not in feat_cache:
                        if fkey == "bert_title":
                            # M3 uses the raw 1024-d bert_title (embedding_dir copy), bypassing
                            # the loader's flat 128-d PCA default (config.BERT_TITLE_EMB_NPY).
                            # See paper Table mood_compactness caption.
                            loader._feature_files["bert_title"] = ds["emb"] / "bert_title_embeddings.npy"
                        feat_cache[fkey] = loader.get_combined_tensor(feat_names, device=device)
                    feat_tensor = feat_cache[fkey]
                feat_dim = feat_tensor.shape[1] if feat_tensor is not None else 0

            if model_name == "lightgcl":
                # set_adj() eagerly materializes a dense SVD (OOM on ML-20M); predict()
                # needs only norm_adj, so build without set_adj and attach it directly.
                model = build_model(model_name, data.n_users, data.n_items, feat_dim, None)
                model.norm_adj = norm_adj
            elif model_name == "hypernet_replacer":
                # R3: not in run_experiment.build_model; same predict/set_features/set_adj
                # interface as LightGCN-SF. embed_dim=128, n_layers=3, hidden=256 (project defaults).
                from models.hypernet_replacer import HypernetReplacer
                model = HypernetReplacer(data.n_users, data.n_items, EMBED_DIM, LIGHTGCN_LAYERS, feat_dim, 256)
                model.set_adj(norm_adj)
            else:
                model = build_model(model_name, data.n_users, data.n_items, feat_dim,
                                    norm_adj if model_name in GNN else None)
            if model_name in FEAT_MODELS and feat_tensor is not None:
                model.set_features(feat_tensor)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.to(device)

            tm = evaluate_model(model, data, split="test", device=device)
            ndcgs.append(tm["NDCG@10"])
            out = res_root / cfg / enc / f"seed-{seed}" / "results.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "experiment": f"{cfg}/{enc}/seed-{seed}",
                "config": RLABEL.get(cfg, get_config_name(model_name, fkey)),
                "seed": seed,
                "test_metrics": {k: float(v) for k, v in tm.items()},
                "_provenance": "eval-only full-ranking re-evaluation of the released best_model.pt (no retraining).",
            }, indent=2) + "\n")
            logger.info(f"  {cfg} seed-{seed}: NDCG@10={tm['NDCG@10']:.4f} Recall@10={tm['Recall@10']:.4f} MRR={tm['MRR']:.4f}")
        if ndcgs:
            mean = float(np.mean(ndcgs)); anchor = ds["anchors"].get(cfg)
            flag = ""
            if anchor is not None and abs(mean - anchor) > 0.0008:
                flag = f"  ⚠ vs paper {anchor:.4f}"; overall_ok = False
            logger.info(f"  == {cfg}: 5-seed NDCG@10 mean={mean:.4f}"
                        f"{f' (paper {anchor:.4f})' if anchor else ''}{flag}")
    logger.info("DONE" + ("" if overall_ok else "  — with anchor mismatches/missing (investigate)"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
