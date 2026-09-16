#!/usr/bin/env bash
# Phase 3 — Amazon-Books Tier-1 benchmark (cross-domain).
#
# Runs the minimal Tier-1 ablation: M1 (LightGCN ID-only), M4 (+ LLM
# profile), M7 (+ LLM profile + mood). 5 seeds each = 15 runs total.
#
# Each run is idempotent: results land at
#   code/benchmark/results_amazon/{m1,m4,m7}/bge-large-en-v1.5/seed-<seed>/
# and an existing results.json is *not* overwritten by run_experiment.py
# (re-running a completed config is a no-op aside from re-loading data).
# To force re-run, delete the corresponding results directory.
#
# Inputs:
#   code/benchmark/data/processed_amazon/                 (Phase 0 output)
#   code/embedding_generator/output_amazon/bge-large-en-v1.5/  (Phase 2 output)
#
# Outputs:
#   code/benchmark/results_amazon/{m1,m4,m7}/bge-large-en-v1.5/seed-<seed>/results.json
#   code/benchmark/checkpoints_amazon/{...}/best.pt
#
# Runtime estimate (single GPU, n=120k train interactions, ~5-10 min/run):
#   15 runs × ~7 min ≈ 1.5-2 hours
#
# Usage:
#   bash scripts/run_amazon_phase3_tier1.sh                   # all 5 seeds
#   bash scripts/run_amazon_phase3_tier1.sh "42 123"          # subset of seeds
#   GPU=1 bash scripts/run_amazon_phase3_tier1.sh             # pin to GPU 1
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SEEDS="${1:-42 123 456 789 2026}"
DATA_DIR="code/benchmark/data/processed_amazon"
EMB_DIR="code/embedding_generator/output_amazon/bge-large-en-v1.5"
RESULTS_DIR="code/benchmark/results_amazon"
CKPT_DIR="code/benchmark/checkpoints_amazon"
LOG_DIR="code/benchmark/results_amazon/logs"
mkdir -p "$LOG_DIR"

if [[ -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "Pinning to GPU $GPU"
fi

# Sanity checks
for required in \
    "$DATA_DIR/train.csv" \
    "$DATA_DIR/item_map.json" \
    "$EMB_DIR/profile_embeddings.npy" \
    "$EMB_DIR/combined_features.npy" \
    "$EMB_DIR/movie_id_index.json"
do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required input missing: $required" >&2
        exit 1
    fi
done

# (model, features, label) — Tier 1 minimal cross-domain ablation
declare -a CONFIGS=(
    "lightgcn      none           M1"
    "lightgcn_sf   llm_profile    M4"
    "lightgcn_sf   llm_prof_mood  M7"
)

echo "═══ Amazon-Books Phase 3: Tier-1 Benchmark ═══"
echo "  Seeds      : $SEEDS"
echo "  Data dir   : $DATA_DIR"
echo "  Emb dir    : $EMB_DIR"
echo "  Results    : $RESULTS_DIR"
echo "  Configs    : M1, M4, M7"
echo ""

t0=$(date +%s)
n_runs=0
n_skip=0
for seed in $SEEDS; do
    for entry in "${CONFIGS[@]}"; do
        read -r MODEL FEATURES LABEL <<<"$entry"
        cfg_lc=$(echo "$LABEL" | tr '[:upper:]' '[:lower:]')
        out="$RESULTS_DIR/$cfg_lc/bge-large-en-v1.5/seed-$seed/results.json"
        log="$LOG_DIR/${cfg_lc}-seed${seed}.log"

        if [[ -f "$out" ]]; then
            echo "  [skip] $LABEL seed=$seed (already done)"
            n_skip=$((n_skip+1))
            continue
        fi

        echo "  [run ] $LABEL seed=$seed → $log"
        python3 code/benchmark/run_experiment.py \
            --model "$MODEL" \
            --features "$FEATURES" \
            --seed "$seed" \
            --data-dir "$DATA_DIR" \
            --embedding-dir "$EMB_DIR" \
            --results-dir "$RESULTS_DIR" \
            --checkpoint-dir "$CKPT_DIR" \
            >"$log" 2>&1 || {
                echo "    FAILED — see $log" >&2
                exit 2
            }
        n_runs=$((n_runs+1))
    done
done

elapsed=$(( $(date +%s) - t0 ))
echo ""
echo "═══ Phase 3 complete ═══"
echo "  Runs executed : $n_runs"
echo "  Runs skipped  : $n_skip"
echo "  Elapsed       : ${elapsed}s ($((elapsed/60)) min)"
echo ""
echo "Aggregating means/stds across seeds..."
python3 scripts/aggregate_amazon_tier1.py "$RESULTS_DIR"
