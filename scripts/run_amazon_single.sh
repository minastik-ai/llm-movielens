#!/usr/bin/env bash
# Run a single Amazon-Books Tier-1 experiment (M1, M4, or M7) with one seed.
#
# Usage:
#   bash scripts/run_amazon_single.sh <CONFIG> [SEED]
#
#   <CONFIG>  one of: M1, M4, M7  (case-insensitive)
#   [SEED]    integer seed, default 42
#
# Examples:
#   bash scripts/run_amazon_single.sh M4              # M4, seed 42
#   bash scripts/run_amazon_single.sh m7 123          # M7, seed 123
#   GPU=0 bash scripts/run_amazon_single.sh M1 456    # pin to GPU 0
#
# Outputs:
#   code/benchmark/results_amazon/<cfg>/bge-large-en-v1.5/seed-<SEED>/results.json
#   code/benchmark/checkpoints_amazon/<cfg>/bge-large-en-v1.5/seed-<SEED>/best.pt
#   code/benchmark/results_amazon/logs/<cfg>-seed<SEED>.log
#
# Idempotent: if results.json already exists, this script exits without
# re-training. Delete the results dir to force re-run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <M1|M4|M7> [SEED]" >&2
    exit 1
fi

CFG_RAW="$1"
SEED="${2:-42}"
CFG_UC=$(echo "$CFG_RAW" | tr '[:lower:]' '[:upper:]')
CFG_LC=$(echo "$CFG_RAW" | tr '[:upper:]' '[:lower:]')

case "$CFG_UC" in
    M1) MODEL="lightgcn"     ; FEATURES="none"          ;;
    M4) MODEL="lightgcn_sf"  ; FEATURES="llm_profile"   ;;
    M7) MODEL="lightgcn_sf"  ; FEATURES="llm_prof_mood" ;;
    *)  echo "ERROR: config must be M1, M4, or M7 (got '$CFG_RAW')" >&2; exit 1 ;;
esac

DATA_DIR="code/benchmark/data/processed_amazon"
EMB_DIR="code/embedding_generator/output_amazon/bge-large-en-v1.5"
RESULTS_DIR="code/benchmark/results_amazon"
CKPT_DIR="code/benchmark/checkpoints_amazon"
LOG_DIR="$RESULTS_DIR/logs"
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
    "$EMB_DIR/movie_id_index.json"
do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required input missing: $required" >&2
        echo "  Run Phase 0 (splits) and Phase 2 (embeddings) first." >&2
        exit 2
    fi
done
if [[ "$CFG_UC" == "M7" && ! -f "$EMB_DIR/combined_features.npy" ]]; then
    echo "ERROR: $EMB_DIR/combined_features.npy missing (M7 needs profile+mood)" >&2
    exit 2
fi

OUT="$RESULTS_DIR/$CFG_LC/bge-large-en-v1.5/seed-$SEED/results.json"
LOG="$LOG_DIR/${CFG_LC}-seed${SEED}.log"

if [[ -f "$OUT" ]]; then
    echo "[skip] $CFG_UC seed=$SEED already complete: $OUT"
    exit 0
fi

echo "═══ Amazon-Books Single Run ═══"
echo "  Config   : $CFG_UC ($MODEL + $FEATURES)"
echo "  Seed     : $SEED"
echo "  Data     : $DATA_DIR"
echo "  Emb      : $EMB_DIR"
echo "  Results  : $OUT"
echo "  Log      : $LOG"
echo ""

python3 code/benchmark/run_experiment.py \
    --model "$MODEL" \
    --features "$FEATURES" \
    --seed "$SEED" \
    --data-dir "$DATA_DIR" \
    --embedding-dir "$EMB_DIR" \
    --results-dir "$RESULTS_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    2>&1 | tee "$LOG"

echo ""
echo "═══ Done: $CFG_UC seed=$SEED ═══"
echo "Top-line metrics:"
python3 - <<EOF
import json
from pathlib import Path
p = Path("$OUT")
if not p.exists():
    print("  (results.json not found — see log)")
    exit(1)
r = json.loads(p.read_text())
test = r.get("test_metrics") or r.get("best_val_metrics") or {}
for k in ("NDCG@10", "Recall@10", "MRR", "HR@10"):
    if k in test:
        print(f"  {k:10s} {test[k]:.4f}")
print(f"  best_epoch  {r.get('best_epoch')}")
EOF
