#!/bin/bash
# Quick ablation: BPR-MF + LightGCN-SF only
#
# Configurable via environment variables:
#   EPOCHS          — number of training epochs (default: 10)
#   SEED            — random seed (default: 42)
#   EMBEDDING_DIR   — path to embedding directory (default: from config.py)
#   DATA_DIR        — path to processed data directory (default: from config.py)
#   RESULTS_DIR     — path to results directory (default: from config.py)
#   CHECKPOINT_DIR  — path to checkpoint directory (default: from config.py)
#
# Resume support:
#   - Interrupted experiments resume from last saved epoch
#   - Completed experiments with enough epochs are skipped
#   - To extend training: increase EPOCHS and re-run
#   - Safe to re-run after interruption
set -e
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-10}
SEED=${SEED:-42}

# Build extra args from env vars
EXTRA_ARGS=""
[ -n "$EMBEDDING_DIR" ]  && EXTRA_ARGS="$EXTRA_ARGS --embedding-dir $EMBEDDING_DIR"
[ -n "$DATA_DIR" ]       && EXTRA_ARGS="$EXTRA_ARGS --data-dir $DATA_DIR"
[ -n "$RESULTS_DIR" ]    && EXTRA_ARGS="$EXTRA_ARGS --results-dir $RESULTS_DIR"
[ -n "$CHECKPOINT_DIR" ] && EXTRA_ARGS="$EXTRA_ARGS --checkpoint-dir $CHECKPOINT_DIR"

echo "=== Quick Ablation: BPR-MF + LightGCN-SF (${EPOCHS} epochs, seed=${SEED}) ==="
echo "    Resume mode: will continue from last checkpoint"
[ -n "$EMBEDDING_DIR" ]  && echo "    Embedding dir: $EMBEDDING_DIR"
[ -n "$RESULTS_DIR" ]    && echo "    Results dir: $RESULTS_DIR"
[ -n "$CHECKPOINT_DIR" ] && echo "    Checkpoint dir: $CHECKPOINT_DIR"
echo ""

run_exp() {
    local idx=$1 label=$2 model=$3 features=$4
    echo "[${idx}/9] ${label}"
    python3 run_experiment.py --model "$model" --features "$features" --seed $SEED --epochs $EPOCHS $EXTRA_ARGS
}

# M0: BPR-MF baseline
run_exp 1 "M0: BPR-MF (ID only)" bpr_mf none

# M2-M9: LightGCN-SF with different features
run_exp 2 "M2: + genome PCA(128)" lightgcn_sf genome
run_exp 3 "M3: + BERT title(128)" lightgcn_sf bert_title
run_exp 4 "M4: + LLM profile(128)" lightgcn_sf llm_profile
run_exp 5 "M5: + LLM mood(10)" lightgcn_sf llm_mood
run_exp 6 "M6: + LLM themes(528)" lightgcn_sf llm_themes
run_exp 7 "M7: + LLM profile+mood(138)" lightgcn_sf llm_prof_mood
run_exp 8 "M8: + LLM all(666)" lightgcn_sf llm_all
run_exp 9 "M9: + genome+mood+themes" lightgcn_sf genome_llm

echo ""
echo "=== Collecting results ==="
python3 run_ablation.py --collect-only $EXTRA_ARGS

echo "=== Done ==="
