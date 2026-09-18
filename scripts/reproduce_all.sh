#!/usr/bin/env bash
# Full reproduction script for LLM-MovieLens benchmark
# Runs all 14 configurations × 5 seeds
#
# Usage:
#   bash scripts/reproduce_all.sh                    # Full run
#   bash scripts/reproduce_all.sh --tier 2           # Only Tier 2
#   bash scripts/reproduce_all.sh --config M4        # Single config
#   bash scripts/reproduce_all.sh --dry-run          # Print commands only

set -euo pipefail

SEEDS=(42 123 456 789 2026)
TIER1=(M0 M1 M1b M1c M1d)
TIER2=(M2 M2b M3 M4 M5 M6 M7 M8 M9)   # M2b = raw 1,128-d genome, the dimensionality control
# No tier 3 here. R2 and R3 are replacer-class configurations that neither the
# resource paper nor its fuller version reports, and their per-seed results are
# no longer staged, so running them would produce numbers no table accounts for.

TIER_FILTER=""
CONFIG_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier) TIER_FILTER="$2"; shift 2 ;;
        --config) CONFIG_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Select configs based on filters
CONFIGS=()
if [ -n "$CONFIG_FILTER" ]; then
    CONFIGS=("$CONFIG_FILTER")
elif [ -n "$TIER_FILTER" ]; then
    case $TIER_FILTER in
        1) CONFIGS=("${TIER1[@]}") ;;
        2) CONFIGS=("${TIER2[@]}") ;;
        3) echo "There is no tier 3 in this release: the R2/R3 replacer configurations" >&2
           echo "are not reported here and their per-seed results are not staged." >&2
           exit 1 ;;
        *) echo "Invalid tier: $TIER_FILTER (use 1 or 2)" >&2; exit 1 ;;
    esac
else
    CONFIGS=("${TIER1[@]}" "${TIER2[@]}")
fi

TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
CURRENT=0
SKIPPED=0
FAILED=0

echo "=== LLM-MovieLens Benchmark Reproduction ==="
echo "Configs: ${CONFIGS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total experiments: ${TOTAL}"
echo ""

# Resolve the benchmark package relative to this script, so the command works
# from any working directory.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/benchmark" 2>/dev/null && pwd)"
if [ -z "$BENCH_DIR" ]; then
  BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../code/benchmark" && pwd)"
fi

# Preflight. Without this the run reaches the first experiment, raises a bare
# FileNotFoundError deep in the loader, and repeats that traceback 70 times -- once
# per experiment -- because the loop treats every failure as a per-experiment error.
# Say what is missing, once, before starting.
# Skipped for --dry-run, which must work with no data at all -- that is its purpose,
# and gating it behind the data check made it print nothing.
DATA_DIR="${DATA_DIR:-$BENCH_DIR/data/processed}"
if [ "$DRY_RUN" != true ] && [ ! -f "$DATA_DIR/train.csv" ]; then
  echo "Missing the processed splits: $DATA_DIR/train.csv" >&2
  echo "" >&2
  echo "The MovieLens source is not redistributed with this repository. Build the" >&2
  echo "splits from your own download first:" >&2
  echo "" >&2
  echo "  bash scripts/download_ml20m.sh" >&2
  echo "  python3 scripts/download_artifacts.py      # the profiles set the item universe" >&2
  echo "  python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m" >&2
  echo "" >&2
  echo "Then re-run this script. Use --dry-run to list the 70 experiments without" >&2
  echo "needing any data." >&2
  exit 1
fi

# The same argument for the features. Without this the run reaches the first
# experiment and dies on a bare FileNotFoundError inside the feature loader, once
# per experiment. Ask the benchmark itself where it will look, so this cannot
# disagree with config.py the way the data path once did.
if [ "$DRY_RUN" != true ]; then
  EMB_DIR="$(cd "$BENCH_DIR" && python3 -c 'import config; print(config.EMBEDDING_DIR)' 2>/dev/null || true)"
  if [ -n "$EMB_DIR" ] && [ ! -f "$EMB_DIR/movie_id_index.json" ]; then
    echo "Missing the encoded features: $EMB_DIR/movie_id_index.json" >&2
    echo "" >&2
    echo "Either fetch the ones we released (~190 MB, no GPU, no API key):" >&2
    echo "" >&2
    echo "  python3 scripts/download_artifacts.py" >&2
    echo "" >&2
    echo "or encode your own profiles first:" >&2
    echo "" >&2
    echo "  python3 src/embedding_generator/main.py" >&2
    echo "" >&2
    echo "Set EMBEDDING_DIR to use a different encoder's output." >&2
    exit 1
  fi
fi

START_TIME=$(date +%s)

# M2b is the raw 1,128-d genome dimensionality control. Its features are built
# from genome-scores.csv by the EVAL path (eval_checkpoints.py); the training
# loader never implemented `genome_raw`, so every M2b run here failed with
# "Unknown feature" and counted as five failures in the summary. Skip it
# explicitly and say why, rather than spending five crashes to say the same.
SKIP_TRAINING="M2b"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        if [[ " ${SKIP_TRAINING} " == *" ${config} "* ]]; then
            if [ "$seed" = "${SEEDS[0]}" ]; then
                echo "[${CURRENT}/${TOTAL}] ${config} -- SKIPPED, not trainable from this script"
                echo "  ${config} is the raw 1,128-d genome control. Its features come from"
                echo "  genome-scores.csv through the evaluation path, not the training loader:"
                echo "      python3 ${BENCH_DIR}/eval_checkpoints.py"
                # ${x,,} is bash 4; macOS ships bash 3.2 and answers "bad
                # substitution" at RUNTIME, which `bash -n` does not catch.
                echo "  Its per-seed results ship in results/$(echo "$config" | tr 'A-Z' 'a-z')/."
            fi
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        # Run the entry point directly. The previous form invoked
        # `python -m llm_movielens.benchmark.run_experiment`, a package that does
        # not exist in the release, so every one of the 70 experiments failed with
        # ModuleNotFoundError. run_experiment.py imports its siblings flatly
        # (`from data.dataset import ...`), so it is invoked from its own directory.
        CMD="python3 ${BENCH_DIR}/run_experiment.py --config ${config} --seed ${seed}"

        echo "[${CURRENT}/${TOTAL}] ${config} seed=${seed}"

        if [ "$DRY_RUN" = true ]; then
            echo "  (dry run) ${CMD}"
        else
            if $CMD; then
                echo "  Done."
            else
                echo "  FAILED!" >&2
                FAILED=$((FAILED + 1))
            fi
        fi
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=== Summary ==="
if [ "$DRY_RUN" = true ]; then
    # Nothing was executed, so reporting runs as completed would be a false
    # statement in the output of the command the README tells a reader to run first.
    echo "Dry run: listed ${TOTAL} experiments, ran none."
else
    echo "Completed: $((CURRENT - FAILED - SKIPPED))/${TOTAL}"
    echo "Failed: ${FAILED}"
    [ "$SKIPPED" -gt 0 ] && echo "Skipped (not trainable here): ${SKIPPED}"
    echo "Time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
fi

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
