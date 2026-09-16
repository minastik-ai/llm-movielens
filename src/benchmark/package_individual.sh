#!/bin/bash
# Package each experiment individually for parallel execution on multiple machines.
#
# Creates: packages/exp_M0.tar.gz, packages/exp_M2.tar.gz, ..., packages/exp_R1.tar.gz
# Each package is self-contained with only the data/embeddings it needs.
#
# Usage:
#   bash package_individual.sh          # package all experiments
#   bash package_individual.sh M4       # package only M4

set -e
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$BENCH_DIR")"
EMB_DIR="$CODE_DIR/embedding_generator/output/bge-large-en-v1.5"
PKG_DIR="$BENCH_DIR/packages"

mkdir -p "$PKG_DIR"

# Temporary staging area
STAGE="/tmp/benchmark_staging"

package_experiment() {
    local label=$1 model=$2 features=$3 seed=$4 epochs=$5
    local pkg_name="exp_${label}"
    local stage="$STAGE/$pkg_name"

    echo "Packaging $pkg_name (model=$model, features=$features)..."

    rm -rf "$stage"
    mkdir -p "$stage/benchmark/data/processed"
    mkdir -p "$stage/benchmark/features"
    mkdir -p "$stage/benchmark/models"
    mkdir -p "$stage/benchmark/data"
    mkdir -p "$stage/embeddings"

    # Core code
    cp "$BENCH_DIR/config.py" "$stage/benchmark/"
    cp "$BENCH_DIR/train.py" "$stage/benchmark/"
    cp "$BENCH_DIR/evaluate.py" "$stage/benchmark/"
    cp "$BENCH_DIR/run_experiment.py" "$stage/benchmark/"
    cp "$BENCH_DIR/requirements.txt" "$stage/benchmark/"

    # Data code
    cp "$BENCH_DIR/data/__init__.py" "$stage/benchmark/data/"
    cp "$BENCH_DIR/data/dataset.py" "$stage/benchmark/data/"

    # Feature code
    cp "$BENCH_DIR/features/__init__.py" "$stage/benchmark/features/"
    cp "$BENCH_DIR/features/loader.py" "$stage/benchmark/features/"

    # Model code
    cp "$BENCH_DIR/models/__init__.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/bpr_mf.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/lightgcn.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/simgcl.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/xsimgcl.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/lightgcl.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/kar.py" "$stage/benchmark/models/"
    cp "$BENCH_DIR/models/sasrec.py" "$stage/benchmark/models/"

    # Processed data (shared by all)
    cp "$BENCH_DIR/data/processed/"*.csv "$stage/benchmark/data/processed/"
    cp "$BENCH_DIR/data/processed/"*.json "$stage/benchmark/data/processed/"

    # Movie ID index (needed by feature loader)
    cp "$EMB_DIR/movie_id_index.json" "$stage/embeddings/"

    # Only copy the embeddings this experiment needs
    case "$features" in
        none)
            ;; # no embeddings needed
        genome)
            cp "$EMB_DIR/genome_embeddings.npy" "$stage/embeddings/"
            ;;
        bert_title)
            cp "$EMB_DIR/bert_title_embeddings.npy" "$stage/embeddings/"
            ;;
        llm_profile)
            cp "$EMB_DIR/profile_embeddings.npy" "$stage/embeddings/"
            ;;
        llm_mood)
            cp "$EMB_DIR/mood_vectors.npy" "$stage/embeddings/"
            ;;
        llm_themes)
            cp "$EMB_DIR/theme_matrix.npy" "$stage/embeddings/"
            ;;
        llm_prof_mood)
            cp "$EMB_DIR/profile_embeddings.npy" "$stage/embeddings/"
            cp "$EMB_DIR/mood_vectors.npy" "$stage/embeddings/"
            ;;
        llm_all)
            cp "$EMB_DIR/profile_embeddings.npy" "$stage/embeddings/"
            cp "$EMB_DIR/mood_vectors.npy" "$stage/embeddings/"
            cp "$EMB_DIR/theme_matrix.npy" "$stage/embeddings/"
            ;;
        genome_llm)
            cp "$EMB_DIR/genome_embeddings.npy" "$stage/embeddings/"
            cp "$EMB_DIR/mood_vectors.npy" "$stage/embeddings/"
            cp "$EMB_DIR/theme_matrix.npy" "$stage/embeddings/"
            ;;
    esac

    # Copy existing results/checkpoints for resume
    local exp_name="${model}__${features}__seed${seed}"
    if [ -d "$BENCH_DIR/results/$exp_name" ]; then
        mkdir -p "$stage/benchmark/results/$exp_name"
        cp -r "$BENCH_DIR/results/$exp_name/"* "$stage/benchmark/results/$exp_name/"
    fi
    if [ -d "$BENCH_DIR/checkpoints/$exp_name" ]; then
        mkdir -p "$stage/benchmark/checkpoints/$exp_name"
        cp -r "$BENCH_DIR/checkpoints/$exp_name/"* "$stage/benchmark/checkpoints/$exp_name/"
    fi

    # Create run script
    cat > "$stage/run.sh" << RUNEOF
#!/bin/bash
# Run experiment: $label ($model + $features)
# Resume-safe: re-run after interruption to continue
set -e
cd "\$(dirname "\$0")/benchmark"

# Point embeddings to local copy
export EMBEDDING_DIR="\$(cd ../embeddings && pwd)"

python3 run_experiment.py \\
    --model $model \\
    --features $features \\
    --seed $seed \\
    --epochs $epochs \\
    --device auto

echo ""
echo "=== Results ==="
cat results/${exp_name}/results.json 2>/dev/null || echo "No results yet"
RUNEOF
    chmod +x "$stage/run.sh"

    # Create setup script
    cat > "$stage/setup.sh" << SETUPEOF
#!/bin/bash
# One-time setup on remote hardware
set -e
pip install -r benchmark/requirements.txt
echo "Setup complete. Run: bash run.sh"
SETUPEOF
    chmod +x "$stage/setup.sh"

    # Create README
    cat > "$stage/README.txt" << READMEEOF
Experiment: $label
Model: $model
Features: $features
Seed: $seed
Epochs: $epochs

Setup:   bash setup.sh
Run:     bash run.sh
Results: benchmark/results/${exp_name}/results.json

Resume: safe to re-run after interruption.
READMEEOF

    # Tar it up
    cd "$STAGE"
    tar czf "$PKG_DIR/${pkg_name}.tar.gz" "$pkg_name/"
    local size=$(ls -lh "$PKG_DIR/${pkg_name}.tar.gz" | awk '{print $5}')
    echo "  → $PKG_DIR/${pkg_name}.tar.gz ($size)"

    rm -rf "$stage"
}

mkdir -p "$STAGE"

FILTER="${1:-all}"

echo "=== Packaging Individual Experiments ==="
echo ""

# Determine which to package
should_package() {
    [ "$FILTER" = "all" ] || [ "$FILTER" = "$1" ]
}

# M0: BPR-MF
should_package M0 && package_experiment M0 bpr_mf none 42 10

# M2-M9: LightGCN-SF variants
should_package M2 && package_experiment M2 lightgcn_sf genome 42 10
should_package M3 && package_experiment M3 lightgcn_sf bert_title 42 10
should_package M4 && package_experiment M4 lightgcn_sf llm_profile 42 10
should_package M5 && package_experiment M5 lightgcn_sf llm_mood 42 10
should_package M6 && package_experiment M6 lightgcn_sf llm_themes 42 10
should_package M7 && package_experiment M7 lightgcn_sf llm_prof_mood 42 10
should_package M8 && package_experiment M8 lightgcn_sf llm_all 42 10
should_package M9 && package_experiment M9 lightgcn_sf genome_llm 42 10

# GNN baselines (for CUDA machines)
should_package M1  && package_experiment M1  lightgcn none 42 10
should_package M1b && package_experiment M1b simgcl none 42 10
should_package M1c && package_experiment M1c xsimgcl none 42 10
should_package M1d && package_experiment M1d lightgcl none 42 10

# Tier 3: LLM-for-RecSys methods
should_package R3  && package_experiment R3  kar llm_prof_mood 42 10

echo ""
echo "=== All packages created in $PKG_DIR/ ==="
echo ""
ls -lhS "$PKG_DIR/"*.tar.gz 2>/dev/null
echo ""
echo "=== Usage on remote hardware ==="
echo "  scp packages/exp_M4.tar.gz user@gpu-server:~/"
echo "  ssh user@gpu-server"
echo "  tar xzf exp_M4.tar.gz && cd exp_M4"
echo "  bash setup.sh && bash run.sh"
