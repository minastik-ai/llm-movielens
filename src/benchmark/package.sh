#!/bin/bash
# Package benchmark code + data for running on remote hardware.
#
# Creates: benchmark_package.tar.gz containing:
#   - All benchmark source code
#   - Processed data splits (train/val/test.csv + mappings)
#   - Pre-computed embeddings (.npy files)
#   - Existing results (for resume)
#   - Existing checkpoints (for resume)
#   - RLMRec code + prepared data
#
# Does NOT include:
#   - Raw ML-20M data (too large, re-download on target)
#   - Python environments (re-install on target)
#   - __pycache__ directories

set -e
cd "$(dirname "$0")/.."  # code/ directory

PACKAGE_NAME="benchmark_package"
OUTPUT="${PACKAGE_NAME}.tar.gz"

echo "=== Packaging benchmark for remote hardware ==="

# Create tarball
tar czf "$OUTPUT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='external/RLMRec/.git' \
    benchmark/ \
    embedding_generator/output/bge-large-en-v1.5/ \
    embedding_generator/config.py \
    embedding_generator/embedder.py \
    embedding_generator/main.py \
    embedding_generator/requirements.txt \
    embedding_generator/README.md

echo ""
echo "=== Package created: $(pwd)/$OUTPUT ==="
echo ""

# Show contents summary
echo "Contents:"
tar tzf "$OUTPUT" | head -5
echo "..."
tar tzf "$OUTPUT" | wc -l | xargs -I{} echo "{} files total"
echo ""
ls -lh "$OUTPUT" | awk '{print "Size: " $5}'

echo ""
echo "=== Transfer to remote hardware ==="
echo "  scp $(pwd)/$OUTPUT user@remote:~/"
echo ""
echo "=== On remote hardware ==="
echo "  tar xzf $OUTPUT"
echo "  cd benchmark"
echo "  pip install -r requirements.txt"
echo "  # Resume all experiments:"
echo "  bash run_quick.sh                    # BPR-MF + LightGCN-SF"
echo "  python run_ablation.py               # full ablation (5 seeds)"
echo "  cd external && bash run_rlmrec.sh    # RLMRec experiments"
