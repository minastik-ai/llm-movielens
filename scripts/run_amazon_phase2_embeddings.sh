#!/usr/bin/env bash
# Phase 2 — Amazon-Books embedding generation (cross-domain).
#
# Inputs (must exist):
#   code/profile_generator/output_amazon/book_profiles.json   (Phase 1 output)
#
# Outputs:
#   code/embedding_generator/output_amazon/bge-large-en-v1.5/
#     profile_embeddings.npy   (9332, 1024)
#     mood_vectors.npy         (9332, 10)
#     theme_matrix.npy         (9332, ?)         ← book-domain themes
#     combined_features.npy    (9332, 1034)      ← profile + mood (M7 input)
#     combined_full.npy        (9332, ?)
#     movie_id_index.json      [0, 1, ..., 9331] ← bench item-id list
#     theme_vocabulary.json
#     embedding_metadata.json
#
# Notes:
#   - --skip-genome: Amazon-Books has no genome tag scores. Tier-1
#     ablation (M1, M4, M7) does not need a genome baseline.
#   - The benchmark FeatureLoader aligns embeddings to bench item IDs via
#     item_map.json (identity for Amazon) + movie_id_index.json. Both
#     sides use the RLMRec iid 0..9331 directly, so no extra mapping is
#     needed.
#   - Runtime: ~2-3 min on a single GPU (9332 profiles × bge-large).
#
# Usage:
#   bash scripts/run_amazon_phase2_embeddings.sh
#   bash scripts/run_amazon_phase2_embeddings.sh /custom/profiles.json /custom/out_dir
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROFILES="${1:-code/profile_generator/output_amazon/book_profiles.json}"
OUT_DIR="${2:-code/embedding_generator/output_amazon/bge-large-en-v1.5}"

if [[ ! -f "$PROFILES" ]]; then
    echo "ERROR: profiles not found: $PROFILES" >&2
    echo "       Run Phase 1 first: python scripts/generate_book_profiles.py --resume" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "═══ Amazon-Books Phase 2: Embedding Generation ═══"
echo "  Profiles : $PROFILES"
echo "  Output   : $OUT_DIR"
echo "  Encoder  : BAAI/bge-large-en-v1.5 (1024-dim, no PCA)"

python3 code/embedding_generator/main.py \
    --profiles "$PROFILES" \
    --output-dir "$OUT_DIR" \
    --skip-genome \
    --model "BAAI/bge-large-en-v1.5"

echo ""
echo "═══ Phase 2 complete ═══"
echo "Verification:"
python3 - <<EOF
import json, numpy as np
from pathlib import Path
out = Path("$OUT_DIR")
prof = np.load(out / "profile_embeddings.npy")
mood = np.load(out / "mood_vectors.npy")
combo = np.load(out / "combined_features.npy")
ids = json.loads((out / "movie_id_index.json").read_text())
meta = json.loads((out / "embedding_metadata.json").read_text())
print(f"  profile_embeddings : {prof.shape}  dtype={prof.dtype}")
print(f"  mood_vectors       : {mood.shape}  dtype={mood.dtype}")
print(f"  combined_features  : {combo.shape}  (M7 input dim)")
print(f"  movie_id_index     : {len(ids)} ids, range [{min(ids)}..{max(ids)}]")
print(f"  encoder            : {meta['sentence_model']}")
assert prof.shape[0] == mood.shape[0] == combo.shape[0] == len(ids), \
    "row-count mismatch across artifacts"
assert combo.shape[1] == prof.shape[1] + mood.shape[1], "M7 dim mismatch"
print("  ✓ shapes consistent")
EOF
