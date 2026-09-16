#!/usr/bin/env bash
# Download MovieLens 20M dataset
# Usage: bash scripts/download_ml20m.sh [--target-dir data/raw]

set -euo pipefail

TARGET_DIR="${1:-data/raw}"
ML20M_URL="https://files.grouplens.org/datasets/movielens/ml-20m.zip"
ZIP_FILE="${TARGET_DIR}/ml-20m.zip"

echo "=== Downloading MovieLens 20M ==="
echo "Target: ${TARGET_DIR}/ml-20m/"

mkdir -p "${TARGET_DIR}"

if [ -d "${TARGET_DIR}/ml-20m" ] && [ -f "${TARGET_DIR}/ml-20m/ratings.csv" ]; then
    echo "MovieLens 20M already exists at ${TARGET_DIR}/ml-20m/. Skipping download."
    exit 0
fi

echo "Downloading from ${ML20M_URL} ..."
if command -v wget &> /dev/null; then
    wget -q --show-progress -O "${ZIP_FILE}" "${ML20M_URL}"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o "${ZIP_FILE}" "${ML20M_URL}"
else
    echo "Error: Neither wget nor curl found. Please install one." >&2
    exit 1
fi

echo "Extracting..."
unzip -q -o "${ZIP_FILE}" -d "${TARGET_DIR}"
rm -f "${ZIP_FILE}"

echo "Done. Files at ${TARGET_DIR}/ml-20m/"
ls -lh "${TARGET_DIR}/ml-20m/"
