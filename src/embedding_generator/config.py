"""
Configuration for Movie Embedding Generator.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
# The generator sits at a different depth in the published repository than in the
# development checkout, so find it by SHAPE rather than by a fixed relative path.
def _find_generator(start):
    for base in [start, *start.parents]:
        for pat in ("profile_generator/llm-movie-profiler-*", "profile_generator",
                    "src/profile_generator"):
            hits = [h for h in sorted(base.glob(pat)) if (h / "config").is_dir() or (h / "main.py").exists()]
            if hits:
                return hits[0]
    return start.parent / "profile_generator"


PROFILE_GENERATOR_ROOT = _find_generator(PROJECT_ROOT)

# Input
PROFILES_JSON = PROFILE_GENERATOR_ROOT / "output" / "movie_profiles.json"
# Same resolution order as the generator and the benchmark: the
# generation layout, then what scripts/download_ml20m.sh creates, then a reader
# who passed --target-dir data, with ML20M_DIR overriding all three.
_ML20M = [PROJECT_ROOT.parent.parent / "data" / "raw" / "ml-20m",  # download_ml20m.sh
          PROJECT_ROOT.parent.parent / "data" / "ml-20m",          # --target-dir data
          PROFILE_GENERATOR_ROOT / "data" / "ml-20m"]              # generation layout
ML20M_DIR = Path(os.environ["ML20M_DIR"]) if os.environ.get("ML20M_DIR") else next(
    (c for c in _ML20M if (c / "genome-scores.csv").exists()), _ML20M[0])
GENOME_SCORES_CSV = ML20M_DIR / "genome-scores.csv"
GENOME_TAGS_CSV = ML20M_DIR / "genome-tags.csv"

# Output
OUTPUT_DIR = PROJECT_ROOT / "output" / "bge-large-en-v1.5"

# ──────────────────────────────────────────────────────────────────────────────
# SENTENCE TRANSFORMER MODELS
# ──────────────────────────────────────────────────────────────────────────────
# Model options (trade-off: quality vs speed vs dimensions):
#   Strong (recommended):
#     - "BAAI/bge-large-en-v1.5":          1024-dim, 335M params, MTEB ~64 (default)
#     - "intfloat/e5-large-v2":            1024-dim, 335M params, MTEB ~62
#     - "nomic-ai/nomic-embed-text-v1.5":  768-dim,  137M params, MTEB ~62 (Matryoshka)
#     - "Alibaba-NLP/gte-large-en-v1.5":   1024-dim, 434M params, MTEB ~65
#   Lightweight (fast, lower quality):
#     - "all-MiniLM-L6-v2":                384-dim, 22M params,  MTEB ~56
#     - "all-MiniLM-L12-v2":               384-dim, 33M params,  MTEB ~57
#     - "all-mpnet-base-v2":               768-dim, 110M params, MTEB ~58
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

# Model registry with native dimensions (for reference/validation)
MODEL_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "Alibaba-NLP/gte-large-en-v1.5": 1024,
}

# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
# PCA target dimensions for profile embeddings (None = no reduction, keep native dims)
# Use --pca-dims 128 or --pca-dims 256 to enable PCA reduction
PCA_DIMS = None

# Batch size for sentence-transformer encoding
ENCODE_BATCH_SIZE = 256

# ──────────────────────────────────────────────────────────────────────────────
# MOOD VECTOR AXES (must match profile_generator config)
# ──────────────────────────────────────────────────────────────────────────────
MOOD_AXES = [
    "dark_light",
    "serious_playful",
    "slow_fast",
    "cerebral_visceral",
    "realistic_fantastical",
    "intimate_epic",
    "conventional_experimental",
    "emotional_detached",
    "nostalgic_contemporary",
    "predictable_subversive",
]

# ──────────────────────────────────────────────────────────────────────────────
# GENOME TAG SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
GENOME_PCA_DIMS = 128  # PCA target for 1128-dim genome vectors

# ──────────────────────────────────────────────────────────────────────────────
# THEME FILTERING
# ──────────────────────────────────────────────────────────────────────────────
THEME_MIN_COUNT = 10  # Only keep themes appearing in >= 10 movies
