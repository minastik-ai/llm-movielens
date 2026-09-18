"""
Benchmark configuration.
All paths, hyperparameters, and experiment configs in one place.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CODE_ROOT = PROJECT_ROOT.parent

# ML-20M raw data. Three locations were in play and none of them agreed:
# scripts/download_ml20m.sh unzips to <repo>/data/raw/ml-20m, docs/REPRODUCIBILITY.md
# passed data/ml-20m, and this constant pointed inside the vendored generator
# directory -- a path nothing in the documented sequence ever creates, so
# features/bert_baseline.py failed for any reader who followed the instructions.
# Resolve in the order a reader will actually have it, env var first.
PROFILE_GEN_ROOT = CODE_ROOT / "profile_generator" / "llm-movie-profiler-v1-20260402"
REPO_ROOT = CODE_ROOT.parent
_ML20M_CANDIDATES = [
    REPO_ROOT / "data" / "raw" / "ml-20m",   # what download_ml20m.sh creates
    REPO_ROOT / "data" / "ml-20m",           # a reader who passed --target-dir data
    PROFILE_GEN_ROOT / "data" / "ml-20m",    # the original generation layout
]
ML20M_DIR = Path(os.environ["ML20M_DIR"]) if os.environ.get("ML20M_DIR") else next(
    (c for c in _ML20M_CANDIDATES if (c / "ratings.csv").exists()), _ML20M_CANDIDATES[0])
RATINGS_CSV = ML20M_DIR / "ratings.csv"
MOVIES_CSV = ML20M_DIR / "movies.csv"
GENOME_SCORES_CSV = ML20M_DIR / "genome-scores.csv"
GENOME_TAGS_CSV = ML20M_DIR / "genome-tags.csv"

# Pre-computed embeddings (overridable via EMBEDDING_DIR env var)
DEFAULT_EMBEDDING_DIR = CODE_ROOT / "embedding_generator" / "output" / "bge-large-en-v1.5"
EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", str(DEFAULT_EMBEDDING_DIR)))
PROFILE_EMB_NPY = EMBEDDING_DIR / "profile_embeddings.npy"
MOOD_VECTORS_NPY = EMBEDDING_DIR / "mood_vectors.npy"
THEME_MATRIX_NPY = EMBEDDING_DIR / "theme_matrix.npy"
GENOME_EMB_NPY = EMBEDDING_DIR / "genome_embeddings.npy"
COMBINED_NPY = EMBEDDING_DIR / "combined_features.npy"
COMBINED_FULL_NPY = EMBEDDING_DIR / "combined_full.npy"
MOVIE_ID_INDEX = EMBEDDING_DIR / "movie_id_index.json"
EMBEDDING_METADATA = EMBEDDING_DIR / "embedding_metadata.json"

# BERT title embeddings live in the parent output/ dir (not per-encoder subdirs)
EMBEDDING_OUTPUT_ROOT = Path(os.environ.get("EMBEDDING_OUTPUT_ROOT", str(CODE_ROOT / "embedding_generator" / "output")))
BERT_TITLE_EMB_NPY = EMBEDDING_OUTPUT_ROOT / "bert_title_embeddings.npy"

# Benchmark outputs (overridable via env vars)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data" / "processed")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJECT_ROOT / "results")))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))

# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
POSITIVE_THRESHOLD = 3.5       # rating >= this → positive interaction
K_CORE = 10                    # iterative k-core filtering threshold

# Temporal split timestamps (Unix epoch)
# Train: before 2014-01-01, Val: 2014-01-01 to 2014-07-01, Test: after 2014-07-01
TRAIN_END_TS = 1388534400      # 2014-01-01 00:00:00 UTC
VAL_END_TS = 1404172800        # 2014-07-01 00:00:00 UTC

# ──────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS (tuned on validation set)
# ──────────────────────────────────────────────────────────────────────────────
EMBED_DIM = 128                # shared embedding dimension for all models
LIGHTGCN_LAYERS = 3            # number of graph convolution layers
LIGHTGCN_DROPOUT = 0.0         # no dropout (per original LightGCN paper)
LIGHTGCL_SVD_Q = 5             # SVD rank for LightGCL contrastive view

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5            # L2 regularization
BATCH_SIZE = 32768
NUM_EPOCHS = 200
PATIENCE = 20                  # early stopping patience (epochs without val improvement)
NUM_NEGATIVES = 1              # negatives per positive for BPR

# Evaluation
EVAL_BATCH_SIZE = 256          # users per evaluation batch
TOP_K = [10, 20, 50]           # K values for NDCG@K, Recall@K, HR@K

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGS
# ──────────────────────────────────────────────────────────────────────────────
NUM_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2026]  # SEEDS[0] is the default single-run seed

# Hyperparameter search grid (for tuning)
HP_GRID = {
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "n_layers": [2, 3, 4],
    "batch_size": [1024, 2048, 4096],
}

# Side feature configurations for ablation (feature_name → npy file or list of npy files)
FEATURE_CONFIGS = {
    "none":           [],                                              # M1: ID only
    "genome":         ["genome"],                                      # M2: genome PCA
    "genome_raw":     ["genome_raw"],                                  # M2b: genome 1128-dim raw
    "bert_title":     ["bert_title"],                                  # M3: BERT(title+genre)
    "llm_profile":    ["profile"],                                     # M4: LLM profile
    "llm_mood":       ["mood"],                                        # M5: mood
    "llm_themes":     ["themes"],                                      # M6: themes
    "llm_prof_mood":  ["profile", "mood"],                             # M7: profile + mood
    "llm_all":        ["profile", "mood", "themes"],                   # M8: all LLM
    "genome_llm":     ["genome", "mood", "themes"],                    # M9: genome + structured LLM
}

# Mapping from (model, features) to paper experiment label (used for output paths).
# Output folders are laid out as: {checkpoints|results}/{config}/{encoder}/seed-{seed}/
# -- config first, which keeps every encoder variant of a config adjacent. This
# comment described {encoder}/{config}-seed-{seed}, a layout that has not existed
# for some time; experiment_path() below is the authority and the shipped
# results/ tree agrees with it, so the comment was the only thing saying otherwise.
CONFIG_NAME_MAP = {
    ("bpr_mf",      "none"):          "M0",
    ("lightgcn",    "none"):          "M1",
    ("simgcl",      "none"):          "M1b",
    ("xsimgcl",     "none"):          "M1c",
    ("lightgcl",    "none"):          "M1d",
    ("lightgcn_sf", "genome"):        "M2",
    ("lightgcn_sf", "genome_raw"):    "M2b",
    ("lightgcn_sf", "bert_title"):    "M3",
    ("lightgcn_sf", "llm_profile"):   "M4",
    ("lightgcn_sf", "llm_mood"):      "M5",
    ("lightgcn_sf", "llm_themes"):    "M6",
    ("lightgcn_sf", "llm_prof_mood"): "M7",
    ("lightgcn_sf", "llm_all"):       "M8",
    ("lightgcn_sf", "genome_llm"):    "M9",
}


def resolve_config(label: str) -> tuple[str, str]:
    """Paper label -> (model, features). The inverse of get_config_name().

    The paper and `reproduce_all.sh` both address configurations the way the tables
    do -- "M4", "M7" -- while run_experiment.py takes --model/--features. Without
    this the documented command (`--config m4`) did not exist, and reproduce_all.sh
    invoked a module path that did not either, so it failed on all 80 experiments.
    """
    inv = {v.lower(): k for k, v in CONFIG_NAME_MAP.items()}
    key = label.strip().lower()
    if key not in inv:
        raise SystemExit(
            f"unknown config {label!r}; known: "
            + ", ".join(sorted(CONFIG_NAME_MAP.values(), key=str)))
    return inv[key]


def get_config_name(model: str, features: str) -> str:
    """Return the paper-style config label (e.g. 'M0', 'M4', 'M7').

    Falls back to a descriptive key if the combination isn't in the map —
    useful for ad-hoc runs that don't correspond to a reported ablation row.
    """
    return CONFIG_NAME_MAP.get((model, features), f"{model}__{features}")


def get_encoder_name(embedding_dir) -> str:
    """Derive the encoder label (e.g. 'bge-large-en-v1.5', 'e5-large-v2')
    from the trailing component of the embedding directory path.
    """
    from pathlib import Path
    return Path(embedding_dir).name


def experiment_path(model: str, features: str, seed: int, embedding_dir) -> str:
    """Build the canonical experiment path used for checkpoints and results.

    Layout: {config_lower}/{encoder}/seed-{seed}
    Example: m4/bge-large-en-v1.5/seed-42

    This config-first layout keeps all encoder variants of a given config
    adjacent on disk, which simplifies encoder-sensitivity analysis.
    """
    cfg = get_config_name(model, features).lower()
    enc = get_encoder_name(embedding_dir)
    return f"{cfg}/{enc}/seed-{seed}"

# Cold-start buckets (by number of training interactions)
COLD_START_BUCKETS = {
    "cold":   (0, 10),      # <10 training interactions
    "medium": (10, 50),     # 10-50
    "warm":   (50, float("inf")),  # >50
}
