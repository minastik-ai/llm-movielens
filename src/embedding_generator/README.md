# Movie Embedding Generator

Converts LLM-generated movie profiles into multiple embedding types for use as item features in recommendation models. Operates on the output of the [profile_generator](../profile_generator/) pipeline.

## Embedding Pipeline

```
movie_profiles.json (10,381 movies)
        │
        ├──→ Profile text ──→ bge-large-en-v1.5 (1024-dim) ──→ profile_embeddings.npy
        │                     (optional: --pca-dims 128/256 to reduce)
        │
        ├──→ Mood vector (10 axes) ──→ direct extraction ──→ mood_vectors.npy
        │
        ├──→ Key themes ──→ frequency filter (≥10) ──→ multi-hot (528-dim) ──→ theme_matrix.npy
        │
        └──→ [Baseline] genome-scores.csv (1128-dim) ──→ PCA (128-dim) ──→ genome_embeddings.npy

Concatenations (dimensions depend on encoder and PCA settings):
  combined_features.npy  = profile(1024) + mood(10)           = 1034-dim  (default, no PCA)
  combined_full.npy      = profile(1024) + mood(10) + themes(528) = 1562-dim  (default, no PCA)
```

## Feature Types

### A. Profile Embeddings (1024-dim default)

The 80-120 word LLM-generated profile text is encoded by a sentence-transformer into a dense vector. PCA reduction is optional.

| Step | Dimensions | Details |
|------|-----------|---------|
| Input | 80-120 words | Embedding-optimized semantic profile (no cast/director/year/ratings) |
| Sentence-transformer | 1024 | `BAAI/bge-large-en-v1.5` with L2 normalization (default) |
| PCA (optional) | 128/256/512 | Enable via `--pca-dims`; off by default |
| Output | 1024 (or PCA target) | L2 re-normalized after PCA if applied |

The profile text follows a consistent semantic flow across all movies: thematic essence, emotional tone, narrative style, genre distinction, audience experience. This consistency ensures analogous information occupies similar positions in the embedding space.

### B. Mood Vectors (10-dim)

Direct extraction of the 10-axis continuous mood vector from each profile. Each axis ranges from -1.0 to 1.0:

| Axis | -1.0 pole | +1.0 pole |
|------|-----------|-----------|
| `dark_light` | Bleak, grim | Bright, uplifting |
| `serious_playful` | Grave, solemn | Comedic, absurd |
| `slow_fast` | Contemplative, still | Frenetic, breathless |
| `cerebral_visceral` | Intellectual | Sensory, action-driven |
| `realistic_fantastical` | Gritty naturalism | Pure fantasy |
| `intimate_epic` | Small-scale, personal | Grand, sweeping |
| `conventional_experimental` | Mainstream | Avant-garde |
| `emotional_detached` | Cold, analytical | Intensely emotional |
| `nostalgic_contemporary` | Classic sensibility | Modern, zeitgeist |
| `predictable_subversive` | Formulaic | Expectation-defying |

No encoding needed — these are direct numeric features usable as-is in any model.

### C. Key Themes (528-dim multi-hot)

Each movie has 3-6 LLM-generated themes (3-5 are requested; 36% of records carry more than three) (e.g., "redemption", "obsession", "identity"). These are encoded as a multi-hot vector over a filtered vocabulary.

- **Raw vocabulary:** 9,183 unique themes
- **After filtering (count >= 10):** 528 themes
- **65.6% of raw themes appeared only once** — these are overly specific compounds like "absurdist cohabitation" that carry no discriminative value as features
- **Top themes:** redemption (597), obsession (445), survival (411), identity (404), mortality (385)

### D. Genome Embeddings (128-dim, baseline)

MovieLens 20M ships with 1,128 genome tag relevance scores per movie. These are reduced via PCA to 128-dim for fair comparison with LLM embeddings.

- **Input:** 1,128-dim relevance scores from `genome-scores.csv`
- **PCA:** 128-dim, ~80% variance retained
- **Purpose:** Baseline to answer "does LLM reasoning over genome tags produce a better representation than feeding the raw 1,128-dim vector?"

### E. BERT Title Baseline (128-dim)

Encodes `"Movie Title (Year) | Genre1, Genre2, ..."` with the same sentence-transformer as the profile embeddings (default: bge-large-en-v1.5). Controls for the embedding method — isolates the value of LLM-generated profile content vs. naive title text.

## Output Files

Shapes shown for default settings (bge-large-en-v1.5, no PCA). Profile-derived dimensions change with encoder and PCA settings.

| File | Shape (default) | Description |
|------|----------------|-------------|
| `profile_embeddings.npy` | (10381, 1024) | LLM profile text embeddings |
| `mood_vectors.npy` | (10381, 10) | 10-axis mood features |
| `theme_matrix.npy` | (10381, 528) | Multi-hot filtered themes |
| `genome_embeddings.npy` | (10381, 128) | Genome PCA baseline |
| `bert_title_embeddings.npy` | (10381, 1024) | BERT title+genre baseline |
| `combined_features.npy` | (10381, 1034) | Profile + mood |
| `combined_full.npy` | (10381, 1562) | Profile + mood + themes |
| `movie_id_index.json` | 10381 IDs | Row-to-movieId mapping |
| `theme_vocabulary.json` | 528 themes | Theme names for multi-hot columns |
| `embedding_metadata.json` | — | Model info, PCA variance, dimensions |

All `.npy` files share the same row ordering defined by `movie_id_index.json`.

## Output layout — flat `output/` vs per-encoder subdirs (read this before "deduplicating")

The default ML-20M run writes **both** a flat `output/*.npy` set **and** per-encoder subdirs (`output/bge-large-en-v1.5/`, `output/e5-large-v2/`). **These are not duplicates** — different (now-frozen) pipeline stages read different copies, and the same-named files genuinely differ:

| Consumer | Reads from | Files |
|---|---|---|
| Main benchmark (M1–M9 content features) | `output/bge-large-en-v1.5/` (`config.EMBEDDING_DIR`) | profile, mood, theme, genome, combined*, movie_id_index, metadata |
| M3 BERT-title baseline | **flat** `output/` (`config.BERT_TITLE_EMB_NPY`) | `bert_title_embeddings.npy` (flat copy; differs from the subdir copy) |

⚠ `output/profile_embeddings.npy` and `output/bert_title_embeddings.npy` (flat) are **load-bearing and differ** from the same-named files under `bge-large-en-v1.5/` — do **not** assume the flat set is a stale duplicate of the subdir. Other datasets use only the subdir layout: sibling `output_amazon/`, `output_ml1m/`, `output_ml20m_gpt4omini/` (each with a `bge-large-en-v1.5/` subdir).

## Supported Models

| Model | Params | Native Dims | MTEB Avg | Notes |
|-------|--------|-------------|----------|-------|
| **`BAAI/bge-large-en-v1.5`** | **335M** | **1024** | **~64** | **Default** |
| `intfloat/e5-large-v2` | 335M | 1024 | ~62 | Instruction-tuned |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 768 | ~62 | Matryoshka (truncatable) |
| `Alibaba-NLP/gte-large-en-v1.5` | 434M | 1024 | ~65 | Near-SOTA open, 2024 |
| `all-MiniLM-L6-v2` | 22M | 384 | ~56 | Fast, lightweight |
| `all-MiniLM-L12-v2` | 33M | 384 | ~57 | Slightly better than L6 |
| `all-mpnet-base-v2` | 110M | 768 | ~58 | Best of the lightweight models |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all embeddings (default: bge-large-en-v1.5, no PCA, 1024-dim)
python main.py

# Apply PCA to reduce dimensions
python main.py --pca-dims 128
python main.py --pca-dims 256

# PCA dimension sweep: generates full-dim + 128/256/512 variants
python main.py --pca-sweep 128 256 512

# Use a different encoder
python main.py --model Alibaba-NLP/gte-large-en-v1.5
python main.py --model all-MiniLM-L6-v2 --pca-dims 128

# Custom output directory
python main.py --output-dir output/custom

# Skip genome baseline (faster)
python main.py --skip-genome

# Generate BERT title baseline (from benchmark directory)
cd ../benchmark && python features/bert_baseline.py
```

### Running with different embeddings in the benchmark

The benchmark reads embeddings from the path in `EMBEDDING_DIR`. By default it points to `output/` (bge-large-en-v1.5, 1024-dim). Override to use other embedding sets:

```bash
cd ../benchmark

# Run benchmark with default embeddings (bge-large, no PCA) — no env var needed
python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42

# Run benchmark with PCA-256 variant
EMBEDDING_DIR=../embedding_generator/output/pca256 \
    python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42

# Run benchmark with a different encoder
EMBEDDING_DIR=../embedding_generator/output/gte-large-v1.5 \
    python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42
```

## Usage in Recommendation Models

```python
import numpy as np
import json

# Load embeddings
profile_emb = np.load("output/profile_embeddings.npy")    # (10381, 1024) default
mood = np.load("output/mood_vectors.npy")                  # (10381, 10)
themes = np.load("output/theme_matrix.npy")                # (10381, 528)
genome = np.load("output/genome_embeddings.npy")           # (10381, 128)

# Load movie ID mapping
with open("output/movie_id_index.json") as f:
    movie_ids = json.load(f)  # movie_ids[i] = original movieId for row i

# Concatenate features for a model
item_features = np.concatenate([profile_emb, mood], axis=1)  # (10381, 1034)
```

## Project Structure

```
embedding_generator/
├── main.py              # Entry point — orchestrates full pipeline
├── config.py            # Paths, model settings, PCA dims, mood axes
├── embedder.py          # Core: encode profiles, extract mood, themes, genome
├── requirements.txt     # Python dependencies
└── output/              # Generated embeddings (auto-created)
```

The BERT title baseline is generated separately via `../benchmark/features/bert_baseline.py` and saved into `output/bert_title_embeddings.npy`.

## Dependencies

- Python 3.10+
- sentence-transformers >= 2.2.0
- numpy, pandas, scikit-learn
