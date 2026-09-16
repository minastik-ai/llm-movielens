# Reproducibility Guide

Step-by-step instructions to reproduce all results from the paper.

## Hardware Requirements

| Stage | Minimum | Recommended | Time Estimate |
|-------|---------|-------------|---------------|
| Profile generation | CPU + internet | CPU + internet | 13 min 12 s |
| Embedding generation | 16GB RAM | 32GB RAM + GPU | ~10 min |
| Benchmark (all 70 runs) | 1× GPU (8GB+) | 1× A100 (40GB) | ~24 hours |
| Single experiment | 1× GPU (8GB+) | 1× GPU (16GB) | ~20 min |

## Prerequisites

```bash
# Python 3.10+
python --version  # Should be 3.10+

# Clone and install
git clone https://github.com/minastik-ai/llm-movielens.git
cd llm-movielens
pip install -r requirements.txt
```

## Option A: Reproduce from Pre-computed Features (Recommended)

Skip stages 1-2 and use our pre-computed embeddings from HuggingFace.

```bash
# 1. Fetch MovieLens 20M under its own terms (we do not redistribute it)
bash scripts/download_ml20m.sh

# 2. Rebuild the temporal splits, SHA-256-checked against ours
python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m

# 2b. Fetch the profiles and embeddings we generated, so stages 1-2 can be skipped
python3 scripts/download_artifacts.py

# 3. Run all experiments (14 configurations × 5 seeds = 70 runs)
bash scripts/reproduce_all.sh --dry-run   # print the plan first; needs no data
bash scripts/reproduce_all.sh

# 4. Generate the results table
python3 scripts/export_results_table.py
```

### Expected Results

After the benchmark finishes you should see results matching the paper's main
results table (within ±0.002 due to hardware/CUDA non-determinism):

| Config | NDCG@10 (expected) |
|--------|-------------------|
| M0: BPR-MF | ~0.1137 |
| M2: +Genome PCA(128) | ~0.1144 |
| M3: +BERT Title(1024) | ~0.1136 |
| M4: +LLM Profile | ~0.1173 |
| M7: +Profile+Mood | ~0.1175 |

The per-seed files these are computed from ship in `results/`, so the whole table
can be re-derived on a CPU in seconds.

One difference worth knowing before you diff your own run against them: the shipped
files are the output of an **eval-only re-evaluation** of the released checkpoints,
not of the training script, which is what their `_provenance` field records. They
therefore carry `seed` and `_provenance` where a fresh training run writes
`best_epoch`, `best_val_metrics`, `stopped_early` and `total_epochs_trained`. The
`test_metrics` block -- the part every number in the paper is computed from -- has
the same keys in both, so a comparison against your own run is like for like.
One trap if you script across the two: `config` is a string label (`"M4"`) in the
shipped files and a dict of hyperparameters (`lr`, `weight_decay`, `num_epochs`,
`patience`, `batch_size`) in a fresh run -- same key, different type.

```bash
python3 tools/verify_paper_numbers.py
```

## Option B: Full Pipeline Reproduction

Regenerate everything from scratch.

### Stage 1: Profile Generation

Requires API keys:
- `ANTHROPIC_API_KEY` — Claude API access
- `TMDB_API_KEY` — TMDb metadata access

```bash
# Set API keys
export ANTHROPIC_API_KEY="your-key-here"
export TMDB_API_KEY="your-key-here"

# Inspect the plan first — costs nothing
python3 src/profile_generator/batch_generate.py --dry-run

# Generate profiles (~$21 API cost via the Batches API, 13 min 12 s)
python3 src/profile_generator/batch_generate.py
```

**Cost:** ~$21 USD (Claude Haiku 4.5 via the Batches API, 50% discount). Prompt
caching does NOT apply: the 1,389-token system prompt is below the model's
4,096-token minimum cacheable prefix, so `cache_read_input_tokens` is 0.

### Stage 2: Embedding Generation

```bash
# Generate all embedding types (~10 min on GPU)
python3 src/embedding_generator/main.py

# The title-embedding baseline (run from the benchmark directory)
cd src/benchmark && python3 features/bert_baseline.py && cd ../..
```

### Stage 3: Benchmark

```bash
# Full reproduction (70 runs)
bash scripts/reproduce_all.sh

# Or one tier, or one configuration
bash scripts/reproduce_all.sh --tier 1     # Pure CF baselines (M0, M1, M1b, M1c, M1d)
bash scripts/reproduce_all.sh --tier 2     # Content-augmented (M2 … M9)
bash scripts/reproduce_all.sh --config M4
```

There is no third tier here: the replacer-class configurations are not reported by
the paper and their per-seed results are not staged, so running them would produce
numbers no table accounts for.

## Verifying Results

```bash
# Recompute every headline number from the shipped per-seed files (CPU, seconds)
python3 tools/verify_paper_numbers.py

# Check that the released code still reproduces the released artifact
# (needs the source data; re-renders all 10,381 prompts and checks their SHA-256)
python3 tools/verify_generator_e2e.py

# Export your own run as a table
python3 scripts/export_results_table.py
```

## Random Seeds

All experiments use 5 seeds: `42, 123, 456, 789, 2026` (see `SEEDS` in `src/benchmark/config.py`; `SEEDS[0] = 42` is the default single-run seed). Seeds control:
- PyTorch weight initialization
- Numpy random sampling (negative sampling)
- Data loader shuffling

Note: Full determinism requires `torch.use_deterministic_algorithms(True)`, which we do not enforce due to performance impact. Results may vary by ±0.002 across hardware.

## Known Issues

1. **CUDA non-determinism:** Results may differ slightly across GPU architectures
2. **TMDb API rate limits:** Profile generation may take longer if rate-limited
