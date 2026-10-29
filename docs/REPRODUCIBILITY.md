# Reproducibility Guide

Step-by-step instructions to reproduce the paper's results, from the shipped files or
from scratch.

## Hardware Requirements

| Stage | Minimum | Recommended | Time Estimate |
|-------|---------|-------------|---------------|
| Profile generation | CPU + internet | CPU + internet | ~13 min (Batches API) |
| Embedding generation | 16 GB RAM | 32 GB RAM + GPU | ~10 min |
| Benchmark (all 70 runs) | 1 GPU (8 GB+) | 1 GPU (40 GB) | ~24 hours |
| Single experiment | 1 GPU (8 GB+) | 1 GPU (40 GB) | ~20 min |

## Prerequisites

```bash
# Python 3.10+
python3 --version  # Should be 3.10+

# Clone and install
git clone https://github.com/minastik-ai/llm-movielens.git
cd llm-movielens
pip install -r requirements.txt
```

## Check the paper first (no GPU)

```bash
pip install -r requirements-verify.txt
python3 tools/verify_paper_numbers.py
```

It recomputes the main results table, the paired *t*-tests and the leakage verdict
from the per-seed files in `results/` and `results_noaggregate/`, on a CPU, in under a
minute (cold imports dominate; a second run takes a second). `tools/verify_mood_agreement.py` does the same for the pairwise mood
agreement, from the human-evaluation sheets (`python3 scripts/download_artifacts.py
--human-eval` fetches them).

## Option A: Reproduce from Pre-computed Features (Recommended)

Skip stages 1-2 for the LLM-generated features and use our pre-computed embeddings
from Hugging Face. Two feature sets encode MovieLens content rather than generated
text -- the genome PCA (M2, M9) and the title embedding (M3) -- and are not ours to
redistribute, so those three configurations need a local Stage 2; step 3 says how.

```bash
# 1. Fetch MovieLens 20M under its own terms (we do not redistribute it)
bash scripts/download_ml20m.sh

# 2. Fetch the profiles and embeddings we generated, so stages 1-2 can be skipped
#    (~190 MB, and about twice that on disk once huggingface_hub has cached it)
python3 scripts/download_artifacts.py

# 2b. Rebuild the temporal splits, SHA-256-checked against ours. This reads the
#     item universe off the profiles, so it has to come after step 2.
python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m

# 3. Run the experiments. Eleven of the fourteen configurations train from what
#    steps 1-2 give you (M2b builds its raw genome matrix from genome-scores.csv
#    itself). M2, M3 and M9 also need the genome PCA and the title embedding:
#    build them from your own MovieLens copy with Stage 2 of Option B. The
#    benchmark names the command if you reach them without it.
bash scripts/reproduce_all.sh --dry-run   # print the plan first; needs no data
bash scripts/reproduce_all.sh

# 4. Tabulate YOUR run. Your results land inside the benchmark package, not in the
#    shipped results/ tree, so the two can be compared.
python3 scripts/export_results_table.py --results-dir src/benchmark/results
```

`tools/verify_paper_numbers.py` always reads the **shipped** `results/`: its job is to
check that the paper agrees with what was released, which is a different question
from whether your re-run agrees with the paper. For that, compare the table step 4
writes against the expected values below.

### Expected Results

`python3 scripts/export_results_table.py` run on the shipped `results/` reproduces the
paper's main table (mean and sample standard deviation over five seeds, bge-large-en-v1.5):

| Config | | NDCG@10 | Recall@10 | MRR |
|--------|---|---------|-----------|-----|
| M0 | BPR-MF | 0.1137 ± 0.0004 | 0.0483 ± 0.0005 | 0.2379 ± 0.0022 |
| M1 | LightGCN | 0.1138 ± 0.0007 | 0.0497 ± 0.0011 | 0.2354 ± 0.0015 |
| M1b | SimGCL | 0.1078 ± 0.0011 | 0.0466 ± 0.0003 | 0.2293 ± 0.0017 |
| M1c | XSimGCL | 0.1108 ± 0.0007 | 0.0462 ± 0.0006 | 0.2359 ± 0.0017 |
| M1d | LightGCL | 0.0888 ± 0.0060 | 0.0365 ± 0.0025 | 0.2006 ± 0.0100 |
| M2 | + genome PCA (128) | 0.1144 ± 0.0004 | 0.0490 ± 0.0006 | 0.2399 ± 0.0014 |
| M2b | + genome raw (1,128) | 0.1140 ± 0.0018 | 0.0487 ± 0.0008 | 0.2373 ± 0.0047 |
| M3 | + title embedding | 0.1142 ± 0.0015 | 0.0481 ± 0.0007 | 0.2377 ± 0.0047 |
| M4 | + LLM profile | 0.1175 ± 0.0006 | 0.0498 ± 0.0007 | 0.2428 ± 0.0011 |
| M5 | + LLM mood (10) | 0.1149 ± 0.0005 | 0.0485 ± 0.0007 | 0.2420 ± 0.0039 |
| M6 | + LLM themes (528) | 0.1102 ± 0.0012 | 0.0481 ± 0.0004 | 0.2305 ± 0.0028 |
| M7 | + profile + mood | 0.1169 ± 0.0010 | 0.0498 ± 0.0007 | 0.2437 ± 0.0021 |

M8 and M9 are sensitivity configurations outside the paper's table; the exporter
lists them after M7. MRR is the reciprocal rank of the first relevant item over the full ranking.

A re-run will not match to the last digit: GPU kernels are not deterministic (see
Random Seeds), so expect differences in the third or fourth decimal of a single seed,
and smaller ones in a five-seed mean.

The shipped per-seed files are the same record a training run writes
(`results/<config>/<encoder>/seed-<n>/results.json`: `best_epoch`,
`total_epochs_trained`, `stopped_early`, `best_val_metrics`, `test_metrics`, the
training settings under `config`, the split's shape under `data`, and the contrastive
models' own settings under `model_hparams`), so your run and ours can be compared key
for key.

### The contrastive baselines' settings

M1b (SimGCL), M1c (XSimGCL) and M1d (LightGCL) use their own hyperparameters,
selected on validation NDCG@10 at seed 42 over the values their authors report and
frozen before any reported run. The frozen choices, the selection trace and their
SHA-256 are in `src/benchmark/contrastive_settings.json`, and `run_experiment.py`
applies them to these three models (with Xavier-uniform initialisation, validation
every epoch through epoch 20, and no early stop before epoch 45), so
`reproduce_all.sh` trains exactly the runs the paper reports. A flag on the command
line overrides a setting. `scripts/contrastive_grid.py` is the search that selected
them, if you want to repeat it.

## Option B: Full Pipeline Reproduction

Regenerate everything from scratch.

### Stage 0: The source catalogue

Every stage below reads MovieLens 20M, which we do not redistribute, so fetch it
first -- including for the dry run, which assembles the real prompts and is
therefore what makes it worth running before you spend anything.

```bash
bash scripts/download_ml20m.sh
```

### Stage 1: Profile Generation

Requires API keys:
- `ANTHROPIC_API_KEY` -- Claude API access
- `TMDB_API_KEY` -- TMDb metadata access (the crawl fills `src/profile_generator/cache/`)

```bash
export ANTHROPIC_API_KEY="your-key-here"
export TMDB_API_KEY="your-key-here"

# Inspect the plan first -- costs nothing
python3 src/profile_generator/batch_generate.py --dry-run

# Generate profiles (~$21 API cost via the Batches API, about 13 minutes)
python3 src/profile_generator/batch_generate.py
```

**Cost:** ~$21 (Claude Haiku 4.5 via the Batches API, 50% discount); `main.py` sends
the same requests synchronously for about twice that. Prompt caching does not apply:
the 1,389-token system prompt is below the model's 4,096-token minimum cacheable
prefix.

**The no-aggregate control (§6 of the paper)** uses the same generator with the one
prompt line that carries the labelled aggregates (TMDb vote average and count,
MovieLens mean rating and rating count) removed. Its profiles go to
`movie_profiles_noaggregate.json`, so they never overwrite the full set:

```bash
PROFILE_PROMPT=no-aggregates python3 src/profile_generator/batch_generate.py
```

Encode them with Stage 2 into a directory of their own (the example below uses
`src/embedding_generator/output_noaggregate/bge-large-en-v1.5`), train M4 and M7 on it with
`EMBEDDING_DIR` (Stage 3), then apply the paper's decision rule and its popularity
probe, which also reads the ML-20M pack's genome and title embeddings from its default
`--emb-dir`:

```bash
python3 scripts/apply_noaggregate_decision_rule.py --noagg-dir results_noaggregate
python3 scripts/probe_popularity.py --noagg-emb-dir src/embedding_generator/output_noaggregate/bge-large-en-v1.5 --tmdb-cache src/profile_generator/cache/tmdb_metadata.json
```

Run on the shipped `results_noaggregate/`, the first prints the paper's verdict (PASS,
93.0% of the effect retained).

### Stage 2: Embedding Generation

```bash
# Profile embeddings, mood vectors, theme matrix and the genome PCA (~10 min on GPU)
python3 src/embedding_generator/main.py

# The title embedding M3 reads, written beside the encoder's other features
cd src/benchmark && python3 features/bert_baseline.py && cd ../..
```

`main.py` writes into `src/embedding_generator/output/bge-large-en-v1.5/`, the same
directory `download_artifacts.py` fills, so it replaces any downloaded copies there.
For the e5-large-v2 sensitivity runs, `python3 scripts/generate_e5_embeddings.py`
builds `output/e5-large-v2/`; select it with
`EMBEDDING_DIR=src/embedding_generator/output/e5-large-v2`.

### Stage 3: Benchmark

The benchmark reads its splits from inside its own package, and nothing earlier in
Option B writes them, so rebuild them first -- from your own MovieLens download and
the profiles Stage 1 produced, whose item universe defines the split.

```bash
python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m

# Full reproduction (70 runs)
bash scripts/reproduce_all.sh

# Or one tier, or one configuration
bash scripts/reproduce_all.sh --tier 1     # Pure CF baselines (M0, M1, M1b, M1c, M1d)
bash scripts/reproduce_all.sh --tier 2     # Content-augmented (M2 ... M9)
bash scripts/reproduce_all.sh --config M4
```

## The other catalogues and the second generating model

The profiles for these ship on the data repository; each also needs its source data from
its owner. Paths are relative to the repository root.

**The GPT-4o-mini arm (provider comparison).** Save
`profiles/ml20m/gpt-4o-mini/movie_profiles.json` from the data repository as
`src/profile_generator/output_ml20m_gpt4omini/movie_profiles.json`, then encode it and
train a cell (training needs the ML-20M splits, `python3 tools/rebuild_splits.py`):

```bash
python3 src/embedding_generator/main.py --profiles src/profile_generator/output_ml20m_gpt4omini/movie_profiles.json --output-dir src/embedding_generator/output_ml20m_gpt4omini/bge-large-en-v1.5
python3 scripts/run_cross_llm_cell.py --config m4 --llm gpt --seed 42 --res_root results_gpt4omini_new
```

A cell writes `results_gpt4omini_new/m4_gpt/bge-large-en-v1.5/seed-42/results.json` (or `<res_root>/...` for another `--res_root`). The shipped
files of the same cells sit under `results_gpt4omini/m4/bge-large-en-v1.5/seed-42/` (their
`experiment` field reads `m4/...`), which is where `scripts/cross_llm_paired_ttest.py`
reads: copy a cell there to include it in the comparison.

**Amazon-Books.** Fetch RLMRec's split (the converter prints the same instructions when
it cannot find it): download `https://archive.org/download/rlmrec_data/data.zip`, copy its
`amazon/` folder to `src/benchmark/external/RLMRec/data/amazon/`, and take
`data/mapper/amazon_item.json` and `amazon_user.json` from github.com/HKUDS/RLMRec into
`src/benchmark/external/RLMRec/data/mapper/`. Save
`profiles/amazon_books_2018/claude-haiku-4-5/book_profiles.json` from the data repository as
`src/profile_generator/output_amazon/book_profiles.json`. Then:

```bash
python3 scripts/convert_rlmrec_amazon_splits.py   # writes src/benchmark/data/processed_amazon/
bash scripts/run_amazon_phase2_embeddings.sh      # needs python3 with sentence-transformers on PATH
bash scripts/run_amazon_single.sh M4 42           # or run_amazon_phase3_tier1.sh: M1, M4, M7 x 5 seeds
```

**MovieLens 1M.** Download `https://files.grouplens.org/datasets/movielens/ml-1m.zip`, unzip
it so that `src/profile_generator/data/ml-1m/ratings.dat` exists, fetch the ML-20M
embeddings (`python3 scripts/download_artifacts.py`), then:

```bash
python3 scripts/prepare_ml1m.py            # percentile temporal split into src/benchmark/data/processed_ml1m/
python3 scripts/build_ml1m_embeddings.py   # slices the ML-20M arrays to the ML-1M films
python3 scripts/run_ml1m_m4.py --seed 42
```

**Figures with no shipped producing script.** The per-axis cross-model mood correlations
over all 10,381 profiles, the confidence interval on the discriminativeness gap between the
two models, and the 20.2% share of profiles over 120 words are stated in the paper from the
authors' analysis code, which is not shipped. The released files hold what they are computed
from (the profiles, `human_eval/` and the sheets); the word-count share and the genre means
in the datasheet recompute from them in a few lines, and the popularity thirds also need the
MovieLens ratings counts.

## Verifying Results

```bash
# Recompute the results tables, paired tests and leakage verdict from the shipped
# per-seed files (CPU, under a minute)
python3 tools/verify_paper_numbers.py

# The paper's comparisons of M4 with every baseline, with the Holm correction
python3 scripts/analyze_contrastive.py

# The provider comparison (§3.3): GPT-4o-mini against Claude profiles, M4 and M7
python3 scripts/cross_llm_paired_ttest.py

# Check that the released code still reproduces the released artifact.
# All three parts need the MovieLens source data (bash scripts/download_ml20m.sh); the
# third also needs the released profiles. The prompt-digest part re-renders the prompts,
# and a prompt embeds a TMDb synopsis and keyword list that are not ours to
# redistribute -- so with only MovieLens fetched it checks the 82 films that have no
# TMDb record and reports CANNOT JUDGE (exit 2), naming what it would need, for the
# rest. It is not a failure. The digests are exact only under the pinned pandas
# (< 3.1); the script tests that first. By default it checks a 416-film sample plus
# those 82; --sample 0 checks all 10,381 where a TMDb cache is present.
python3 tools/verify_generator_e2e.py

# Export your own run as a table. A fresh run writes inside the benchmark
# package, so point the exporter there; bare, it reads the shipped results/.
python3 scripts/export_results_table.py --results-dir src/benchmark/results
```

The analysis scripts write their outputs inside the checkout: `scripts/export_results_table.py`
to `main_results/`, `scripts/analyze_contrastive.py` to `contrastive_analysis/`, and
`scripts/cross_llm_paired_ttest.py` to `results_gpt4omini/analysis/`, where it regenerates
`cross_llm_summary.json` byte for byte and also writes `cross_llm_paper_table.md`. None of
the new files is listed in `SHA256SUMS`, so `shasum -c` still passes after a run.

## Random Seeds

All experiments use 5 seeds: `42, 123, 456, 789, 2026` (see `SEEDS` in
`src/benchmark/config.py`; `SEEDS[0] = 42` is the default single-run seed). Seeds
control:
- PyTorch weight initialization
- NumPy random sampling (negative sampling)
- Data loader shuffling

Full determinism would require `torch.use_deterministic_algorithms(True)`, which we
do not enforce because of its cost, so a re-run differs slightly from the shipped
files.

## Known Issues

1. **GPU non-determinism:** a re-run differs from the shipped files in the third or
   fourth decimal (see Random Seeds).
2. **TMDb API rate limits:** profile generation may take longer if rate-limited.
3. **The contrastive baselines need GPU memory in proportion to the batch.** M1b and
   M1c compute an InfoNCE loss over the batch's unique users and items, and M1d over
   every user or item, so at the default `BATCH_SIZE` of 32,768 they need a large
   card (the recommended 40 GB). On a smaller one, reduce the batch rather than
   giving up on the configuration:

   ```bash
   cd src/benchmark && python3 run_experiment.py --config M1d --seed 42 --batch-size 4096
   ```

   The optimiser sees more, smaller steps, so a reduced batch is a change to the
   training recipe, not just to memory use: expect a small shift from the reported
   numbers and say so if you report your own.
4. **Other accelerators.** The device selector falls back `cuda -> mps -> cpu`, so
   the benchmark runs on any PyTorch-supported device, more slowly than on the
   recommended GPU. `--epochs 1` makes a quick smoke test.
