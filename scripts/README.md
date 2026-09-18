# `scripts/` - what each one is

Every run script behind the paper. They are invoked directly rather than
imported, so they are grouped by the pipeline stage they belong to instead of
listed alphabetically. Each description is the script's own first docstring line,
and this file is generated from the shipped directory, so it cannot name a script
that is not here.

Only the four under **Start here** are needed for the paper's headline numbers.
The rest each produced one result and are kept so each is separately checkable.

## Start here

| Script | What it does |
|---|---|
| `reproduce_all.sh` | Full reproduction script for LLM-MovieLens benchmark |
| `download_ml20m.sh` | Download MovieLens 20M dataset |
| `download_artifacts.py` | Fetch the released profiles and embeddings so a reader can enter at stage 2 or 3. |
| `export_results_table.py` | Export benchmark results to CSV and LaTeX table. |

## Data preparation

| Script | What it does |
|---|---|
| `prepare_ml1m.py` | ML-1M preprocessing for the LLM-MovieLens benchmark. |
| `fetch_amazon_books_metadata.py` | Fetch Amazon-Books metadata via OpenLibrary API. |
| `rebuild_metrics_from_seeds.py` | Rebuild each results*/<cfg>/metrics.json from its own seed-*/results.json files. |

## Stage 1 - profile generation

| Script | What it does |
|---|---|
| `generate_book_profiles.py` | Stage-1 LLM profile generation for Amazon-Books (cross-domain experiment). |
| `generate_steam_profiles.py` | Stage-1 LLM profile generation for Steam (cross-domain, third catalogue). |
| `generate_gpt4o_profiles.py` | Generate 500 movie profiles using GPT-4o-mini for LLM sensitivity analysis (cross-LLM). |
| `gpt4omini_calibrate.py` | cross-LLM Option B — Step 1: GPT-4o-mini token calibration on N movies. |
| `gpt4omini_batch.py` | cross-LLM Option B — Steps 2/3: Submit OpenAI Batch API request for full ML-20M |
| `gpt4omini_fill_82.py` | cross-LLM fix: generate the 82 missing GPT-4o-mini profiles using Claude's |
| `ml20m_masked_pilot.py` | ML-20M identity-handle ablation — the other half of the Steam comparison. |

## Stage 2 - embeddings

| Script | What it does |
|---|---|
| `generate_e5_embeddings.py` | Generate profile embeddings using e5-large-v2 for encoder sensitivity analysis. |
| `build_ml1m_embeddings.py` | Build ML-1M profile + mood + themes embeddings by subsetting the existing |
| `run_amazon_phase2_embeddings.sh` | Phase 2 — Amazon-Books embedding generation (cross-domain). |

## Stage 3 - benchmark, Amazon-Books

| Script | What it does |
|---|---|
| `run_amazon_m1.py` | Run M1 (LightGCN, ID-only) on Amazon-Books with a single seed. |
| `run_amazon_m4.py` | Run M4 (LightGCN-SF + LLM profile) on Amazon-Books with a single seed. |
| `run_amazon_m5.py` | Run M5 (LightGCN-SF + LLM mood only) on Amazon-Books with a single seed. |
| `run_amazon_m6.py` | Run M6 (LightGCN-SF + LLM themes only) on Amazon-Books with a single seed. |
| `run_amazon_m7.py` | Run M7 (LightGCN-SF + LLM profile + mood) on Amazon-Books with a single seed. |
| `run_amazon_m8.py` | Run M8 (LightGCN-SF + LLM profile + mood + themes, full feature set) on Amazon-Books. |
| `run_amazon_single.sh` | Run a single Amazon-Books Tier-1 experiment (M1, M4, or M7) with one seed. |
| `run_amazon_phase3_tier1.sh` | Phase 3 — Amazon-Books Tier-1 benchmark (cross-domain). |
| `aggregate_amazon_tier1.py` | Aggregate Amazon-Books Tier-1 results (M1, M4, M7 × 5 seeds). |
| `eval_amazon_m1.py` | Backfill per-seed test-metric JSONs for Amazon-Books M1 (LightGCN, ID-only). |

## Stage 3 - benchmark, ML-1M

| Script | What it does |
|---|---|
| `run_ml1m_m1.py` | Run M1 (LightGCN, ID-only) on ML-1M with a single seed. |
| `run_ml1m_m4.py` | Run M4 (LightGCN-SF + LLM profile) on ML-1M with a single seed. |
| `run_ml1m_m7.py` | Run M7 (LightGCN-SF + LLM profile + mood) on ML-1M with a single seed. |

## Cold start

| Script | What it does |
|---|---|
| `run_cold_start_5seeds.py` | 5-seed cold-start evaluation for M1, M4, M7 (bge-large-en-v1.5). |
| `run_cold_start_eval.py` | Run cold-start evaluation from saved checkpoints for M1, M2b, M4, M7. |
| `run_cold_start_amazon.py` | 5-seed cold-start evaluation for M1/M4/M7 on Amazon-Books. |
| `run_cold_start_m235_for_mood_exp.py` | Experiment B for §7 (Mood as a Controllable-Retrieval Primitive): cold-start |
| `bootstrap_cold_start.py` | User-level bootstrap 95% CI on cold-start metrics for M1, M4, M7. |
| `plot_cold_start.py` | Regenerate cold_start_barchart.pdf, the cold-start bucket figure. |

## Cross-LLM sensitivity

| Script | What it does |
|---|---|
| `run_cross_llm_sweep.py` | cross-LLM sweep orchestrator: 2 configs × 2 LLMs × 5 seeds = 20 cells. |
| `run_cross_llm_retrain.py` | cross-LLM closure — single-cell retrain harness for M4/M7 with Claude OR GPT-4o-mini features. |
| `run_cross_llm_colab.py` | cross-LLM single-cell retrain for Colab Pro — resume-friendly across disconnects. |

## Mood as a controllable-retrieval primitive

| Script | What it does |
|---|---|
| `run_mood_controllable_retrieval.py` | Experiment A for §7 (Mood as a Controllable-Retrieval Primitive): zero-shot |

## Human evaluation

| Script | What it does |
|---|---|
| `prepare_human_eval.py` | Prepare human evaluation materials. |
| `build_human_eval_sheets.py` | Build two annotator-facing sheets for the human evaluation study described |
| `regenerate_consolidated.py` | Regenerate human_eval/consolidated/ from the filled annotator sheets. |

## Shared helpers - imported, not run

| Script | What it does |
|---|---|
| `_paths.py` | Resolve the project's trees in BOTH the development and the published layout. |
| `_amazon_run_helper.py` | Shared helper for Amazon-Books single-experiment runners. |
| `_ml1m_run_helper.py` | Shared helper for ML-1M single-experiment runners. |

