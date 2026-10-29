# LLM-MovieLens — pipeline, harness, and the code behind the paper

Having a language model read an item's structured metadata and write prose about it
produces a **better item representation than consuming that metadata directly**. On
MovieLens 20M the synthesized profile beats the 1,128-dimensional tag genome, the
same sentence encoder applied to titles, and pure collaborative filtering — and the
gain is not dimensionality, because it survives against the genome at both 128 and
its raw 1,128 dimensions.

This repository is the pipeline that produces those profiles and the harness that
tests whether they helped. The generated artifacts themselves — profiles,
embeddings, mood vectors, human-evaluation sheets — live in the companion data
release: **https://huggingface.co/datasets/minastik-ai/llm-movielens**

> **Supplementary material.** Five things the paper states in a sentence are
> documented in full here: the
> [dataset survey](docs/DATASET_SURVEY.md) behind the choice of MovieLens 20M, the
> [licensing analysis](docs/LICENSING_SURVEY.md) behind what we do and do not
> redistribute, the [artifact verification](docs/ARTIFACT_VERIFICATION.md), the
> [datasheet](docs/DATASHEET.md), and
> [how to read the benchmark](docs/READING_THE_BENCHMARK.md) — what each of the
> configurations controls for, and why the list is a set of matched
> pairs rather than a leaderboard.

---

## Start here: check the paper without a GPU

The fastest way to see whether this resource is what it claims is to recompute the
paper's numbers yourself. It needs no GPU, no model download and no API key.

```bash
python3 --version                          # 3.10 or newer
pip install -r requirements-verify.txt     # numpy and scipy, ~150 MB
python3 tools/verify_paper_numbers.py
```

It reads the per-seed result files shipped in `results/` and `results_noaggregate/`
and recomputes the main results table, the paired *t*-tests and the pre-registered
leakage verdict, in under a minute on a first run (cold imports dominate). Exit 0 means every number it
checks matched the paper; 1, the files and the paper disagree; 2, an input is missing,
so it did not judge. The verifier needs only those two packages; the full
`requirements.txt` pulls in several dozen, including `torch`, because it also covers
generation, encoding and the benchmark. Install that one when you intend to run
the pipeline, not to check the paper. Keeping it CPU-only is deliberate: a resource whose claims can
only be re-checked on a cluster stops being re-checked.

## What is in here

| Path | What it is |
|---|---|
| `src/profile_generator/` | Stage 1 — reads a catalogue's structured metadata, writes a profile (80–120 words requested, 95–135 realised — the distribution is in [docs/DATASHEET.md](docs/DATASHEET.md)), a 10-axis mood vector and key themes (ML-20M: no count requested, 3–6 realised, 64% have three, four records carry six; the book and game configurations validate 3–5). Two transports: `batch_generate.py` (Batches API, the price the paper reports) and `main.py` (synchronous, ~2×). |
| `src/embedding_generator/` | Stage 2 — encodes the generated text into the feature primitives the benchmark consumes. |
| `src/benchmark/` | Stage 3 — models, features and hyperparameters for the benchmark configurations. |
| `scripts/` | Every run script behind the paper, including `reproduce_all.sh`. |
| `tools/` | `verify_paper_numbers.py` (results tables, paired tests and the leakage verdict), `verify_mood_agreement.py` (the pairwise mood numbers, from the data repository's sheets), `rebuild_splits.py` (deterministic splits), `verify_generator_e2e.py` (the released code reproduces the released artifact), and `make_pipeline_resource.py` (the pipeline figure). |
| `results/` | Per-seed result files of the fourteen benchmark configurations, five seeds each: the verifier's input. Each is the record `run_experiment.py` writes (best epoch, validation and test metrics, the training settings), so a run you train yourself produces the same file. |
| `results_gpt4omini/` | The provider-sensitivity arm: M4 and M7 trained on profiles generated with GPT-4o-mini, five paired seeds. These are the inputs behind the provider comparison in §3.3 of the paper; `analysis/cross_llm_summary.json` holds the paired *t*-tests, and `scripts/cross_llm_paired_ttest.py` recomputes it. |
| `results_noaggregate/` | The leakage arm (§6 of the paper): M4 and M7 trained on five seeds on profiles generated with the labelled popularity fields removed from the prompt. `scripts/apply_noaggregate_decision_rule.py` applies the pre-registered decision rule to these files and the main table's; the verifier checks the same verdict. |
| `paper/generated/` | The macro files the paper's numbers are generated into — the verifier's expectations. |
| `tests/` | The test suite. Most tests run on the code alone; those that need the artifacts, or torch, skip without them, so a fresh clone is green either way. `pip install -r requirements.txt -r tests/requirements.txt && pytest tests/`. |
| `manifest/` | `prompt_manifest.jsonl` — the SHA-256 of all 10,381 rendered prompts, which `tools/verify_generator_e2e.py` re-renders and checks against. The same file ships in the dataset repository under `metadata/`; the two are byte-identical, and a check holds them so. |
| `docs/` | Datasheet, reproducibility guide, human-evaluation protocol, the benchmark reading guide, and the two surveys the paper had no room for. |

## This repository is the artifact, not one paper's supplement

The testbed is shared. This paper is its descriptor, but the harness, the profiles
and the result files are cited by other work through the dataset DOI, which is why
they carry a persistent identifier of their own rather than living inside a paper's
supplementary material.

One consequence is visible in `results/`: it holds the resource's **fourteen**
benchmark configurations, and `analysis/` (the cold-start, popularity-probe and mood
analyses), and nothing else — M0 through M9, including the M1b/M1c/M1d
contrastive baselines and the M2b dimensionality control. The paper's main table prints
twelve of them. The other two, M8 (profile, mood and themes) and M9 (genome, mood and
themes), are sensitivity configurations: their per-seed results are here, and
[docs/READING_THE_BENCHMARK.md](docs/READING_THE_BENCHMARK.md) says what each one
controls for, but no claim in the paper rests on them. The harness can run more than
that, and other studies on this testbed do, but their results are not shipped here.

A second dimension is visible one level down. Four configurations — `m3`, `m4`,
`m7`, `m8` — carry results under **two** sentence encoders, so `results/m4/` holds
both `bge-large-en-v1.5/` and `e5-large-v2/`. Those four are the encoder-dependent
ones: their feature input is a sentence-transformer encoding of text, so swapping the
encoder changes what the downstream model sees. The rest pass no text through an
encoder and are identical under either, so they need no second run. **All headline
numbers in the paper come from `bge-large-en-v1.5`**; the `e5-large-v2` files back the
encoder-sensitivity study alone, and the verifier reads only the bge tree. They show
that the profile-over-title margin, the paper's cleanest comparison, is +0.0033
NDCG@10 (+2.9%, p = 0.015) under bge-large and +0.0008 (+0.70%, p = 0.47) under
e5-large: the profile leads under both encoders; only the separation is
encoder-dependent.

## Reproducing, in the order the pipeline runs

```bash
# 0. Dependencies and source data. We do not redistribute the data; you fetch it under its own terms.
pip install -r requirements.txt
bash scripts/download_ml20m.sh   # fetches https://files.grouplens.org/datasets/movielens/ml-20m.zip (works when the grouplens.org website is down)

# 1. Generate profiles (needs ANTHROPIC_API_KEY). ~$21 via the Batches API.
python3 src/profile_generator/batch_generate.py --dry-run   # inspect first, costs nothing
python3 src/profile_generator/batch_generate.py

# 1-2 (alternative). Skip both stages: fetch what we already generated. ~190 MB.
python3 scripts/download_artifacts.py --dry-run   # show what lands where, and how big
python3 scripts/download_artifacts.py

# 2. Encode them (skip if you took the alternative above)
python3 src/embedding_generator/main.py

# 2b. Splits. AFTER the profiles exist -- the item universe is read off them, so
# this cannot run before step 1 or its alternative.
python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m   # deterministic, SHA-256-checked against ours

# 3. Evaluate: fourteen configurations x five seeds, temporal split, full ranking.
# 70 runs, about 24 hours on one GPU (40 GB); 8 GB is enough but slower. docs/REPRODUCIBILITY.md
# has the table, and --config M4 runs one configuration instead of all fourteen.
bash scripts/reproduce_all.sh --dry-run                      # print the plan
bash scripts/reproduce_all.sh
```

Every stage is skippable: the released artifacts let you enter at stage 2 or 3
without an API key or a GPU-hour.

**Extending it is a path override, not a fork.** A new sentence encoder, generating
model or prompt plugs in by pointing the configuration at a different path — a new
model architecture is a class plus a branch in `build_model`, not a path; the harness runs whatever feature it is given through the same temporal split,
five seeds and full-ranking protocol.

## What is deliberately not in here

- **Source data.** No MovieLens interaction data, no tag genome or any
  transformation of it, no genre strings, no cached third-party item metadata. The
  reasoning is in [docs/LICENSING_SURVEY.md](docs/LICENSING_SURVEY.md); the practical
  consequence is that `tools/rebuild_splits.py` reconstructs the splits from your own
  download and verifies them byte-for-byte against ours.
- **Vendored baselines.** Third-party implementations are fetched, not shipped —
  their licences are theirs to grant.
- **Trained checkpoints.** GB-scale, and the paper's claims rest on the per-seed
  results in `results/`, which are here.

## Further documentation

**[The dataset survey](docs/DATASET_SURVEY.md).** MovieLens 20M was not the default
choice, it was the surviving one. Testing a claim about representation needs three
properties in one catalogue — a structured description worth beating, density enough
for a credible collaborative baseline, and metadata substantial enough to synthesise
from. This document records every candidate measured against all three, including
the two that came closest and failed on opposite criteria.

**[The licensing analysis](docs/LICENSING_SURVEY.md).** What comparable releases
redistribute, what their licences actually permit, and why this release publishes
only what it made.

**[Artifact verification](docs/ARTIFACT_VERIFICATION.md).** Row-order proofs,
recomputed self-describing fields, and what each check establishes.

**[The datasheet](docs/DATASHEET.md).** Composition, collection and uses, plus the
measured bias audit and the identity-handle ablation, both with their caveats.

**[The reproducibility guide](docs/REPRODUCIBILITY.md).** Step-by-step instructions
to reproduce the paper's results from the shipped files or from scratch, with the
hardware each step needs.

**[Reading the benchmark](docs/READING_THE_BENCHMARK.md).** What each of the fourteen
configurations controls for, and how to read a difference between two rows.

**[The human-evaluation protocol](docs/HUMAN_EVAL_PROTOCOL.md).** The procedure behind
the 500-profile study, so it can be repeated on another corpus.

## Maintenance and versioning

**The resource is archived.** It is deposited on Zenodo under a DOI that resolves to
the newest archived record, not to a moving branch. Each archived record has its own
DOI, listed on Zenodo, which resolves to the exact code and data it contains. A result
cited with that DOI stays checkable after the repository moves on, and a correction
is recorded with the change, never a silent overwrite.

**The harness is the maintained surface, not the data file.** The pipeline reads
whatever structured description a catalogue has; the harness runs any feature
through the same protocol. That is what lets the resource absorb a change in encoder
or model without the benchmark drifting underneath it, and the cross-provider
regeneration reported in the paper is evidence that it works rather than an
intention.

**Anyone can check the release still reproduces itself, without a GPU.**

```bash
# No download, no GPU, no API key -- but numpy and scipy must be installed
# (pip install -r requirements-verify.txt). Under a minute.
python3 tools/verify_paper_numbers.py

# The test suite. The tests that need the artifacts, or torch, SKIP without them,
# so a fresh clone is green either way. The project's own requirements are part
# of the install -- the suite exercises the shipped code, so a test runner alone
# leaves the tests that import numpy or torch skipping.
pip install -r requirements.txt -r tests/requirements.txt
pytest tests/

# Needs the source data as well (bash scripts/download_ml20m.sh first): it re-renders
# a 416-film sample of the prompts (--sample 0 for all 10,381), plus the 82 films with no
# TMDb record, and checks their SHA-256 against the released manifest. The digests are
# exact only under the pinned pandas (< 3.1); the script tests that and reports CANNOT
# JUDGE otherwise. A prompt that embeds TMDb text is checked only where your own TMDb
# cache holds that film; the rest report CANNOT JUDGE. Still CPU only.
python3 tools/verify_generator_e2e.py
```

Keeping these GPU-free is deliberate. A resource whose claims can only be re-checked
on a cluster stops being re-checked, and a resource nobody re-checks quietly stops
being correct.

**Scope of what is promised.** We maintain the harness, the verifier and the released
artifacts, and record corrections with the change. We do not promise to track
upstream API changes in third-party services; where a dependency's behaviour is
load-bearing, its version is pinned with the reason recorded in `requirements.txt`.

## Known issues

Some profiles were generated from another film's metadata, because MovieLens's own
`links.csv` names a different film's TMDb record for a few films. Ten are confirmed
by hand after screening all 10,299 joined films, and six of their profiles describe
the other film. The data repository lists them in
`profiles/ml20m/claude-haiku-4-5/known_issues.json`, and
[docs/DATASHEET.md](docs/DATASHEET.md) gives the screens and their limits, with the
other known issues.

## Licences

Code, metadata, the datasheet and the rest of the human-evaluation data are **CC BY 4.0**. The generated
profiles and their embeddings, and the six `human_eval/` files that reproduce profile text or mood vectors (named in `LICENSE`), are **CC BY-NC 4.0** — our output, but generated from
inputs whose terms are non-commercial (MovieLens, TMDb) or unstated (RLMRec's book
descriptions), so the condition reflects the upstream position rather than a
preference of ours. Research use is permitted under both.

This product uses the TMDb API but is not endorsed or certified by TMDb.

## Citation

See `CITATION.cff`. Please cite the paper and the DOI of the archived record you
used, so that a result cited from this resource stays checkable after the resource
moves on.
