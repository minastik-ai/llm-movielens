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

> **This README is also where the paper's page limit went.** Five things the
> 12-page version could only state in a sentence are documented in full here: the
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
pip install -r requirements.txt
python3 tools/verify_paper_numbers.py
```

It reads the per-seed result files shipped in `results/` and recomputes the main
results table, the paired *t*-tests and the pre-registered leakage verdict, in a
couple of seconds. Keeping it CPU-only is deliberate: a resource whose claims can
only be re-checked on a cluster stops being re-checked.

## What is in here

| Path | What it is |
|---|---|
| `src/profile_generator/` | Stage 1 — reads a catalogue's structured metadata, writes a profile (80–120 words requested, 95–135 realised — the distribution is in [docs/DATASHEET.md](docs/DATASHEET.md)), a 10-axis mood vector and key themes: 3–5 requested, 3–6 realised (64% have three, four records carry six). Two transports: `batch_generate.py` (Batches API, the price the paper reports) and `main.py` (synchronous, ~2×). |
| `src/embedding_generator/` | Stage 2 — encodes the generated text into the feature primitives the benchmark consumes. |
| `src/benchmark/` | Stage 3 — models, features and hyperparameters for the benchmark configurations. |
| `scripts/` | Every run script behind the paper, including `reproduce_all.sh`. |
| `tools/` | `verify_paper_numbers.py` (headline numbers), `rebuild_splits.py` (deterministic splits), `verify_generator_e2e.py` (the released code reproduces the released artifact), and the figure and table generators. |
| `results/` | Per-seed result files: the verifier's input. Per-seed result files for the fourteen configurations this paper reports. |
| `results_gpt4omini/` | The provider-sensitivity arm: the same M4 and M7 configurations retrained on profiles regenerated with GPT-4o-mini, five paired seeds. These are the per-seed inputs behind the provider table in the paper's appendix, and `analysis/cross_llm_summary.json` records the protocol and the paired *t*-tests. |
| `paper/generated/` | The macro files the paper's numbers are generated into — the verifier's expectations. |
| `tests/` | The test suite. 27 tests run on the code alone; 21 more need the artifacts and skip without them, so a fresh clone is green either way. `pip install -r requirements.txt -r tests/requirements.txt && pytest tests/`. |
| `manifest/` | `prompt_manifest.jsonl` — the SHA-256 of all 10,381 rendered prompts, which `tools/verify_generator_e2e.py` re-renders and checks against. The same file ships in the dataset repository under `metadata/`; the two are byte-identical and a gate holds them so. |
| `docs/` | Datasheet, reproducibility guide, human-evaluation protocol, the benchmark reading guide, and the two surveys the paper had no room for. |

## This repository is the artifact, not one paper's supplement

The testbed is shared. This paper is its descriptor, but the harness, the profiles
and the result files are cited by other work through the dataset DOI, which is why
they carry a persistent identifier of their own rather than living inside a paper's
supplementary material.

One consequence is visible in `results/`: it holds exactly the configurations this
paper reports and nothing else — **fourteen**, M0 through M9 including the M1b/M1c/M1d
contrastive variants and the M2b dimensionality control. The harness can run more than
that, and other studies on this testbed do, but shipping their per-seed results here
would put numbers in the artifact that no table in this paper accounts for. The
verifier reads the same fourteen, so every file in `results/` backs a printed claim.

A second dimension is visible one level down. Four configurations — `m3`, `m4`,
`m7`, `m8` — carry results under **two** sentence encoders, so `results/m4/` holds
both `bge-large-en-v1.5/` and `e5-large-v2/`. Those four are the encoder-dependent
ones: their feature input is a sentence-transformer encoding of text, so swapping the
encoder changes what the downstream model sees. The rest pass no text through an
encoder and are identical under either, which is why they are not re-run. **All
headline numbers in the paper come from `bge-large-en-v1.5`**; the `e5-large-v2`
files back the encoder-sensitivity table alone, and the verifier reads only the bge
tree. They are worth opening for one reason: they are what shows that the
profile-over-title margin, the paper's cleanest comparison, is `+3.3%` under
bge-large and not significant under e5-large.

## Reproducing, in the order the pipeline runs

```bash
# 0. Source data. We do not redistribute it; you fetch it under its own terms.
bash scripts/download_ml20m.sh
python3 tools/rebuild_splits.py --ml20m-dir data/raw/ml-20m   # deterministic, SHA-256-checked against ours

# 1. Generate profiles (needs ANTHROPIC_API_KEY). ~$21 via the Batches API.
python3 src/profile_generator/batch_generate.py --dry-run   # inspect first, costs nothing
python3 src/profile_generator/batch_generate.py

# 1-2 (alternative). Skip both stages: fetch what we already generated.
python3 scripts/download_artifacts.py --dry-run   # show what lands where
python3 scripts/download_artifacts.py

# 2. Encode them
python3 src/embedding_generator/main.py

# 3. Evaluate: fourteen configurations x five seeds, temporal split, full ranking.
# 70 runs, about 24 hours on one A100; 8 GB is enough but slower. docs/REPRODUCIBILITY.md
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

## What the paper could not fit

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
recomputed self-describing fields, and the defects those checks found.

**[The datasheet](docs/DATASHEET.md).** Composition, collection and uses, plus the
measured bias audit and the identity-handle ablation, both with their caveats.

## Maintenance and versioning

**Releases are archived, not just tagged.** Every release is deposited under a DOI.
The DOI in `CITATION.cff` resolves to the exact code and data the paper's numbers
were computed on, not to a moving branch, so a result cited from this repository
stays checkable after the repository moves on. Later versions receive their own DOI
under the same concept record; a correction is a new version with the change
recorded, never a silent overwrite.

**The harness is the maintained surface, not the data file.** The pipeline reads
whatever structured description a catalogue has; the harness runs any feature
through the same protocol. That is what lets the resource absorb a change in encoder
or model without the benchmark drifting underneath it, and the cross-provider
regeneration reported in the paper is evidence that it works rather than an
intention.

**Anyone can check the release still reproduces itself, without a GPU.**

```bash
# No download, no GPU, no API key -- but numpy and scipy must be installed
# (pip install -r requirements.txt). Seconds.
python3 tools/verify_paper_numbers.py

# The test suite: 48 tests. 27 run on the code alone; the other 21 need the
# artifacts and SKIP without them, so a fresh clone is green either way. The
# project's own requirements are part of the install -- the suite exercises the
# shipped code, so a runner alone leaves every one of the 27 skipping.
pip install -r requirements.txt -r tests/requirements.txt
pytest tests/

# Needs the source data as well (bash scripts/download_ml20m.sh first), because it
# re-renders all 10,381 prompts and checks their SHA-256 against the released
# manifest. Still CPU only.
python3 tools/verify_generator_e2e.py
```

Keeping these GPU-free is deliberate. A resource whose claims can only be re-checked
on a cluster stops being re-checked, and a resource nobody re-checks quietly stops
being correct.

**Scope of what is promised.** We maintain the harness, the verifier and the released
artifacts, and publish corrections as new versions. We do not promise to track
upstream API changes in third-party services; where a dependency's behaviour is
load-bearing, its version is pinned with the reason recorded in `requirements.txt`.

## Licences

Code, annotations, metadata and the datasheet are **CC BY 4.0**. The generated
profiles and their embeddings are **CC BY-NC 4.0** — our output, but generated from
sources whose own terms are non-commercial, so the condition reflects the upstream
position rather than a preference of ours. Research use is unrestricted under both.

## Citation

See `CITATION.cff`. Please cite the paper and the DOI of the release version you
used, so that a result cited from this resource stays checkable after the resource
moves on.
