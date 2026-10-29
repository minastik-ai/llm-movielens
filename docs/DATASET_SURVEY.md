# Dataset survey: why MovieLens 20M, and what else was measured

Testing a claim about item representation needs three properties in one catalogue: a
structured description worth beating, enough density for a credible collaborative
baseline, and metadata substantial enough to synthesise from. This survey measures
candidate catalogues against all three with [`tools/recon_amazon.py`](../tools/recon_amazon.py)
and [`tools/recon_steam.py`](../tools/recon_steam.py), which apply the *exact* protocol of
`rebuild_splits.py` (positive threshold 3.5, iterative k-core over users and items, the
paper's 98.99/0.43/0.58 temporal shares) to a candidate dataset. No training, no
generation: the point is to price a catalogue before spending anything on it.

## Amazon Reviews 2023 / Video Games: rich metadata, too sparse

| k-core | interactions | users | items | density | int/item | test set |
|--:|--:|--:|--:|--:|--:|--:|
| **10** (the paper's) | 101,392 | 6,211 | 3,165 | 0.516% | 32.0 | **589** |
| 5 | 533,133 | 63,383 | 19,020 | 0.044% | 28.0 | 3,097 |
| 3 | 1,118,150 | 222,104 | 41,497 | 0.012% | 26.9 | 6,494 |
| *ML-20M, for reference* | 11,499,778 | 127,371 | 9,906 | **0.911%** | 1,161 | 67,466 |

At the paper's own k=10, Video Games retains **2.2% of interactions, 0.2% of users and 2.3%
of items**, leaving a test set of 589 interactions, far too small to support a paired
*t*-test on NDCG@10 differences of ~0.003. Lowering k rescues the size but destroys the
density: at k=5 the benchmark is **21× sparser than ML-20M**.

That is disqualifying, not merely inconvenient, for two reasons:

1. **It contradicts the reason for choosing ML-20M.** ML-20M's density gives a content
   feature a credible collaborative baseline to beat; a 0.044% arm would test the
   feature against a weak one.
2. **It would need a framework this resource does not build.** A datapoint 21× off ML-20M
   cannot be interpreted without a treatment of how paradigm choice varies with interaction
   density, and the paper is a resource descriptor, not that argument.

The metadata, by contrast, is excellent: 93.0% of surviving items carry a free-text
`description`, 94.5% bullet `features`, 98.5% hierarchical `categories` at median depth 5,
97.7% a `details` dict; the median prompt payload is ~509 tokens against ML-20M's ~365 on
the same chars/3.8 basis (ML-20M's 2,012 is total billed input, system prompt included).
Generation would cost about **$6** (payload plus system prompt, which is billed on every
call) and training 1–3 GPU-hours. The problem is not cost.

## The constraint is structural across Amazon

All 33 named categories, ranked on the two binding axes: interactions per user (which
decides k-core survival) and catalogue size (which decides ranking cost), from the
dataset's own per-category table, its "Unknown" bucket excluded:

- **Zero categories** combine ≥2.5 int/user with ≤200K items.
- The categories dense enough (Kindle Store 4.57, Clothing 2.92, Home & Kitchen 2.91,
  Books 2.86, CDs & Vinyl 2.67, Movies & TV 2.66, Automotive 2.50) carry **0.7M–7.2M
  items**; generation alone would run $1,400–$14,000 before any GPU time.
- Every Amazon category sits at **1.1–4.6 int/user**. ML-20M sits at **90.3**.

**Steam beats every Amazon category on both axes at once**: 3.03 int/user (bettered only by
Kindle Store) and 15,474 items (about 45× smaller than the smallest dense category, 0.7M items). It is the only
candidate examined that is in the same *regime* as ML-20M rather than an order of magnitude
away. Its weakness is the opposite one: thin metadata and no free-text description.

## Steam: the right shape, too little substance

Steam needs its own loader: the files are Python-repr rather than JSON, there are no star
ratings, and the ownership file (`australian_users_items`) carries **no timestamps at
all**, so the paper's temporal split is only possible from `steam_reviews.json.gz`, where a
review is itself the positive signal. `recon_steam.py` imports the tested k-core and split
from `recon_amazon.py`, so both candidates are judged by identical code, and it
cross-checks its fast regex parser against `ast.literal_eval` before trusting it.

| | **Steam** | ML-20M | Amazon VG (k=10) |
|---|--:|--:|--:|
| interactions | 2,802,435 | 11,499,778 | 101,392 |
| users | **118,367** | 127,371 | 6,211 |
| items | **9,346** | 9,906 | 3,165 |
| density | 0.253% | 0.911% | 0.516% |
| int / item | 299.9 | 1,161 | 32.0 |
| test set | **18,560** | 67,466 | 589 |
| prompt payload (chars / 3.8) | **~61 tokens** | ~365 | ~509 |
| generation, one full arm | **$14.71** (measured) | $21 | ~$6 |
| GPU, 20 runs | 20–65 h | — | 1–3 h |

**The shape is excellent.** After the paper's own 10-core, Steam holds 94% of ML-20M's item
count and 93% of its users, at a density within 3.6× rather than Amazon's 21×. The temporal
split lands at 2017-12-28 / 2017-12-31, giving val 11,342 and test 18,560, large enough for
paired *t*-tests. Metadata coverage is essentially total (tags 100%, genres 99.1%, specs
98.8%, developer 98.6%). 0.84% of review lines are unparsable and are dropped.

**The metadata is the problem.** The median prompt payload is **230 characters, about 61
tokens**: about 6× less than ML-20M and 8× less than Amazon Video Games. It is a title, a
median of 8 tags, genres, specs and the developer/publisher; Steam ships no free-text
description. Asking for an 80–120 word profile from 61 tokens means the output is roughly
twice the length of its input, so most of it must come from the model's pretrained
knowledge of the game rather than from the supplied metadata. That is recall, not
reasoning over structured metadata, and it is not what the paper measures. The paper's
identity-handle ablation quantifies exactly this (datasheet, "Identity-handle ablation"):
with title, developer and publisher withheld, a Steam profile retrieves its own unmasked
version at rank 1 only 26.3% of the time, against 94.1% on ML-20M.

**So the two candidates fail in opposite directions:** Steam has the right shape and too
little substance; Amazon has the right substance and the wrong shape.

**What a Steam arm costs.** Measured from 2,014 real calls: 1,327 input and 364 output
tokens per profile, `cache_read_input_tokens = 0` throughout (the system prompt is below
the model's minimum cacheable prefix, so it is billed in full on every call). That is
**$14.71 per full Steam arm at the batch price**, $29 for a masked/unmasked pair, plus
20–65 GPU-hours to train. The generator records its own usage and prints the measured
per-profile cost and full-catalogue projection at the end of every run.

## What this establishes

**No Amazon category combines the per-user density and the catalogue size that a
full-ranking content benchmark needs**: 33 categories checked, none qualifying. Together
with the tag genome, which has no widely used structured equivalent elsewhere, that is
the case for MovieLens 20M a decade on.

## Reproducing

```bash
python3 tools/recon_amazon.py --category Video_Games --k-core 10
```

Caches ~700 MB on first run (`benchmark/0core/rating_only/<Cat>.csv` and
`raw/meta_categories/meta_<Cat>.jsonl`), then runs offline. Writes `recon_<Cat>.json`.
