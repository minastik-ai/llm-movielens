# Stage-0 recon — can a third dataset join the benchmark?

**Run 2026-09-01** with [`tools/recon_amazon.py`](../tools/recon_amazon.py), which applies the
*exact* protocol of `rebuild_splits.py` (positive threshold 3.5, iterative k-core over users
and items, the paper's 98.99/0.43/0.58 temporal shares) to a candidate dataset. No training,
no generation: the point is to price the decision before spending anything on it.

**Result: Amazon Reviews 2023 / Video Games does not work.** The recon overturned the
prior recommendation, which had been made on metadata richness without checking whether the
benchmark protocol survives contact with the data.

## What was measured

| k-core | interactions | users | items | density | int/item | test set |
|--:|--:|--:|--:|--:|--:|--:|
| **10** (the paper's) | 101,392 | 6,211 | 3,165 | 0.516% | 32.0 | **589** |
| 5 | 533,133 | 63,383 | 19,020 | 0.044% | 28.0 | 3,097 |
| 3 | 1,118,150 | 222,104 | 41,497 | 0.012% | 26.9 | 6,494 |
| *ML-20M, for reference* | 11,499,778 | 127,371 | 9,906 | **0.910%** | 1,161 | 67,466 |

At the paper's own k=10, Video Games retains **2.2% of interactions, 0.2% of users and 2.3%
of items**, leaving a test set of 589 interactions — far too small to support a paired
*t*-test on NDCG@10 differences of ~0.003. Lowering k rescues the size but destroys the
density: at k=5 the benchmark is **21× sparser than ML-20M**.

Two reasons that is disqualifying, not merely inconvenient:

1. **It contradicts the paper's own argument.** §1 justifies ML-20M partly because at 0.91%
   density "a content feature meets a credible collaborative baseline rather than a weak
   one", and §2 criticises prior LLM-recsys work for being "evaluated where the collaborative
   baseline is weak". Adding a 0.044% arm would place this paper in the position it faults.
2. **It would need a framework this paper does not build.** A datapoint 21× off ML-20M
   cannot be interpreted without a treatment of how paradigm choice varies with interaction
   density, and this paper is a resource descriptor, not that argument.

The metadata, by contrast, was excellent — 93.0% of surviving items carry a free-text
`description`, 94.5% bullet `features`, 98.5% hierarchical `categories` at median depth 5,
97.7% a `details` dict; median prompt payload ~509 tokens against ML-20M's 2,012. Generation
would have cost about **$2.80** and training 1–3 GPU-hours. The problem was never cost.

## The constraint is structural across Amazon

Ranking all 28 named categories on the two binding axes — interactions per user (which
decides k-core survival) and catalogue size (which decides ranking cost):

- **Zero categories** combine ≥2.5 int/user with ≤200K items.
- The categories dense enough (Kindle Store 4.57, Clothing 2.92, Home & Kitchen 2.91,
  Books 2.86, CDs & Vinyl 2.67, Movies & TV 2.66) carry **0.5M–7.2M items** — generation
  alone would run $1,000–$14,000 before any GPU time.
- Every Amazon category sits at **1.1–4.6 int/user**. ML-20M sits at **90.3**.

**Steam beats every Amazon category on both axes at once**: 3.03 int/user (bettered only by
Kindle Store) and 15,474 items (9× smaller than the smallest dense category). It is the only
candidate examined that is in the same *regime* as ML-20M rather than an order of magnitude
away. Its known weakness is the opposite one — thin metadata, no free-text description
(see the comparison in the session notes) — so it needs the equivalent recon before use.

## Steam — measured 2026-09-01 with `tools/recon_steam.py`

Steam needed its own loader: the files are Python-repr rather than JSON, there are no
star ratings, and the ownership file (`australian_users_items`) carries **no timestamps at
all**, so the paper's temporal split is only possible from `steam_reviews.json.gz`, where a
review is itself the positive signal. The script imports the tested k-core and split from
`recon_amazon.py` so both candidates are judged by identical code, and it cross-checks its
fast regex parser against `ast.literal_eval` before trusting it.

| | **Steam** | ML-20M | Amazon VG (k=10) |
|---|--:|--:|--:|
| interactions | 2,802,435 | 11,499,778 | 101,392 |
| users | **118,367** | 127,371 | 6,211 |
| items | **9,346** | 9,906 | 3,165 |
| density | 0.253% | 0.910% | 0.516% |
| int / item | 299.9 | 1,161 | 32.0 |
| test set | **18,560** | 67,466 | 589 |
| prompt payload | **~61 tokens** | ~2,012 | ~509 |
| generation | **$5.4** | $21 | $2.8 |
| GPU, 20 runs (CUDA) | 15–53 h | — | 1–3 h |

**The shape is excellent.** After the paper's own 10-core, Steam holds 94% of ML-20M's item
count and 93% of its users, at a density within 3.6× rather than Amazon's 21×. The temporal
split lands at 2017-12-28 / 2017-12-31 giving val 11,342 and test 18,560 — large enough for
the paired *t*-tests the paper reports. Metadata coverage is essentially total (tags 100%,
genres 99.1%, specs 98.8%, developer 98.6%). 0.84% of review lines are unparsable and are
dropped.

**The metadata is the problem, and it is worse than expected.** The median prompt payload is
**230 characters, about 61 tokens** — 33× less than ML-20M and 8× less than Amazon Video
Games. It is a title, a median of 8 tags, genres, specs and the developer/publisher. Steam
ships no free-text description. Asking for an 80–120 word profile from 61 tokens means the
output is roughly twice the length of its input, so most of it must come from the model's
pretrained knowledge of the game rather than from the supplied metadata. That is recall, not
reasoning over structured metadata, and it is not what the paper claims to measure.

**So the two candidates fail in opposite directions:** Steam has the right shape and too
little substance; Amazon has the right substance and the wrong shape.

**What would make Steam usable.** The confound is cheap to control: generate a second Steam
arm with the game title, developer and publisher removed, so the model sees only tags, genres
and specs. The gap between the two arms *is* the world-knowledge share. Full Steam arm with
that control: about **$11 and 20–65 CUDA-hours**.

**Cost model: wrong twice, then measured (2026-09-03).** This note first said $1 per
Steam arm, then $5.38. Both were modelled rather than measured, and both were low.
The first scaled the known $/profile by input length alone, ignoring that the output
is pinned at 80–120 words whatever the input, so on a metadata-poor catalogue the
output dominates. The second fixed that but still costed the *payload* only, omitting
the ~1,210-token system prompt — which is below the model's minimum cacheable prefix
and is therefore billed in full on every single call.

**Measured, from 2,014 real calls:** 1,327 input and 364 output tokens per profile,
`cache_read_input_tokens = 0` throughout. That is **$14.71 per Steam arm at the batch
price**, $29 for the masked/unmasked pair — about 2.7x the modelled figure. The
generator now records its own usage and prints the measured per-profile cost and
full-catalogue projection at the end of every run, so the next estimate is an
observation rather than a third model.

## By-product worth keeping even if no dataset is added

This survey is a directly usable defence for §1's "why MovieLens 20M, a decade on". The
paper currently asserts the genome is unmatched; it can now also state, from measurement,
that **no Amazon category combines the per-user density and the catalogue size that a
full-ranking content benchmark needs** — 28 categories checked, zero qualifying.

## Reproducing

```bash
python3 tools/recon_amazon.py --category Video_Games --k-core 10
```

Caches ~700 MB on first run (`benchmark/0core/rating_only/<Cat>.csv` and
`raw/meta_categories/meta_<Cat>.jsonl`), then runs offline. Writes `recon_<Cat>.json`.
