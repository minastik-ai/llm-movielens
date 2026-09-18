# Artifact verification — 2026-09-02

Ran against the released data files and the generating code. Everything below was
*measured*, not read off a README. One defect was found, in the paper's prose
rather than in the data.

**Status: the defect this audit found has since been corrected.** This file is kept
as the dated record of what was measured, not as a list of open problems, so the
section below still reads as it did on the day. The wording it proposed is the
wording the paper and the datasheet now carry — see "Output and validation" in the
paper and the generation section of [DATASHEET.md](DATASHEET.md). The one item still
open is the missing `batch_generator.py`, at the end of this file.

## Structural invariants — all hold

| check | result |
|---|---|
| **Row order** of every ML-20M array | `mood_vectors.npy` reconstructs from the profiles JSON in **sorted-movieId** order, **10,381/10,381 rows, max abs diff 0.0**. Insertion order matches only 162 — the gap is what makes this conclusive |
| Theme matrix | reconstructs from the profiles + vocabulary, **10,381/10,381** |
| Theme vocabulary | 528 entries; independently, exactly **528** raw themes occur in ≥10 films — the stated rule, confirmed from the data |
| `\numthemesraw` | 9,191 distinct raw theme strings — matches |
| Profile embeddings | L2 norm **min = max = 1.000000**, exactly as the paper states |
| Mood vectors across encoders | byte-identical between the two encoder directories (correct — mood is encoder-independent) |
| Amazon-Books counts | 9,289 profiles + 43 failed = **9,332 attempted** — matches all three macros |
| ML-1M profile index | 2,807 rows, **19 all-zero** (uncovered items) ⇒ 2,807 − 19 = **2,788** covered — matches `\numprofilesmlm` |
| Shapes | 1024-d profile, 10-d mood, 528-d themes; 1034 = 1024+10; 1562 = 1024+10+528 |
| `SHA256SUMS` | **55/55** verify after the derived-field repair |
| `croissant.json` | 5 hashed entries, **0 stale** |

## Defect: the "twenty-two regex repairs" claim is wrong

The paper (the profile-generation section) and the datasheet stated:

> Twenty-two of 10,381 records needed a regular-expression repair first, because a
> numeric movie title had corrupted the identifier field.

Three things are wrong with that.

**The code does not do what the sentence describes.** `generator.py` runs

```python
text = re.sub(r'"movieId"\s*:\s*[^,}\]]+', f'"movieId": {movie_id}', text, count=1)
```

**unconditionally on every record**, before JSON parsing. It neither detects nor
counts malformations, so it cannot produce a count of 22.

**The rate is not 22 of 10,381; it is effectively all of them.** In the 766
first-attempt responses surviving in `logs/prompts.jsonl`, the model emitted a
wrong identifier **766 times — 100%**. Deduplicated to 721 distinct films, still
100%.

**The mechanism is not "a numeric movie title".** The observed values are nulls,
title slugs (`"beverly_hills_cop_iii"`, `"la_confidential_1997"`,
`"the-black-hole-1979"`) and year-derived numbers (`1995001` for movie 175,
`1989001`, `3456`, `0`).

**The number 22 has no source.** It appears as a literal in three places and is
generated from nothing. The substitution never logged when it fired,
and no run log contains a repair record.

**The artifact is not affected.** Because the substitution is unconditional, the
released `movie_profiles.json` has **0/10,381** identifier mismatches. The data is
right; only the explanation is wrong — and the released code contradicts it in
three lines a reviewer reads for free.

### Suggested replacement — adopted

> The generating model does not reliably reproduce the item identifier: in a
> 766-response sample it never did, emitting nulls, title slugs or year-derived
> numbers instead. The pipeline therefore overwrites the identifier field with the
> catalogue's own id for every record before parsing, so identifiers in the release
> come from MovieLens rather than from the model.

That is accurate, it matches the code, and it documents a failure mode anyone
reusing the pipeline needs to know. The same failure mode, uncontrolled, is what
put `movieId: 99997` into all 10,381 GPT-4o-mini records (see the derived-field
repair of 2026-09-02).

## Generation-log audit (added 2026-09-02, second pass)

Prompted by the author's recollection that the first pass did not generate every
profile and that retries plus a regex fix were involved. **The recollection is
correct on all three counts**, and the audit also corrects an error in the section
above.

### Retries were real and substantial

From all 11 run logs, by distinct movie:

| problem class | distinct movies |
|---|--:|
| batch result `errored` | 261 |
| `FAILED after 3 attempts` (streaming) | 103 |
| batch validation error | 84 |
| API / connection error | 80 |
| streaming validation error | 33 |
| **union — any problem** | **423 of 10,381 (4.1%)** |

Validation errors by attempt number: **55 / 48 / 46** at attempts 1 / 2 / 3 — the
retry ladder the author remembered. Error taxonomy: **225** `JSON parse error:
Expecting ',' delimiter` and **56** `Profile too long: N words (max 120)`. All 125
`FAILED after N` lines are `N = 3`, so nothing was abandoned before exhausting the
retry budget, and the final artifact has all 10,381 profiles.

### CORRECTION: the paper's *mechanism* is right; I said otherwise above

The earlier section claimed the "numeric movie title" mechanism was wrong. **It is
right.** Movie 277 is *Miracle on 34th Street (1994)* and the model emitted
`"movieId": 34street1994` — an unquoted token that fails JSON parsing with exactly
the `Expecting ',' delimiter` error the logs are full of. Every breaking token
observed begins with a digit, **29 of 29 (100%)**:

```
  277 -> 34street1994        2019 -> 7samurai1954      2144 -> 16candles1984
  334 -> 42nd_street_...     2398 -> 34street1947      4907 -> 110th_street_1972
 1759 -> 4daysinSeptember1997  3634 -> 7daysinmay1964  4876 -> 13ghosts2001
```

The distinction that matters, and that the first pass missed: a *quoted* slug
(`"la_confidential_1997"`) and `null` are **valid JSON** — the regex silently
corrects them but parsing never fails. Only the unquoted, digit-initial tokens
break the parse, and those are exactly the numeric-title cases.

### The number 22 is nonetheless too low

**29 distinct movies with a JSON-breaking identifier are directly observable in
`logs/prompts.jsonl`, which covers only 721 of 10,381 movies (6.9%).** A total of
22 is impossible when 29 are visible in 7% of the catalogue. Scaling the observed
rate suggests a few hundred, but the log's sampling rule is undocumented, so treat
that as an estimate and 29 as the floor.

### Provenance is asymmetric between the two models

| | GPT-4o-mini | Claude (primary) |
|---|---|---|
| batch record | `batch_info.json`: 10,299 requests, 0 failed | 97 bytes, 103 requests |
| usage / cost | `usage.json` with full token + $ breakdown | absent |
| failure list | `failed_iids.json` | absent |
| retry record | `retry_via_standard_api` (1 movie), `fill_82_missing` | log lines only |

The **82** the paper cites *is* traceable: `failed_iids.json` lists 82
`missing_inputs_at_build`, and `scripts/gpt4omini_fill_82.py` generated them using
Claude's documented TMDb-fallback convention (10,299 + 82 = 10,381). The primary
run has no equivalent summary, which is precisely why "22" cannot be traced.

### `batch_generator.py` is missing from the release

The logs attribute 261 errored movies, every batch status poll, and 84 validation
errors to a `batch_generator` module. There is no `batch_generator.py` in the
source tree and no `batch_generator.*.pyc` in `__pycache__`. **A script that
produced part of the released artifact is not in the released code** — for a paper
whose contribution is the pipeline, that is a reproducibility gap independent of
the wording issues above.

What the release does carry for that stage is `batch_generate.py`, a Batches-API
driver for the same generation step, and `tools/verify_generator_e2e.py`, which
re-renders all 10,381 prompts with the shipped code and checks them against the
SHA-256 values in the released manifest. So the prompts behind the artifact are
reproducible from the released code; what is missing is the historical submission
and retry driver those logs were written by, and with it any way to re-derive the
per-attempt failure counts tabulated above.

## Reproducing

<!-- The released copy of this document ships in the code repository, where the
     tool is staged under tools/ and the data repository is a separate download. -->
```bash
# Both commands audit DATA, so they need a checkout or download of the dataset
# repository -- the code repository does not contain these files. Without one,
# the first reports CANNOT JUDGE (exit 2) rather than a silent pass over nothing.
python3 tools/fix_derived_fields.py --check --release-dir DATASET_DIR

# From inside DATASET_DIR:
shasum -a 256 -c SHA256SUMS                    # 55/55
```
