# Artifact verification

What was checked against the released data files and the generating code, and what
each check establishes. Every figure below was measured from the files, not read off
a README.

## Structural invariants

| check | result |
|---|---|
| **Row order** of every ML-20M array | `mood_vectors.npy` reconstructs from the profiles JSON in **sorted-movieId** order, **10,381/10,381 rows, max abs diff 0.0**. Insertion order matches only 162 rows, which is what makes the check conclusive |
| Theme matrix | reconstructs from the profiles + vocabulary, **10,381/10,381** |
| Theme vocabulary | 528 entries; independently, exactly **528** raw themes occur in ≥10 films, the stated rule, confirmed from the data |
| Raw themes | 9,191 distinct raw theme strings |
| Profile embeddings | L2 norm **min = max = 1.000000**, as the paper states |
| Mood vectors across encoders | byte-identical between the two encoder directories (mood is encoder-independent) |
| Amazon-Books counts | 9,289 profiles + 43 failed = **9,332 attempted** |
| ML-1M profile index | 2,807 rows, **19 all-zero** (uncovered items), so 2,807 − 19 = **2,788** covered |
| Shapes | 1024-d profile, 10-d mood, 528-d themes; 1034 = 1024+10; 1562 = 1024+10+528 |
| `SHA256SUMS` | **60/60** verify (every file in the data repository) |
| `croissant.json` | every hashed distribution matches its file |

## The item identifier is the catalogue's, not the model's

The generating model does not reliably reproduce the item identifier. In the 766
first-attempt responses kept in `logs/prompts.jsonl` (721 distinct films), it emitted
a wrong identifier every time: nulls, title slugs (`"beverly_hills_cop_iii"`,
`"la_confidential_1997"`, `"the-black-hole-1979"`) and year-derived numbers
(`1995001` for movie 175, `1989001`, `3456`, `0`). `generator.py` therefore runs

```python
text = re.sub(r'"movieId"\s*:\s*[^,}\]]+', f'"movieId": {movie_id}', text, count=1)
```

**on every record**, before JSON parsing, so identifiers in the release come from
MovieLens rather than from the model. The released `movie_profiles.json` has
**0/10,381** identifier mismatches.

Most of those wrong identifiers are still valid JSON (a quoted slug, `null`). The ones
that break parsing are unquoted tokens that begin with a digit, and they come from
titles that contain a number: movie 277, *Miracle on 34th Street (1994)*, came back
as `"movieId": 34street1994`, which fails with `Expecting ',' delimiter`. Every
JSON-breaking identifier visible in the log sample is of this kind, **29 of 29**:

```
  277 -> 34street1994        2019 -> 7samurai1954      2144 -> 16candles1984
  334 -> 42nd_street_...     2398 -> 34street1947      4907 -> 110th_street_1972
 1759 -> 4daysinSeptember1997  3634 -> 7daysinmay1964  4876 -> 13ghosts2001
```

The log covers 721 of 10,381 films (6.9%) under an undocumented sampling rule, so 29
is a floor for the catalogue, not a count. Without the overwrite, the same failure
mode leaves the model's value in the record: the GPT-4o-mini records carried a
few-shot example's id, which `tools/fix_derived_fields.py` replaces with each
record's own key.

## Generation retries

From all 11 run logs, by distinct movie:

| problem class | distinct movies |
|---|--:|
| batch result `errored` | 261 |
| `FAILED after 3 attempts` (streaming) | 103 |
| batch validation error | 84 |
| API / connection error | 80 |
| streaming validation error | 33 |
| **union: any problem** | **423 of 10,381 (4.1%)** |

Validation errors by attempt number: **55 / 48 / 46** at attempts 1 / 2 / 3. Error
taxonomy: **225** `JSON parse error: Expecting ',' delimiter` and **56** `Profile too
long: N words (max 120)`. All 125 `FAILED after N` lines are `N = 3`, so nothing was
abandoned before exhausting the retry budget, and the final artifact has all 10,381
profiles.

## Provenance of the two models' runs

| | GPT-4o-mini | Claude (primary) |
|---|---|---|
| batch record | `batch_info.json`: 10,299 requests, 0 failed | 97 bytes, 103 requests |
| usage / cost | `usage.json` with full token + $ breakdown | the cost record in `src/generation_economics.json` |
| failure list | `failed_iids.json` | the run logs |
| retry record | `retry_via_standard_api` (1 movie), `fill_82_missing` | log lines |

The GPT-4o-mini set has 10,381 profiles: `failed_iids.json` lists the 82 films whose
inputs were missing when the batch was built, and `scripts/gpt4omini_fill_82.py`
generated them with Claude's TMDb-fallback convention (10,299 + 82 = 10,381).

The per-attempt failure counts above come from the logs of the submission-and-retry
driver used for the Claude run, which is not part of the release. The release carries
`src/profile_generator/batch_generate.py`, a Batches-API driver for the same
generation step, and `tools/verify_generator_e2e.py`, which re-renders the prompts
with the shipped code and checks them against the SHA-256 values in the released
manifest. So the prompts behind the artifact can be re-rendered with the released code
and checked: exactly, under the pinned pandas (< 3.1), for the 82 films with no TMDb
record and for the system-prompt and template digests; for the other films only where the
reader's own TMDb crawl matches the one used at generation, which a later crawl usually
does not. The retry history is documented here rather than re-derivable.

## Reproducing

<!-- The released copy of this document ships in the code repository, where the
     tool is staged under tools/ and the data repository is a separate download. -->
```bash
# Both commands check DATA, so they need a checkout or download of the dataset
# repository -- the code repository does not contain these files. Without one,
# the first reports CANNOT JUDGE (exit 2) rather than a silent pass over nothing.
python3 tools/fix_derived_fields.py --check --release-dir DATASET_DIR

# From inside DATASET_DIR:
shasum -a 256 -c SHA256SUMS                    # 60/60
```
