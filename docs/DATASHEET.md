# Datasheet: LLM-MovieLens

Following the framework of [Gebru et al. (2021)](https://arxiv.org/abs/1803.09010).

**Scope.** This datasheet documents the **ML-20M instantiation** — 10,381 film
profiles, the catalogue every result in the paper is computed on. The release also
ships a **second instantiation on a catalogue with no shared metadata source**:
9,289 Amazon-Books profiles (of 9,332 items; the 43 that failed all three attempts,
most for want of a title and a usable description, are listed in `failed_iids.json`), which exist as portability evidence rather than as a
second set of results. Their composition follows the same schema and is inventoried
in the dataset repository's `RELEASE_MANIFEST.md`. A third directory,
`embeddings/ml1m/`, is a re-indexing of the ML-20M profiles onto the MovieLens 1M
item set (2,807 rows, 19 of them all-zero where no ML-20M profile maps) and not a
separate generation.

## Motivation

**For what purpose was the dataset created?**
To provide a benchmark for evaluating LLM-generated features in collaborative filtering recommendation systems. The dataset enables researchers to study whether LLM reasoning over metadata produces better item representations than traditional content features.

**Who created the dataset and on behalf of which entity?**
Tan Nghia Duong and Manh Hoang Tran (Hanoi University of Science and Technology; Minastik Technology JSC).

**Who funded the creation of the dataset?**
Minastik Technology JSC funded this work in full, including the LLM API costs (~$21 for Claude Haiku 4.5 over the whole catalogue, via the Batches API). The funder placed no restriction on publication or on release of the benchmark. See the paper's Declarations for competing interests.

## Composition

**What do the instances that comprise the dataset represent?**
Each instance is a movie with:
- An LLM-generated semantic profile (80–120 words requested; realised 95–135, median 115)
- A 10-axis continuous mood vector
- 3-6 key discriminative themes - the prompt sets no theme count and shows three in each
  of its two examples; the realised counts are 3 for 6,688 records, 4 for 2,857, 5 for 832
  and 6 for four of them
- Pre-computed embeddings (1024-dim profile, 10-dim mood, 528-dim themes)

**How many instances are there in total?**
10,381 movie profiles covering genome-tagged movies from MovieLens 20M.

**Does the dataset contain all possible instances or is it a sample?**
It covers all 10,381 movies in MovieLens 20M that have genome tag annotations. This represents ~38% of movies but ~99% of all ratings in the dataset.

**What data does each instance consist of?**
These are the keys the shipped file actually has, not the schema the pipeline
works with internally:

- `movieId`: Integer identifier from MovieLens 20M
- `title`: Movie title, the release year stripped (e.g., `"Toy Story"`) — except in 160
  records, whose title keeps a trailing year
- `profile`: LLM-generated semantic description - 80-120 words requested,
  95-135 realised (see the realised distribution below)
- `word_count`: the realised length of `profile`, so the distribution above can be
  re-derived without re-tokenising
- `mood_vector`: 10 floats in [−1, 1], one per named bipolar axis (e.g.
  `dark_light`), stored as a dict keyed by axis name. About half the values are
  negative (51,365 of 103,810); nothing rescales them, in the JSON or the arrays
- `key_themes`: 3-6 string labels identifying dominant themes. **Variable length** -
  35.6% of ML-20M records carry more than three, so do not unpack to a fixed 3

**There is no `genres` field**, and there is no other MovieLens-derived content:
genre strings are upstream data this release does not redistribute, and
`RELEASE_MANIFEST.md` records where the raw multi-label string is dropped. The
Croissant record in `metadata/croissant.json` declares the four scalar keys as fields
and describes the two nested ones, `mood_vector` and `key_themes`, in its record-set
description.

The parallel Amazon-Books file (`profiles/amazon_books_2018/...`) carries the same
generated fields with `asin` and `itemId` in place of `movieId`. It is less complete:
861 of its 9,289 records have a null `title`, including all 430 whose ASIN begins
with "B"; 10 have an all-zero mood vector, 9 of them null-titled; and one of those
carries an extra `note` field in which the model says it set the mood vector to zero
for lack of metadata. Those 10 were written from almost no input and their text is
generic; they are listed in `profiles/amazon_books_2018/claude-haiku-4-5/known_issues.json`.

**Is there a label or target associated with each instance?**
No explicit labels. The profiles are features, not targets. Evaluation is performed using user interaction data from MovieLens 20M.

**Is any information missing from individual instances?**
Not in the MovieLens file: all 10,381 profiles are complete. The Amazon-Books file
has the null titles and all-zero mood vectors described above.

**Are there recommended data splits?**
Yes, and they are specified here rather than shipped: the MovieLens interactions
are used, not redistributed, so rebuild them with `tools/rebuild_splits.py`, which
reproduces these cuts and SHA-256-checks the result against ours.
- Train: < 2014-01-01 (11,499,778 interactions)
- Validation: 2014-01-01 to 2014-07-01 (49,668 interactions)
- Test: > 2014-07-01 (67,466 interactions)

**Are there any errors, sources of noise, or redundancies?**
The generating model does not reliably reproduce the item identifier: in a
766-response sample it never did, emitting nulls, title slugs or year-derived
numbers instead. The pipeline therefore overwrites the identifier field with the
catalogue's own id for every record before parsing, so identifiers in the release
come from MovieLens rather than from the model, and the released
`movie_profiles.json` has 0/10,381 mismatches. LLM-generated text may still
contain minor factual inaccuracies for obscure films.

Some profiles were generated from **another film's metadata**. The TMDb record is
joined through MovieLens's own `links.csv`, and for a few films that file names a
different film's record. Five screens compared MovieLens's record of each of the
10,299 joined films with the joined record (release year, the language of the
original title, popularity, a documentary label, runtime); each of the 194 flagged
was read against MovieLens's title, genres and genome tags. Ten carry another
film's record, and in six the profile describes that film — *Finding North* (1998),
a comedy-drama, carries a profile of a 2012 documentary on food insecurity. They are
listed, with how each was found, in
`profiles/ml20m/claude-haiku-4-5/known_issues.json`. A wrong record that agrees with
MovieLens on all five screens passes them, so ten is a floor, not a count. They are
not regenerated: a regenerated profile would no longer match the released
embeddings and results.

The prompt bans seven generic phrases ("must-see", "iconic", ...), but nothing checks
the output against the list: 51 released profiles contain one of them, a string count
that includes narrative uses such as "her beloved". Theme count is neither requested
nor checked (the prompt's two examples carry three), and four records carry six.

**Is the dataset self-contained?**
Partly, and the split is deliberate. Everything we *made* is self-contained: the
profiles, the mood vectors, the theme matrices and the profile embeddings. Two
classes of input are not, because they are transformations of someone else's data
rather than things we made, and redistributing them is not ours to grant. The
**baseline features** — genome-PCA, genome-raw and the title embedding — are
regenerated by the shipped harness (`src/embedding_generator/main.py`) from the
reader's own download, as are the **benchmark splits** (`tools/rebuild_splits.py`,
which verifies the result against the SHA-256 of the files the experiments used). The benchmark evaluation requires MovieLens 20M interaction data, which must be downloaded separately from GroupLens (free, non-commercial license).

## Collection Process

**How was the data associated with each instance acquired?**
Profiles were generated by Claude Haiku 4.5 (Anthropic API). Each prompt names the film
by title, year and genres (from MovieLens 20M), then carries four inputs:
1. Top-30 genome tags with relevance scores (from MovieLens 20M)
2. TMDb metadata: plot overview, keywords, director, cast, runtime, vote average and
   vote count
3. Two MovieLens aggregates: mean rating and rating count
4. A summary of the film's MovieLens user tags

The labelled aggregates (TMDb vote average and count, MovieLens mean rating and
count) are the subject of the paper's §6 control, which regenerates every profile
without them (`PROFILE_PROMPT=no-aggregates`).

**What mechanisms or procedures were used to collect the data?**
An automated pipeline: TMDb metadata was crawled via the TMDb API, combined with the MovieLens fields above, and sent to Claude Haiku 4.5 with a structured system prompt requesting embedding-optimized output.

**Who was involved in the data collection process?**
The process was fully automated. No human annotators were involved in profile generation.

**Over what timeframe was the data collected?**
- TMDb metadata: March 2026
- LLM profile generation: April 2026 (13 min 12 s measured for a 10,000-request batch)
- MovieLens 20M interactions: 1995-2015 (original dataset timeframe)

**Were any ethical review processes conducted?**
No formal IRB review was required as no human subjects data was collected. The dataset contains only item-side features (movie descriptions).

## Preprocessing/Cleaning/Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
1. The item identifier is overwritten with the catalogue's own id for every
   record before parsing (the model does not reliably reproduce it)
2. Profile text was generated as embedding-optimized (high semantic density)
3. Mood vectors are the model's own values in [−1, 1]; nothing rescales them
4. Themes were aggregated into a 528-word vocabulary across all profiles

## Uses

**Has the dataset been used for any tasks already?**
Yes, in the accompanying paper for benchmarking fourteen recommendation configurations across five CF architectures.

**What (other) tasks could the dataset be used for?**
- Cold-start recommendation research
- Movie similarity and clustering
- Multi-modal recommendation (combining with visual features)
- LLM prompt engineering research
- Mood-based recommendation systems

**Is there anything about the composition of the dataset or the way it was collected that might impact future uses?**
- Profiles reflect Claude Haiku 4.5's knowledge and biases (April 2026)
- English-only coverage limits cross-lingual research
- Genome tag coverage (10,381 of 27,278 movies) biases toward popular/well-annotated films

**Realised distribution vs. the requested spec.** The prompt asks for 80–120 words
and sets no theme count (its two examples show three themes each). The released ML-20M
profiles realise **95–135 words** (median 115; 20.2% exceed 120, none fall short of 80)
and **3–6 themes** (four records carry six). The generator's validator accepts a small
tolerance either side of the word target, which is why the word counts differ from the
request; the ML-20M validator checks no theme count (the book and game validators accept 3 to 5). Stated here because a user filtering on the spec would
otherwise be surprised by a fifth of the catalogue.

**Identity-handle ablation.** The first 1,000 items of each catalogue
by identifier were generated a second time with the item's structured identity
withheld — title, year, genres, director and cast on ML-20M (the genres share the
title line); title, developer and publisher on Steam — keeping every other field. Encoding both arms with `bge-large-en-v1.5`, we ask whether a profile written
without the identity still retrieves its own unmasked version.

| | ML-20M | Steam |
|---|--:|--:|
| median item payload (chars / 3.8) | ~365 tok | ~61 tok |
| paired items | 1,000 | 998 of 1,000 attempted |
| masked → own profile, recall@1 | 94.1% | 26.3% |
| recall@10 | 99.3% | 66.1% |
| median rank | 1 | 4 |
| MRR | 0.963 | 0.393 |

With identity withheld, an ML-20M profile is still determined by the rest of its
metadata; a Steam profile is not. The payload row excludes the system prompt, which
is billed on every call: total input per ML-20M profile is 2,012 tokens. Below
roughly a hundred tokens of item metadata, an 80–120-word request asks the model to
supply most of the content itself.

*Why 998.* Two games produced no masked profile. With identity withheld, the whole
remaining input for *Hot Dish* is `genres: none, tags: [Casual], specs:
[Single-player]` and for *Mightier* `genres: [Indie], tags: [Indie], specs:
[Single-player, Game demo]`; the model declined, saying the input was insufficient.
Both generated normally with the title present. Reported as refusals rather than
pressed past — they are the boundary condition at its limit.

*Caveat.* This withholds **structured** identity only. ML-20M free text still leaks
it (the plot names characters, user tags carry actor names), so 94.1% means the
title is redundant when the description already determines the profile — not that
the model uses no prior knowledge. Source record:
`src/identity_ablation.json`.

**Measured bias audit** (from the 500-profile / three-annotator human
evaluation = 1,500 judgements, each the mean of the five rated axes; overall 4.27):

| split | mean quality |
|---|--:|
| least-rated third of items | 4.28 |
| middle third | 4.28 |
| most-rated third | 4.24 |
| comedy (weakest genre) | 4.10 |
| drama | 4.26 |
| documentary | 4.32 |
| horror | 4.33 |
| action (strongest genre) | 4.34 |

The popularity thirds are thirds of the 500 sampled films, ordered by their number of
ratings in the full ML-20M ratings file (500 films, 167/167/166 per third); the genre
rows use the sheet's `primary_genre`. Quality is **flat across item popularity** — no penalty on cold items, which is
where a content feature is most needed — and **varies by genre**, a 0.24-point
spread on a five-point scale. Small but systematic; a user should know which end
their catalogue sits on.

Three exposures we can name but not quantify: profiles are English-only, written by
an English-centric model over a catalogue whose composition skews Western; the three
annotators were not demographically sampled, and the blinded columns kept a
fixed order rather than a randomised one;
and the ten mood axes are researcher-chosen bipolar scales encoding one framing of
how items differ. Source record: `src/fairness_audit.json`.

## Distribution

**How will the dataset be distributed?**
- HuggingFace Hub: https://huggingface.co/datasets/minastik-ai/llm-movielens
- GitHub (code only): https://github.com/minastik-ai/llm-movielens

**When will the dataset be distributed?**
It is public, archived on Zenodo under DOI 10.5281/zenodo.22766653, which always
resolves to the newest archived record.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license?**
- LLM-generated profiles and embeddings (`profiles/`, `embeddings/`), and the six `human_eval/` files that reproduce generated profile text or mood vectors (`annotator_sheet.csv`, the three `annotator_sheet_filled_A*.csv`, `gpt4o_mini_profiles.json` and `mood_pairwise_key.json`): CC BY-NC 4.0.
  They are our output, but generated from inputs whose terms are non-commercial
  (MovieLens, TMDb) or unstated (RLMRec's book descriptions); research use is
  unrestricted
- Everything else we made — code, harness, results, the rest of the human evaluation, metadata and
  documentation: CC BY 4.0
- MovieLens 20M: GroupLens Research terms; not redistributed
- TMDb metadata: TMDb Terms of Use; not redistributed. This product uses the TMDb API
  but is not endorsed or certified by TMDb.

`LICENSE` in either repository is the authority.

## Maintenance

**Who is supporting/hosting/maintaining the dataset?**
The authors, via HuggingFace Hub and GitHub.

**How can the owner/curator/manager of the dataset be contacted?**
Via GitHub issues or the corresponding author's email (listed in the paper).

**Will the dataset be updated?**
We plan to release embeddings from additional encoders and updated profiles from newer LLMs as future versions.
