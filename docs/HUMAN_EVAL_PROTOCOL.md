# Human Evaluation Protocol — Task Assignment Guide

> **Purpose:** This document is the step-by-step operational protocol behind the human evaluation in §4 of the paper (Table 3). It is reproduced here so that other researchers can replicate the evaluation on their own LLM-generated profile corpora.
>
> **Timeline:** about two weeks (1 day prep, 1 day pilot, ~33 hours of main-round rating spread over 8-10 days, 1 day analysis)
>
> **Cost:** ~$1,300 total (3 annotators × ~36 hours × $12/hour). An annotator rates BOTH
> providers' profile for each of the 500 sampled films, so 1,000 profiles and 5,000
> ratings; at the two-minutes-per-profile pace budgeted below that is ~33 hours of rating,
> plus the pilot, the calibration meeting and the mood task.

---

## Phase 0: Preparation (Day 1)

### 0.1 Generate the Sample

Run the sampling script to select 500 profiles + 30 pilot profiles:

```bash
python3 scripts/prepare_human_eval.py
```

This creates seven files; the six sheets are CSV — the release ships no spreadsheet:
- `human_eval/sample_500_movie_ids.json` — the 500 films of the main round, in order.
  It is byte-identical to the released file of the same name: the draw reproduces the
  sample our annotators rated.
- `human_eval/pilot_30.csv` — 30 profiles for calibration (same for all 3 annotators)
- `human_eval/main_500.csv` — 500 profiles for the main round
- `human_eval/mood_pairs_100.csv` — 100 movie pairs for mood validation
- `human_eval/mood_pairs_answer_key.csv` — the signed key those pairs are scored against
- `human_eval/pilot_annotation_template.csv` — blank annotation sheet for the pilot round
- `human_eval/main_annotation_template.csv` — blank annotation sheet for the main round

The sample is stratified by genre, and genres are MovieLens's: the profile records
carry none, so fetch MovieLens first (`bash scripts/download_ml20m.sh`).

The filled sheets from our own three annotators ship on the data repository as
`human_eval/annotator_sheet_filled_A{1,2,3}.csv` and
`human_eval/mood_pairwise_sheet_filled_A{1,2,3}.csv`; the blanks above are what
this script regenerates so a reader can run the protocol themselves.

### 0.2 Prepare Reference Materials

For each of the 500 + 30 profiles, `prepare_human_eval.py` writes into
`main_500.csv` and `pilot_30.csv`:
- The generated `profile_text`
- The movie title and year
- A TMDb reference summary (`tmdb_overview`)
- The LLM-assigned `mood_vector`

The rating sheets themselves (`scripts/build_human_eval_sheets.py`, see Annotation
Spreadsheet Format) carry the two profiles and the rating columns. The TMDb summary is filled from the TMDb cache that
`src/profile_generator/tmdb_crawler.py` builds with your own TMDb API key: TMDb text
is not redistributed, so the profile records do not carry it, and without the cache
the `tmdb_overview` column is left empty and annotators look the film up on TMDb.

### 0.3 Recruit Annotators

**Requirements:**
- Graduate students in CS, Information Science, or related field
- Familiarity with movies (no domain expertise required — a TMDb reference is provided; see 0.2)
- Available for ~36 hours over about two weeks (see the cost note above)
- Can work independently (no collaboration during annotation)

**Compensation:** $12/hour (document local minimum wage compliance)

---

## Phase 1: Pilot Round (Day 2)

### 1.1 Distribute Pilot Materials

Give each annotator:
1. **This rating guide** (print or share digitally — see §Rating Scale below)
2. **`pilot_30.csv`** — same 30 profiles for all 3 annotators
3. **Blank annotation spreadsheet** (one per annotator)

### 1.2 Independent Rating

Each annotator independently rates all 30 pilot profiles on 5 axes (1-5 scale). Expected time: ~60-70 minutes.

**Important:** Annotators must NOT discuss ratings with each other before submitting.

### 1.3 Calibration Meeting (30-60 minutes)

After all 3 submit pilot ratings:

1. **Compute initial Fleiss' kappa** per axis. The released agreement figures come from the filled sheets under `human_eval/` on the HuggingFace dataset; any standard implementation (e.g. `statsmodels.stats.inter_rater.fleiss_kappa`) reproduces them from those files.
2. **Identify disagreement cases** — profiles where annotators differ by ≥2 points
3. **Discuss disagreements as a group** — understand why annotators diverged
4. **Clarify edge cases** — document decisions as addenda to the guidelines:
   - If annotator hasn't seen the movie → rate based on TMDb reference summary
   - Borderline generic terms like "compelling" or "haunting" → NOT rule violations
   - Multi-genre movies → rate thematic accuracy based on primary genre characterization
5. **DO NOT change existing guidelines** — only add clarifications

Record the calibration outcomes. The pilot exists to reconcile anchor
interpretation; its ratings are **discarded rather than reported**, so no pilot
figure appears in the paper and none is released.

---

## Phase 2: Main Annotation Round (about 8-10 days)

### 2.1 Distribute Main Materials

Give each annotator:
- **The rating sheet** for all 500 films, each carrying two profiles to rate (one per
  provider, blinded as `profile_a` / `profile_b`), so 1,000 profiles per annotator; same
  set for all 3. `scripts/build_human_eval_sheets.py` builds it from the sample
  `prepare_human_eval.py` draws and from both providers' profiles; `main_500.csv` itself
  carries one profile per row, with its TMDb summary and mood vector
- **Updated guidelines** with calibration addenda
- **Annotation spreadsheet** (one per annotator)

### 2.2 Annotation Schedule

Each profile takes ~2 minutes to rate across all 5 axes, and each row carries two of them,
so budget ~4 minutes per row and ~33 hours for the 500 rows. Pace it to avoid fatigue: about
4 hours a day over 8-10 days, in blocks of no more than 2 hours, with 10-second breaks between
profiles. Do the mood pairwise task (~45 minutes) last.

### 2.3 Mood Pairwise Task (last day of the main round)

For 100 movie pairs, each annotator answers:

> "Which movie should score higher on [axis]?"
> Options: Movie A / Movie B / Too close to call

This is done for all 10 mood axes per pair = 1,000 judgments per annotator. Expected time: ~45 minutes.

---

## Rating Scale (Give This to Annotators)

### What You Will See

For each movie, you receive:
1. **Title and year** (e.g., "Toy Story (1995)")
2. **Two generated profiles** — one per system, each a short semantic
   description (95-135 words in this corpus), in the columns
   `profile_a_text` and `profile_b_text` --- you are not told which model wrote
   which, and you do not need to know
3. **TMDb reference** — a brief official plot summary for fact-checking
4. **Mood vector** — 10 numerical values in [-1, 1] describing the movie's tone
   (listed in `main_500.csv`; no rated axis asks about it)

### What You Rate

Rate **each of the two profiles** on the same 5 axes using a 1-5 scale — ten
ratings per movie. The comparison is blinded to model identity: the columns are named `profile_a` and `profile_b`, and the mapping is applied by the script and held in `annotator_sheet_key.json`, which annotators do not receive. The mapping is fixed across rows, so identity is blinded and column order is not randomised.

---

#### Axis 1: Thematic Accuracy
*"Does the profile correctly capture the movie's themes?"*

| Score | Meaning | Example |
|-------|---------|---------|
| 1 | Completely wrong themes; describes a different type of movie | Profile describes a romantic comedy, but the movie is a war drama |
| 2 | Mostly incorrect; one minor element may be right | Mentions "family" but misses that the movie is primarily about political corruption |
| 3 | Partially correct; captures some themes but misses key aspects | Gets the genre right but misses the central conflict or message |
| 4 | Mostly correct; accurately captures the main themes with minor omissions | Captures the main themes well, might miss a subtle secondary theme |
| 5 | Fully accurate; the thematic characterization is precise and complete | Nails every major theme and their interplay |

---

#### Axis 2: Discriminativeness
*"Could you identify which movie this profile describes?"*

| Score | Meaning | Example |
|-------|---------|---------|
| 1 | Completely generic; could describe hundreds of movies | "A thrilling adventure with memorable characters" |
| 2 | Mostly generic with one vaguely specific detail | "A space adventure involving a father-son relationship" |
| 3 | Somewhat specific; narrows to a small group of similar movies | "An animated adventure about living toys who fear being replaced" — could be Toy Story or a few others |
| 4 | Highly specific; a knowledgeable viewer could likely identify the movie | Mentions specific plot dynamics that strongly suggest one film |
| 5 | Uniquely identifying; can only refer to this particular movie | Describes unique plot elements, setting, and tone combination that matches exactly one film |

---

#### Axis 3: Rule Compliance
*"Does the profile follow the generation rules?"*

**Rules:** The profile must NOT contain: actor/actress names, director names, release years, numerical ratings, box office figures, or banned generic filler phrases.

| Score | Meaning | Example violation |
|-------|---------|-------------------|
| 1 | Multiple violations | "Tom Hanks stars in this Spielberg-directed 1998 war epic rated 8.5/10" |
| 2 | One major violation | "Leonardo DiCaprio's portrayal of..." |
| 3 | One minor violation | Uses a slightly generic filler phrase |
| 4 | No violations; follows all rules correctly | Clean, rule-compliant text |
| 5 | No violations and exemplary adherence to semantic flow | Perfect structure and no borderline cases |

---

#### Axis 4: Factual Consistency
*"Any hallucinated or incorrect claims?"*

Use the TMDb reference to check. If you know the movie, use your knowledge too.

| Score | Meaning | Example |
|-------|---------|---------|
| 1 | Multiple factual errors or hallucinated plot elements | Claims the movie is set in space when it's set in New York |
| 2 | One significant factual error | Says the protagonist is a doctor when they're a lawyer |
| 3 | Mostly accurate but one minor inaccuracy | Slightly wrong about a secondary plot point |
| 4 | Fully accurate; all claims match the movie | Everything checks out against TMDb |
| 5 | Fully accurate with notable precision | Impressively detailed and correct characterization |

---

#### Axis 5: Coherence & Fluency
*"Is the text well-structured and readable?"*

| Score | Meaning |
|-------|---------|
| 1 | Incoherent; sentences don't connect logically |
| 2 | Poorly structured; awkward phrasing or disjointed flow |
| 3 | Adequate; readable but with noticeable structural issues |
| 4 | Well-written; clear semantic flow with minor room for improvement |
| 5 | Excellent; polished, natural flow |

---

## Phase 3: Analysis (1 day)

### 3.1 Collect Spreadsheets

Collect completed annotation spreadsheets from all 3 annotators. Verify completeness:
- 30 pilot profiles × 5 axes × 2 systems × 3 annotators = 900 pilot judgments,
  **discarded after calibration** — they are not merged into the results and are
  not released
- 500 main profiles × 5 axes × **2 systems** × 3 annotators = **15,000** main judgments
- 100 pairs × 10 axes × 3 annotators = 3,000 pairwise judgments

The released sheets satisfy the last two exactly: `annotator_sheet_filled_A{1,2,3}.csv`
carry 500 rows × 10 rating columns each, and `mood_pairwise_sheet_filled_A{1,2,3}.csv`
carry 100 pairs × 10 axes each, with no blank and no out-of-range cell.

### 3.2 Compute Metrics

The figures of the paper's human-evaluation table are reproducible from the filled
sheets, and a script ships that recomputes them:

```bash
python3 scripts/download_artifacts.py --human-eval   # the filled sheets, into data/human_eval/
python3 scripts/regenerate_consolidated.py           # writes data/human_eval/consolidated/
```

It decodes the blinded columns through `annotator_sheet_key.json` and writes two
files. `summary_table5.csv` holds, per system and axis, the mean, standard
deviation, Fleiss' kappa, linearly weighted kappa and ≥4 rate;
`per_annotator_ratings.csv` holds every rating in long form. The paper's
human-evaluation table (`paper/generated/human_eval_table.tex`) is generated from
`summary_table5.csv`: its cells are that file's figures for the released
profiles (Claude Haiku 4.5), and Δ mean is the difference between their mean and
the comparator's. Fleiss' kappa can also be checked with a standard
implementation, independently of the script:

```python
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
# per axis, per system: stack the three annotators' columns and aggregate
```

Verified against the released sheets: Fleiss' kappa reproduces the paper's five
per-axis figures exactly, and both reported ranges.

The pairwise mood task has its own checker, because its agreement has two
definitions: the paper reports the figure where at least two annotators chose the
same side, and the checker also computes the stricter majority-of-three figure:

```bash
python3 scripts/download_artifacts.py --human-eval   # the filled sheets, into data/human_eval/
python3 tools/verify_mood_agreement.py
```

It recomputes the pairwise mood numbers from the three pairwise sheets and the key
and checks each against the paper's value or the generated macro behind it (exit 0
all match, 1 a disagreement, 2 a sheet missing).

The key's `llm_ordering` field is informational and the checker does not read it. It
labels an axis `T` (too close to call) when the two profiles' mood values differ by
less than 0.1 in floating point, so pairs that differ by exactly 0.10 carry either
label (62 of the 99 such cells are `T`).

What to compute:
1. **Fleiss' kappa** per axis (inter-annotator agreement)
2. **Mean ± std** per axis (profile quality)
3. **≥4 rate** per axis (fraction rated good or excellent)
4. **Mood pairwise accuracy** (% agreement with LLM-assigned ordering), over the cells
   where the model's two values differ, two ways: where at least two annotators chose
   the same side, how often that side is the model's (a side is binary, so chance is
   50%); and under a majority of all three votes, which counts a "too close" majority
   as disagreement and so has no 50% chance line
5. **One-way ANOVA** per axis by genre group (genre effect test)
6. **Score distribution histograms** per axis

### 3.3 Interpret Results

| Fleiss' κ | Interpretation |
|-----------|----------------|
| < 0.20 | Poor |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

**What the paper's evaluation found** (Table 3): Fleiss' κ from 0.27
(discriminativeness) to 0.50 (coherence and fluency), fair to moderate, with rule
compliance and coherence the most agreed on and the two judgement-heavy axes
(thematic accuracy, discriminativeness) the least; means from 3.96 to 4.58. The
linearly weighted κ (0.40-0.54) credits near-misses that Fleiss' κ counts as
disagreement.

### 3.4 Paper integration

After analysis, the per-axis means, κ and ≥4 rates feed the human-evaluation table
(`paper/generated/human_eval_table.tex`, generated from `summary_table5.csv`) and the
section text. Distributions and the genre breakdown are not printed in the paper;
`per_annotator_ratings.csv` holds everything needed to compute them.

---

## Annotation Spreadsheet Format

Each annotator fills in one spreadsheet with these columns:

`eval_id`, `movieId`, `title`, then the two profile texts, then five rating
columns per system, then `notes` and `primary_genre`:

| eval_id | movieId | title | profile_a_text | profile_b_text | profile_a_thematic_accuracy | … | profile_b_coherence_fluency | notes | primary_genre |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | Toy Story (1995) | … | … | 5 | … | 4 | | other |

The five axis names repeat under each system prefix, so there are ten rating
columns: `profile_a_*` and `profile_b_*` for thematic_accuracy, discriminativeness,
rule_compliance, factual_consistency and coherence_fluency.

The `notes` column is optional — annotators can record edge cases or justifications for borderline ratings.

---

## Checklist

- [ ] Sample 500 + 30 profiles (`scripts/prepare_human_eval.py`)
- [ ] Prepare annotation spreadsheets (3 copies)
- [ ] Recruit 3 graduate student annotators
- [ ] Distribute pilot materials (30 profiles)
- [ ] Collect pilot annotations
- [ ] Run calibration meeting, document addenda
- [ ] Distribute main materials (500 profiles + mood pairs)
- [ ] Collect main annotations (check completeness)
- [ ] Run analysis script
- [ ] Update the paper's human-evaluation table and section with the analysis output (κ per axis, ≥4 rate)
- [ ] Archive raw annotation data for reproducibility
