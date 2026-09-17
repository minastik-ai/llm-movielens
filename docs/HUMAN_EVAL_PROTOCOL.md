# Human Evaluation Protocol — Task Assignment Guide

> **Purpose:** This document is the step-by-step operational protocol used for the human evaluation reported in the paper (the human-evaluation section and its appendix). The paper's human-evaluation
> table and that appendix carry the final 500-profile / 30-pilot results from applying this protocol. The protocol is reproduced here so that other researchers can replicate the evaluation methodology on their own LLM-generated profile corpora.
>
> **Timeline:** 5-7 days total (1 day prep, 1 day pilot, 3-4 days main round, 1 day analysis)
>
> **Cost:** ~$270 total (3 annotators × ~6 hours × $15/hour)

---

## Phase 0: Preparation (Day 1)

### 0.1 Generate the Sample

Run the sampling script to select 500 profiles + 30 pilot profiles:

```bash
python scripts/prepare_human_eval.py
```

This creates six files, all CSV — the release ships no spreadsheet:
- `human_eval/pilot_30.csv` — 30 profiles for calibration (same for all 3 annotators)
- `human_eval/main_500.csv` — 500 profiles for the main round
- `human_eval/mood_pairs_100.csv` — 100 movie pairs for mood validation
- `human_eval/mood_pairs_answer_key.csv` — the signed key those pairs are scored against
- `human_eval/pilot_annotation_template.csv` — blank annotation sheet for the pilot round
- `human_eval/main_annotation_template.csv` — blank annotation sheet for the main round

The filled sheets from our own three annotators ship on the data repository as
`human_eval/annotator_sheet_filled_A{1,2,3}.csv` and
`human_eval/mood_pairwise_sheet_filled_A{1,2,3}.csv`; the blanks above are what
this script regenerates so a reader can run the protocol themselves.

### 0.2 Prepare Reference Materials

For each of the 500 + 30 profiles, you need:
- The generated `profile_text`
- The movie title and year
- A TMDb reference summary (already in `movie_profiles.json` as `tmdb_overview`)
- The LLM-assigned `mood_vector`

These are all included in the CSV exports.

### 0.3 Recruit Annotators

**Requirements:**
- Graduate students in CS, Information Science, or related field
- Familiarity with movies (no domain expertise required — TMDb reference is provided)
- Available for ~6 hours over 5 days
- Can work independently (no collaboration during annotation)

**Compensation:** $15/hour (document local minimum wage compliance)

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

## Phase 2: Main Annotation Round (Days 3-5)

### 2.1 Distribute Main Materials

Give each annotator:
- **`main_500.csv`** — all 500 profiles (same set for all 3)
- **Updated guidelines** with calibration addenda
- **Annotation spreadsheet** (one per annotator)

### 2.2 Annotation Schedule

Recommended pacing (to avoid fatigue):
- **Day 3:** Profiles 1-200 (~3 hours with breaks)
- **Day 4:** Profiles 201-400 (~3 hours)
- **Day 5:** Profiles 401-500 + mood pairwise task (~2 hours)

Each profile takes ~2 minutes to rate across all 5 axes. Budget 10-second breaks between profiles.

### 2.3 Mood Pairwise Task (Day 5)

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
4. **Mood vector** — 10 numerical values (0-1) describing the movie's tone

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

## Phase 3: Analysis (Day 6)

### 3.1 Collect Spreadsheets

Collect completed annotation spreadsheets from all 3 annotators. Verify completeness:
- 30 pilot profiles × 5 axes × 2 systems × 3 annotators = 900 pilot judgments,
  **discarded after calibration** — they are not merged into the results and are
  not released, which is what the paper states
- 500 main profiles × 5 axes × **2 systems** × 3 annotators = **15,000** main judgments
- 100 pairs × 10 axes × 3 annotators = 3,000 pairwise judgments

The released sheets satisfy the last two exactly: `annotator_sheet_filled_A{1,2,3}.csv`
carry 500 rows × 10 rating columns each, and `mood_pairwise_sheet_filled_A{1,2,3}.csv`
carry 100 pairs × 10 axes each, with no blank and no out-of-range cell.

### 3.2 Compute Metrics

The released figures are reproducible from the filled sheets with any standard
implementation — no bespoke script is required, and none is shipped, because one
written for an earlier design is worse than none:

```python
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
# per axis, per system: stack the three annotators' columns and aggregate
```

Verified against the released sheets: Fleiss' kappa reproduces the paper's five
per-axis figures exactly, and both reported ranges.

What to compute:
1. **Fleiss' kappa** per axis (inter-annotator agreement)
2. **Mean ± std** per axis (profile quality)
3. **≥4 rate** per axis (fraction rated good or excellent)
4. **Mood pairwise accuracy** (% agreement with LLM-assigned ordering)
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

**Expected outcomes** (based on similar LLM evaluation studies):
- Rule compliance: highest kappa (most objective axis) — expect κ ≥ 0.70
- Discriminativeness: lowest kappa (most subjective) — expect κ ≥ 0.50
- Thematic accuracy: expect κ ≥ 0.60
- Mean scores: expect ≥ 4.0 for most axes
- Mood pairwise: expect ≥ 75% majority-vote accuracy

### 3.4 Paper integration

After analysis, the per-axis κ + ≥4 rate + sample-mean distributions feed:
- the human-evaluation results table — Likert means + Fleiss' kappa
- the human-evaluation section text — kappa values and interpretation
- the human-evaluation appendix — pilot kappa range, compensation hours, distributions, genre breakdown

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
- [ ] Update the paper's human-evaluation table, section and appendix with the analysis output (κ per axis, ≥4 rate, distributions, genre breakdown)
- [ ] Archive raw annotation data for reproducibility
