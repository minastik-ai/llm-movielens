# LLM Movie Profile Generator for MovieLens 20M

Generate **embedding-optimized** 80–120 word movie profiles for the **10,381 genome-covered movies** in MovieLens 20M using **Claude Haiku 4.5**, integrating top-30 genome tags by relevance score with crawled TMDb metadata. Each profile is designed to maximize downstream sentence-transformer embedding quality for collaborative filtering augmentation.

Embedding-optimized profiles with 10-axis mood vector.

## Architecture

```
┌───────────────────┐     ┌──────────────┐     ┌───────────────────────────┐
│   ML-20M Data     │     │   TMDb API   │     │  Claude Haiku 4.5         │
│                   │     │              │     │                           │
│ genome-scores.csv │     │ overview     │     │  System Prompt (1,389 tok)│
│ genome-tags.csv   │────▶│ cast/crew    │────▶│  + 2 few-shot examples    │
│ movies.csv        │     │ keywords     │     │  + User Prompt (per movie)│
│ ratings.csv       │     │ runtime      │     │                           │
│ tags.csv          │     │ vote_average │     │  Batches API, 50% off     │
│ links.csv         │     └──────────────┘     │  (caching inert: see Cost)│
└───────────────────┘                          └─────────────┬─────────────┘
                                                             │
                        ┌────────────────────────────────────┘
                        ▼
              ┌───────────────────────────────┐
              │      JSON Output (per movie)  │
              │                               │
              │  profile     80–120 words     │ → sentence-transformer → embedding
              │  key_themes  3 strings        │ → categorical features
              │  mood_vector 10 float axes    │ → direct 10-dim numeric feature
              │  word_count  integer          │ → quality metric
              └───────────────────────────────┘
```

## Profile Design Principles

The profile text is **not** a movie summary. It is an **embedding-optimized semantic fingerprint** engineered for a sentence-transformer to produce maximally discriminative item vectors.

| Principle | Implementation | Why It Matters |
|---|---|---|
| Lead with thematic essence | Sentence 1 = movie's core identity | Sentence-transformers weight early tokens more heavily; the first sentence anchors the embedding |
| Exclude structured metadata | No cast, director, year, ratings, runtime | These are separate feature columns; repeating them wastes embedding capacity on redundant information |
| Consistent semantic flow | theme → tone → style → distinction → audience | Ensures analogous information occupies similar positions across all 10,381 embeddings |
| Tight word count (80–120) | Well within sentence-transformer token windows (256–512 tokens) | Denser text produces sharper, less diluted embeddings than padding to 200 words |
| Discriminative vocabulary | Bans: "great film", "well-made", "iconic", "masterpiece" | Generic praise carries zero embedding information; every word must distinguish this movie from others |

## Quick Start

```bash
# 1. Install dependencies
cd llm-movie-profiler
pip install -r requirements.txt

# 2. Download MovieLens 20M
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
cd ..

# 3. Set API keys in .env file (loaded automatically via python-dotenv)
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
echo 'TMDB_API_KEY=your-tmdb-api-key' >> .env   # Free at https://www.themoviedb.org/settings/api

# 4. Dry run — assemble every prompt, call nothing
python batch_generate.py --dry-run

# 5. Full run — all 10,381 movies through the Batches API (~$21; see Cost)
python batch_generate.py

# 6. Resume: submit only movies not already in the output file
python batch_generate.py --resume

# --- synchronous path: ~2x the cost, but no batch queue latency -------------
# Better for a handful of movies; see Cost for when to prefer which.
python main.py --dry-run --limit 3
python main.py --movie-ids 1 50 318 593 2571
python main.py --skip-tmdb --resume
```

## Output Format

Each movie produces a JSON entry in `output/movie_profiles.json`:

```json
{
  "1": {
    "movieId": 1,
    "title": "Toy Story",
    "profile": "A luminous exploration of possessiveness and loyalty within the secret emotional lives of childhood playthings. The narrative channels rivalry and insecurity through a buddy-film structure where a displaced cowboy doll confronts an oblivious space ranger, generating tension from identity crisis rather than external threat. Tonally it balances slapstick physical comedy with genuine existential anxiety about obsolescence and replacement. Its fantasy world-building operates through strict internal rules about toy consciousness, grounding whimsy in emotional realism. Pacing moves briskly through escalating misadventures externalizing inner jealousy. Sits at the intersection of family animation and psychological character study. Rewards audiences who appreciate layered storytelling beneath accessible surfaces.",
    "word_count": 103,
    "key_themes": ["jealousy", "identity", "friendship"],
    "mood_vector": {
      "dark_light": 0.6,
      "serious_playful": 0.4,
      "slow_fast": 0.3,
      "cerebral_visceral": -0.2,
      "realistic_fantastical": 0.7,
      "intimate_epic": -0.3,
      "conventional_experimental": -0.4,
      "emotional_detached": 0.6,
      "nostalgic_contemporary": 0.1,
      "predictable_subversive": 0.2
    }
  }
}
```

Note: the profile contains **no cast names, no director, no year, no ratings** — only thematic and tonal synthesis. Structured metadata is provided as separate features downstream.

## 10-Axis Mood Vector

Each movie receives a 10-dimensional continuous vector on a `[-1.0, 1.0]` scale. This is a **direct numeric feature** — no embedding needed.

| Axis | -1.0 (left pole) | +1.0 (right pole) | Discriminative purpose |
|---|---|---|---|
| `dark_light` | Bleak, grim | Bright, uplifting | Separates *Requiem for a Dream* from *Toy Story* |
| `serious_playful` | Grave, solemn | Comedic, absurd | Separates *Schindler's List* from *Airplane!* |
| `slow_fast` | Contemplative, still | Frenetic, breathless | Separates *2001* from *Mad Max: Fury Road* |
| `cerebral_visceral` | Intellectual, philosophical | Sensory, action-driven | Separates *Primer* from *John Wick* |
| `realistic_fantastical` | Gritty naturalism | Pure fantasy/surreal | Separates *Manchester by the Sea* from *Lord of the Rings* |
| `intimate_epic` | Small-scale, personal | Grand, sweeping | Separates *Before Sunrise* from *Lawrence of Arabia* |
| `conventional_experimental` | Mainstream, familiar | Avant-garde, unconventional | Separates *Marvel* films from *Mulholland Drive* |
| `emotional_detached` | Cold, analytical | Intensely emotional | Separates Kubrick from Spielberg |
| `nostalgic_contemporary` | Classic, retro sensibility | Modern, zeitgeist | Separates period nostalgia from contemporary urgency |
| `predictable_subversive` | Formulaic, safe | Expectation-defying | Separates standard sequels from genre-bending films |

## Prompt Documentation

All prompts are logged to `logs/prompts.jsonl` for full reproducibility.

### System Prompt (1,389 tokens; see the cost section — it is not cacheable)

```
You produce embedding-optimized movie profiles for a recommendation system.
Each profile will be encoded by a sentence-transformer into a dense vector
used as an item feature in collaborative filtering. Your output quality
directly determines recommendation accuracy.

RULES — follow exactly:
1. The "profile" field is 80–120 words, one paragraph, no line breaks.
2. NEVER mention: actor/cast names, director names, release year, ratings,
   vote counts, runtime, box office, or awards. These exist as separate
   structured features. Your profile captures ONLY what requires semantic synthesis.
3. Sentence 1: the movie's thematic core and emotional identity.
4. Sentences 2–4: tonal texture, narrative approach, genre positioning,
   and what makes this film distinct from superficially similar ones.
5. Final sentence: the viewing experience or audience sensibility.
6. Every word must be discriminative. Ban: "a great film", "well-made",
   "entertaining", "must-see", "beloved", "iconic", "masterpiece".
7. Synthesize genome tags and plot into insight — never list tags or
   parrot the plot synopsis.

SEMANTIC FLOW (same order for every movie):
  thematic essence → emotional tone → narrative/visual style
  → genre distinction → audience experience

MOOD VECTOR — 10 axes, each float from -1.0 to 1.0:
  [dark_light, serious_playful, slow_fast, cerebral_visceral,
   realistic_fantastical, intimate_epic, conventional_experimental,
   emotional_detached, nostalgic_contemporary, predictable_subversive]

+ 2 few-shot examples (sci-fi and romcom) demonstrating exact output format
```

The full prompt text including few-shot examples is in `config/settings.py` → `SYSTEM_PROMPT`.

### User Prompt Template (per movie, ~306 tokens)

```
MOVIE: {title} ({year}) | {genres}

GENOME TAGS (top-30 by relevance):
   1. animation: 0.9876
   2. pixar: 0.9654
   ... (30 tags total)

PLOT: {overview}
KEYWORDS: {keywords}
DIRECTOR: {directors} | CAST: {cast} | RUNTIME: {runtime}min
SCORES: TMDb {vote_average}/10 ({vote_count}v) | ML {ml_avg_rating}/5 ({ml_rating_count}r)
USER TAGS: {user_tags_summary}

Generate the JSON profile. 80–120 words. No cast/director/year/ratings in profile text.
```

All metadata is provided so the LLM can *reason about it*, but the system prompt rules ensure only thematic/tonal synthesis appears in the profile text.

## Cost

Measured, not estimated. Every figure below comes from
[`../generation_economics.json`](../generation_economics.json), which records how
each one was obtained.

| Path | Script | Cost for 10,381 profiles |
|---|---|---|
| **Batches API (50% discount)** | `batch_generate.py` | **~$21** — the figure reported in the paper |
| Synchronous API | `main.py` | ~$42 |

TMDb calls are free (40 req/10s). Per profile: ~2,012 input and ~400 output
tokens, measured on a 5-movie test batch with fresh, uncached input. Projected
batch spend was $20.82; actual spend on a full regeneration was $20.66.

**Use `batch_generate.py` for a full run.** It is the path the paper's cost
figure assumes, and the only one that reproduces it. `main.py` submits the same
requests synchronously at roughly twice the price; it is the better choice for a
small test, because the batch queue's start-up latency dominates at low volume
(5 requests took 43m43s, while 10,000 took 13m12s).

### Prompt caching does not apply to this workload

An earlier version of this file claimed the synchronous path was cheaper than
batch, because a cached system prompt would be re-read at 0.1x input price. **That
is wrong, and the correction matters enough to record here rather than quietly
delete.**

Prompt caching never engages. Claude Haiku 4.5 has a **4,096-token minimum
cacheable prefix**, and this system prompt is **1,389 tokens** — below it. The
`cache_control` marker is accepted but creates no cache entry, and
`cache_read_input_tokens` summed **0** across all 10,381 requests. The saving the
old argument depended on does not exist, in either transport, so the 50% batch
discount is the only lever available and batch is unambiguously cheaper.

The retracted figures were `~$14` total and a `~1,327`-token system prompt; the
correct ones are `~$21` and `1,389` tokens, the latter measured with the
provider's `count_tokens` endpoint.

## Project Structure

```
llm-movie-profiler/
├── batch_generate.py        # Entry point (recommended) — Batches API, ~$21
├── main.py                  # Entry point — synchronous, ~$42; also hosts prepare_movie_data()
├── config/
│   ├── __init__.py
│   └── settings.py          # All configuration: prompts, API params, mood axes, paths
├── data_loader.py           # ML-20M data loading & top-30 genome tag extraction
├── tmdb_crawler.py          # Async TMDb API crawler with disk caching
├── generator.py     # Claude API client with prompt caching, validation, checkpointing
├── requirements.txt         # Pinned Python dependencies
├── data/
│   └── ml-20m/              # Place ML-20M CSV files here (6 files)
├── cache/
│   └── tmdb_metadata.json   # Cached TMDb responses (auto-generated)
├── output/
│   ├── movie_profiles.json          # Final output — all 10,381 profiles
│   └── movie_profiles_partial.json  # Checkpoint file (auto-generated)
└── logs/
    ├── prompts.jsonl         # Every prompt + response logged (reproducibility)
    └── run_*.log             # Timestamped execution logs
```

## Resume & Fault Tolerance

- **Prompt caching:** requested but inert. The 1,389-token system prompt is below Claude Haiku 4.5's 4,096-token minimum cacheable prefix, so no cache entry is created and `cache_read_input_tokens` is 0 on every request. See the cost section.
- **Disk checkpointing:** Profiles are saved every 50 movies (configurable via `CHECKPOINT_EVERY`). Resume with `python main.py --resume`.
- **TMDb cache:** All TMDb API responses are cached to `cache/tmdb_metadata.json`. Re-runs skip previously fetched movies.
- **Retry with backoff:** Failed Claude API calls are retried up to 3 times. Rate limit errors trigger progressive wait (30s, 60s, 90s).
- **Output validation:** Every LLM response is parsed and validated against:
  - JSON schema (6 required fields)
  - Word count (80–120, with ±10/15 tolerance)
  - Mood vector completeness (all 10 axes present)
  - Mood value range (each axis within [-1.0, 1.0])
  - Failed validation triggers automatic retry with the same prompt.

## Known Issue: the model does not reliably reproduce `movieId`

During generation, 22 out of 10,381 movies consistently produced invalid JSON on all retry attempts. The root cause: **when a movie title contains a number, Haiku generates a slug-like string instead of the integer `movieId`**.

**Example — "Gone in 60 Seconds" (movieId: 26322):**

What the model **should** output:
```json
{"movieId": 26322, "title": "Gone in 60 Seconds", ...}
```

What the model **actually** output:
```json
{"movieId": 60seconds_1974, "title": "Gone in 60 Seconds", ...}
```

`60seconds_1974` is not valid JSON — it's a bare unquoted string mixing the title's number with the year. This causes `json.loads()` to fail with `Expecting ',' delimiter` at character 17.

**Corrected 2026-09-15.** This section originally read as though the problem were
confined to 22 films with numbers in their titles. A later audit of the surviving
first-attempt responses (`logs/prompts.jsonl`) found the model emitted a wrong
identifier in **766 of 766** — 100%, and 100% again after deduplicating to 721
distinct films. The observed values are nulls, title slugs
(`the-black-hole-1979`) and year-derived numbers (`1995001`), not only the
numeric-title confusion below. The count 22 has no source in any run log.
The examples below are real; the scope they implied was not:**

| Movie ID | Title | Bad `movieId` output |
|---|---|---|
| 277 | Miracle on **34th** Street | `34street1994` |
| 2019 | **Seven** Samurai | `7samurai1954` |
| 2144 | **Sixteen** Candles | `16candles1984` |
| 4876 | **Thirteen** Ghosts | `13ghosts2001` |
| 26322 | Gone in **60** Seconds | `60seconds_1974` |
| 54997 | **3:10** to Yuma | `310toyuma2007` |
| 77846 | **12** Angry Men | `12_angry_men_1997` |
| ... | *(15 more movies)* | *(same pattern)* |

**Why retries didn't help:** The same movie + same prompt = same confusion. Haiku consistently made this mistake for these specific titles across all 3 retry attempts.

**Fix applied in `generator.py`:** Since we already override `movieId` with the correct value after parsing, a regex pre-processor replaces whatever the model put in the `movieId` field with the correct integer *before* JSON parsing:

```python
text = re.sub(r'"movieId"\s*:\s*[^,}\]]+', f'"movieId": {movie_id}', text, count=1)
```

This turns `"movieId": 60seconds_1974` into `"movieId": 26322`, making the JSON
valid. The substitution is **unconditional** — it rewrites the field on every
record, so it neither detects nor counts malformations. That is why the released
`movie_profiles.json` has 0/10,381 identifier mismatches: the identifiers come
from the MovieLens catalogue, not from the model.

## Downstream Usage

The generated `movie_profiles.json` provides **two complementary feature types** for recommendation models:

### Feature A: Profile Text → Dense Embedding (semantic)

```python
from sentence_transformers import SentenceTransformer
import json, numpy as np

# Load profiles
with open("output/movie_profiles.json") as f:
    profiles = json.load(f)

# Encode with sentence-transformer (default: bge-large-en-v1.5, 1024-dim)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
texts = [profiles[mid]["profile"] for mid in sorted(profiles.keys())]
embeddings = model.encode(texts, show_progress_bar=True)  # (10381, 1024)

# Optional: reduce dimensionality via PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=128)
# embeddings = pca.fit_transform(embeddings)  # (10381, 128)
```

Use as item side features in LightGCN, DCN-V2, or two-tower models.

### Feature B: Mood Vector → Direct 10-Dim Numeric Feature

```python
# Extract mood vectors — ready-to-use, no embedding needed
mood_matrix = np.array([
    [profiles[mid]["mood_vector"][axis] for axis in [
        "dark_light", "serious_playful", "slow_fast", "cerebral_visceral",
        "realistic_fantastical", "intimate_epic", "conventional_experimental",
        "emotional_detached", "nostalgic_contemporary", "predictable_subversive",
    ]]
    for mid in sorted(profiles.keys())
])  # (10381, 10)
```

Usage in recommendation models:

| Method | How | When |
|---|---|---|
| **Direct concatenation** | Append 10-dim mood to item embeddings | Default — always include |
| **User preference matching** | Avg mood of user's liked items → cosine sim with candidates | Cold-start users, content-based fallback |
| **Cluster-gated routing** | K-means on mood vectors → per-cluster models | Heterogeneous user populations |
| **Interaction features** | Element-wise `|user_avg_mood - candidate_mood|` → ranking MLP input | Feature-crossing in DCN-V2 |

### Recommended Ablation Structure

| Experiment | Item Features | Added Dims | Expected Signal |
|---|---|---|---|
| Baseline: LightGCN (ID only) | Collaborative signals only | d | Baseline |
| + Genome raw (1128-dim PCA→128) | ID + genome vector | d + 128 | Known-strong content signal |
| + BERT(title+genre) | ID + BERT encoding | d + 1024 | Simpler text encoder baseline |
| + **LLM profile embedding** | ID + profile embedding | d + 1024 | Core contribution |
| + **LLM mood vector** | ID + 10-dim mood | d + 10 | Lightweight structured signal |
| + **Profile + mood combined** | ID + profile + mood | d + 1034 | Full LLM feature set |
| LLM profile only (no CF) | Profile embedding only | 1024 | Content-only floor |

The critical comparison is **LLM profile embedding vs. genome raw PCA** — this directly answers: *does having an LLM reason about genome tags produce a better representation than just feeding the 1,128-dim vector as-is?*

## Research Context

This pipeline addresses a gap in the released resources: we found no prior release that applies LLM feature synthesis to the genome-annotated portion of MovieLens 20M — the 10,381 items this resource covers in full — or that uses the tag genome as structured input to the generation. The closest works (LLMRec at WSDM 2024, RLMRec at WWW 2024, A-LLMRec at KDD 2024) were built on MovieLens 1M, a filtered MovieLens 10M, or Amazon subsets — and none touch the genome tag data.
