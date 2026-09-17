"""
Configuration for LLM Movie Profile Generator.
All tunable parameters in one place.

Version: 2.0 — Embedding-optimized profiles with 10-axis mood vector
Model: Claude Haiku 4.5
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
# Raw MovieLens. In the generation tree this sits under the generator; in the
# released layout scripts/download_ml20m.sh writes <repo>/data/raw/ml-20m
# instead, so resolve in the order a reader will actually have it and honour the
# same ML20M_DIR override the benchmark uses. One dataset once had four
# different hard-coded locations across the packages that read it.
_ML20M = [PROJECT_ROOT.parent.parent / "data" / "raw" / "ml-20m",  # download_ml20m.sh
          PROJECT_ROOT.parent.parent / "data" / "ml-20m",          # --target-dir data
          PROJECT_ROOT / "data" / "ml-20m"]                        # generation layout
DATA_DIR = Path(os.environ["ML20M_DIR"]) if os.environ.get("ML20M_DIR") else next(
    (c for c in _ML20M if (c / "genome-tags.csv").exists()), _ML20M[0])
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# MovieLens 20M files
RATINGS_CSV = DATA_DIR / "ratings.csv"
MOVIES_CSV = DATA_DIR / "movies.csv"
TAGS_CSV = DATA_DIR / "tags.csv"
LINKS_CSV = DATA_DIR / "links.csv"
GENOME_SCORES_CSV = DATA_DIR / "genome-scores.csv"
GENOME_TAGS_CSV = DATA_DIR / "genome-tags.csv"

# Output files
PROFILES_OUTPUT = OUTPUT_DIR / "movie_profiles.json"
PROFILES_PARTIAL = OUTPUT_DIR / "movie_profiles_partial.json"
METADATA_CACHE = CACHE_DIR / "tmdb_metadata.json"
PROMPT_LOG = LOG_DIR / "prompts.jsonl"
RUN_LOG = LOG_DIR / "run.log"

# ──────────────────────────────────────────────────────────────────────────────
# GENOME TAG SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
TOP_K_GENOME_TAGS = 30  # Number of top genome tags per movie by relevance score

# ──────────────────────────────────────────────────────────────────────────────
# TMDb API
# ──────────────────────────────────────────────────────────────────────────────
TMDB_API_KEY = ""  # Set via environment variable TMDB_API_KEY
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_RATE_LIMIT = 40  # requests per 10 seconds (TMDb limit is 40/10s)
TMDB_RETRY_MAX = 3
TMDB_RETRY_DELAY = 2.0  # seconds

# ──────────────────────────────────────────────────────────────────────────────
# CLAUDE API (Anthropic)
# ──────────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = ""  # Set via environment variable ANTHROPIC_API_KEY
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 600  # headroom for JSON structure + 10-axis mood vector
CLAUDE_TEMPERATURE = 0.3  # low for structural consistency, slight variation in vocabulary
CLAUDE_BATCH_SIZE = 10  # concurrent requests (Haiku supports higher throughput)
CLAUDE_RATE_LIMIT_RPM = 200  # requests per minute (Haiku tier)
CHECKPOINT_EVERY = 50  # save partial results every N movies
USE_PROMPT_CACHING = True  # cache system prompt across requests (~90% input cost savings)

# ──────────────────────────────────────────────────────────────────────────────
# PROFILE CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────────
PROFILE_MIN_WORDS = 80
PROFILE_MAX_WORDS = 120

# ──────────────────────────────────────────────────────────────────────────────
# 10-AXIS MOOD VECTOR SPECIFICATION
# Defined centrally — shared by validation, prompting, and downstream code
# ──────────────────────────────────────────────────────────────────────────────
MOOD_AXES = [
    "dark_light",                # -1.0 = very dark/bleak → 1.0 = very light/bright
    "serious_playful",           # -1.0 = grave/solemn → 1.0 = comedic/whimsical
    "slow_fast",                 # -1.0 = contemplative/meditative → 1.0 = frenetic/breathless
    "cerebral_visceral",         # -1.0 = intellectual/abstract → 1.0 = sensory/physical
    "realistic_fantastical",     # -1.0 = gritty realism → 1.0 = pure fantasy/surrealism
    "intimate_epic",             # -1.0 = personal/chamber → 1.0 = grand/sweeping scale
    "conventional_experimental", # -1.0 = mainstream/formulaic → 1.0 = avant-garde
    "emotional_detached",        # -1.0 = cold/clinical → 1.0 = deeply emotional/sentimental
    "nostalgic_contemporary",    # -1.0 = backward-looking/classic → 1.0 = modern/zeitgeist
    "predictable_subversive",    # -1.0 = formulaic/safe → 1.0 = defies expectations
]

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT v2.0 — Embedding-Optimized with Few-Shot Examples
#
# Design principles:
#   P1. LEAD WITH THEMATIC ESSENCE — sentence-transformers weight early tokens
#       more heavily; the first sentence anchors the embedding.
#   P2. EXCLUDE STRUCTURED METADATA — cast, director, year, ratings, runtime are
#       separate feature columns; repeating them wastes embedding capacity.
#   P3. CONSISTENT SEMANTIC STRUCTURE — every profile follows the same flow so
#       analogous information aligns across all 10,381 embeddings.
#   P4. 80-120 WORDS — fits well within sentence-transformer token windows
#       (256–512 tokens); tighter text produces sharper, less diluted embeddings.
#   P5. 10-AXIS MOOD VECTOR — continuous structured features usable directly as
#       10-dim side features without embedding; complements the semantic profile.
#
# Few-shot examples included because Haiku 4.5 produces significantly more
# consistent output quality when pattern-matching against demonstrations.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You produce embedding-optimized movie profiles for a recommendation system. Each profile will be encoded by a sentence-transformer into a dense vector used as an item feature in collaborative filtering. Your output quality directly determines recommendation accuracy.

RULES — follow exactly:
1. The "profile" field is 80–120 words, one paragraph, no line breaks.
2. NEVER mention: actor/cast names, director names, release year, ratings, vote counts, runtime, box office, or awards. These exist as separate structured features. Your profile captures ONLY what requires semantic synthesis.
3. Sentence 1: the movie's thematic core and emotional identity — this anchors the embedding.
4. Sentences 2–4: tonal texture, narrative approach, genre positioning, and what makes this film distinct from superficially similar ones.
5. Final sentence: the viewing experience or audience sensibility this film rewards.
6. Every word must be discriminative. Ban: "a great film", "well-made", "entertaining", "must-see", "beloved", "iconic", "masterpiece". These carry zero embedding information.
7. Synthesize the genome tags and plot into insight — never list tags or parrot the plot synopsis.

SEMANTIC FLOW (same order for every movie):
  thematic essence → emotional tone → narrative/visual style → genre distinction → audience experience

MOOD VECTOR — 10 axes, each a float from -1.0 to 1.0:
  dark_light:                 -1=bleak/grim → 1=bright/uplifting
  serious_playful:            -1=grave/solemn → 1=comedic/absurd
  slow_fast:                  -1=contemplative/still → 1=frenetic/breathless
  cerebral_visceral:          -1=intellectual/philosophical → 1=sensory/action-driven
  realistic_fantastical:      -1=gritty naturalism → 1=pure fantasy/surreal
  intimate_epic:              -1=small-scale/personal → 1=grand/sweeping
  conventional_experimental:  -1=mainstream/familiar → 1=avant-garde/unconventional
  emotional_detached:         -1=cold/analytical → 1=intensely emotional
  nostalgic_contemporary:     -1=classic/retro sensibility → 1=modern/current
  predictable_subversive:     -1=formulaic/safe → 1=expectation-defying

Output valid JSON only — no markdown, no backticks, no commentary outside the JSON.

EXAMPLE 1 — input tags: [sci-fi:0.97, dystopia:0.94, thought-provoking:0.91, futuristic:0.88, technology:0.85, visually appealing:0.82, twist ending:0.78, action:0.75, philosophical:0.72, loneliness:0.68] — plot about a dystopian future where memories are commodified:
{"movieId":99999,"title":"Example Sci-Fi","profile":"A claustrophobic meditation on commodified memory and manufactured identity in a surveillance state where consciousness itself becomes currency. The narrative operates through deliberate disorientation, withholding information to mirror the protagonist's fractured self-knowledge before detonating a structural revelation that recontextualizes everything preceding it. Tonally it sustains oppressive dread beneath sleek technological surfaces, finding horror not in violence but in the erasure of authentic experience. The dystopian architecture feels plausible rather than spectacular, grounding speculative elements in recognizable institutional control. Pacing alternates between hypnotic stillness and sudden visceral confrontation. Occupies the intersection of cerebral science fiction and existential thriller, distinguishing itself from action-oriented genre entries through philosophical density. Rewards viewers who engage actively with unreliable narrative structures.","word_count":112,"key_themes":["identity","surveillance","memory"],"mood_vector":{"dark_light":-0.7,"serious_playful":-0.8,"slow_fast":-0.3,"cerebral_visceral":-0.6,"realistic_fantastical":0.3,"intimate_epic":-0.2,"conventional_experimental":0.4,"emotional_detached":-0.3,"nostalgic_contemporary":0.5,"predictable_subversive":0.6}}

EXAMPLE 2 — input tags: [comedy:0.96, romantic:0.93, witty:0.89, feel-good:0.86, quirky:0.82, friendship:0.78, indie:0.75, dialogue-driven:0.72, charming:0.68, bittersweet:0.65] — plot about two strangers navigating an unexpected connection during a layover:
{"movieId":99998,"title":"Example Romcom","profile":"A bittersweet exploration of transient human connection and the weight of unlived possibilities, structured around compressed time that heightens emotional stakes through imposed expiration. The conversational rhythm drives momentum rather than plot mechanics, with naturalistic verbal sparring that reveals character through deflection and vulnerability in equal measure. Tonally it balances genuine warmth with an undertow of melancholy, acknowledging that charm and longing are inseparable. The romantic framework subverts genre expectations by privileging ambiguity over resolution, treating uncertainty as emotionally richer than closure. Visual storytelling favors observational intimacy over stylistic flourish. Appeals to audiences who find romance more compelling when shadowed by impermanence.","word_count":103,"key_themes":["connection","impermanence","vulnerability"],"mood_vector":{"dark_light":0.3,"serious_playful":0.2,"slow_fast":-0.4,"cerebral_visceral":-0.3,"realistic_fantastical":-0.8,"intimate_epic":-0.7,"conventional_experimental":0.3,"emotional_detached":0.7,"nostalgic_contemporary":0.1,"predictable_subversive":0.4}}"""

# ──────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE
#
# Compact format minimizes input tokens. All metadata is provided so the LLM
# has full context, but the system prompt rules ensure only thematic/tonal
# synthesis appears in the profile text.
# ──────────────────────────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """MOVIE: {title} ({year}) | {genres}

GENOME TAGS (top-30 by relevance):
{genome_tags_formatted}

PLOT: {overview}
KEYWORDS: {keywords}
DIRECTOR: {directors} | CAST: {cast} | RUNTIME: {runtime}min
SCORES: TMDb {vote_average}/10 ({vote_count}v) | ML {ml_avg_rating}/5 ({ml_rating_count}r)
USER TAGS: {user_tags_summary}

Generate the JSON profile. 80–120 words. No cast/director/year/ratings in profile text."""
