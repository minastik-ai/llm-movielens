"""Configuration for LLM Book Profile Generator (Amazon-Books cross-domain).

Adapts the movie-profile pipeline to books while preserving every design
principle: the system prompt's 5 rules, the 10-axis mood specification, the
80-120 word constraint, the discriminative-vocabulary ban list, and the
semantic-flow ordering. Input fields are book-domain analogs of the movie
fields:
    movie title          → book title
    movie genres         → book subjects
    movie genome tags    → (none — books have no genome equivalent)
    movie plot synopsis  → RLMRec-generated profile text (description surrogate)
    movie cast/director  → book authors
    movie runtime        → book page count
    movie ratings        → (omitted; not available for our subset)

Design intent: keep the system prompt almost identical to the movie version so
that the cross-domain experiment isolates the *content type* variable, not the
prompt variant.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT.parent / "output_amazon"
LOG_DIR = OUTPUT_DIR / "logs"

# Inputs
BOOK_METADATA_JSON = OUTPUT_DIR / "book_metadata.json"  # produced by scripts/fetch_amazon_books_metadata.py
RLMREC_DATA_DIR = PROJECT_ROOT.parent.parent / "benchmark" / "external" / "RLMRec" / "data" / "amazon"

# Outputs
PROFILES_OUTPUT = OUTPUT_DIR / "book_profiles.json"
PROFILES_PARTIAL = OUTPUT_DIR / "book_profiles_partial.json"
PROMPT_LOG = LOG_DIR / "prompts.jsonl"
RUN_LOG = LOG_DIR / "run.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CLAUDE API
# ──────────────────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_TEMPERATURE = 0.3
ANTHROPIC_MAX_TOKENS = 600
ANTHROPIC_RATE_LIMIT_RPM = 200  # requests / minute

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
PROFILE_WORD_MIN = 80
PROFILE_WORD_MAX = 130           # +10 tolerance over 120 (matches movie pipeline)
KEY_THEMES_MIN = 3
KEY_THEMES_MAX = 5
MOOD_AXIS_RANGE = (-1.0, 1.0)
MAX_RETRIES = 3

# ──────────────────────────────────────────────────────────────────────────────
# MOOD AXES — identical to movie pipeline (perceptual axes are domain-general)
# ──────────────────────────────────────────────────────────────────────────────
MOOD_AXES = [
    ("dark_light",                "bleak/grim", "bright/uplifting"),
    ("serious_playful",           "grave/solemn", "comedic/absurd"),
    ("slow_fast",                 "contemplative/still", "frenetic/breathless"),
    ("cerebral_visceral",         "intellectual/philosophical", "sensory/action-driven"),
    ("realistic_fantastical",     "gritty naturalism", "pure fantasy/surreal"),
    ("intimate_epic",             "small-scale/personal", "grand/sweeping"),
    ("conventional_experimental", "mainstream/familiar", "avant-garde/unconventional"),
    ("emotional_detached",        "cold/analytical", "intensely emotional"),
    ("nostalgic_contemporary",    "classic/retro sensibility", "modern/current"),
    ("predictable_subversive",    "formulaic/safe", "expectation-defying"),
]

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — books variant
#
# Structurally identical to the movie system prompt: 5 design rules, 10-axis
# mood spec, JSON-only output, two few-shot examples. Only domain-specific
# wording (movie → book, cast/director → author, genome tags → subjects,
# runtime → pages, etc.) is changed.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You produce embedding-optimized book profiles for a recommendation system. Each profile will be encoded by a sentence-transformer into a dense vector used as an item feature in collaborative filtering. Your output quality directly determines recommendation accuracy.

RULES — follow exactly:
1. The "profile" field is 80–120 words, one paragraph, no line breaks.
2. NEVER mention: author names, publisher names, publication year, page count, ratings, review counts, or awards. These exist as separate structured features. Your profile captures ONLY what requires semantic synthesis.
3. Sentence 1: the book's thematic core and emotional identity — this anchors the embedding.
4. Sentences 2–4: tonal texture, narrative approach, genre positioning, and what makes this book distinct from superficially similar ones.
5. Final sentence: the reading experience or audience sensibility this book rewards.
6. Every word must be discriminative. Ban: "a great book", "well-written", "engaging", "must-read", "beloved", "iconic", "masterpiece". These carry zero embedding information.
7. Synthesize subjects, description, and contextual cues into insight — never list subjects or parrot the description.

SEMANTIC FLOW (same order for every book):
  thematic essence → emotional tone → narrative/prose style → genre distinction → reader experience

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

EXAMPLE 1 — input subjects: [literary fiction, postmodern, identity, surveillance, dystopia, philosophical, twist ending] — description: a near-future novel about a clerk whose memories are commodified by a corporate state:
{"itemId":99999,"title":"Example Lit-Fic","profile":"A claustrophobic meditation on commodified memory and manufactured identity in a surveillance state where consciousness itself becomes currency. The narrative operates through deliberate disorientation, withholding information to mirror the protagonist's fractured self-knowledge before detonating a structural revelation that recontextualizes everything preceding it. Tonally it sustains oppressive dread beneath sleek institutional surfaces, finding horror not in violence but in the erasure of authentic experience. The dystopian architecture feels plausible rather than spectacular, grounding speculative elements in recognizable bureaucratic control. Prose alternates between hypnotic stillness and sudden confrontation. Occupies the intersection of cerebral speculative fiction and existential thriller, distinguishing itself from plot-driven genre entries through philosophical density. Rewards readers who engage actively with unreliable narrators.","word_count":118,"key_themes":["identity","surveillance","memory"],"mood_vector":{"dark_light":-0.7,"serious_playful":-0.8,"slow_fast":-0.3,"cerebral_visceral":-0.6,"realistic_fantastical":0.3,"intimate_epic":-0.2,"conventional_experimental":0.4,"emotional_detached":-0.3,"nostalgic_contemporary":0.5,"predictable_subversive":0.6}}

EXAMPLE 2 — input subjects: [contemporary romance, dialogue-driven, friendship, charming, bittersweet, indie] — description: two strangers navigate an unexpected connection during a long airport layover:
{"itemId":99998,"title":"Example Romance","profile":"A bittersweet exploration of transient human connection and the weight of unlived possibilities, structured around compressed time that heightens emotional stakes through imposed expiration. The conversational rhythm drives momentum rather than plot mechanics, with naturalistic verbal sparring that reveals character through deflection and vulnerability in equal measure. Tonally it balances genuine warmth with an undertow of melancholy, acknowledging that charm and longing are inseparable. The romantic framework subverts genre expectations by privileging ambiguity over resolution, treating uncertainty as emotionally richer than closure. Prose favors observational intimacy over stylistic flourish. Appeals to readers who find romance more compelling when shadowed by impermanence.","word_count":103,"key_themes":["connection","impermanence","vulnerability"],"mood_vector":{"dark_light":0.3,"serious_playful":0.2,"slow_fast":-0.4,"cerebral_visceral":-0.3,"realistic_fantastical":-0.8,"intimate_epic":-0.7,"conventional_experimental":0.3,"emotional_detached":0.7,"nostalgic_contemporary":0.1,"predictable_subversive":0.4}}"""

# ──────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE — books
# ──────────────────────────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """BOOK: {title} ({publish_date}) | subjects: {subjects}

DESCRIPTION (community-generated content cue): {description}

AUTHORS: {authors} | PUBLISHER: {publishers} | PAGES: {pages}

Generate the JSON profile. 80–120 words. No author/publisher/year/ratings in profile text."""

# Item-id field name (books use "itemId" instead of "movieId")
ID_FIELD_NAME = "itemId"
