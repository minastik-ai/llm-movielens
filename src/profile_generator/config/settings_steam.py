"""Configuration for LLM Game Profile Generator (Steam cross-domain).

Third instantiation of the movie-profile pipeline, after books. Every design
principle is preserved verbatim: the system prompt's 5 rules, the 10-axis mood
specification, the 80-120 word constraint, the discriminative-vocabulary ban
list, and the semantic-flow ordering. Only the domain nouns change:

    movie title          -> game title
    movie genres         -> Steam genres
    movie genome tags    -> Steam community tags (a 376-tag vocabulary, UNSCORED)
    movie plot synopsis  -> (none -- Steam ships no free-text description)
    movie cast/director  -> developer / publisher
    movie runtime        -> (none)
    movie ratings        -> (omitted; the store's sentiment label is not used)

READ THIS BEFORE RUNNING. Steam is deliberately the metadata-POOREST catalogue
in the release, and that is the point of including it -- but it also bounds what
the result can mean. The median prompt payload is ~61 tokens against ML-20M's
~2,012, so an 80-120 word profile is roughly twice the length of its own input.
Whatever the model writes beyond the tag list therefore comes from its pretrained
knowledge of the game, not from the metadata it was shown. That makes Steam a
test of generation portability (which is what the paper claims for it) and NOT,
on its own, a test of reasoning over supplied structured metadata.

The masked variant below exists to measure exactly that gap. Running the pipeline
twice -- once normally, once with MASK_IDENTITY -- and comparing the two profile
sets quantifies how much of a Steam profile is recall rather than synthesis. Do
not report the unmasked Steam arm as evidence about the reasoning step without it.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT.parent / "output_steam"
LOG_DIR = OUTPUT_DIR / "logs"

# Input, produced by ecir-.../src/prepare_steam_metadata.py
GAME_METADATA_JSON = OUTPUT_DIR / "game_metadata.json"

# Outputs
PROFILES_OUTPUT = OUTPUT_DIR / "game_profiles.json"
PROFILES_PARTIAL = OUTPUT_DIR / "game_profiles_partial.json"
PROFILES_MASKED = OUTPUT_DIR / "game_profiles_masked.json"
PROMPT_LOG = LOG_DIR / "prompts.jsonl"
RUN_LOG = LOG_DIR / "run.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CLAUDE API — identical to the movie and book pipelines
# ──────────────────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_TEMPERATURE = 0.3
ANTHROPIC_MAX_TOKENS = 600
ANTHROPIC_RATE_LIMIT_RPM = 200

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT VALIDATION — identical
# ──────────────────────────────────────────────────────────────────────────────
PROFILE_WORD_MIN = 80
PROFILE_WORD_MAX = 130
KEY_THEMES_MIN = 3
KEY_THEMES_MAX = 5
MOOD_AXIS_RANGE = (-1.0, 1.0)
MAX_RETRIES = 3

# ──────────────────────────────────────────────────────────────────────────────
# MOOD AXES — identical to movie and book pipelines
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
# SYSTEM PROMPT — games variant
#
# Structurally identical to the movie and book system prompts: 5 design rules,
# 10-axis mood spec, JSON-only output, two few-shot examples. Only domain
# wording changes (movie -> game, cast/director -> developer, genome tags ->
# community tags, runtime -> omitted).
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You produce embedding-optimized game profiles for a recommendation system. Each profile will be encoded by a sentence-transformer into a dense vector used as an item feature in collaborative filtering. Your output quality directly determines recommendation accuracy.

RULES — follow exactly:
1. The "profile" field is 80–120 words, one paragraph, no line breaks.
2. NEVER mention: developer names, publisher names, release year, price, review counts, or store ratings. These exist as separate structured features. Your profile captures ONLY what requires semantic synthesis.
3. Sentence 1: the game's thematic core and experiential identity — this anchors the embedding.
4. Sentences 2–4: tonal texture, mechanical approach, genre positioning, and what makes this game distinct from superficially similar ones.
5. Final sentence: the play experience or player sensibility this game rewards.
6. Every word must be discriminative. Ban: "a great game", "fun", "addictive", "must-play", "beloved", "iconic", "masterpiece". These carry zero embedding information.
7. Synthesize tags, genres and modes into insight — never list tags or parrot them back.

SEMANTIC FLOW (same order for every game):
  thematic essence → emotional tone → mechanical/aesthetic style → genre distinction → player experience

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

EXAMPLE 1 — input tags: [Roguelike, Deck Building, Turn-Based, Strategy, Difficult, Indie, Replay Value] — genres: [Indie, Strategy] — modes: [Single-player]:
{"itemId":999999,"title":"Example Roguelike","profile":"A compulsive exercise in incremental mastery where each defeat converts into knowledge and every run rewrites the calculus of risk. Combat resolves as a tightening lattice of card synergies, asking the player to commit to a strategic identity early and then defend it against escalating variance. The tone stays austere and systems-forward rather than narrative, treating difficulty as instruction rather than punishment. It distinguishes itself from broader strategy entries through brevity of the individual run and depth across many, so failure costs minutes and teaches hours. Rewards players who enjoy reading probability and building engines under constraint.","word_count":97,"key_themes":["mastery","risk","synergy"],"mood_vector":{"dark_light":-0.3,"serious_playful":-0.2,"slow_fast":-0.1,"cerebral_visceral":-0.8,"realistic_fantastical":0.4,"intimate_epic":-0.6,"conventional_experimental":0.2,"emotional_detached":-0.6,"nostalgic_contemporary":0.3,"predictable_subversive":0.3}}

EXAMPLE 2 — input tags: [Atmospheric, Exploration, Story Rich, Walking Simulator, Emotional, Beautiful, Relaxing] — genres: [Adventure, Indie] — modes: [Single-player]:
{"itemId":999998,"title":"Example Atmospheric","profile":"A contemplative drift through inhabited landscapes where environmental detail carries the narrative that dialogue withholds. Progression is measured in noticing rather than solving, trading challenge for attention and letting pacing settle into something closer to walking than playing. Tonally it sustains a gentle melancholy without tipping into despair, using light, weather and silence as its primary expressive instruments. It sits apart from adventure entries built on puzzles by refusing friction almost entirely, making its emotional argument through accumulation rather than climax. Rewards players willing to move slowly and read a place as text.","word_count":95,"key_themes":["solitude","place","memory"],"mood_vector":{"dark_light":0.2,"serious_playful":-0.5,"slow_fast":-0.9,"cerebral_visceral":-0.4,"realistic_fantastical":0.1,"intimate_epic":-0.5,"conventional_experimental":0.5,"emotional_detached":0.6,"nostalgic_contemporary":0.2,"predictable_subversive":0.2}}"""

# ──────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATES — games
#
# Two variants. The default mirrors the movie and book pipelines. The masked
# variant removes the game's identity (title, developer, publisher) so the model
# has ONLY the structured metadata to work from; the gap between the two profile
# sets measures how much of a Steam profile is pretrained recall rather than
# synthesis from the supplied fields. See the module docstring.
# ──────────────────────────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """GAME: {title} | genres: {genres}

COMMUNITY TAGS: {tags}

MODES: {specs} | DEVELOPER: {developer} | PUBLISHER: {publisher}

Generate the JSON profile. 80–120 words. No developer/publisher/year/price/ratings in profile text."""

USER_PROMPT_TEMPLATE_MASKED = """GAME: (identity withheld) | genres: {genres}

COMMUNITY TAGS: {tags}

MODES: {specs}

Generate the JSON profile from the tags and genres alone. 80–120 words. No developer/publisher/year/price/ratings in profile text."""

# Item-id field name (games use "itemId", as books do)
ID_FIELD_NAME = "itemId"
