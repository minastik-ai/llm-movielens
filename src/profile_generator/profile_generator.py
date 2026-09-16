"""
LLM Profile Generator.
Constructs prompts from movie data, calls Claude Haiku 4.5 via Anthropic API,
validates JSON output, handles retries, and saves checkpoints.
"""

import json
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from anthropic import Anthropic, APIError, RateLimitError

from config.settings import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE,
    CLAUDE_BATCH_SIZE, CLAUDE_RATE_LIMIT_RPM, CHECKPOINT_EVERY,
    PROFILE_MIN_WORDS, PROFILE_MAX_WORDS, MOOD_AXES, USE_PROMPT_CACHING,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    PROFILES_OUTPUT, PROFILES_PARTIAL, PROMPT_LOG,
)
from data_loader import format_genome_tags_for_prompt

logger = logging.getLogger(__name__)


class PromptLogger:
    """Logs every prompt sent to the LLM for reproducibility."""

    def __init__(self, log_path: Path = PROMPT_LOG):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Log system prompt once at initialization
        self._log_entry({
            "type": "system_prompt",
            "timestamp": datetime.utcnow().isoformat(),
            "model": CLAUDE_MODEL,
            "temperature": CLAUDE_TEMPERATURE,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
        })
        logger.info(f"Prompt logger initialized: {self.log_path}")

    def log_request(self, movie_id: int, user_prompt: str, response_text: str,
                    input_tokens: int, output_tokens: int, latency_ms: float,
                    attempt: int, success: bool):
        self._log_entry({
            "type": "request",
            "timestamp": datetime.utcnow().isoformat(),
            "movie_id": movie_id,
            "user_prompt": user_prompt,
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "attempt": attempt,
            "success": success,
        })

    def _log_entry(self, entry: Dict[str, Any]):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_user_prompt(
    movie_id: int,
    title: str,
    year: str,
    genres: str,
    genome_tags: List[Dict[str, Any]],
    tmdb_meta: Dict[str, Any],
    rating_stats: Dict[str, float],
    user_tags_summary: str,
) -> str:
    """Construct the user prompt from all data sources."""
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        year=year or "Unknown",
        genres=genres,
        genome_tags_formatted=format_genome_tags_for_prompt(genome_tags),
        overview=tmdb_meta["overview"],
        directors=tmdb_meta["directors"],
        cast=tmdb_meta["cast"],
        runtime=tmdb_meta["runtime"],
        vote_average=tmdb_meta["vote_average"],
        vote_count=tmdb_meta["vote_count"],
        keywords=tmdb_meta["keywords"],
        ml_avg_rating=rating_stats.get("avg_rating", "N/A"),
        ml_rating_count=rating_stats.get("rating_count", 0),
        user_tags_summary=user_tags_summary or "No user tags available",
    )


def validate_profile_json(raw_text: str, movie_id: int) -> Tuple[Optional[Dict], str]:
    """
    Parse and validate the LLM response.
    Returns (parsed_dict, error_message). error_message is "" if valid.
    """
    # Strip common LLM artifacts
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Fix malformed movieId (e.g. "movieId": 60seconds_1974 → "movieId": <movie_id>)
    import re
    text = re.sub(r'"movieId"\s*:\s*[^,}\]]+', f'"movieId": {movie_id}', text, count=1)

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    # Validate required fields
    required = ["movieId", "title", "profile", "word_count", "key_themes", "mood_vector"]
    missing = [f for f in required if f not in data]
    if missing:
        return None, f"Missing fields: {missing}"

    # Validate profile word count
    profile = data.get("profile", "")
    actual_word_count = len(profile.split())
    data["word_count"] = actual_word_count  # override with actual count

    if actual_word_count < PROFILE_MIN_WORDS - 10:  # allow small tolerance
        return None, f"Profile too short: {actual_word_count} words (min {PROFILE_MIN_WORDS})"
    if actual_word_count > PROFILE_MAX_WORDS + 15:  # allow small tolerance
        return None, f"Profile too long: {actual_word_count} words (max {PROFILE_MAX_WORDS})"

    # Validate mood_vector (10 axes from centralized config)
    mood = data.get("mood_vector", {})
    for axis in MOOD_AXES:
        if axis not in mood:
            return None, f"Missing mood axis: {axis}"
        val = mood[axis]
        if not isinstance(val, (int, float)) or val < -1.0 or val > 1.0:
            return None, f"Invalid mood value for {axis}: {val}"

    # Ensure movieId matches
    data["movieId"] = movie_id

    return data, ""


class ProfileGenerator:
    """
    Generates movie profiles using Claude Haiku 4.5.
    Supports batched concurrent requests, retries, and checkpointing.
    """

    def __init__(self, api_key: str = ""):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Provide via environment variable or config."
            )
        self.client = Anthropic(api_key=self.api_key)
        self.prompt_logger = PromptLogger()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0

    def generate_single_profile(
        self,
        movie_id: int,
        user_prompt: str,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single movie profile with retry logic.
        Uses prompt caching for the system prompt to reduce input costs.
        Returns validated profile dict or None on failure.
        """
        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                # Build system message with optional prompt caching
                if USE_PROMPT_CACHING:
                    system_msg = [{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }]
                else:
                    system_msg = SYSTEM_PROMPT

                response = self.client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    temperature=CLAUDE_TEMPERATURE,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                raw_text = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                latency_ms = (time.time() - start_time) * 1000

                # Track costs (Claude Haiku 4.5: $1/MTok input, $5/MTok output)
                # Cache reads: 0.1× base input price; cache writes: 1.25× base input price
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cache_read_tokens += cache_read
                self.total_cache_write_tokens += cache_write

                # Validate
                profile, error = validate_profile_json(raw_text, movie_id)

                self.prompt_logger.log_request(
                    movie_id=movie_id,
                    user_prompt=user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt,
                    response_text=raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    attempt=attempt,
                    success=profile is not None,
                )

                if profile is not None:
                    return profile
                else:
                    logger.warning(
                        f"Movie {movie_id}, attempt {attempt}: validation error: {error}"
                    )
                    if attempt < max_retries:
                        time.sleep(1.0)

            except RateLimitError as e:
                wait_time = 30 * attempt
                logger.warning(f"Rate limit hit for movie {movie_id}, waiting {wait_time}s...")
                time.sleep(wait_time)

            except APIError as e:
                logger.error(f"API error for movie {movie_id}: {e}")
                time.sleep(5 * attempt)

            except Exception as e:
                logger.error(f"Unexpected error for movie {movie_id}: {type(e).__name__}: {e}")
                time.sleep(2)

        logger.error(f"Movie {movie_id}: FAILED after {max_retries} attempts")
        return None

    def generate_all_profiles(
        self,
        movie_data: List[Dict[str, Any]],
        existing_profiles: Optional[Dict[int, Any]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate profiles for all movies with checkpointing.
        movie_data: list of dicts, each with movie_id, user_prompt, etc.
        existing_profiles: previously generated profiles to resume from.
        """
        profiles = dict(existing_profiles or {})
        total = len(movie_data)
        skipped = 0
        generated = 0
        failed = 0
        start_time = time.time()

        # Calculate delay between requests to respect rate limit
        min_delay = 60.0 / CLAUDE_RATE_LIMIT_RPM

        logger.info(f"Starting profile generation for {total:,} movies")
        logger.info(f"  Model: {CLAUDE_MODEL}")
        logger.info(f"  Temperature: {CLAUDE_TEMPERATURE}")
        logger.info(f"  Rate limit: {CLAUDE_RATE_LIMIT_RPM} RPM → {min_delay:.2f}s between requests")
        logger.info(f"  Checkpoint every: {CHECKPOINT_EVERY} movies")
        logger.info(f"  Already completed: {len(profiles):,} movies")

        for i, movie in enumerate(movie_data):
            mid = movie["movie_id"]

            # Skip already generated
            if mid in profiles:
                skipped += 1
                continue

            # Generate
            profile = self.generate_single_profile(mid, movie["user_prompt"])

            if profile:
                profiles[mid] = profile
                generated += 1
            else:
                failed += 1

            # Progress logging
            total_processed = skipped + generated + failed
            if total_processed % 100 == 0 or total_processed == total:
                elapsed = time.time() - start_time
                rate = generated / elapsed * 3600 if elapsed > 0 else 0
                est_cost_input = self.total_input_tokens / 1e6 * 1.0  # $1/MTok base input
                est_cost_output = self.total_output_tokens / 1e6 * 5.0  # $5/MTok output
                est_cost_cache_read = self.total_cache_read_tokens / 1e6 * 0.1  # $0.10/MTok cache reads
                est_cost_cache_write = self.total_cache_write_tokens / 1e6 * 1.25  # $1.25/MTok cache writes
                est_cost = est_cost_input + est_cost_output + est_cost_cache_read + est_cost_cache_write
                logger.info(
                    f"Progress: {total_processed}/{total} "
                    f"(gen={generated}, skip={skipped}, fail={failed}) | "
                    f"Rate: {rate:.0f}/hr | "
                    f"Tokens: {self.total_input_tokens:,}in + {self.total_output_tokens:,}out "
                    f"(cache: {self.total_cache_read_tokens:,}r/{self.total_cache_write_tokens:,}w) | "
                    f"Est. cost: ${est_cost:.2f}"
                )

            # Checkpoint
            if generated > 0 and generated % CHECKPOINT_EVERY == 0:
                self._save_checkpoint(profiles)

            # Rate limiting
            time.sleep(min_delay)

        # Final save
        self._save_checkpoint(profiles)
        self._save_final(profiles)

        elapsed = time.time() - start_time
        logger.info(f"═══ GENERATION COMPLETE ═══")
        logger.info(f"  Total: {total:,} movies")
        logger.info(f"  Generated: {generated:,}")
        logger.info(f"  Skipped (cached): {skipped:,}")
        logger.info(f"  Failed: {failed:,}")
        logger.info(f"  Elapsed: {elapsed/3600:.1f} hours")
        logger.info(f"  Input tokens: {self.total_input_tokens:,}")
        logger.info(f"  Output tokens: {self.total_output_tokens:,}")
        logger.info(f"  Cache read tokens: {self.total_cache_read_tokens:,}")
        logger.info(f"  Cache write tokens: {self.total_cache_write_tokens:,}")

        return profiles

    def _save_checkpoint(self, profiles: Dict[int, Any]):
        """Save partial results to disk."""
        PROFILES_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILES_PARTIAL, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in profiles.items()},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"  → Checkpoint saved: {len(profiles):,} profiles")

    def _save_final(self, profiles: Dict[int, Any]):
        """Save final results."""
        PROFILES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILES_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in profiles.items()},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"  → Final output saved: {PROFILES_OUTPUT}")


def load_existing_profiles() -> Dict[int, Any]:
    """Load previously generated profiles for resume support."""
    for path in [PROFILES_PARTIAL, PROFILES_OUTPUT]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = {int(k): v for k, v in data.items()}
            logger.info(f"Loaded {len(result):,} existing profiles from {path}")
            return result
    return {}
