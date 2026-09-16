"""
TMDb metadata crawler.
Fetches movie details (overview, cast, crew, keywords) from TMDb API.
Includes rate limiting, disk caching, and retry with exponential backoff.
"""

import json
import ssl
import time
import logging
import asyncio
import aiohttp
import certifi
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.settings import (
    TMDB_API_KEY, TMDB_BASE_URL, TMDB_RATE_LIMIT,
    TMDB_RETRY_MAX, TMDB_RETRY_DELAY, METADATA_CACHE,
)

logger = logging.getLogger(__name__)


def load_cache() -> Dict[str, Any]:
    """Load cached TMDb metadata from disk."""
    if METADATA_CACHE.exists():
        with open(METADATA_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache):,} cached TMDb entries")
        return cache
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    """Save TMDb metadata cache to disk."""
    METADATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    logger.info(f"Saved {len(cache):,} entries to TMDb cache")


async def fetch_tmdb_movie(
    session: aiohttp.ClientSession,
    tmdb_id: int,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """
    Fetch movie details + credits + keywords from TMDb API in a single call
    using append_to_response.
    """
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": api_key,
        "append_to_response": "credits,keywords",
        "language": "en-US",
    }

    for attempt in range(TMDB_RETRY_MAX):
        async with semaphore:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return _parse_tmdb_response(data)
                    elif resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning(f"TMDb rate limit hit, waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                    elif resp.status == 404:
                        logger.debug(f"TMDb ID {tmdb_id}: not found (404)")
                        return None
                    else:
                        logger.warning(f"TMDb ID {tmdb_id}: HTTP {resp.status}")
                        await asyncio.sleep(TMDB_RETRY_DELAY * (attempt + 1))
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"TMDb ID {tmdb_id}: {type(e).__name__} on attempt {attempt + 1}")
                await asyncio.sleep(TMDB_RETRY_DELAY * (attempt + 1))

    logger.error(f"TMDb ID {tmdb_id}: failed after {TMDB_RETRY_MAX} retries")
    return None


def _parse_tmdb_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant fields from TMDb API response."""
    # Directors
    crew = data.get("credits", {}).get("crew", [])
    directors = [p["name"] for p in crew if p.get("job") == "Director"]

    # Top 5 cast members
    cast = data.get("credits", {}).get("cast", [])
    top_cast = [p["name"] for p in cast[:5]]

    # Keywords
    keywords = data.get("keywords", {}).get("keywords", [])
    keyword_names = [k["name"] for k in keywords[:15]]

    return {
        "overview": data.get("overview", ""),
        "directors": directors,
        "cast": top_cast,
        "runtime": data.get("runtime"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "original_language": data.get("original_language", "en"),
        "keywords": keyword_names,
        "release_date": data.get("release_date", ""),
        "budget": data.get("budget"),
        "revenue": data.get("revenue"),
    }


async def crawl_tmdb_batch(
    tmdb_ids: Dict[int, int],  # movieId -> tmdbId mapping
    api_key: str,
    cache: Dict[str, Any],
    max_concurrent: int = 10,
) -> Dict[int, Dict[str, Any]]:
    """
    Crawl TMDb metadata for a batch of movies.
    tmdb_ids: {movieId: tmdbId}
    Returns: {movieId: metadata_dict}
    """
    # Filter out already cached movies
    to_fetch = {
        mid: tid for mid, tid in tmdb_ids.items()
        if str(mid) not in cache and tid is not None
    }
    logger.info(f"TMDb crawl: {len(to_fetch)} to fetch, {len(tmdb_ids) - len(to_fetch)} cached")

    if not to_fetch:
        return {int(k): v for k, v in cache.items() if int(k) in tmdb_ids}

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = {}
        for movie_id, tmdb_id in to_fetch.items():
            task = asyncio.create_task(
                fetch_tmdb_movie(session, tmdb_id, api_key, semaphore)
            )
            tasks[movie_id] = task

        completed = 0
        total = len(tasks)
        for movie_id, task in tasks.items():
            result = await task
            if result:
                results[movie_id] = result
                cache[str(movie_id)] = result
            completed += 1
            if completed % 500 == 0:
                logger.info(f"  TMDb progress: {completed}/{total} ({completed/total*100:.1f}%)")
                save_cache(cache)  # periodic save

    save_cache(cache)  # final save

    # Merge cached + freshly fetched
    all_results = {}
    for mid in tmdb_ids:
        if mid in results:
            all_results[mid] = results[mid]
        elif str(mid) in cache:
            all_results[mid] = cache[str(mid)]

    logger.info(f"TMDb crawl complete: {len(all_results)} movies with metadata")
    return all_results


def get_metadata_for_movie(
    movie_id: int,
    tmdb_metadata: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Get TMDb metadata for a movie, with safe defaults for missing data.
    """
    meta = tmdb_metadata.get(movie_id, {})
    return {
        "overview": meta.get("overview", "No overview available."),
        "directors": ", ".join(meta.get("directors", [])) or "Unknown",
        "cast": ", ".join(meta.get("cast", [])) or "Unknown",
        "runtime": meta.get("runtime") or "Unknown",
        "vote_average": meta.get("vote_average") or "N/A",
        "vote_count": meta.get("vote_count") or 0,
        "original_language": meta.get("original_language", "en"),
        "keywords": ", ".join(meta.get("keywords", [])) or "None available",
    }
