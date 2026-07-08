"""
api/cache.py
─────────────
Async Redis cache layer for:
  - Prediction results  (TTL: 5 min)
  - WHOIS lookups       (TTL: 24 hr)
  - DNS queries         (TTL: 1 hr)

Gracefully degrades (logs warning, continues without cache)
if Redis is unavailable.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from utils.config import settings
from utils.helpers import url_cache_key
from utils.logger import get_logger

logger = get_logger(__name__)

_redis_client = None


async def get_redis():
    """Return the async Redis client, initialising on first call."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await _redis_client.ping()
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning(f"Redis unavailable: {exc} — running without cache")
        _redis_client = None
    return _redis_client


async def cache_get(prefix: str, url: str) -> Optional[dict]:
    """Retrieve a cached value for *url*.

    Parameters
    ----------
    prefix:
        Cache namespace (e.g. ``"prediction"``).
    url:
        The URL key.

    Returns
    -------
    dict or None
    """
    client = await get_redis()
    if client is None:
        return None
    try:
        key  = url_cache_key(prefix, url)
        data = await client.get(key)
        if data:
            logger.debug(f"Cache HIT  [{prefix}] {url[:50]}")
            return json.loads(data)
    except Exception as exc:
        logger.debug(f"Cache GET error: {exc}")
    return None


async def cache_set(
    prefix: str,
    url: str,
    value: dict,
    ttl: Optional[int] = None,
) -> bool:
    """Store a value in cache.

    Parameters
    ----------
    prefix:
        Cache namespace.
    url:
        The URL key.
    value:
        JSON-serialisable dict to store.
    ttl:
        Time-to-live in seconds.  Defaults to settings value for *prefix*.

    Returns
    -------
    bool
        True if stored successfully.
    """
    client = await get_redis()
    if client is None:
        return False

    # Default TTLs by prefix
    if ttl is None:
        ttl = {
            "prediction": settings.REDIS_TTL_PREDICTION,
            "whois":      settings.REDIS_TTL_WHOIS,
            "dns":        settings.REDIS_TTL_DNS,
        }.get(prefix, 300)

    try:
        key = url_cache_key(prefix, url)
        await client.setex(key, ttl, json.dumps(value, default=str))
        logger.debug(f"Cache SET  [{prefix}] {url[:50]}  TTL={ttl}s")
        return True
    except Exception as exc:
        logger.debug(f"Cache SET error: {exc}")
        return False


async def cache_delete(prefix: str, url: str) -> bool:
    """Delete a cached entry."""
    client = await get_redis()
    if client is None:
        return False
    try:
        key = url_cache_key(prefix, url)
        await client.delete(key)
        return True
    except Exception:
        return False


async def redis_ping() -> bool:
    """Return True if Redis is reachable."""
    client = await get_redis()
    if client is None:
        return False
    try:
        return await client.ping()
    except Exception:
        return False
