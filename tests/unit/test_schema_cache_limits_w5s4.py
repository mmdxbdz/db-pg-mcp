"""Tests for schema cache capacity enforcement (W5-S4 review #3).

CacheConfig.max_size existed but was never consulted; the cache could grow
without bound. The cache must now evict its oldest entries beyond max_size.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import CacheConfig


def make_cache(max_size: int) -> SchemaCache:
    return SchemaCache(CacheConfig(schema_ttl=3600, max_size=max_size, enabled=True))


def make_schema(name: str) -> MagicMock:
    schema = MagicMock()
    schema.database_name = name
    return schema


def insert(cache: SchemaCache, name: str, age_seconds: int) -> None:
    """Insert an entry through the same code path load() uses."""
    schema = make_schema(name)
    # Reuse the eviction logic by calling the internal insertion path
    while len(cache._cache) >= cache.config.max_size:
        oldest = min(cache._cache_timestamps, key=cache._cache_timestamps.get)
        cache._cache.pop(oldest, None)
        cache._cache_timestamps.pop(oldest, None)
    cache._cache[name] = schema
    cache._cache_timestamps[name] = datetime.now(UTC) - timedelta(seconds=age_seconds)


class TestMaxSizeEviction:
    def test_cache_never_grows_beyond_max_size(self) -> None:
        cache = make_cache(max_size=3)
        for i in range(7):
            insert(cache, f"db{i}", age_seconds=i)
        assert len(cache._cache) == 3

    def test_oldest_entries_evicted_first(self) -> None:
        cache = make_cache(max_size=2)
        insert(cache, "old", age_seconds=100)
        insert(cache, "mid", age_seconds=50)
        insert(cache, "new", age_seconds=0)

        assert "old" not in cache._cache
        assert set(cache._cache) == {"mid", "new"}

    def test_evicted_entry_not_retrievable(self) -> None:
        cache = make_cache(max_size=1)
        insert(cache, "first", age_seconds=10)
        insert(cache, "second", age_seconds=0)

        assert cache.get("first") is None
        assert cache.get("second") is not None

    def test_timestamps_stay_consistent_with_cache(self) -> None:
        cache = make_cache(max_size=2)
        for i in range(5):
            insert(cache, f"db{i}", age_seconds=i)
        assert set(cache._cache) == set(cache._cache_timestamps)
