"""
Unit tests for Full-Text Search caching.
"""

from unittest.mock import AsyncMock

import pytest

import yar.cache.fts_cache as fts_cache_module
from yar.cache.fts_cache import (
    _build_local_cache_key,
    _build_redis_cache_key,
    _compute_cache_key,
    _fts_cache,
    get_cached_fts_results,
    get_fts_cache_stats,
    invalidate_fts_cache_for_workspace,
    store_fts_results,
)


class TestFTSCacheKey:
    """Test cache key computation."""

    def test_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        key1 = _compute_cache_key('test query', 'default', 10, 'english')
        key2 = _compute_cache_key('test query', 'default', 10, 'english')
        assert key1 == key2

    def test_cache_key_differs_by_query(self):
        """Different queries produce different keys."""
        key1 = _compute_cache_key('query A', 'default', 10, 'english')
        key2 = _compute_cache_key('query B', 'default', 10, 'english')
        assert key1 != key2

    def test_cache_key_differs_by_workspace(self):
        """Different workspaces produce different keys."""
        key1 = _compute_cache_key('test', 'workspace1', 10, 'english')
        key2 = _compute_cache_key('test', 'workspace2', 10, 'english')
        assert key1 != key2

    def test_cache_key_differs_by_limit(self):
        """Different limits produce different keys."""
        key1 = _compute_cache_key('test', 'default', 10, 'english')
        key2 = _compute_cache_key('test', 'default', 20, 'english')
        assert key1 != key2

    def test_cache_key_differs_by_language(self):
        """Different languages produce different keys."""
        key1 = _compute_cache_key('test', 'default', 10, 'english')
        key2 = _compute_cache_key('test', 'default', 10, 'french')
        assert key1 != key2

    def test_cache_key_length(self):
        """Cache key is 16 characters (SHA256[:16])."""
        key = _compute_cache_key('test', 'default', 10, 'english')
        assert len(key) == 16

    def test_redis_cache_key_contains_workspace_prefix(self):
        """Redis cache key encodes workspace for selective invalidation."""
        redis_key = _build_redis_cache_key('workspaceA', 'abcd1234')
        assert redis_key == 'yar:fts:workspaceA:abcd1234'


class TestFTSCacheOperations:
    """Test cache operations."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        _fts_cache.clear()
        yield
        _fts_cache.clear()

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        result = await get_cached_fts_results('new query', 'default', 10, 'english')
        assert result is None

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Stored results can be retrieved."""
        test_results = [{'id': 'chunk1', 'content': 'test', 'score': 0.9}]

        await store_fts_results('test query', 'default', 10, 'english', test_results)
        cached = await get_cached_fts_results('test query', 'default', 10, 'english')

        assert cached == test_results

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Expired entries are not returned."""
        test_results = [{'id': 'chunk1', 'content': 'test', 'score': 0.9}]

        await store_fts_results('test query', 'default', 10, 'english', test_results)

        # Manually expire the entry
        cache_hash = _compute_cache_key('test query', 'default', 10, 'english')
        local_key = _build_local_cache_key('default', cache_hash)
        _fts_cache[local_key] = (test_results, 0)  # timestamp = 0 (expired)

        cached = await get_cached_fts_results('test query', 'default', 10, 'english')
        assert cached is None

    @pytest.mark.asyncio
    async def test_invalidation_clears_cache(self):
        """Invalidation clears cache entries."""
        test_results = [{'id': 'chunk1', 'content': 'test', 'score': 0.9}]

        await store_fts_results('test query', 'default', 10, 'english', test_results)
        assert len(_fts_cache) > 0

        await invalidate_fts_cache_for_workspace('default')
        assert len(_fts_cache) == 0

    @pytest.mark.asyncio
    async def test_workspace_invalidation_is_scoped(self):
        """Invalidation only clears entries for the target workspace."""
        ws1_results = [{'id': 'ws1-chunk', 'content': 'test', 'score': 0.9}]
        ws2_results = [{'id': 'ws2-chunk', 'content': 'test', 'score': 0.8}]

        await store_fts_results('test query', 'workspace1', 10, 'english', ws1_results)
        await store_fts_results('test query', 'workspace2', 10, 'english', ws2_results)

        assert len(_fts_cache) == 2
        await invalidate_fts_cache_for_workspace('workspace1')

        ws1_cached = await get_cached_fts_results('test query', 'workspace1', 10, 'english')
        ws2_cached = await get_cached_fts_results('test query', 'workspace2', 10, 'english')

        assert ws1_cached is None
        assert ws2_cached == ws2_results
        assert len(_fts_cache) == 1

    @pytest.mark.asyncio
    async def test_workspace_invalidation_with_empty_workspace_is_scoped(self):
        """Empty workspace invalidation only clears empty-workspace entries."""
        empty_ws_results = [{'id': 'empty', 'content': 'test', 'score': 0.7}]
        ws_results = [{'id': 'ws', 'content': 'test', 'score': 0.6}]

        await store_fts_results('test query', '', 10, 'english', empty_ws_results)
        await store_fts_results('test query', 'workspace1', 10, 'english', ws_results)

        invalidated = await invalidate_fts_cache_for_workspace('')

        assert invalidated == 1
        empty_ws_cached = await get_cached_fts_results('test query', '', 10, 'english')
        ws_cached = await get_cached_fts_results('test query', 'workspace1', 10, 'english')
        assert empty_ws_cached is None
        assert ws_cached == ws_results

    @pytest.mark.asyncio
    async def test_local_cache_key_contains_workspace_prefix(self):
        """Local cache key encodes workspace for targeted invalidation."""
        await store_fts_results('test query', 'workspaceA', 10, 'english', [{'id': 'chunk'}])

        assert any(key.startswith('workspaceA:') for key in _fts_cache)

    @pytest.mark.asyncio
    async def test_workspace_invalidation_scoped_for_redis(self, monkeypatch):
        """Redis invalidation only deletes keys for the targeted workspace."""
        _fts_cache['workspace1:key1'] = ([{'id': '1'}], 1.0)
        _fts_cache['workspace2:key2'] = ([{'id': '2'}], 1.0)

        class _FakeRedis:
            def __init__(self):
                self.deleted_keys = []

            async def scan(self, cursor, match, count):
                if match == 'yar:fts:workspace1:*':
                    return 0, ['yar:fts:workspace1:r1']
                if match == 'yar:fts:workspace2:*':
                    return 0, ['yar:fts:workspace2:r2']
                return 0, []

            async def delete(self, *keys):
                self.deleted_keys.extend(keys)

        fake_redis = _FakeRedis()
        monkeypatch.setattr(fts_cache_module, 'REDIS_FTS_CACHE_ENABLED', True)
        monkeypatch.setattr(
            fts_cache_module,
            '_get_redis_client',
            AsyncMock(return_value=fake_redis),
        )

        invalidated = await invalidate_fts_cache_for_workspace('workspace1')

        assert invalidated == 2  # 1 local + 1 redis
        assert _fts_cache == {'workspace2:key2': ([{'id': '2'}], 1.0)}
        assert fake_redis.deleted_keys == ['yar:fts:workspace1:r1']

    @pytest.mark.asyncio
    async def test_empty_results_not_cached(self):
        """Empty results are not cached (by design of caller)."""
        # The store function should handle empty results
        # but the main function only calls store if results exist
        await store_fts_results('test query', 'default', 10, 'english', [])

        # Empty results are still stored (for negative caching)
        cached = await get_cached_fts_results('test query', 'default', 10, 'english')
        assert cached == []


class TestFTSCacheStats:
    """Test cache statistics."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear cache before each test."""
        _fts_cache.clear()
        yield
        _fts_cache.clear()

    @pytest.mark.asyncio
    async def test_stats_empty_cache(self):
        """Stats for empty cache."""
        stats = get_fts_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['valid_entries'] == 0

    @pytest.mark.asyncio
    async def test_stats_with_entries(self):
        """Stats reflect cache entries."""
        await store_fts_results('q1', 'ws', 10, 'en', [{'id': '1'}])
        await store_fts_results('q2', 'ws', 10, 'en', [{'id': '2'}])

        stats = get_fts_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 2

    @pytest.mark.asyncio
    async def test_stats_expired_entries(self):
        """Stats correctly count expired entries."""
        await store_fts_results('q1', 'ws', 10, 'en', [{'id': '1'}])

        # Manually expire one entry
        cache_hash = _compute_cache_key('q1', 'ws', 10, 'en')
        local_key = _build_local_cache_key('ws', cache_hash)
        _fts_cache[local_key] = ([{'id': '1'}], 0)

        await store_fts_results('q2', 'ws', 10, 'en', [{'id': '2'}])

        stats = get_fts_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 1  # Only q2 is valid
