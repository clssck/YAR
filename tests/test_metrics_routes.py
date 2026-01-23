"""Tests for metrics_routes.py - Metrics API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, Query
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from yar.api.routers.metrics_routes import (
    CacheStats,
    EmbedStats,
    FTSCacheStats,
    LatencyPercentiles,
    MetricsResponse,
    QueryMetricItem,
    RecentQueriesResponse,
)

# =============================================================================
# Response Model Validation Tests
# =============================================================================


class TestLatencyPercentilesValidation:
    """Test LatencyPercentiles model validation."""

    @pytest.mark.offline
    def test_valid_percentiles(self):
        """Test valid latency percentiles."""
        percs = LatencyPercentiles(
            p50=234.5,
            p95=890.2,
            p99=1200.0,
            avg=345.6,
            min=100.0,
            max=2500.0,
        )
        assert percs.p50 == 234.5
        assert percs.p95 == 890.2
        assert percs.p99 == 1200.0

    @pytest.mark.offline
    def test_missing_required_field(self):
        """Test that missing required field raises validation error."""
        with pytest.raises(ValidationError):
            LatencyPercentiles(  # type: ignore[call-arg]
                p50=234.5,
                p95=890.2,
                # Missing p99, avg, min, max
            )


class TestCacheStatsValidation:
    """Test CacheStats model validation."""

    @pytest.mark.offline
    def test_valid_cache_stats(self):
        """Test valid cache statistics."""
        stats = CacheStats(
            total_llm_calls=100,
            total_cache_hits=50,
            hit_rate=0.5,
        )
        assert stats.total_llm_calls == 100
        assert stats.total_cache_hits == 50
        assert stats.hit_rate == 0.5

    @pytest.mark.offline
    def test_zero_values(self):
        """Test zero values are valid."""
        stats = CacheStats(
            total_llm_calls=0,
            total_cache_hits=0,
            hit_rate=0.0,
        )
        assert stats.hit_rate == 0.0


class TestEmbedStatsValidation:
    """Test EmbedStats model validation."""

    @pytest.mark.offline
    def test_valid_embed_stats(self):
        """Test valid embedding statistics."""
        stats = EmbedStats(total_calls=200)
        assert stats.total_calls == 200


class TestFTSCacheStatsValidation:
    """Test FTSCacheStats model validation."""

    @pytest.mark.offline
    def test_valid_fts_cache_stats(self):
        """Test valid FTS cache statistics."""
        stats = FTSCacheStats(
            enabled=True,
            total_entries=150,
            valid_entries=140,
            max_size=5000,
            ttl_seconds=300,
            redis_enabled=False,
        )
        assert stats.enabled is True
        assert stats.total_entries == 150
        assert stats.redis_enabled is False

    @pytest.mark.offline
    def test_disabled_cache(self):
        """Test disabled FTS cache."""
        stats = FTSCacheStats(
            enabled=False,
            total_entries=0,
            valid_entries=0,
            max_size=0,
            ttl_seconds=0,
            redis_enabled=False,
        )
        assert stats.enabled is False


class TestMetricsResponseValidation:
    """Test MetricsResponse model validation."""

    @pytest.mark.offline
    def test_valid_metrics_response_with_latency(self):
        """Test valid metrics response with latency data."""
        resp = MetricsResponse(
            uptime_seconds=3600.5,
            total_queries=150,
            queries_in_window=45,
            window_seconds=3600.0,
            latency_percentiles=LatencyPercentiles(
                p50=234.5, p95=890.2, p99=1200.0, avg=345.6, min=100.0, max=2500.0
            ),
            cache_stats=CacheStats(total_llm_calls=100, total_cache_hits=50, hit_rate=0.33),
            embed_stats=EmbedStats(total_calls=200),
            fts_cache_stats=FTSCacheStats(
                enabled=True,
                total_entries=150,
                valid_entries=140,
                max_size=5000,
                ttl_seconds=300,
                redis_enabled=False,
            ),
            mode_distribution={'mix': 30, 'local': 10, 'global': 5},
            legacy_stats={'llm_call': 100, 'llm_cache': 50},
        )
        assert resp.uptime_seconds == 3600.5
        assert resp.total_queries == 150
        assert resp.latency_percentiles is not None
        assert resp.latency_percentiles.p50 == 234.5

    @pytest.mark.offline
    def test_valid_metrics_response_without_latency(self):
        """Test valid metrics response without latency data (no queries)."""
        resp = MetricsResponse(
            uptime_seconds=100.0,
            total_queries=0,
            queries_in_window=0,
            window_seconds=3600.0,
            latency_percentiles=None,
            cache_stats=CacheStats(total_llm_calls=0, total_cache_hits=0, hit_rate=0.0),
            embed_stats=EmbedStats(total_calls=0),
            fts_cache_stats=FTSCacheStats(
                enabled=False,
                total_entries=0,
                valid_entries=0,
                max_size=0,
                ttl_seconds=0,
                redis_enabled=False,
            ),
            mode_distribution={},
            legacy_stats={},
        )
        assert resp.latency_percentiles is None
        assert resp.total_queries == 0


class TestQueryMetricItemValidation:
    """Test QueryMetricItem model validation."""

    @pytest.mark.offline
    def test_valid_query_metric_item(self):
        """Test valid query metric item."""
        item = QueryMetricItem(
            timestamp=1705320000.0,
            duration_ms=234.5,
            mode='mix',
            cache_hit=True,
            entities_count=10,
            relations_count=5,
            chunks_count=3,
            tokens_used=1500,
        )
        assert item.timestamp == 1705320000.0
        assert item.mode == 'mix'
        assert item.cache_hit is True

    @pytest.mark.offline
    def test_missing_required_fields(self):
        """Test that missing required fields raises validation error."""
        with pytest.raises(ValidationError):
            QueryMetricItem(  # type: ignore[call-arg]
                timestamp=1705320000.0,
                duration_ms=234.5,
                # Missing mode, cache_hit, etc.
            )


class TestRecentQueriesResponseValidation:
    """Test RecentQueriesResponse model validation."""

    @pytest.mark.offline
    def test_valid_recent_queries_response(self):
        """Test valid recent queries response."""
        resp = RecentQueriesResponse(
            queries=[
                QueryMetricItem(
                    timestamp=1705320000.0,
                    duration_ms=234.5,
                    mode='mix',
                    cache_hit=True,
                    entities_count=10,
                    relations_count=5,
                    chunks_count=3,
                    tokens_used=1500,
                ),
            ],
            count=1,
        )
        assert resp.count == 1
        assert len(resp.queries) == 1

    @pytest.mark.offline
    def test_empty_queries_response(self):
        """Test empty queries response."""
        resp = RecentQueriesResponse(queries=[], count=0)
        assert resp.count == 0
        assert len(resp.queries) == 0


# =============================================================================
# Test Route Factory for Endpoint Tests
# =============================================================================


def create_test_metrics_routes(
    mock_collector: MagicMock,
    mock_fts_stats: dict,
    mock_statistic_data: dict,
) -> APIRouter:
    """Create simplified test routes that mirror metrics_routes.py structure."""
    router = APIRouter(tags=['metrics'])

    @router.get('/metrics')
    async def get_metrics_(
        window: float = Query(default=3600.0, ge=60.0, le=86400.0),
    ):
        stats = mock_collector.compute_stats(window_seconds=window)
        fts_stats = mock_fts_stats

        latency = None
        if stats['latency_percentiles']:
            latency = LatencyPercentiles(**stats['latency_percentiles'])

        return MetricsResponse(
            uptime_seconds=stats['uptime_seconds'],
            total_queries=stats['total_queries'],
            queries_in_window=stats['queries_in_window'],
            window_seconds=stats['window_seconds'],
            latency_percentiles=latency,
            cache_stats=CacheStats(**stats['cache_stats']),
            embed_stats=EmbedStats(**stats['embed_stats']),
            fts_cache_stats=FTSCacheStats(**fts_stats),
            mode_distribution=stats['mode_distribution'],
            legacy_stats=mock_statistic_data.copy(),
        )

    @router.get('/metrics/queries')
    async def get_recent_queries_(
        limit: int = Query(default=10, ge=1, le=100),
    ):
        queries = mock_collector.get_recent_queries(limit=limit)
        return RecentQueriesResponse(
            queries=[QueryMetricItem(**q) for q in queries],
            count=len(queries),
        )

    return router


# =============================================================================
# Endpoint Tests
# =============================================================================


@pytest.fixture
def mock_collector():
    """Create a mock metrics collector."""
    collector = MagicMock()
    collector.compute_stats = MagicMock(
        return_value={
            'uptime_seconds': 3600.5,
            'total_queries': 150,
            'queries_in_window': 45,
            'window_seconds': 3600.0,
            'latency_percentiles': {
                'p50': 234.5,
                'p95': 890.2,
                'p99': 1200.0,
                'avg': 345.6,
                'min': 100.0,
                'max': 2500.0,
            },
            'cache_stats': {
                'total_llm_calls': 100,
                'total_cache_hits': 50,
                'hit_rate': 0.33,
            },
            'embed_stats': {'total_calls': 200},
            'mode_distribution': {'mix': 30, 'local': 10, 'global': 5},
        }
    )
    collector.get_recent_queries = MagicMock(
        return_value=[
            {
                'timestamp': 1705320000.0,
                'duration_ms': 234.5,
                'mode': 'mix',
                'cache_hit': True,
                'entities_count': 10,
                'relations_count': 5,
                'chunks_count': 3,
                'tokens_used': 1500,
            },
            {
                'timestamp': 1705320100.0,
                'duration_ms': 345.6,
                'mode': 'local',
                'cache_hit': False,
                'entities_count': 5,
                'relations_count': 2,
                'chunks_count': 1,
                'tokens_used': 800,
            },
        ]
    )
    return collector


@pytest.fixture
def mock_fts_stats():
    """Create mock FTS cache stats."""
    return {
        'enabled': True,
        'total_entries': 150,
        'valid_entries': 140,
        'max_size': 5000,
        'ttl_seconds': 300,
        'redis_enabled': False,
    }


@pytest.fixture
def mock_statistic_data():
    """Create mock legacy statistics data."""
    return {'llm_call': 100, 'llm_cache': 50, 'embed_call': 200}


@pytest.fixture
def test_app(mock_collector, mock_fts_stats, mock_statistic_data):
    """Create FastAPI test app with metrics routes."""
    from fastapi import FastAPI

    app = FastAPI()
    router = create_test_metrics_routes(mock_collector, mock_fts_stats, mock_statistic_data)
    app.include_router(router)
    return app


class TestGetMetricsEndpoint:
    """Test GET /metrics endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_default_window(self, test_app, mock_collector):
        """Test GET /metrics with default window."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics')

        assert response.status_code == 200
        data = response.json()
        assert data['uptime_seconds'] == 3600.5
        assert data['total_queries'] == 150
        assert data['queries_in_window'] == 45
        assert data['latency_percentiles']['p50'] == 234.5
        mock_collector.compute_stats.assert_called_once_with(window_seconds=3600.0)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_custom_window(self, test_app, mock_collector):
        """Test GET /metrics with custom window."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics?window=7200')

        assert response.status_code == 200
        mock_collector.compute_stats.assert_called_once_with(window_seconds=7200.0)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_min_window(self, test_app, mock_collector):
        """Test GET /metrics with minimum window (60 seconds)."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics?window=60')

        assert response.status_code == 200
        mock_collector.compute_stats.assert_called_once_with(window_seconds=60.0)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_max_window(self, test_app, mock_collector):
        """Test GET /metrics with maximum window (24 hours)."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics?window=86400')

        assert response.status_code == 200
        mock_collector.compute_stats.assert_called_once_with(window_seconds=86400.0)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_window_too_small(self, test_app):
        """Test GET /metrics with window below minimum returns 422."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics?window=30')

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_window_too_large(self, test_app):
        """Test GET /metrics with window above maximum returns 422."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics?window=100000')

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_no_latency_data(self, test_app, mock_collector):
        """Test GET /metrics when no queries in window (null latency)."""
        mock_collector.compute_stats.return_value = {
            'uptime_seconds': 100.0,
            'total_queries': 0,
            'queries_in_window': 0,
            'window_seconds': 3600.0,
            'latency_percentiles': None,
            'cache_stats': {'total_llm_calls': 0, 'total_cache_hits': 0, 'hit_rate': 0.0},
            'embed_stats': {'total_calls': 0},
            'mode_distribution': {},
        }

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics')

        assert response.status_code == 200
        data = response.json()
        assert data['latency_percentiles'] is None
        assert data['total_queries'] == 0

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_metrics_response_structure(self, test_app):
        """Test GET /metrics returns all expected fields."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics')

        assert response.status_code == 200
        data = response.json()

        # Check top-level fields
        assert 'uptime_seconds' in data
        assert 'total_queries' in data
        assert 'queries_in_window' in data
        assert 'window_seconds' in data
        assert 'latency_percentiles' in data
        assert 'cache_stats' in data
        assert 'embed_stats' in data
        assert 'fts_cache_stats' in data
        assert 'mode_distribution' in data
        assert 'legacy_stats' in data

        # Check nested structures
        assert 'p50' in data['latency_percentiles']
        assert 'p95' in data['latency_percentiles']
        assert 'p99' in data['latency_percentiles']
        assert 'total_llm_calls' in data['cache_stats']
        assert 'hit_rate' in data['cache_stats']
        assert 'enabled' in data['fts_cache_stats']


class TestGetRecentQueriesEndpoint:
    """Test GET /metrics/queries endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_default_limit(self, test_app, mock_collector):
        """Test GET /metrics/queries with default limit."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries')

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 2
        assert len(data['queries']) == 2
        mock_collector.get_recent_queries.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_custom_limit(self, test_app, mock_collector):
        """Test GET /metrics/queries with custom limit."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries?limit=5')

        assert response.status_code == 200
        mock_collector.get_recent_queries.assert_called_once_with(limit=5)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_min_limit(self, test_app, mock_collector):
        """Test GET /metrics/queries with minimum limit (1)."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries?limit=1')

        assert response.status_code == 200
        mock_collector.get_recent_queries.assert_called_once_with(limit=1)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_max_limit(self, test_app, mock_collector):
        """Test GET /metrics/queries with maximum limit (100)."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries?limit=100')

        assert response.status_code == 200
        mock_collector.get_recent_queries.assert_called_once_with(limit=100)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_limit_too_small(self, test_app):
        """Test GET /metrics/queries with limit below minimum returns 422."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries?limit=0')

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_limit_too_large(self, test_app):
        """Test GET /metrics/queries with limit above maximum returns 422."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries?limit=101')

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_empty(self, test_app, mock_collector):
        """Test GET /metrics/queries when no queries available."""
        mock_collector.get_recent_queries.return_value = []

        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries')

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 0
        assert data['queries'] == []

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_recent_queries_response_structure(self, test_app):
        """Test GET /metrics/queries returns all expected fields."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url='http://test'
        ) as client:
            response = await client.get('/metrics/queries')

        assert response.status_code == 200
        data = response.json()

        # Check top-level fields
        assert 'queries' in data
        assert 'count' in data

        # Check query item structure
        assert len(data['queries']) > 0
        query = data['queries'][0]
        assert 'timestamp' in query
        assert 'duration_ms' in query
        assert 'mode' in query
        assert 'cache_hit' in query
        assert 'entities_count' in query
        assert 'relations_count' in query
        assert 'chunks_count' in query
        assert 'tokens_used' in query
