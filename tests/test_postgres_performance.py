"""PostgreSQL performance benchmarks for LightRAG.

Measures latency percentiles, throughput, and resource usage for:
- Vector similarity search
- Hybrid search (BM25 + vector with RRF)
- Entity rebuild operations
- Bulk insert operations

Usage:
    pytest tests/test_postgres_performance.py -v --run-integration

Environment:
    LIGHTRAG_BENCHMARK_SIZE: small|medium|large (default: small)
    LIGHTRAG_BENCHMARK_ITERATIONS: int (default: 50)

    Required database connection:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE
"""

from __future__ import annotations

import contextlib
import os
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from lightrag.kg.postgres_impl import PGVectorStorage, PostgreSQLDB


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    iterations: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    throughput_ops_per_sec: float

    def __str__(self) -> str:
        return (
            f'{self.operation}:\n'
            f'  Iterations: {self.iterations}\n'
            f'  Latency: p50={self.latency_p50_ms:.2f}ms, p95={self.latency_p95_ms:.2f}ms, '
            f'p99={self.latency_p99_ms:.2f}ms\n'
            f'  Mean: {self.latency_mean_ms:.2f}ms, Min: {self.latency_min_ms:.2f}ms, '
            f'Max: {self.latency_max_ms:.2f}ms\n'
            f'  Throughput: {self.throughput_ops_per_sec:.1f} ops/sec'
        )


def calculate_percentile(data: list[float], percentile: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def compute_benchmark_result(operation: str, latencies_ms: list[float]) -> BenchmarkResult:
    """Compute benchmark statistics from latency measurements."""
    if not latencies_ms:
        return BenchmarkResult(
            operation=operation,
            iterations=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            latency_mean_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            throughput_ops_per_sec=0,
        )

    total_time_sec = sum(latencies_ms) / 1000
    return BenchmarkResult(
        operation=operation,
        iterations=len(latencies_ms),
        latency_p50_ms=calculate_percentile(latencies_ms, 50),
        latency_p95_ms=calculate_percentile(latencies_ms, 95),
        latency_p99_ms=calculate_percentile(latencies_ms, 99),
        latency_mean_ms=statistics.mean(latencies_ms),
        latency_min_ms=min(latencies_ms),
        latency_max_ms=max(latencies_ms),
        throughput_ops_per_sec=len(latencies_ms) / total_time_sec if total_time_sec > 0 else 0,
    )


# Test data size configurations
TEST_DATA_SIZES = {
    'small': {'chunks': 100, 'entities': 50, 'relations': 100},
    'medium': {'chunks': 1000, 'entities': 500, 'relations': 1000},
    'large': {'chunks': 10000, 'entities': 2000, 'relations': 5000},
}


def get_benchmark_config() -> tuple[dict[str, int], int]:
    """Get benchmark configuration from environment."""
    size = os.getenv('LIGHTRAG_BENCHMARK_SIZE', 'small')
    iterations = int(os.getenv('LIGHTRAG_BENCHMARK_ITERATIONS', '50'))

    if size not in TEST_DATA_SIZES:
        size = 'small'

    return TEST_DATA_SIZES[size], iterations


def generate_random_embedding(dim: int | None = None) -> list[float]:
    """Generate a random normalized embedding vector."""
    if dim is None:
        dim = EMBEDDING_DIM
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def generate_test_content(idx: int) -> str:
    """Generate test content for a chunk."""
    topics = ['machine learning', 'database optimization', 'distributed systems', 'natural language processing']
    topic = topics[idx % len(topics)]
    return f'This is test chunk {idx} about {topic}. It contains relevant information for benchmarking purposes.'


@pytest.fixture(scope='function')
async def db_connection():
    """Create a database connection for benchmarks."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    # Check required environment variables
    required_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DATABASE']
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        pytest.skip(f'Missing required environment variables: {missing}')

    db = PostgreSQLDB(
        config={
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'database': os.getenv('POSTGRES_DATABASE'),
            'workspace': f'benchmark_{int(time.time())}',
            'max_connections': int(os.getenv('POSTGRES_MAX_CONNECTIONS', '10')),
            'min_connections': 1,
            'connection_retry_attempts': 3,
            'connection_retry_backoff': 0.5,
            'connection_retry_backoff_max': 5.0,
            'pool_close_timeout': 5.0,
        }
    )

    try:
        await db.initdb()
        yield db
    finally:
        if db.pool:
            with contextlib.suppress(Exception):
                await db.pool.close()


# Default embedding dimension (matches typical OpenAI models)
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '1536'))


async def mock_embedding_func(texts: list[str]) -> np.ndarray:
    """Mock embedding function that returns random vectors."""
    return np.random.randn(len(texts), EMBEDDING_DIM).astype(np.float32)


@pytest.fixture(scope='function')
async def vector_storage(db_connection: PostgreSQLDB):
    """Create a vector storage instance for benchmarks."""
    from lightrag.kg.postgres_impl import PGVectorStorage

    workspace = f'benchmark_test_{int(time.time())}'

    storage = PGVectorStorage(
        namespace='chunks',
        global_config={
            'embedding_dim': EMBEDDING_DIM,
            'embedding_batch_num': 32,
            'vector_db_storage_cls_kwargs': {
                'cosine_better_than_threshold': 0.2,
            },
        },
        workspace=workspace,
        embedding_func=mock_embedding_func,
    )
    storage.db = db_connection

    yield storage

    # Cleanup: delete test data
    with contextlib.suppress(Exception):
        await db_connection.execute(
            'DELETE FROM LIGHTRAG_VDB_CHUNKS WHERE workspace = $1', data={'workspace': workspace}
        )


@pytest.fixture(scope='function')
async def seeded_storage(vector_storage: PGVectorStorage):
    """Seed the vector storage with test data and return it."""
    config, _ = get_benchmark_config()

    # Generate and insert test chunks
    chunks_data = {}
    for i in range(config['chunks']):
        chunk_id = f'chunk_{i:06d}'
        chunks_data[chunk_id] = {
            'content': generate_test_content(i),
            'content_vector': generate_random_embedding(),
            'full_doc_id': f'doc_{i // 10:04d}',
            'chunk_order_index': i % 10,
            'tokens': 50 + (i % 100),
            'file_path': f'/test/doc_{i // 10:04d}.txt',
        }

    # Upsert in batches
    await vector_storage.upsert(chunks_data)

    return vector_storage


@pytest.mark.integration
@pytest.mark.requires_db
class TestVectorSearchPerformance:
    """Benchmarks for vector similarity search."""

    @pytest.mark.asyncio
    async def test_vector_search_latency(self, seeded_storage: PGVectorStorage):
        """Measure latency percentiles for vector similarity search."""
        _, iterations = get_benchmark_config()
        latencies: list[float] = []

        # Warm up
        query_embedding = generate_random_embedding()
        await seeded_storage.query('test query', top_k=10, query_embedding=query_embedding)

        # Benchmark
        for _ in range(iterations):
            query_embedding = generate_random_embedding()
            start = time.perf_counter()
            await seeded_storage.query('benchmark query', top_k=10, query_embedding=query_embedding)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        result = compute_benchmark_result('Vector Search (top_k=10)', latencies)
        print(f'\n{result}')

        # Assertions - these are sanity checks, not strict requirements
        assert result.latency_p99_ms < 5000, f'p99 latency too high: {result.latency_p99_ms}ms'

    @pytest.mark.asyncio
    async def test_vector_search_varying_top_k(self, seeded_storage: PGVectorStorage):
        """Measure how latency scales with top_k parameter."""
        iterations = 20
        top_k_values = [5, 10, 25, 50, 100]

        print('\nVector Search Latency vs top_k:')
        for top_k in top_k_values:
            latencies: list[float] = []
            for _ in range(iterations):
                query_embedding = generate_random_embedding()
                start = time.perf_counter()
                await seeded_storage.query('benchmark query', top_k=top_k, query_embedding=query_embedding)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            result = compute_benchmark_result(f'top_k={top_k}', latencies)
            print(f'  top_k={top_k:3d}: p50={result.latency_p50_ms:.2f}ms, p95={result.latency_p95_ms:.2f}ms')


@pytest.mark.integration
@pytest.mark.requires_db
class TestHybridSearchPerformance:
    """Benchmarks for hybrid search (BM25 + vector with RRF fusion)."""

    @pytest.mark.asyncio
    async def test_hybrid_search_latency(self, seeded_storage: PGVectorStorage):
        """Measure latency for hybrid search combining BM25 and vector."""
        _, iterations = get_benchmark_config()
        latencies: list[float] = []

        queries = ['machine learning optimization', 'database performance', 'distributed computing']

        # Warm up
        query_embedding = generate_random_embedding()
        await seeded_storage.hybrid_search(queries[0], top_k=10, query_embedding=query_embedding)

        # Benchmark
        for i in range(iterations):
            query = queries[i % len(queries)]
            query_embedding = generate_random_embedding()
            start = time.perf_counter()
            await seeded_storage.hybrid_search(query, top_k=10, query_embedding=query_embedding)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        result = compute_benchmark_result('Hybrid Search (BM25 + Vector)', latencies)
        print(f'\n{result}')

        assert result.latency_p99_ms < 10000, f'p99 latency too high: {result.latency_p99_ms}ms'

    @pytest.mark.asyncio
    async def test_hybrid_vs_vector_comparison(self, seeded_storage: PGVectorStorage):
        """Compare hybrid search latency against pure vector search."""
        iterations = 30
        vector_latencies: list[float] = []
        hybrid_latencies: list[float] = []

        for _ in range(iterations):
            query_embedding = generate_random_embedding()

            # Vector search
            start = time.perf_counter()
            await seeded_storage.query('benchmark query', top_k=10, query_embedding=query_embedding)
            vector_latencies.append((time.perf_counter() - start) * 1000)

            # Hybrid search
            start = time.perf_counter()
            await seeded_storage.hybrid_search('benchmark query', top_k=10, query_embedding=query_embedding)
            hybrid_latencies.append((time.perf_counter() - start) * 1000)

        vector_result = compute_benchmark_result('Vector Only', vector_latencies)
        hybrid_result = compute_benchmark_result('Hybrid (BM25+Vector)', hybrid_latencies)

        print('\nVector vs Hybrid Search Comparison:')
        print(f'  Vector: p50={vector_result.latency_p50_ms:.2f}ms, p95={vector_result.latency_p95_ms:.2f}ms')
        print(f'  Hybrid: p50={hybrid_result.latency_p50_ms:.2f}ms, p95={hybrid_result.latency_p95_ms:.2f}ms')
        print(f'  Overhead: {(hybrid_result.latency_p50_ms / vector_result.latency_p50_ms - 1) * 100:.1f}% at p50')


@pytest.mark.integration
@pytest.mark.requires_db
class TestBulkInsertPerformance:
    """Benchmarks for bulk insert operations."""

    @pytest.mark.asyncio
    async def test_bulk_upsert_throughput(self, vector_storage: PGVectorStorage):
        """Measure throughput for bulk upsert operations."""
        batch_sizes = [50, 100, 250, 500]

        print('\nBulk Upsert Throughput:')
        for batch_size in batch_sizes:
            # Generate test data
            chunks_data = {}
            for i in range(batch_size):
                chunk_id = f'throughput_test_{batch_size}_{i:06d}'
                chunks_data[chunk_id] = {
                    'content': generate_test_content(i),
                    'content_vector': generate_random_embedding(),
                    'full_doc_id': f'throughput_doc_{i // 10:04d}',
                    'chunk_order_index': i % 10,
                    'tokens': 50 + (i % 100),
                    'file_path': f'/throughput/doc_{i // 10:04d}.txt',
                }

            # Benchmark
            start = time.perf_counter()
            await vector_storage.upsert(chunks_data)
            elapsed_sec = time.perf_counter() - start

            throughput = batch_size / elapsed_sec
            print(f'  Batch size {batch_size:4d}: {throughput:.1f} rows/sec ({elapsed_sec * 1000:.1f}ms total)')

        # Cleanup
        try:
            db = vector_storage._db_required()
            await db.execute(
                "DELETE FROM LIGHTRAG_VDB_CHUNKS WHERE id LIKE 'throughput_test_%' AND workspace = $1",
                data={'workspace': vector_storage.workspace},
            )
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.requires_db
class TestGraphOperationsPerformance:
    """Benchmarks for graph operations (entity rebuild, edge fetching)."""

    @pytest.fixture
    async def graph_storage(self, db_connection: PostgreSQLDB):
        """Create a graph storage instance for benchmarks."""
        from lightrag.kg.postgres_impl import PGGraphStorage

        workspace = f'graph_benchmark_{int(time.time())}'

        storage = PGGraphStorage(
            namespace='chunk_entity_relation',
            global_config={'embedding_dim': EMBEDDING_DIM},
            workspace=workspace,
            embedding_func=mock_embedding_func,
        )
        storage.db = db_connection
        storage.graph_name = f'{workspace}_chunk_entity_relation'
        config, _ = get_benchmark_config()

        for i in range(config['entities']):
            entity_id = f'entity_{i:04d}'
            await storage.upsert_node(
                node_id=entity_id,
                node_data={
                    'entity_id': entity_id,
                    'entity_type': 'TEST_ENTITY',
                    'description': f'Test entity {i} for benchmarking',
                    'source_id': f'chunk_{i % 100:06d}',
                },
            )

        # Create relations (connect entities in a graph pattern)
        for i in range(config['relations']):
            src = f'entity_{i % config["entities"]:04d}'
            tgt = f'entity_{(i + 1) % config["entities"]:04d}'
            await storage.upsert_edge(
                source_node_id=src,
                target_node_id=tgt,
                edge_data={
                    'weight': 1.0,
                    'description': f'Relation from {src} to {tgt}',
                    'keywords': 'test,benchmark',
                    'source_id': f'chunk_{i % 100:06d}',
                },
            )

        yield storage

        # Cleanup
        with contextlib.suppress(Exception):
            await storage.delete_graph()

    @pytest.mark.asyncio
    async def test_get_edge_single_vs_batch(self, graph_storage):
        """Compare single edge fetch vs batch edge fetch performance."""
        config, _ = get_benchmark_config()
        iterations = min(20, config['relations'] // 10)

        # Get some edge pairs to test
        edge_pairs = [
            {'src': f'entity_{i % config["entities"]:04d}', 'tgt': f'entity_{(i + 1) % config["entities"]:04d}'}
            for i in range(10)
        ]

        # Single fetch timing
        single_latencies: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            for pair in edge_pairs:
                await graph_storage.get_edge(pair['src'], pair['tgt'])
            elapsed_ms = (time.perf_counter() - start) * 1000
            single_latencies.append(elapsed_ms)

        # Batch fetch timing
        batch_latencies: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            await graph_storage.get_edges_batch(edge_pairs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            batch_latencies.append(elapsed_ms)

        single_result = compute_benchmark_result(f'Single ({len(edge_pairs)} edges)', single_latencies)
        batch_result = compute_benchmark_result(f'Batch ({len(edge_pairs)} edges)', batch_latencies)

        print('\nSingle vs Batch Edge Fetch:')
        print(f'  Single: p50={single_result.latency_p50_ms:.2f}ms, p95={single_result.latency_p95_ms:.2f}ms')
        print(f'  Batch:  p50={batch_result.latency_p50_ms:.2f}ms, p95={batch_result.latency_p95_ms:.2f}ms')

        if single_result.latency_p50_ms > 0:
            speedup = single_result.latency_p50_ms / batch_result.latency_p50_ms
            print(f'  Speedup: {speedup:.1f}x at p50')

    @pytest.mark.asyncio
    async def test_get_node_edges_latency(self, graph_storage):
        """Measure latency for getting all edges connected to a node."""
        config, _ = get_benchmark_config()
        iterations = 30
        latencies: list[float] = []

        for i in range(iterations):
            entity_id = f'entity_{(i * 7) % config["entities"]:04d}'  # Spread across entities
            start = time.perf_counter()
            await graph_storage.get_node_edges(entity_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        result = compute_benchmark_result('Get Node Edges', latencies)
        print(f'\n{result}')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--run-integration', '-s'])
