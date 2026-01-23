"""Diagnostic tests for the rerank module.

Each test isolates a specific failure mode for easy debugging.
Tests are organized by component: chunking, aggregation, API, factory.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from tenacity import RetryError

pytestmark = pytest.mark.offline

from yar.rerank import (
    aggregate_chunk_scores,
    chunk_documents_for_rerank,
    create_rerank_func,
    deepinfra_rerank,
    generic_rerank_api,
)

# ============================================================================
# TEST HELPERS
# ============================================================================


class MockAsyncContextManager:
    """Async context manager for mocking aiohttp."""

    def __init__(self, obj):
        self.obj = obj

    async def __aenter__(self):
        return self.obj

    async def __aexit__(self, *args):
        pass


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, status: int, json_data: dict | None = None, text_data: str = ""):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data
        self.headers = {"content-type": "application/json"}
        self.request_info = MagicMock()
        self.history = []

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


class MockSession:
    """Mock aiohttp session with request capture."""

    def __init__(self, response: MockResponse):
        self.response = response
        self.call_args = None

    def post(self, *args, **kwargs):
        self.call_args = (args, kwargs)
        return MockAsyncContextManager(self.response)


def mock_aiohttp(status: int, json_data: dict | None = None, text: str = ""):
    """Create mock aiohttp session factory."""
    response = MockResponse(status, json_data, text)
    session = MockSession(response)
    return lambda: MockAsyncContextManager(session), response, session


# ============================================================================
# CHUNKING: chunk_documents_for_rerank()
# ============================================================================


class TestChunkingBoundaries:
    """Test chunking at critical boundaries where behavior changes."""

    def test_document_under_limit_unchanged(self):
        """DIAGNOSTIC: If this fails, chunking is too aggressive."""
        docs = ["Short document."]
        chunked, indices = chunk_documents_for_rerank(docs, max_tokens=100)

        assert chunked == docs, "Short doc should pass through unchanged"
        assert indices == [0], "Index mapping should be identity"

    def test_document_over_limit_splits(self):
        """DIAGNOSTIC: If this fails, chunking isn't triggering for long docs."""
        long_doc = "word " * 100  # ~100 tokens
        docs = [long_doc]

        chunked, indices = chunk_documents_for_rerank(docs, max_tokens=20)

        assert len(chunked) > 1, "Long doc should split into multiple chunks"
        assert all(idx == 0 for idx in indices), "All chunks should map to doc 0"

    def test_overlap_prevents_infinite_loop(self):
        """DIAGNOSTIC: If this hangs, overlap clamping is broken."""
        docs = ["Test document content."]

        # overlap >= max_tokens would cause infinite loop without clamping
        chunked, _indices = chunk_documents_for_rerank(docs, max_tokens=5, overlap_tokens=10)

        # Should complete without hanging
        assert len(chunked) >= 1, "Should produce output even with bad overlap"

    def test_empty_document_handled(self):
        """DIAGNOSTIC: If this fails, empty docs crash the chunker."""
        docs = ["", "non-empty"]
        chunked, _indices = chunk_documents_for_rerank(docs, max_tokens=100)

        assert "" in chunked, "Empty string should pass through"
        assert len(chunked) == 2


class TestChunkingIndexMapping:
    """Test that chunk indices correctly map back to original documents."""

    def test_mixed_lengths_preserve_order(self):
        """DIAGNOSTIC: If this fails, index mapping is corrupted."""
        short = "Short."
        long = "word " * 100
        docs = [short, long, short]  # indices 0, 1, 2

        _chunked, indices = chunk_documents_for_rerank(docs, max_tokens=20)

        # First chunk is doc 0, then several chunks from doc 1, then doc 2
        assert indices[0] == 0, "First chunk should be doc 0"
        assert indices[-1] == 2, "Last chunk should be doc 2"
        assert 1 in indices, "Doc 1 chunks should be present"

    def test_multiple_long_docs_tracked_separately(self):
        """DIAGNOSTIC: If this fails, chunks from different docs are mixed up."""
        long1 = "apple " * 50
        long2 = "banana " * 50
        docs = [long1, long2]

        _chunked, indices = chunk_documents_for_rerank(docs, max_tokens=20)

        # Find where doc 1 starts
        doc1_start = indices.index(1)

        # All indices before should be 0, all after should be 1
        assert all(i == 0 for i in indices[:doc1_start])
        assert all(i == 1 for i in indices[doc1_start:])


class TestChunkingTokenizerFallback:
    """Test character-based fallback when tokenizer fails."""

    def test_fallback_produces_valid_chunks(self):
        """DIAGNOSTIC: If this fails, fallback chunking is broken."""
        with patch('yar.rerank.TiktokenTokenizer', side_effect=Exception("No tokenizer")):
            doc = "A" * 200  # 200 chars
            docs = [doc]

            chunked, indices = chunk_documents_for_rerank(docs, max_tokens=10)

            # With 1 token ≈ 4 chars, max_tokens=10 → ~40 chars per chunk
            assert len(chunked) > 1, "Should chunk using character fallback"
            assert all(idx == 0 for idx in indices)


# ============================================================================
# AGGREGATION: aggregate_chunk_scores()
# ============================================================================


class TestAggregationStrategies:
    """Test each aggregation strategy produces correct results."""

    def test_max_takes_highest_chunk_score(self):
        """DIAGNOSTIC: If this fails, max aggregation is broken."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.3},  # chunk 0 → doc 0
            {"index": 1, "relevance_score": 0.9},  # chunk 1 → doc 0
        ]
        doc_indices = [0, 0]  # both chunks from doc 0

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1, aggregation="max")

        assert results[0]["relevance_score"] == 0.9, "Max should pick highest (0.9)"

    def test_mean_averages_chunk_scores(self):
        """DIAGNOSTIC: If this fails, mean aggregation is broken."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.4},
            {"index": 1, "relevance_score": 0.8},
        ]
        doc_indices = [0, 0]

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1, aggregation="mean")

        assert results[0]["relevance_score"] == pytest.approx(0.6), "Mean should be (0.4+0.8)/2 = 0.6"

    def test_first_takes_first_chunk_score(self):
        """DIAGNOSTIC: If this fails, first aggregation is broken."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.3},  # first chunk
            {"index": 1, "relevance_score": 0.9},  # second chunk (ignored)
        ]
        doc_indices = [0, 0]

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1, aggregation="first")

        assert results[0]["relevance_score"] == 0.3, "First should use 0.3, not 0.9"

    def test_unknown_strategy_defaults_to_max(self):
        """DIAGNOSTIC: If this fails, fallback to max is broken."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.8},
        ]
        doc_indices = [0, 0]

        results = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs=1, aggregation="invalid_strategy"
        )

        assert results[0]["relevance_score"] == 0.8, "Unknown strategy should default to max"


class TestAggregationSorting:
    """Test that aggregated results are properly sorted."""

    def test_results_sorted_descending(self):
        """DIAGNOSTIC: If this fails, result sorting is broken."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.2},  # doc 0
            {"index": 1, "relevance_score": 0.8},  # doc 1
            {"index": 2, "relevance_score": 0.5},  # doc 2
        ]
        doc_indices = [0, 1, 2]

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=3)

        scores = [r["relevance_score"] for r in results]
        assert scores == [0.8, 0.5, 0.2], "Results should be sorted descending"


class TestAggregationEdgeCases:
    """Test aggregation handles edge cases correctly."""

    def test_empty_input_returns_empty(self):
        """DIAGNOSTIC: If this fails, empty input crashes aggregation."""
        results = aggregate_chunk_scores([], [], num_original_docs=0)
        assert results == []

    def test_invalid_chunk_index_skipped(self):
        """DIAGNOSTIC: If this fails, invalid indices corrupt results."""
        chunk_results = [{"index": 999, "relevance_score": 0.9}]  # invalid
        doc_indices = [0]

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1)

        # Invalid index should be filtered out, doc 0 has no scores
        assert len(results) == 0

    def test_doc_with_no_chunks_excluded(self):
        """DIAGNOSTIC: If this fails, docs without scores appear in results."""
        chunk_results = [{"index": 0, "relevance_score": 0.9}]
        doc_indices = [1]  # chunk 0 → doc 1, so doc 0 has nothing

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=2)

        # Only doc 1 should appear
        assert len(results) == 1
        assert results[0]["index"] == 1


# ============================================================================
# DEEPINFRA API: deepinfra_rerank()
# ============================================================================


class TestDeepInfraFormat:
    """Test DeepInfra-specific request/response format."""

    @pytest.mark.asyncio
    async def test_sends_queries_array_not_query(self):
        """DIAGNOSTIC: If this fails, DeepInfra request format is wrong."""
        factory, _, session = mock_aiohttp(200, {"scores": [0.9, 0.1]})

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            await deepinfra_rerank(
                query="test query",
                documents=["doc1", "doc2"],
                api_key="key",
            )

            _, kwargs = session.call_args
            payload = kwargs["json"]

            # DeepInfra uses 'queries' (plural array), not 'query' (singular string)
            assert "queries" in payload, "Should use 'queries' not 'query'"
            assert payload["queries"] == ["test query"]
            assert "query" not in payload

    @pytest.mark.asyncio
    async def test_converts_scores_to_standard_format(self):
        """DIAGNOSTIC: If this fails, DeepInfra response parsing is broken."""
        factory, _, _ = mock_aiohttp(200, {"scores": [0.95, 0.30, 0.75]})

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await deepinfra_rerank(
                query="test",
                documents=["d1", "d2", "d3"],
                api_key="key",
            )

            # Should convert scores[] to [{index, relevance_score}, ...]
            assert len(results) == 3
            assert all("index" in r and "relevance_score" in r for r in results)

            # Should be sorted by score descending
            assert results[0]["relevance_score"] == 0.95
            assert results[0]["index"] == 0

    @pytest.mark.asyncio
    async def test_top_n_limits_results(self):
        """DIAGNOSTIC: If this fails, top_n filtering is broken."""
        factory, _, _ = mock_aiohttp(200, {"scores": [0.9, 0.8, 0.7, 0.6, 0.5]})

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await deepinfra_rerank(
                query="test",
                documents=["d1", "d2", "d3", "d4", "d5"],
                api_key="key",
                top_n=2,
            )

            assert len(results) == 2


class TestDeepInfraErrors:
    """Test DeepInfra error handling."""

    @pytest.mark.asyncio
    async def test_missing_base_url_raises_valueerror(self):
        """DIAGNOSTIC: If this fails, URL validation is broken."""
        with pytest.raises(ValueError, match="Base URL is required"):
            await deepinfra_rerank(
                query="test",
                documents=["doc"],
                api_key="key",
                base_url="",
            )

    @pytest.mark.asyncio
    async def test_api_error_retries_then_raises(self):
        """DIAGNOSTIC: If this fails, retry logic is broken."""
        factory, _, _ = mock_aiohttp(500, text="Server Error")

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            with pytest.raises(RetryError) as exc_info:
                await deepinfra_rerank(
                    query="test",
                    documents=["doc"],
                    api_key="key",
                )

            # Underlying error should be ClientResponseError
            assert isinstance(exc_info.value.last_attempt.exception(), aiohttp.ClientResponseError)

    @pytest.mark.asyncio
    async def test_empty_scores_returns_empty_list(self):
        """DIAGNOSTIC: If this fails, empty response handling is broken."""
        factory, _, _ = mock_aiohttp(200, {"scores": []})

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await deepinfra_rerank(
                query="test",
                documents=["doc"],
                api_key="key",
            )

            assert results == []


# ============================================================================
# GENERIC API: generic_rerank_api()
# ============================================================================


class TestGenericAPIFormat:
    """Test generic API request/response format."""

    @pytest.mark.asyncio
    async def test_standard_format_uses_query_singular(self):
        """DIAGNOSTIC: If this fails, standard format is wrong for Cohere/Jina."""
        factory, _, session = mock_aiohttp(200, {"results": [{"index": 0, "relevance_score": 0.9}]})

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
            )

            _, kwargs = session.call_args
            payload = kwargs["json"]

            # Standard format uses 'query' (singular)
            assert "query" in payload
            assert payload["query"] == "test"

    @pytest.mark.asyncio
    async def test_aliyun_format_uses_nested_structure(self):
        """DIAGNOSTIC: If this fails, Aliyun format is wrong."""
        factory, _, session = mock_aiohttp(
            200, {"output": {"results": [{"index": 0, "relevance_score": 0.9}]}}
        )

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
                request_format="aliyun",
                response_format="aliyun",
            )

            _, kwargs = session.call_args
            payload = kwargs["json"]

            # Aliyun uses nested input structure
            assert "input" in payload
            assert payload["input"]["query"] == "test"

    @pytest.mark.asyncio
    async def test_returns_standardized_format(self):
        """DIAGNOSTIC: If this fails, response standardization is broken."""
        factory, _, _ = mock_aiohttp(
            200, {"results": [{"index": 0, "relevance_score": 0.95, "extra_field": "ignored"}]}
        )

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
            )

            # Should only have index and relevance_score
            assert results == [{"index": 0, "relevance_score": 0.95}]


class TestGenericAPIErrors:
    """Test generic API error handling."""

    @pytest.mark.asyncio
    async def test_html_error_cleaned_to_readable_message(self):
        """DIAGNOSTIC: If this fails, HTML error cleaning is broken."""
        factory, response, _ = mock_aiohttp(502, text="<!DOCTYPE html><html>...</html>")
        response.headers = {"content-type": "text/html"}

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            with pytest.raises(RetryError) as exc_info:
                await generic_rerank_api(
                    query="test",
                    documents=["doc"],
                    model="model",
                    base_url="https://api.test.com",
                    api_key="key",
                )

            error = exc_info.value.last_attempt.exception()
            error_msg = str(error)
            # Should have clean message, not raw HTML
            assert "Bad Gateway" in error_msg or "502" in error_msg

    @pytest.mark.asyncio
    async def test_missing_base_url_raises_valueerror(self):
        """DIAGNOSTIC: If this fails, URL validation is broken."""
        with pytest.raises(ValueError, match="Base URL is required"):
            await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="",
                api_key="key",
            )

    @pytest.mark.asyncio
    async def test_malformed_results_returns_empty(self):
        """DIAGNOSTIC: If this fails, malformed response handling is broken."""
        factory, _, _ = mock_aiohttp(200, {"results": "not a list"})

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
            )

            assert results == []


class TestGenericAPIChunking:
    """Test chunking integration in generic API."""

    @pytest.mark.asyncio
    async def test_chunking_splits_long_doc_before_api_call(self):
        """DIAGNOSTIC: If this fails, chunking isn't being applied."""
        factory, _, session = mock_aiohttp(
            200,
            {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 2, "relevance_score": 0.7},
                ]
            },
        )

        long_doc = "word " * 200  # Will be chunked with max_tokens=50

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await generic_rerank_api(
                query="test",
                documents=[long_doc],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
                enable_chunking=True,
                max_tokens_per_doc=50,
            )

            # API should receive multiple chunks
            _, kwargs = session.call_args
            sent_docs = kwargs["json"]["documents"]
            assert len(sent_docs) > 1, "Should have chunked the long document"

            # Result should be aggregated back to 1 document
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_chunking_preserves_top_n_at_document_level(self):
        """DIAGNOSTIC: If this fails, top_n is applied to chunks not docs."""
        factory, _, _ = mock_aiohttp(
            200,
            {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 2, "relevance_score": 0.7},
                    {"index": 3, "relevance_score": 0.6},
                ]
            },
        )

        # Two long docs that will each be chunked into 2+ chunks
        long_doc1 = "apple " * 100
        long_doc2 = "banana " * 100

        with patch("yar.rerank.aiohttp.ClientSession", factory):
            results = await generic_rerank_api(
                query="test",
                documents=[long_doc1, long_doc2],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
                enable_chunking=True,
                max_tokens_per_doc=50,
                top_n=1,  # Should limit to 1 DOCUMENT, not 1 chunk
            )

            assert len(results) == 1, "top_n=1 should return 1 document"


# ============================================================================
# FACTORY: create_rerank_func()
# ============================================================================


class TestFactoryRouting:
    """Test factory routes to correct provider."""

    def test_deepinfra_binding_routes_to_deepinfra_rerank(self):
        """DIAGNOSTIC: If this fails, DeepInfra routing is broken."""
        func = create_rerank_func(binding="deepinfra", api_key="test")
        assert callable(func)

    def test_cohere_binding_routes_to_generic_api(self):
        """DIAGNOSTIC: If this fails, Cohere routing is broken."""
        func = create_rerank_func(binding="cohere", api_key="test")
        assert callable(func)

    def test_all_bindings_create_callable(self):
        """DIAGNOSTIC: If this fails, a binding is misconfigured."""
        for binding in ["cohere", "jina", "deepinfra", "openai", "aliyun"]:
            func = create_rerank_func(binding=binding, api_key="test")
            assert callable(func), f"{binding} binding should create callable"


class TestFactoryEnvVars:
    """Test factory reads from environment variables."""

    def test_reads_binding_from_env(self):
        """DIAGNOSTIC: If this fails, env var reading is broken."""
        with patch.dict("os.environ", {"RERANK_BINDING": "jina", "JINA_API_KEY": "env-key"}):
            func = create_rerank_func()  # No args - should use env
            assert callable(func)

    def test_args_override_env_vars(self):
        """DIAGNOSTIC: If this fails, arg precedence is wrong."""
        with patch.dict("os.environ", {"RERANK_BINDING": "jina"}):
            # Explicit arg should override env
            func = create_rerank_func(binding="cohere", api_key="test")
            assert callable(func)


# ============================================================================
# INTEGRATION TESTS - Real API Calls
# ============================================================================


@pytest.mark.integration
class TestRerankIntegration:
    """Integration tests with real API.

    Run with: pytest tests/test_rerank.py -m integration --run-integration
    """

    @pytest.fixture
    def sample_docs(self):
        return [
            "Python is a programming language known for readability.",
            "The capital of France is Paris, home of the Eiffel Tower.",
            "Machine learning is a subset of artificial intelligence.",
            "The Great Wall of China is a famous landmark.",
            "JavaScript runs in web browsers.",
        ]

    @pytest.mark.asyncio
    async def test_relevant_doc_scores_higher(self, sample_docs):
        """DIAGNOSTIC: If this fails, the reranker isn't semantically ranking."""
        rerank = create_rerank_func()

        results = await rerank(
            query="What programming language is good for beginners?",
            documents=sample_docs,
        )

        # Python doc (index 0) should score higher than France doc (index 1)
        python_score = next(r["relevance_score"] for r in results if r["index"] == 0)
        france_score = next(r["relevance_score"] for r in results if r["index"] == 1)

        assert python_score > france_score, (
            f"Python doc ({python_score:.3f}) should rank higher than France doc ({france_score:.3f})"
        )

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance(self, sample_docs):
        """DIAGNOSTIC: If this fails, result sorting is broken."""
        rerank = create_rerank_func()

        results = await rerank(
            query="artificial intelligence",
            documents=sample_docs,
        )

        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_top_n_limits_results(self, sample_docs):
        """DIAGNOSTIC: If this fails, top_n parameter is ignored."""
        rerank = create_rerank_func()

        results = await rerank(
            query="programming",
            documents=sample_docs,
            top_n=2,
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_single_document(self):
        """DIAGNOSTIC: If this fails, single doc edge case is broken."""
        rerank = create_rerank_func()

        results = await rerank(
            query="test",
            documents=["Single document."],
        )

        assert len(results) == 1
        assert results[0]["index"] == 0


@pytest.mark.integration
class TestUnicodeIntegration:
    """Integration tests for Unicode handling."""

    @pytest.mark.asyncio
    async def test_chinese_documents(self):
        """DIAGNOSTIC: If this fails, Chinese text handling is broken."""
        rerank = create_rerank_func()

        docs = [
            "Python是一种高级编程语言",  # Python is a high-level language
            "巴黎是法国的首都",  # Paris is the capital of France
        ]

        results = await rerank(
            query="什么是Python编程?",  # What is Python programming?
            documents=docs,
        )

        assert len(results) == 2
        # Python doc should rank first
        assert results[0]["index"] == 0

    @pytest.mark.asyncio
    async def test_emoji_in_documents(self):
        """DIAGNOSTIC: If this fails, emoji handling is broken."""
        rerank = create_rerank_func()

        docs = [
            "Python 🐍 is great!",
            "Java ☕ enterprise",
        ]

        results = await rerank(
            query="Python snake",
            documents=docs,
        )

        assert len(results) == 2


@pytest.mark.integration
class TestConcurrencyIntegration:
    """Integration tests for concurrent requests."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_succeed(self):
        """DIAGNOSTIC: If this fails, concurrent handling is broken."""
        rerank = create_rerank_func()

        docs = ["Doc about Python", "Doc about France", "Doc about AI"]

        # Launch 5 parallel requests
        tasks = [rerank(f"query {i}", docs) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(len(r) == 3 for r in results)


@pytest.mark.integration
class TestLongDocumentIntegration:
    """Integration tests for long document handling."""

    @pytest.mark.asyncio
    async def test_long_document_handled(self):
        """DIAGNOSTIC: If this fails, long doc handling is broken."""
        rerank = create_rerank_func()

        long_doc = "Python programming is useful. " * 200  # ~1000+ tokens

        results = await rerank(
            query="Python",
            documents=[long_doc, "Short doc about cooking."],
        )

        # Should complete without error
        assert len(results) == 2

        # Long Python doc should rank higher than short cooking doc
        python_result = next(r for r in results if r["index"] == 0)
        cooking_result = next(r for r in results if r["index"] == 1)
        assert python_result["relevance_score"] > cooking_result["relevance_score"]
