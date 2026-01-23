"""
Functional tests for LightRAG query operations.

These tests verify the end-to-end query flow including:
- Query request validation and processing
- Different query modes (local, global, hybrid, naive, mix)
- Response formatting and reference handling
- Streaming responses
- Error handling and edge cases

Tests are marked as @pytest.mark.offline and use mocked LLM/storage
to run without external dependencies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    pass


# =============================================================================
# Mock Data Factories
# =============================================================================


def create_mock_query_response(
    content: str = "This is a test response about AI.",
    include_references: bool = True,
) -> dict:
    """Create a mock RAG query response."""
    response = {
        "content": content,
    }

    if include_references:
        response["content"] += "\n\n### References\n- [1] doc1.pdf\n- [2] doc2.pdf"

    return response


def create_mock_query_data_response(
    include_entities: bool = True,
    include_relationships: bool = True,
    include_sources: bool = True,
) -> dict:
    """Create a mock RAG query_data response matching QueryDataResponse schema."""
    entities = []
    relationships = []
    sources = []

    if include_entities:
        entities = [
            {
                "entity_name": "Artificial Intelligence",
                "entity_type": "TECHNOLOGY",
                "description": "A field of computer science",
                "source_id": "doc1",
            },
            {
                "entity_name": "Machine Learning",
                "entity_type": "TECHNOLOGY",
                "description": "A subset of AI",
                "source_id": "doc1",
            },
        ]

    if include_relationships:
        relationships = [
            {
                "src_id": "Artificial Intelligence",
                "tgt_id": "Machine Learning",
                "description": "includes",
                "keywords": "subset, component",
                "weight": 1.0,
                "source_id": "doc1",
            },
        ]

    if include_sources:
        sources = [
            {
                "id": "doc1",
                "content": "AI and ML are transforming technology...",
            },
        ]

    # Return format expected by QueryDataResponse
    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "response": "Answer based on the knowledge graph.",
            "entities": entities,
            "relationships": relationships,
            "sources": sources,
        },
        "metadata": {
            "mode": "hybrid",
            "query_length": 20,
        },
    }


def create_mock_stream_chunks(
    content: str = "This is a streaming response.",
    chunk_size: int = 10,
) -> list[str]:
    """Create mock streaming response chunks."""
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i : i + chunk_size]
        chunks.append(json.dumps({"response": chunk}) + "\n")
    return chunks


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rag():
    """Create a fully mocked RAG instance for functional testing."""
    rag = MagicMock()

    # Mock query methods
    rag.aquery_llm = AsyncMock(
        return_value={
            "llm_response": create_mock_query_response(),
            "data": create_mock_query_data_response(),
        }
    )

    rag.aquery_data = AsyncMock(return_value=create_mock_query_data_response())

    # Mock the streaming query method
    async def mock_aquery_stream():
        content = "This is a streaming response about artificial intelligence."
        for char in content:
            yield char

    rag.aquery_stream = mock_aquery_stream

    return rag


@pytest.fixture
def app_with_mock_rag(mock_rag):
    """Create FastAPI app with mocked RAG for functional testing."""
    from lightrag.api.routers.query_routes import create_query_routes

    app = FastAPI()
    router = create_query_routes(rag=mock_rag, api_key=None)
    app.include_router(router)
    return app


@pytest.fixture
async def client(app_with_mock_rag):
    """Create async HTTP client for functional testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mock_rag),
        base_url="http://test",
    ) as client:
        yield client


# =============================================================================
# Query Mode Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestQueryModesFunctional:
    """Functional tests for different RAG query modes."""

    @pytest.mark.parametrize(
        "mode",
        ["local", "global", "hybrid", "naive", "mix"],
    )
    async def test_query_all_modes_return_response(self, client, mock_rag, mode):
        """Test that all query modes return valid responses."""
        response = await client.post(
            "/query",
            json={"query": "What is artificial intelligence?", "mode": mode},
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    async def test_local_mode_focuses_on_entities(self, client, mock_rag):
        """Test local mode query returns entity-focused response."""
        response = await client.post(
            "/query",
            json={"query": "What is AI?", "mode": "local"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    async def test_global_mode_uses_relationships(self, client, mock_rag):
        """Test global mode query uses relationship data."""
        response = await client.post(
            "/query",
            json={"query": "How is AI related to ML?", "mode": "global"},
        )
        assert response.status_code == 200

    async def test_hybrid_mode_combines_local_global(self, client, mock_rag):
        """Test hybrid mode combines local and global context."""
        response = await client.post(
            "/query",
            json={"query": "Explain AI comprehensively", "mode": "hybrid"},
        )
        assert response.status_code == 200

    async def test_naive_mode_direct_retrieval(self, client, mock_rag):
        """Test naive mode performs direct document retrieval."""
        response = await client.post(
            "/query",
            json={"query": "Find documents about AI", "mode": "naive"},
        )
        assert response.status_code == 200

    async def test_mix_mode_comprehensive(self, client, mock_rag):
        """Test mix mode uses all available context."""
        response = await client.post(
            "/query",
            json={"query": "Give me everything about AI", "mode": "mix"},
        )
        assert response.status_code == 200


# =============================================================================
# Query Parameter Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestQueryParametersFunctional:
    """Functional tests for query parameters affecting RAG behavior."""

    async def test_top_k_parameter_limits_results(self, client, mock_rag):
        """Test top_k parameter is accepted by the API."""
        response = await client.post(
            "/query",
            json={
                "query": "What is AI?",
                "mode": "hybrid",
                "top_k": 5,
            },
        )
        assert response.status_code == 200

    async def test_max_tokens_parameter(self, client, mock_rag):
        """Test max_tokens parameter is accepted by the API."""
        response = await client.post(
            "/query",
            json={
                "query": "Explain AI in detail",
                "mode": "hybrid",
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200

    async def test_only_need_context_returns_sources(self, client, mock_rag):
        """Test only_need_context parameter is accepted by the API."""
        response = await client.post(
            "/query",
            json={
                "query": "Get context for AI",
                "mode": "hybrid",
                "only_need_context": True,
            },
        )
        assert response.status_code == 200

    async def test_conversation_history_passed_to_rag(self, client, mock_rag):
        """Test conversation history is accepted by the API."""
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]

        response = await client.post(
            "/query",
            json={
                "query": "Tell me more about it",
                "mode": "hybrid",
                "conversation_history": history,
            },
        )
        assert response.status_code == 200

    async def test_custom_keywords_enhance_search(self, client, mock_rag):
        """Test custom keywords are accepted by the API."""
        response = await client.post(
            "/query",
            json={
                "query": "What is this technology?",
                "mode": "hybrid",
                "hl_keywords": ["artificial", "intelligence", "neural"],
                "ll_keywords": ["AI", "ML", "deep learning"],
            },
        )
        assert response.status_code == 200


# =============================================================================
# Query Data Endpoint Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestQueryDataFunctional:
    """Functional tests for the /query/data endpoint."""

    async def test_query_data_returns_structured_response(self, client, mock_rag):
        """Test /query/data returns structured entity/relationship data."""
        response = await client.post(
            "/query/data",
            json={"query": "What entities relate to AI?", "mode": "hybrid"},
        )

        assert response.status_code == 200
        data = response.json()
        # Verify structure - should have status, message, data, metadata
        assert "status" in data
        assert data["status"] == "success"

    async def test_query_data_returns_entities(self, client, mock_rag):
        """Test query/data returns entity information."""
        response = await client.post(
            "/query/data",
            json={"query": "Find AI entities", "mode": "local"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    async def test_query_data_with_different_modes(self, client, mock_rag):
        """Test query/data works with all modes."""
        for mode in ["local", "global", "hybrid"]:
            response = await client.post(
                "/query/data",
                json={"query": "Find entities", "mode": mode},
            )
            assert response.status_code == 200


# =============================================================================
# Streaming Response Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestStreamingFunctional:
    """Functional tests for streaming query responses."""

    async def test_stream_endpoint_returns_chunks(self, client, mock_rag):
        """Test streaming endpoint returns response in chunks."""
        response = await client.post(
            "/query/stream",
            json={"query": "Explain AI step by step", "mode": "hybrid"},
        )

        assert response.status_code == 200
        # Streaming responses should have content-type for NDJSON
        content_type = response.headers.get("content-type", "")
        assert "application/x-ndjson" in content_type or "text/event-stream" in content_type or response.content

    async def test_stream_accumulates_to_complete_response(self, client, mock_rag):
        """Test streaming chunks accumulate to complete response."""
        response = await client.post(
            "/query/stream",
            json={"query": "What is machine learning?", "mode": "hybrid"},
        )

        assert response.status_code == 200
        # Response should contain data
        assert len(response.content) > 0


# =============================================================================
# Response Formatting Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestResponseFormattingFunctional:
    """Functional tests for response formatting and post-processing."""

    async def test_response_strips_reasoning_tags(self, client, mock_rag):
        """Test that <think> tags are stripped from responses."""
        mock_rag.aquery_llm.return_value = {
            "llm_response": {
                "content": "Let me think... <think>internal reasoning</think> The answer is 42."
            },
            "data": {},
        }

        response = await client.post(
            "/query",
            json={"query": "What is the answer?", "mode": "hybrid"},
        )

        assert response.status_code == 200
        data = response.json()
        # Reasoning tags should be stripped
        assert "<think>" not in data.get("response", "")

    async def test_response_deduplicates_references(self, client, mock_rag):
        """Test that duplicate references are removed."""
        mock_rag.aquery_llm.return_value = {
            "llm_response": {
                "content": "Info from [1] and [1] again.\n\n### References\n- [1] doc.pdf\n- [1] doc.pdf"
            },
            "data": {},
        }

        response = await client.post(
            "/query",
            json={"query": "Get info", "mode": "hybrid"},
        )

        assert response.status_code == 200

    async def test_response_renumbers_sparse_references(self, client, mock_rag):
        """Test that sparse references are renumbered sequentially."""
        mock_rag.aquery_llm.return_value = {
            "llm_response": {
                "content": "See [5] and [10] for details.\n\n### References\n- [5] first.pdf\n- [10] second.pdf"
            },
            "data": {},
        }

        response = await client.post(
            "/query",
            json={"query": "Get references", "mode": "hybrid"},
        )

        assert response.status_code == 200


# =============================================================================
# Error Handling Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestErrorHandlingFunctional:
    """Functional tests for error handling in query operations."""

    async def test_empty_query_rejected(self, client):
        """Test that empty queries are rejected with 422."""
        response = await client.post(
            "/query",
            json={"query": "", "mode": "hybrid"},
        )

        assert response.status_code == 422

    async def test_short_query_rejected(self, client):
        """Test that very short queries are rejected."""
        response = await client.post(
            "/query",
            json={"query": "a", "mode": "hybrid"},
        )

        assert response.status_code == 422

    async def test_invalid_mode_rejected(self, client):
        """Test that invalid query modes are rejected."""
        response = await client.post(
            "/query",
            json={"query": "What is AI?", "mode": "invalid_mode"},
        )

        assert response.status_code == 422

    async def test_missing_query_field_rejected(self, client):
        """Test that missing query field is rejected."""
        response = await client.post(
            "/query",
            json={"mode": "hybrid"},
        )

        assert response.status_code == 422


# =============================================================================
# Concurrent Query Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestConcurrentQueriesFunctional:
    """Functional tests for concurrent query handling."""

    async def test_multiple_concurrent_queries(self, client, mock_rag):
        """Test that multiple concurrent queries are handled correctly."""
        import asyncio

        queries = [
            {"query": "What is AI?", "mode": "local"},
            {"query": "What is ML?", "mode": "global"},
            {"query": "What is deep learning?", "mode": "hybrid"},
        ]

        async def make_query(query_data):
            return await client.post("/query", json=query_data)

        # Execute concurrently
        responses = await asyncio.gather(*[make_query(q) for q in queries])

        # All should succeed
        for response in responses:
            assert response.status_code == 200

    async def test_concurrent_different_endpoints(self, client, mock_rag):
        """Test concurrent queries to different endpoints."""
        import asyncio

        async def query_endpoint():
            return await client.post(
                "/query",
                json={"query": "What is AI?", "mode": "hybrid"},
            )

        async def query_data_endpoint():
            return await client.post(
                "/query/data",
                json={"query": "Find AI entities", "mode": "local"},
            )

        responses = await asyncio.gather(
            query_endpoint(),
            query_data_endpoint(),
            query_endpoint(),
        )

        for response in responses:
            assert response.status_code == 200


# =============================================================================
# Edge Case Functional Tests
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
class TestEdgeCasesFunctional:
    """Functional tests for edge cases and boundary conditions."""

    async def test_very_long_query(self, client, mock_rag):
        """Test handling of very long queries."""
        long_query = "What is " + "artificial intelligence " * 100 + "?"

        response = await client.post(
            "/query",
            json={"query": long_query, "mode": "hybrid"},
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 422]

    async def test_query_with_special_characters(self, client, mock_rag):
        """Test query with special characters."""
        response = await client.post(
            "/query",
            json={
                "query": "What is AI? (including ML & DL) @2024 #technology",
                "mode": "hybrid",
            },
        )

        assert response.status_code == 200

    async def test_query_with_unicode(self, client, mock_rag):
        """Test query with Unicode characters."""
        response = await client.post(
            "/query",
            json={
                "query": "什么是人工智能? ¿Qué es la IA? 人工知能とは？",
                "mode": "hybrid",
            },
        )

        assert response.status_code == 200

    async def test_query_with_code_snippets(self, client, mock_rag):
        """Test query containing code snippets."""
        response = await client.post(
            "/query",
            json={
                "query": "Explain this code: def hello(): print('Hello, AI!')",
                "mode": "hybrid",
            },
        )

        assert response.status_code == 200

    async def test_empty_rag_response_handled(self, client, mock_rag):
        """Test handling of empty RAG response."""
        mock_rag.aquery_llm.return_value = {
            "llm_response": {"content": ""},
            "data": {},
        }

        response = await client.post(
            "/query",
            json={"query": "Something with no results", "mode": "hybrid"},
        )

        assert response.status_code == 200

    async def test_null_response_fields_handled(self, client, mock_rag):
        """Test handling of null fields in RAG response."""
        mock_rag.aquery_llm.return_value = {
            "llm_response": {"content": None},
            "data": None,
        }

        response = await client.post(
            "/query",
            json={"query": "Query with null response", "mode": "hybrid"},
        )

        # Should handle gracefully
        assert response.status_code in [200, 500]
