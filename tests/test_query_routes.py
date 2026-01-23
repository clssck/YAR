"""Tests for query routes in lightrag/api/routers/query_routes.py.

This module tests the three main query endpoints:
- POST /query - Non-streaming RAG query
- POST /query/stream - Streaming RAG query (NDJSON format)
- POST /query/data - Data retrieval without LLM generation

Uses httpx AsyncClient with FastAPI's TestClient pattern and mocked RAG instance.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field

# =============================================================================
# Response Models (simplified for testing)
# =============================================================================


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description='Unique reference identifier')
    file_path: str = Field(description='Path to the source file')
    document_title: str | None = Field(default=None)
    s3_key: str | None = Field(default=None)
    presigned_url: str | None = Field(default=None)
    content: list[str] | None = Field(default=None)


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""

    response: str = Field(description='The generated response')
    references: list[ReferenceItem] | None = Field(default=None)


class QueryDataResponse(BaseModel):
    """Response model for /query/data endpoint."""

    status: str = Field(description='Query execution status')
    message: str = Field(description='Status message')
    data: dict[str, Any] = Field(description='Query result data')
    metadata: dict[str, Any] = Field(description='Query metadata')


# =============================================================================
# Test Route Factory
# =============================================================================


def create_test_query_routes(rag: Any, api_key: str | None = None):
    """Create query routes for testing (simplified version without auth)."""
    from lightrag.api.routers.query_routes import (
        QueryRequest,
        deduplicate_references_section,
        renumber_references_sequential,
        strip_reasoning_tags,
    )

    router = APIRouter(tags=['query'])
    # Note: api_key parameter kept for interface compatibility but not used in tests

    @router.post('/query', response_model=QueryResponse)
    async def query_text(request: QueryRequest):
        """Non-streaming query endpoint."""
        try:
            param = request.to_query_params(False)
            param.stream = False

            result = await rag.aquery_llm(request.query, param=param)

            llm_response = result.get('llm_response', {})
            data = result.get('data', {})
            references = data.get('references', [])

            response_content = llm_response.get('content', '')
            if not response_content:
                response_content = 'No relevant context found for the query.'

            response_content = strip_reasoning_tags(response_content)
            response_content = deduplicate_references_section(response_content)
            response_content = renumber_references_sequential(response_content)

            if request.include_references:
                return QueryResponse(response=response_content, references=references)
            else:
                return QueryResponse(response=response_content, references=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post('/query/data', response_model=QueryDataResponse)
    async def query_data(request: QueryRequest):
        """Data retrieval endpoint."""
        try:
            param = request.to_query_params(False)
            response = await rag.aquery_data(request.query, param=param)

            if isinstance(response, dict):
                return QueryDataResponse(**response)
            else:
                return QueryDataResponse(
                    status='failure',
                    message='Invalid response type',
                    data={},
                    metadata={},
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rag():
    """Create a mock RAG instance."""
    rag = MagicMock()
    rag.aquery_llm = AsyncMock()
    rag.aquery_data = AsyncMock()
    return rag


@pytest.fixture
def app(mock_rag):
    """Create FastAPI app with query routes."""
    app = FastAPI()
    router = create_test_query_routes(rag=mock_rag, api_key=None)
    app.include_router(router)
    return app


@pytest.fixture
async def client(app):
    """Create async HTTP client for testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url='http://test',
    ) as client:
        yield client


# =============================================================================
# Helper Functions Tests
# =============================================================================


@pytest.mark.offline
class TestHelperFunctions:
    """Tests for helper functions in query_routes."""

    def test_strip_reasoning_tags_removes_think_blocks(self):
        """Test that <think>...</think> blocks are removed."""
        from lightrag.api.routers.query_routes import strip_reasoning_tags

        text = 'Hello <think>internal reasoning here</think> World'
        result = strip_reasoning_tags(text)
        assert result == 'Hello  World'

    def test_strip_reasoning_tags_handles_multiline(self):
        """Test that multiline think blocks are removed."""
        from lightrag.api.routers.query_routes import strip_reasoning_tags

        text = 'Start <think>\nLine 1\nLine 2\n</think> End'
        result = strip_reasoning_tags(text)
        assert result == 'Start  End'

    def test_strip_reasoning_tags_empty_input(self):
        """Test that empty input returns empty string."""
        from lightrag.api.routers.query_routes import strip_reasoning_tags

        assert strip_reasoning_tags('') == ''
        # Note: function signature says str, but implementation handles None
        assert strip_reasoning_tags(None) is None  # type: ignore[arg-type]

    def test_deduplicate_references_section_removes_duplicates(self):
        """Test that duplicate reference entries are removed."""
        from lightrag.api.routers.query_routes import deduplicate_references_section

        text = """Some content here.

### References
- [1] Document.pdf
- [1] Document.pdf
- [2] Another.pdf
"""
        result = deduplicate_references_section(text)
        # Should have only one [1] entry
        assert result.count('[1] Document.pdf') == 1
        assert '[2] Another.pdf' in result

    def test_renumber_references_sequential(self):
        """Test that sparse references are renumbered sequentially."""
        from lightrag.api.routers.query_routes import renumber_references_sequential

        text = 'See [2] and [5] and [9] for details.'
        result = renumber_references_sequential(text)
        assert '[1]' in result
        assert '[2]' in result
        assert '[3]' in result
        assert '[5]' not in result
        assert '[9]' not in result

    def test_renumber_references_preserves_order(self):
        """Test that renumbering preserves first-appearance order."""
        from lightrag.api.routers.query_routes import renumber_references_sequential

        text = 'First [5], then [2], then [5] again.'
        result = renumber_references_sequential(text)
        # [5] appears first -> becomes [1], [2] appears second -> becomes [2]
        assert result == 'First [1], then [2], then [1] again.'

    def test_renumber_references_empty_input(self):
        """Test that empty input is handled correctly."""
        from lightrag.api.routers.query_routes import renumber_references_sequential

        assert renumber_references_sequential('') == ''
        assert renumber_references_sequential('No refs here') == 'No refs here'


# =============================================================================
# QueryRequest Model Tests
# =============================================================================


@pytest.mark.offline
class TestQueryRequestModel:
    """Tests for QueryRequest model validation."""

    def test_query_request_valid_basic(self):
        """Test that valid basic query is accepted."""
        from lightrag.api.routers.query_routes import QueryRequest

        req = QueryRequest(query='What is AI?')
        assert req.query == 'What is AI?'
        assert req.mode == 'mix'  # default

    def test_query_request_strips_whitespace(self):
        """Test that query whitespace is stripped."""
        from lightrag.api.routers.query_routes import QueryRequest

        req = QueryRequest(query='  What is AI?  ')
        assert req.query == 'What is AI?'

    def test_query_request_too_short_rejected(self):
        """Test that query < 3 chars is rejected."""
        from pydantic import ValidationError

        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query='ab')

    def test_query_request_valid_modes(self):
        """Test all valid query modes."""
        from lightrag.api.routers.query_routes import QueryRequest

        # Test each mode individually to satisfy type checker
        assert QueryRequest(query='test query', mode='local').mode == 'local'
        assert QueryRequest(query='test query', mode='global').mode == 'global'
        assert QueryRequest(query='test query', mode='hybrid').mode == 'hybrid'
        assert QueryRequest(query='test query', mode='naive').mode == 'naive'
        assert QueryRequest(query='test query', mode='mix').mode == 'mix'
        assert QueryRequest(query='test query', mode='bypass').mode == 'bypass'

    def test_query_request_invalid_mode_rejected(self):
        """Test that invalid mode is rejected."""
        from pydantic import ValidationError

        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query='test query', mode='invalid_mode')  # type: ignore[arg-type]

    def test_query_request_conversation_history_valid(self):
        """Test valid conversation history format."""
        from lightrag.api.routers.query_routes import QueryRequest

        history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
        ]
        req = QueryRequest(query='Continue please', conversation_history=history)
        assert req.conversation_history is not None
        assert len(req.conversation_history) == 2

    def test_query_request_conversation_history_missing_role(self):
        """Test that conversation history without role is rejected."""
        from pydantic import ValidationError

        from lightrag.api.routers.query_routes import QueryRequest

        history = [{'content': 'Hello'}]  # missing 'role'
        with pytest.raises(ValidationError):
            QueryRequest(query='test query', conversation_history=history)

    def test_query_request_to_query_params(self):
        """Test conversion to QueryParam."""
        from lightrag.api.routers.query_routes import QueryRequest

        req = QueryRequest(query='test query', mode='local', top_k=5)
        param = req.to_query_params(is_stream=False)
        assert param.mode == 'local'
        assert param.top_k == 5
        assert param.stream is False


# =============================================================================
# POST /query Endpoint Tests
# =============================================================================


@pytest.mark.offline
class TestQueryEndpoint:
    """Tests for POST /query endpoint."""

    @pytest.mark.asyncio
    async def test_query_returns_response(self, client, mock_rag):
        """Test that query returns properly formatted response."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {
                'content': 'AI is artificial intelligence.',
                'is_streaming': False,
            },
            'data': {
                'references': [
                    {'reference_id': '1', 'file_path': '/docs/ai.pdf'},
                ],
            },
        }

        response = await client.post(
            '/query',
            json={'query': 'What is AI?', 'include_references': True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['response'] == 'AI is artificial intelligence.'
        assert len(data['references']) == 1
        assert data['references'][0]['reference_id'] == '1'

    @pytest.mark.asyncio
    async def test_query_without_references(self, client, mock_rag):
        """Test query response without references."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Answer here.'},
            'data': {'references': []},
        }

        response = await client.post(
            '/query',
            json={'query': 'Test query', 'include_references': False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['response'] == 'Answer here.'
        assert data['references'] is None

    @pytest.mark.asyncio
    async def test_query_strips_reasoning_tags(self, client, mock_rag):
        """Test that <think> tags are stripped from response."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Start <think>reasoning</think> End'},
            'data': {'references': []},
        }

        response = await client.post('/query', json={'query': 'Test query'})

        assert response.status_code == 200
        data = response.json()
        assert '<think>' not in data['response']
        assert 'reasoning' not in data['response']

    @pytest.mark.asyncio
    async def test_query_empty_response_fallback(self, client, mock_rag):
        """Test fallback message when LLM returns empty content."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': ''},
            'data': {'references': []},
        }

        response = await client.post('/query', json={'query': 'Test query'})

        assert response.status_code == 200
        data = response.json()
        assert 'No relevant context found' in data['response']

    @pytest.mark.asyncio
    async def test_query_with_all_modes(self, client, mock_rag):
        """Test query works with all valid modes."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Response'},
            'data': {'references': []},
        }

        modes = ['local', 'global', 'hybrid', 'naive', 'mix', 'bypass']
        for mode in modes:
            response = await client.post(
                '/query',
                json={'query': 'Test query', 'mode': mode},
            )
            assert response.status_code == 200, f'Failed for mode: {mode}'


# =============================================================================
# POST /query Validation Tests
# =============================================================================


@pytest.mark.offline
class TestQueryValidation:
    """Tests for /query endpoint validation.

    Note: mock_rag is required by client fixture chain but not directly used
    in validation tests since validation happens before RAG is invoked.
    """

    @pytest.mark.asyncio
    async def test_query_empty_rejected(self, client):
        """Test that empty query is rejected with 422."""
        response = await client.post('/query', json={'query': ''})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_too_short_rejected(self, client):
        """Test that query < 3 chars is rejected."""
        response = await client.post('/query', json={'query': 'ab'})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_missing_rejected(self, client):
        """Test that missing query field is rejected."""
        response = await client.post('/query', json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_invalid_mode_rejected(self, client):
        """Test that invalid mode is rejected."""
        response = await client.post(
            '/query',
            json={'query': 'Test query', 'mode': 'invalid'},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_invalid_top_k_rejected(self, client):
        """Test that top_k < 1 is rejected."""
        response = await client.post(
            '/query',
            json={'query': 'Test query', 'top_k': 0},
        )
        assert response.status_code == 422


# =============================================================================
# POST /query Error Handling Tests
# =============================================================================


@pytest.mark.offline
class TestQueryErrors:
    """Tests for /query endpoint error handling."""

    @pytest.mark.asyncio
    async def test_query_rag_error_returns_500(self, client, mock_rag):
        """Test that RAG errors return 500."""
        mock_rag.aquery_llm.side_effect = Exception('LLM service unavailable')

        response = await client.post('/query', json={'query': 'Test query'})

        assert response.status_code == 500
        assert 'LLM service unavailable' in response.json()['detail']


# =============================================================================
# POST /query/data Endpoint Tests
# =============================================================================


@pytest.mark.offline
class TestQueryDataEndpoint:
    """Tests for POST /query/data endpoint."""

    @pytest.mark.asyncio
    async def test_query_data_returns_structured_response(self, client, mock_rag):
        """Test that /query/data returns structured data response."""
        mock_rag.aquery_data.return_value = {
            'status': 'success',
            'message': 'Query executed successfully',
            'data': {
                'entities': [
                    {
                        'entity_name': 'Neural Networks',
                        'entity_type': 'CONCEPT',
                        'description': 'Computational models',
                    }
                ],
                'relationships': [],
                'chunks': [
                    {
                        'content': 'Neural networks are...',
                        'file_path': '/docs/nn.pdf',
                        'chunk_id': 'chunk-1',
                    }
                ],
                'references': [{'reference_id': '1', 'file_path': '/docs/nn.pdf'}],
            },
            'metadata': {
                'query_mode': 'local',
                'keywords': {'high_level': ['neural'], 'low_level': ['network']},
            },
        }

        response = await client.post(
            '/query/data',
            json={'query': 'What are neural networks?', 'mode': 'local'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert len(data['data']['entities']) == 1
        assert data['data']['entities'][0]['entity_name'] == 'Neural Networks'
        assert data['metadata']['query_mode'] == 'local'

    @pytest.mark.asyncio
    async def test_query_data_naive_mode_no_entities(self, client, mock_rag):
        """Test that naive mode returns empty entities/relationships."""
        mock_rag.aquery_data.return_value = {
            'status': 'success',
            'message': 'Query executed successfully',
            'data': {
                'entities': [],
                'relationships': [],
                'chunks': [{'content': 'Some text', 'chunk_id': 'c1'}],
                'references': [],
            },
            'metadata': {'query_mode': 'naive'},
        }

        response = await client.post(
            '/query/data',
            json={'query': 'Test query', 'mode': 'naive'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['data']['entities'] == []
        assert data['data']['relationships'] == []
        assert len(data['data']['chunks']) == 1

    @pytest.mark.asyncio
    async def test_query_data_global_mode_returns_relationships(self, client, mock_rag):
        """Test that global mode returns relationships."""
        mock_rag.aquery_data.return_value = {
            'status': 'success',
            'message': 'Query executed successfully',
            'data': {
                'entities': [],
                'relationships': [
                    {
                        'src_id': 'AI',
                        'tgt_id': 'ML',
                        'description': 'AI encompasses ML',
                        'weight': 0.9,
                    }
                ],
                'chunks': [],
                'references': [],
            },
            'metadata': {'query_mode': 'global'},
        }

        response = await client.post(
            '/query/data',
            json={'query': 'AI trends', 'mode': 'global'},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data['data']['relationships']) == 1
        assert data['data']['relationships'][0]['src_id'] == 'AI'


# =============================================================================
# POST /query/data Validation Tests
# =============================================================================


@pytest.mark.offline
class TestQueryDataValidation:
    """Tests for /query/data endpoint validation."""

    @pytest.mark.asyncio
    async def test_query_data_empty_query_rejected(self, client):
        """Test that empty query is rejected."""
        response = await client.post('/query/data', json={'query': ''})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_data_short_query_rejected(self, client):
        """Test that query < 3 chars is rejected."""
        response = await client.post('/query/data', json={'query': 'ab'})
        assert response.status_code == 422


# =============================================================================
# POST /query/data Error Handling Tests
# =============================================================================


@pytest.mark.offline
class TestQueryDataErrors:
    """Tests for /query/data endpoint error handling."""

    @pytest.mark.asyncio
    async def test_query_data_rag_error_returns_500(self, client, mock_rag):
        """Test that RAG errors return 500."""
        mock_rag.aquery_data.side_effect = Exception('Knowledge graph unavailable')

        response = await client.post('/query/data', json={'query': 'Test query'})

        assert response.status_code == 500
        assert 'Knowledge graph unavailable' in response.json()['detail']

    @pytest.mark.asyncio
    async def test_query_data_invalid_response_format(self, client, mock_rag):
        """Test handling of unexpected response format."""
        mock_rag.aquery_data.return_value = 'not a dict'

        response = await client.post('/query/data', json={'query': 'Test query'})

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'failure'
        assert data['message'] == 'Invalid response type'


# =============================================================================
# Integration-like Tests (still offline)
# =============================================================================


@pytest.mark.offline
class TestQueryIntegration:
    """Integration-like tests for query routes."""

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, client, mock_rag):
        """Test query with conversation history context."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'More details about AI...'},
            'data': {'references': []},
        }

        response = await client.post(
            '/query',
            json={
                'query': 'Tell me more',
                'conversation_history': [
                    {'role': 'user', 'content': 'What is AI?'},
                    {'role': 'assistant', 'content': 'AI is...'},
                ],
            },
        )

        assert response.status_code == 200
        # Verify the RAG was called with proper params
        mock_rag.aquery_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_custom_keywords(self, client, mock_rag):
        """Test query with user-provided keywords."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Response about RAG'},
            'data': {'references': []},
        }

        response = await client.post(
            '/query',
            json={
                'query': 'What is RAG?',
                'hl_keywords': ['machine learning', 'NLP'],
                'll_keywords': ['retrieval', 'augmented', 'generation'],
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_with_token_limits(self, client, mock_rag):
        """Test query with token limit parameters."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Concise response'},
            'data': {'references': []},
        }

        response = await client.post(
            '/query',
            json={
                'query': 'Explain something',
                'max_total_tokens': 1000,
                'max_entity_tokens': 500,
                'max_relation_tokens': 300,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_with_reranking_options(self, client, mock_rag):
        """Test query with reranking options."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Reranked response'},
            'data': {'references': []},
        }

        response = await client.post(
            '/query',
            json={
                'query': 'Test reranking',
                'enable_rerank': True,
                'chunk_top_k': 10,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_with_bm25_fusion(self, client, mock_rag):
        """Test query with BM25 fusion enabled."""
        mock_rag.aquery_llm.return_value = {
            'llm_response': {'content': 'Hybrid search response'},
            'data': {'references': []},
        }

        response = await client.post(
            '/query',
            json={
                'query': 'Drug name XYZ-123',
                'enable_bm25_fusion': True,
                'bm25_weight': 0.5,
            },
        )

        assert response.status_code == 200
