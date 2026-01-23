"""
Live integration tests for LightRAG API.

These tests query the actual running LightRAG server and validate real responses.
They require the server to be running with a configured database and LLM.

Run with:
    pytest tests/test_rag_live_integration.py -v --run-integration

Requirements:
    - LightRAG server running on localhost:9621
    - PostgreSQL database with data
    - Valid LLM API keys configured
"""

import os
import time

import pytest
import requests

# Configuration
BASE_URL = os.getenv('LIGHTRAG_TEST_URL', 'http://localhost:9621')
API_KEY = os.getenv('LIGHTRAG_API_KEY', None)
TIMEOUT = 60  # LLM queries can take time


def get_headers():
    """Get request headers with optional API key."""
    headers = {'Content-Type': 'application/json'}
    if API_KEY:
        headers['X-API-Key'] = API_KEY
    return headers


def server_is_available():
    """Check if the server is available."""
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Skip all tests if server is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.skipif(
        not server_is_available(),
        reason='LightRAG server not available at ' + BASE_URL,
    ),
]


# =============================================================================
# Health & Status Tests
# =============================================================================


class TestServerHealth:
    """Tests for server health and status endpoints."""

    def test_health_endpoint(self):
        """Test /health endpoint returns healthy status."""
        response = requests.get(f'{BASE_URL}/health', timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'configuration' in data
        assert 'core_version' in data

    def test_health_contains_storage_info(self):
        """Test health endpoint includes storage configuration."""
        response = requests.get(f'{BASE_URL}/health', timeout=TIMEOUT)

        data = response.json()
        config = data['configuration']
        assert 'kv_storage' in config
        assert 'vector_storage' in config
        assert 'graph_storage' in config

    def test_health_contains_llm_info(self):
        """Test health endpoint includes LLM configuration."""
        response = requests.get(f'{BASE_URL}/health', timeout=TIMEOUT)

        data = response.json()
        config = data['configuration']
        assert 'llm_binding' in config
        assert 'llm_model' in config
        assert 'embedding_model' in config


# =============================================================================
# Query Endpoint Tests
# =============================================================================


class TestQueryEndpoint:
    """Tests for the /query endpoint with real LLM calls."""

    def test_simple_query(self):
        """Test a simple query returns a response."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'What is LightRAG?', 'mode': 'naive'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data
        assert len(data['response']) > 0

    def test_query_with_local_mode(self):
        """Test query with local mode."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'What entities exist in the knowledge graph?', 'mode': 'local'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data

    def test_query_with_global_mode(self):
        """Test query with global mode."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'Give me a high-level overview', 'mode': 'global'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data

    def test_query_with_hybrid_mode(self):
        """Test query with hybrid mode."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'Explain the main concepts', 'mode': 'hybrid'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data

    def test_query_with_mix_mode(self):
        """Test query with mix mode (default)."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'Tell me about the system', 'mode': 'mix'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data

    def test_query_with_references(self):
        """Test query with references enabled."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={
                'query': 'What information is stored?',
                'mode': 'hybrid',
                'include_references': True,
            },
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data
        # References may or may not be present depending on data
        if 'references' in data:
            assert isinstance(data['references'], list)

    def test_query_references_contain_file_path(self):
        """Test that references include file_path for source traceability."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={
                'query': 'LightRAG features',
                'mode': 'hybrid',
                'include_references': True,
            },
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()

        # If we got references, validate their structure
        if data.get('references'):
            ref = data['references'][0]
            assert 'reference_id' in ref
            assert 'file_path' in ref
            # Should have excerpt for context
            assert 'excerpt' in ref or 'document_title' in ref

    def test_short_answer_has_inline_citation(self):
        """Test that SHORT answers (1-2 sentences) have inline [n] citations.

        This is a critical test for citation consistency - short answers were
        the main failure mode where LLMs would skip inline citations.
        """
        import re

        response = requests.post(
            f'{BASE_URL}/query',
            json={
                'query': 'What is LightRAG?',  # Expects a concise answer
                'mode': 'naive',
                'response_type': 'Single Paragraph',  # Encourage short response
            },
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data

        # Extract main text (before References section if present)
        main_text = data['response'].split('### References')[0].strip()

        # Skip if no content (empty database)
        if not main_text or 'insufficient information' in main_text.lower():
            pytest.skip('No relevant data available for query')

        # Check for at least one inline citation [n]
        has_inline_citation = bool(re.search(r'\[\d+\]', main_text))

        assert has_inline_citation, (
            f'Short answer missing inline citation [n]. '
            f'Response: {main_text[:200]}...'
        )

    def test_query_with_top_k_parameter(self):
        """Test query with custom top_k parameter."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'List some entities', 'mode': 'local', 'top_k': 5},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'response' in data

    def test_query_with_only_need_context(self):
        """Test query with only_need_context=True (no LLM call)."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'What is stored?', 'mode': 'naive', 'only_need_context': True},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        # Should return context without LLM response
        assert 'response' in data


# =============================================================================
# Streaming Query Tests
# =============================================================================


class TestStreamingQuery:
    """Tests for streaming query endpoint."""

    def test_streaming_query(self):
        """Test /query/stream returns streaming response."""
        response = requests.post(
            f'{BASE_URL}/query/stream',
            json={'query': 'Explain briefly', 'mode': 'naive'},
            headers=get_headers(),
            timeout=TIMEOUT,
            stream=True,
        )

        assert response.status_code == 200
        # LightRAG uses NDJSON format for streaming
        content_type = response.headers.get('content-type', '')
        assert 'ndjson' in content_type or 'json' in content_type

        # Collect streamed chunks
        chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                chunks.append(line)

        # Should have received some data
        assert len(chunks) > 0

    def test_streaming_query_ndjson_format(self):
        """Test streaming response is valid NDJSON."""
        import json

        response = requests.post(
            f'{BASE_URL}/query/stream',
            json={'query': 'Brief answer', 'mode': 'naive'},
            headers=get_headers(),
            timeout=TIMEOUT,
            stream=True,
        )

        assert response.status_code == 200

        valid_json_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            # Handle both str and bytes
            line_str = line if isinstance(line, str) else line.decode('utf-8')
            if line_str.startswith(':'):  # Skip SSE comments
                continue
            # Remove 'data: ' prefix if present
            if line_str.startswith('data: '):
                line_str = line_str[6:]
            if line_str:
                try:
                    json.loads(line_str)
                    valid_json_count += 1
                except json.JSONDecodeError:
                    pass  # Some lines may not be JSON

        assert valid_json_count > 0


# =============================================================================
# Query Data Endpoint Tests
# =============================================================================


class TestQueryDataEndpoint:
    """Tests for the /query/data endpoint."""

    def test_query_data_endpoint(self):
        """Test /query/data returns structured data."""
        response = requests.post(
            f'{BASE_URL}/query/data',
            json={'query': 'What entities are there?', 'mode': 'hybrid'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure - status can be 'success' or 'failure' (no data)
        assert 'status' in data
        assert data['status'] in ['success', 'failure']
        # data field should be present even if empty
        assert 'data' in data or 'message' in data

    def test_query_data_references_structure(self):
        """Test /query/data returns properly structured references."""
        response = requests.post(
            f'{BASE_URL}/query/data',
            json={'query': 'LightRAG', 'mode': 'hybrid'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()

        if data['status'] == 'success' and 'data' in data:
            inner = data['data']
            # Check expected fields exist
            assert 'entities' in inner or 'chunks' in inner or 'references' in inner

            # Validate references structure if present
            if inner.get('references'):
                ref = inner['references'][0]
                assert 'reference_id' in ref
                assert 'file_path' in ref

            # Validate entities have file_path for traceability
            if inner.get('entities'):
                entity = inner['entities'][0]
                assert 'file_path' in entity
                assert 'source_id' in entity

    def test_query_data_contains_entities(self):
        """Test /query/data returns entities when available."""
        response = requests.post(
            f'{BASE_URL}/query/data',
            json={'query': 'List all entities', 'mode': 'local'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()

        # Entities may be in data.data.entities or data.entities
        inner_data = data.get('data', data)
        if 'entities' in inner_data:
            assert isinstance(inner_data['entities'], list)

    def test_query_data_with_exclude_entities(self):
        """Test /query/data with entities excluded."""
        response = requests.post(
            f'{BASE_URL}/query/data',
            json={
                'query': 'What is stored?',
                'mode': 'hybrid',
                'include_entities': False,
            },
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        assert 'status' in data


# =============================================================================
# Graph Endpoints Tests
# =============================================================================


class TestGraphEndpoints:
    """Tests for graph-related endpoints."""

    def test_graph_label_endpoint(self):
        """Test /graphs/label returns graph statistics."""
        response = requests.get(
            f'{BASE_URL}/graphs/label',
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        # May return 200 or 404 depending on data
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_graph_nodes_endpoint(self):
        """Test /graphs endpoint returns nodes with label parameter."""
        response = requests.get(
            f'{BASE_URL}/graphs',
            params={'label': 'entity', 'limit': 10},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        # Should return nodes and edges structure
        assert 'nodes' in data
        assert 'edges' in data


# =============================================================================
# Document Endpoints Tests
# =============================================================================


class TestDocumentEndpoints:
    """Tests for document-related endpoints."""

    def test_documents_list_endpoint(self):
        """Test /documents/list endpoint returns document list."""
        response = requests.post(
            f'{BASE_URL}/documents/list',
            json={'limit': 10},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        # May return 200 or 404 depending on data/endpoint availability
        assert response.status_code in [200, 404, 422]
        if response.status_code == 200:
            data = response.json()
            assert 'documents' in data or isinstance(data, (list, dict))

    def test_documents_status_counts(self):
        """Test /documents/status_counts returns status summary."""
        response = requests.get(
            f'{BASE_URL}/documents/status_counts',
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()
        # Should have status counts
        assert isinstance(data, dict)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_query_mode(self):
        """Test invalid mode returns validation error."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'test', 'mode': 'invalid_mode'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 422

    def test_empty_query(self):
        """Test empty query returns validation error."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': '', 'mode': 'naive'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        # Should return 422 or handle gracefully
        assert response.status_code in [200, 422]

    def test_missing_query_field(self):
        """Test missing query field returns validation error."""
        response = requests.post(
            f'{BASE_URL}/query',
            json={'mode': 'naive'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 422

    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404."""
        response = requests.get(
            f'{BASE_URL}/nonexistent_endpoint',
            headers=get_headers(),
            timeout=TIMEOUT,
        )

        assert response.status_code == 404


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Basic performance tests."""

    def test_health_endpoint_latency(self):
        """Test health endpoint responds quickly."""
        start = time.time()
        response = requests.get(f'{BASE_URL}/health', timeout=TIMEOUT)
        latency = time.time() - start

        assert response.status_code == 200
        assert latency < 2.0  # Health should respond in under 2 seconds

    def test_query_response_time(self):
        """Test query responds within reasonable time."""
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/query',
            json={'query': 'Quick test', 'mode': 'naive'},
            headers=get_headers(),
            timeout=TIMEOUT,
        )
        latency = time.time() - start

        assert response.status_code == 200
        # LLM queries should complete within timeout
        assert latency < TIMEOUT


# =============================================================================
# Concurrent Request Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent request handling."""

    def test_concurrent_health_checks(self):
        """Test server handles concurrent health checks."""
        import concurrent.futures

        def check_health():
            response = requests.get(f'{BASE_URL}/health', timeout=10)
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_health) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status in results)

    def test_concurrent_queries(self):
        """Test server handles concurrent queries."""
        import concurrent.futures

        def make_query():
            response = requests.post(
                f'{BASE_URL}/query',
                json={'query': 'Brief answer', 'mode': 'naive'},
                headers=get_headers(),
                timeout=TIMEOUT,
            )
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_query) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--run-integration'])
