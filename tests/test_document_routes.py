"""Tests for document routes in yar/api/routers/document_routes.py.

This module tests document management endpoints:
- POST /documents/scan - Scan for new documents
- POST /documents/upload - Upload a file
- POST /documents/text - Insert single text
- POST /documents/texts - Insert multiple texts
- DELETE /documents - Clear all documents
- GET /documents/pipeline_status - Get pipeline status
- DELETE /documents/delete_document - Delete documents by ID

Uses httpx AsyncClient with FastAPI's TestClient pattern and mocked dependencies.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from yar.api.routers.document_routes import create_document_routes

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rag():
    """Create a mock RAG instance."""
    rag = MagicMock()
    rag.doc_status = MagicMock()
    rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    rag.doc_status.get_by_id = AsyncMock(return_value=None)
    rag.doc_status.get_status_counts = AsyncMock(return_value={})
    rag.workspace = 'default'
    rag.text_chunks = MagicMock()
    rag.ainsert = AsyncMock(return_value='track_123')
    rag.apipeline_enqueue_documents = AsyncMock()
    rag.apipeline_process_enqueue_documents = AsyncMock()
    return rag


@pytest.fixture
def mock_doc_manager():
    """Create a mock DocumentManager instance."""
    manager = MagicMock()
    manager.input_dir = Path('/tmp/test_input')
    manager.supported_extensions = ['.txt', '.pdf', '.md']
    manager.is_supported_file = MagicMock(return_value=True)
    return manager


@pytest.fixture
def app(mock_rag, mock_doc_manager):
    """Create FastAPI app with document routes."""
    app = FastAPI()
    with patch('yar.api.routers.document_routes.global_args') as mock_global_args:
        mock_global_args.max_upload_size_mb = 100
        router = create_document_routes(
            rag=mock_rag, doc_manager=mock_doc_manager, api_key=None
        )
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
# Helper Function Tests
# =============================================================================


@pytest.mark.offline
class TestHelperFunctions:
    """Tests for helper functions in document_routes."""

    def test_format_datetime_with_datetime(self):
        """Test formatting datetime object."""
        from datetime import datetime, timezone

        from yar.api.routers.document_routes import format_datetime

        dt = datetime(2025, 1, 19, 12, 0, 0, tzinfo=timezone.utc)
        result = format_datetime(dt)
        assert result is not None and isinstance(result, str)
        assert '2025-01-19' in result
        assert '12:00:00' in result

    def test_format_datetime_with_naive_datetime(self):
        """Test formatting naive datetime (no timezone)."""
        from datetime import datetime

        from yar.api.routers.document_routes import format_datetime

        dt = datetime(2025, 1, 19, 12, 0, 0)
        result = format_datetime(dt)
        assert result is not None and isinstance(result, str)
        assert '2025-01-19' in result
        # Should add UTC timezone
        assert '+00:00' in result or 'Z' in result

    def test_format_datetime_with_string(self):
        """Test that string input is returned as-is."""
        from yar.api.routers.document_routes import format_datetime

        result = format_datetime('2025-01-19T12:00:00Z')
        assert result == '2025-01-19T12:00:00Z'

    def test_format_datetime_with_none(self):
        """Test that None returns None."""
        from yar.api.routers.document_routes import format_datetime

        assert format_datetime(None) is None

    def test_sanitize_filename_removes_path_separators(self):
        """Test that path separators are removed."""
        from yar.api.routers.document_routes import sanitize_filename

        result = sanitize_filename('path/to/file.txt', Path('/tmp'))
        assert '/' not in result
        assert '\\' not in result

    def test_sanitize_filename_removes_traversal(self):
        """Test that path traversal sequences are removed."""
        from yar.api.routers.document_routes import sanitize_filename

        result = sanitize_filename('..file.txt', Path('/tmp'))
        assert '..' not in result

    def test_sanitize_filename_empty_rejected(self):
        """Test that empty filename is rejected."""
        from yar.api.routers.document_routes import sanitize_filename

        with pytest.raises(HTTPException) as exc_info:
            sanitize_filename('', Path('/tmp'))
        assert exc_info.value.status_code == 400

    def test_sanitize_filename_whitespace_only_rejected(self):
        """Test that whitespace-only filename is rejected."""
        from yar.api.routers.document_routes import sanitize_filename

        with pytest.raises(HTTPException) as exc_info:
            sanitize_filename('   ', Path('/tmp'))
        assert exc_info.value.status_code == 400


# =============================================================================
# Request Model Tests
# =============================================================================


@pytest.mark.offline
class TestRequestModels:
    """Tests for request model validation."""

    def test_insert_text_request_valid(self):
        """Test valid InsertTextRequest."""
        from yar.api.routers.document_routes import InsertTextRequest

        req = InsertTextRequest(text='This is a test document.')
        assert req.text == 'This is a test document.'
        assert req.file_source is None

    def test_insert_text_request_with_source(self):
        """Test InsertTextRequest with file_source."""
        from yar.api.routers.document_routes import InsertTextRequest

        req = InsertTextRequest(text='Test', file_source='source.txt')
        assert req.file_source == 'source.txt'

    def test_insert_text_request_strips_whitespace(self):
        """Test that text whitespace is stripped."""
        from yar.api.routers.document_routes import InsertTextRequest

        req = InsertTextRequest(text='  Test text  ')
        assert req.text == 'Test text'

    def test_insert_text_request_empty_rejected(self):
        """Test that empty text is rejected."""
        from pydantic import ValidationError

        from yar.api.routers.document_routes import InsertTextRequest

        with pytest.raises(ValidationError):
            InsertTextRequest(text='')

    def test_insert_texts_request_valid(self):
        """Test valid InsertTextsRequest."""
        from yar.api.routers.document_routes import InsertTextsRequest

        req = InsertTextsRequest(texts=['Text 1', 'Text 2'])
        assert len(req.texts) == 2

    def test_insert_texts_request_strips_whitespace(self):
        """Test that all texts are stripped."""
        from yar.api.routers.document_routes import InsertTextsRequest

        req = InsertTextsRequest(texts=['  Text 1  ', '  Text 2  '])
        assert req.texts == ['Text 1', 'Text 2']

    def test_insert_texts_request_empty_list_rejected(self):
        """Test that empty texts list is rejected."""
        from pydantic import ValidationError

        from yar.api.routers.document_routes import InsertTextsRequest

        with pytest.raises(ValidationError):
            InsertTextsRequest(texts=[])

    def test_delete_doc_request_valid(self):
        """Test valid DeleteDocRequest."""
        from yar.api.routers.document_routes import DeleteDocRequest

        req = DeleteDocRequest(doc_ids=['doc-1', 'doc-2'])
        assert len(req.doc_ids) == 2

    def test_delete_doc_request_strips_ids(self):
        """Test that doc_ids are stripped."""
        from yar.api.routers.document_routes import DeleteDocRequest

        req = DeleteDocRequest(doc_ids=['  doc-1  ', '  doc-2  '])
        assert req.doc_ids == ['doc-1', 'doc-2']

    def test_delete_doc_request_empty_rejected(self):
        """Test that empty doc_ids list is rejected."""
        from pydantic import ValidationError

        from yar.api.routers.document_routes import DeleteDocRequest

        with pytest.raises(ValidationError):
            DeleteDocRequest(doc_ids=[])

    def test_delete_doc_request_duplicates_rejected(self):
        """Test that duplicate doc_ids are rejected."""
        from pydantic import ValidationError

        from yar.api.routers.document_routes import DeleteDocRequest

        with pytest.raises(ValidationError):
            DeleteDocRequest(doc_ids=['doc-1', 'doc-1'])


# =============================================================================
# POST /documents/scan Tests
# =============================================================================


@pytest.mark.offline
class TestScanEndpoint:
    """Tests for POST /documents/scan endpoint."""

    @pytest.mark.asyncio
    async def test_scan_returns_success(self, client):
        """Test that scan endpoint returns success status."""
        response = await client.post('/documents/scan')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'scanning_started'
        assert 'track_id' in data
        assert data['track_id'].startswith('scan_')

    @pytest.mark.asyncio
    async def test_scan_returns_track_id(self, client):
        """Test that scan returns a valid track_id."""
        response = await client.post('/documents/scan')

        assert response.status_code == 200
        data = response.json()
        assert data['track_id']
        # Track ID format: scan_YYYYMMDD_HHMMSS_random
        parts = data['track_id'].split('_')
        assert parts[0] == 'scan'


# =============================================================================
# POST /documents/text Tests
# =============================================================================


@pytest.mark.offline
class TestInsertTextEndpoint:
    """Tests for POST /documents/text endpoint."""

    @pytest.mark.asyncio
    async def test_insert_text_success(self, client, mock_rag):
        """Test successful text insertion."""
        mock_rag.doc_status.get_doc_by_file_path.return_value = None
        mock_rag.doc_status.get_by_id.return_value = None

        response = await client.post(
            '/documents/text',
            json={'text': 'This is a test document for RAG system.'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'track_id' in data

    @pytest.mark.asyncio
    async def test_insert_text_with_source(self, client, mock_rag):
        """Test text insertion with file_source."""
        mock_rag.doc_status.get_doc_by_file_path.return_value = None
        mock_rag.doc_status.get_by_id.return_value = None

        response = await client.post(
            '/documents/text',
            json={
                'text': 'Test document content.',
                'file_source': 'manual_input.txt',
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'

    @pytest.mark.asyncio
    async def test_insert_text_duplicate_detected(self, client, mock_rag):
        """Test that duplicate file_source is detected."""
        mock_rag.doc_status.get_doc_by_file_path.return_value = {
            'status': 'completed',
            'track_id': 'existing_123',
        }

        response = await client.post(
            '/documents/text',
            json={
                'text': 'Duplicate content.',
                'file_source': 'existing_source.txt',
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'duplicated'


# =============================================================================
# POST /documents/text Validation Tests
# =============================================================================


@pytest.mark.offline
class TestInsertTextValidation:
    """Tests for /documents/text validation."""

    @pytest.mark.asyncio
    async def test_insert_text_empty_rejected(self, client):
        """Test that empty text is rejected."""
        response = await client.post('/documents/text', json={'text': ''})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_insert_text_missing_rejected(self, client):
        """Test that missing text field is rejected."""
        response = await client.post('/documents/text', json={})
        assert response.status_code == 422


# =============================================================================
# POST /documents/texts Tests
# =============================================================================


@pytest.mark.offline
class TestInsertTextsEndpoint:
    """Tests for POST /documents/texts endpoint."""

    @pytest.mark.asyncio
    async def test_insert_texts_success(self, client, mock_rag):
        """Test successful multiple texts insertion."""
        mock_rag.doc_status.get_doc_by_file_path.return_value = None
        mock_rag.doc_status.get_by_id.return_value = None

        response = await client.post(
            '/documents/texts',
            json={
                'texts': [
                    'First document content.',
                    'Second document content.',
                    'Third document content.',
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'

    @pytest.mark.asyncio
    async def test_insert_texts_with_sources(self, client, mock_rag):
        """Test multiple texts with file_sources."""
        mock_rag.doc_status.get_doc_by_file_path.return_value = None
        mock_rag.doc_status.get_by_id.return_value = None

        response = await client.post(
            '/documents/texts',
            json={
                'texts': ['Content 1', 'Content 2'],
                'file_sources': ['source1.txt', 'source2.txt'],
            },
        )

        assert response.status_code == 200


# =============================================================================
# POST /documents/texts Validation Tests
# =============================================================================


@pytest.mark.offline
class TestInsertTextsValidation:
    """Tests for /documents/texts validation."""

    @pytest.mark.asyncio
    async def test_insert_texts_empty_list_rejected(self, client):
        """Test that empty texts list is rejected."""
        response = await client.post('/documents/texts', json={'texts': []})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_insert_texts_missing_rejected(self, client):
        """Test that missing texts field is rejected."""
        response = await client.post('/documents/texts', json={})
        assert response.status_code == 422


# =============================================================================
# GET /documents/pipeline_status Tests
# =============================================================================


@pytest.mark.offline
class TestPipelineStatusEndpoint:
    """Tests for GET /documents/pipeline_status endpoint."""

    @pytest.mark.asyncio
    async def test_pipeline_status_returns_data(self, client):
        """Test that pipeline status returns expected fields."""
        with patch(
            'yar.kg.shared_storage.get_namespace_data',
            new_callable=AsyncMock,
            return_value={
                'autoscanned': True,
                'busy': False,
                'job_name': '',
                'docs': 0,
                'batches': 0,
                'cur_batch': 0,
                'request_pending': False,
                'latest_message': '',
            },
        ), patch(
            'yar.kg.shared_storage.get_namespace_lock',
            return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
        ), patch(
            'yar.kg.shared_storage.get_all_update_flags_status',
            new_callable=AsyncMock,
            return_value={},
        ):
            response = await client.get('/documents/pipeline_status')

        assert response.status_code == 200
        data = response.json()
        assert 'autoscanned' in data
        assert 'busy' in data
        assert isinstance(data['busy'], bool)


# =============================================================================
# Integration-like Tests
# =============================================================================


@pytest.mark.offline
class TestDocumentIntegration:
    """Integration-like tests for document routes."""

    @pytest.mark.asyncio
    async def test_text_insertion_workflow(self, client, mock_rag):
        """Test a typical text insertion workflow."""
        # First insertion should succeed
        mock_rag.doc_status.get_doc_by_file_path.return_value = None
        mock_rag.doc_status.get_by_id.return_value = None

        response1 = await client.post(
            '/documents/text',
            json={'text': 'Document content here.', 'file_source': 'doc1.txt'},
        )
        assert response1.status_code == 200
        assert response1.json()['status'] == 'success'

        # Second insertion with same source should be detected as duplicate
        mock_rag.doc_status.get_doc_by_file_path.return_value = {
            'status': 'completed',
            'track_id': 'text_123',
        }

        response2 = await client.post(
            '/documents/text',
            json={'text': 'Same source different content.', 'file_source': 'doc1.txt'},
        )
        assert response2.status_code == 200
        assert response2.json()['status'] == 'duplicated'

    @pytest.mark.asyncio
    async def test_batch_text_insertion(self, client, mock_rag):
        """Test batch text insertion."""
        mock_rag.doc_status.get_doc_by_file_path.return_value = None
        mock_rag.doc_status.get_by_id.return_value = None

        texts = [f'Document {i} content.' for i in range(5)]

        response = await client.post(
            '/documents/texts',
            json={'texts': texts},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
