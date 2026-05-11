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

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from yar.api.routers.document_routes import (
    _canonical_markdown_object_name,
    _extraction_manifest_object_name,
    _processed_markdown_object_name,
    _update_db_with_s3_keys,
    _upload_extraction_artifacts_to_s3,
    _upload_processed_text_to_s3,
    create_document_routes,
    pipeline_process_bytes_with_s3,
)

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
    rag.doc_status.delete = AsyncMock()
    rag.full_docs = MagicMock()
    rag.full_docs.get_by_id = AsyncMock(return_value=None)
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
        router = create_document_routes(rag=mock_rag, doc_manager=mock_doc_manager, api_key=None)
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

    def test_processed_markdown_object_name_uses_source_stem(self):
        assert _processed_markdown_object_name('Q1 Research Deck.pptx') == 'Q1 Research Deck.processed.md'

    def test_canonical_and_manifest_object_names_use_source_stem(self):
        assert _canonical_markdown_object_name('Q1 Research Deck.pptx') == 'Q1 Research Deck.canonical.md'
        assert _extraction_manifest_object_name('Q1 Research Deck.pptx') == 'Q1 Research Deck.extraction.json'

    @pytest.mark.parametrize(
        ('source_filename', 'expected'),
        [
            ('reports/Q1\\Slides\x01.PDF', 'reportsQ1Slides.processed.md'),
            ('\x00/\\..', 'document.processed.md'),
            (None, 'document.processed.md'),
        ],
    )
    def test_processed_markdown_object_name_strips_unsafe_chars_and_falls_back(
        self, source_filename: str | None, expected: str
    ):
        assert _processed_markdown_object_name(source_filename) == expected

    @pytest.mark.asyncio
    async def test_upload_processed_text_to_s3_uses_source_derived_object_name(self):
        s3_client = MagicMock()
        s3_client.upload_object = AsyncMock()

        s3_key = await _upload_processed_text_to_s3(
            s3_client=s3_client,
            workspace='default',
            doc_id='doc_123',
            extracted_text='# Extracted',
            source_filename='Q1 Research Deck.pptx',
        )

        assert s3_key == 'default/doc_123/Q1 Research Deck.processed.md'
        s3_client.upload_object.assert_awaited_once_with(
            key=s3_key,
            data=b'# Extracted',
            content_type='text/markdown; charset=utf-8',
        )

    @pytest.mark.asyncio
    async def test_upload_extraction_artifacts_to_s3_writes_processed_canonical_and_manifest(self):
        s3_client = MagicMock()
        s3_client.upload_object = AsyncMock()

        keys = await _upload_extraction_artifacts_to_s3(
            s3_client=s3_client,
            workspace='default',
            doc_id='doc_123',
            extracted_text='# Processed',
            source_filename='Q1 Research Deck.pptx',
            metadata={
                'canonical_content': '# Canonical',
                'extraction_manifest': {'schema_version': 1, 'quality_report': {'page_count': 1}},
            },
        )

        assert keys == {
            'processed_s3_key': 'default/doc_123/Q1 Research Deck.processed.md',
            'canonical_s3_key': 'default/doc_123/Q1 Research Deck.canonical.md',
            'extraction_manifest_s3_key': 'default/doc_123/Q1 Research Deck.extraction.json',
        }
        uploaded_keys = [call.kwargs['key'] for call in s3_client.upload_object.await_args_list]
        assert uploaded_keys == list(keys.values())
        manifest_call = s3_client.upload_object.await_args_list[2]
        assert manifest_call.kwargs['content_type'] == 'application/json; charset=utf-8'
        assert b'"page_count": 1' in manifest_call.kwargs['data']

    @pytest.mark.asyncio
    async def test_upload_extraction_artifacts_cleans_partial_uploads_on_failure(self):
        s3_client = MagicMock()
        s3_client.upload_object = AsyncMock(side_effect=[None, RuntimeError('canonical failed')])
        s3_client.delete_object = AsyncMock()

        with pytest.raises(RuntimeError, match='canonical failed'):
            await _upload_extraction_artifacts_to_s3(
                s3_client=s3_client,
                workspace='default',
                doc_id='doc_123',
                extracted_text='# Processed',
                source_filename='Q1 Research Deck.pptx',
                metadata={'canonical_content': '# Canonical'},
            )

        s3_client.delete_object.assert_awaited_once_with('default/doc_123/Q1 Research Deck.processed.md')

    @pytest.mark.asyncio
    async def test_update_db_with_s3_keys_keeps_original_chunk_file_path(self):
        """Processed Markdown S3 keys must not replace chunk source identity."""
        rag = MagicMock()
        rag.doc_status = MagicMock()
        rag.doc_status.update_s3_key = AsyncMock()
        rag.text_chunks = MagicMock()
        rag.text_chunks.update_s3_key_by_doc_id = AsyncMock(return_value=2)

        await _update_db_with_s3_keys(
            rag=rag,
            doc_id='doc-1',
            original_s3_key='default/doc-1/Q1 Research Deck.pptx',
            processed_s3_key='default/doc-1/Q1 Research Deck.processed.md',
        )

        rag.doc_status.update_s3_key.assert_not_awaited()
        rag.text_chunks.update_s3_key_by_doc_id.assert_awaited_once_with(
            full_doc_id='doc-1',
            s3_key='default/doc-1/Q1 Research Deck.processed.md',
        )
        assert 'archive_url' not in rag.text_chunks.update_s3_key_by_doc_id.await_args.kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('upload_result', 'upload_error', 'expected_chunk_s3_key'),
        [
            (
                'default/doc_123/Q1 Research Deck.processed.md',
                None,
                'default/doc_123/Q1 Research Deck.processed.md',
            ),
            (
                None,
                RuntimeError('upload failed'),
                'default/doc_123/Q1 Research Deck.pptx',
            ),
        ],
    )
    async def test_pipeline_process_bytes_uses_processed_chunk_key_or_original_fallback(
        self,
        upload_result: str | None,
        upload_error: Exception | None,
        expected_chunk_s3_key: str,
    ):
        from yar.api.routers import document_routes as routes

        rag = MagicMock()
        rag.workspace = 'default'
        rag.chunk_token_size = 1200
        rag.chunk_overlap_token_size = 100
        rag.apipeline_enqueue_documents = AsyncMock()
        rag.apipeline_enqueue_error_documents = AsyncMock()
        rag.apipeline_process_enqueue_documents = AsyncMock()
        rag.doc_status = MagicMock()
        rag.text_chunks = MagicMock()
        rag.text_chunks.update_s3_key_by_doc_id = AsyncMock(return_value=1)

        upload_mock = AsyncMock(return_value=upload_result)
        if upload_error is not None:
            upload_mock.side_effect = upload_error

        with (
            patch.object(
                routes,
                '_dispatch_document_extraction',
                AsyncMock(return_value=routes.DocumentExtractionResult(content='Extracted content')),
            ),
            patch.object(routes, '_upload_processed_text_to_s3', upload_mock),
        ):
            result = await pipeline_process_bytes_with_s3(
                rag=rag,
                file_content=b'%PDF-1.7',
                filename='Q1 Research Deck.pptx',
                mime_type='application/pdf',
                s3_client=MagicMock(),
                s3_doc_id='doc_123',
                s3_original_key='default/doc_123/Q1 Research Deck.pptx',
                track_id='upload_test',
            )

        assert result == upload_result
        rag.apipeline_enqueue_documents.assert_awaited_once()
        enqueue_metadata = rag.apipeline_enqueue_documents.await_args.kwargs['metadata']
        assert enqueue_metadata['s3_key'] == expected_chunk_s3_key
        assert set(enqueue_metadata) >= {'s3_key', 'chunk_token_size'}

    @pytest.mark.asyncio
    async def test_pipeline_process_bytes_uploads_spreadsheet_sidecars_and_uses_processed_key(self):
        from yar.api.routers import document_routes as routes

        rag = MagicMock()
        rag.workspace = 'default'
        rag.chunk_token_size = 1200
        rag.chunk_overlap_token_size = 100
        rag.apipeline_enqueue_documents = AsyncMock()
        rag.apipeline_enqueue_error_documents = AsyncMock()
        rag.apipeline_process_enqueue_documents = AsyncMock()
        rag.doc_status = MagicMock()
        rag.text_chunks = MagicMock()
        rag.text_chunks.update_s3_key_by_doc_id = AsyncMock(return_value=1)

        extraction_metadata = {
            'canonical_content': '# Canonical spreadsheet',
            'extraction_manifest': {'schema_version': 1, 'extractor': 'vision'},
        }
        extraction_result = routes.DocumentExtractionResult(
            content='# Processed spreadsheet',
            extractor='vision',
            metadata=extraction_metadata,
        )
        artifact_keys = {
            'processed_s3_key': 'default/doc_123/budget.processed.md',
            'canonical_s3_key': 'default/doc_123/budget.canonical.md',
            'extraction_manifest_s3_key': 'default/doc_123/budget.extraction.json',
        }

        with (
            patch.object(
                routes,
                '_dispatch_document_extraction',
                AsyncMock(return_value=extraction_result),
            ),
            patch.object(
                routes,
                '_upload_extraction_artifacts_to_s3',
                AsyncMock(return_value=artifact_keys),
            ) as upload_artifacts_mock,
        ):
            result = await pipeline_process_bytes_with_s3(
                rag=rag,
                file_content=b'xlsx-bytes',
                filename='budget.xlsx',
                mime_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                s3_client=MagicMock(),
                s3_doc_id='doc_123',
                s3_original_key='default/doc_123/budget.xlsx',
                track_id='upload_test',
            )

        assert result == 'default/doc_123/budget.processed.md'
        upload_artifacts_mock.assert_awaited_once()
        assert upload_artifacts_mock.await_args.kwargs['metadata'] == extraction_metadata
        enqueue_metadata = rag.apipeline_enqueue_documents.await_args.kwargs['metadata']
        assert enqueue_metadata['s3_key'] == 'default/doc_123/budget.processed.md'
        assert enqueue_metadata['canonical_s3_key'] == 'default/doc_123/budget.canonical.md'
        assert enqueue_metadata['extraction_manifest_s3_key'] == 'default/doc_123/budget.extraction.json'

    @pytest.mark.asyncio
    async def test_pipeline_process_bytes_swallows_shutdown_cancellation(self):
        from yar.api.routers import document_routes as routes

        rag = MagicMock()
        rag.workspace = 'default'
        rag.chunk_token_size = 1200
        rag.chunk_overlap_token_size = 100
        rag.apipeline_enqueue_error_documents = AsyncMock()

        with patch.object(
            routes,
            '_dispatch_document_extraction',
            AsyncMock(side_effect=asyncio.CancelledError),
        ):
            result = await pipeline_process_bytes_with_s3(
                rag=rag,
                file_content=b'%PDF-1.7',
                filename='Q1 Research Deck.pptx',
                mime_type='application/pdf',
                s3_client=MagicMock(),
                s3_doc_id='doc_123',
                s3_original_key='default/doc_123/Q1 Research Deck.pptx',
                track_id='upload_test',
            )

        assert result is None
        rag.apipeline_enqueue_error_documents.assert_not_awaited()

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


@pytest.mark.offline
class TestUploadEndpointS3Keys:
    @pytest.mark.asyncio
    async def test_upload_s3_original_object_uses_sanitized_uploaded_filename(self, mock_rag, mock_doc_manager):
        s3_client = MagicMock()
        s3_client.upload_object = AsyncMock()

        app = FastAPI()
        with (
            patch('yar.api.routers.document_routes.global_args') as mock_global_args,
            patch(
                'yar.api.routers.document_routes.pipeline_process_bytes_with_s3',
                new_callable=AsyncMock,
            ) as process_mock,
        ):
            mock_global_args.max_upload_size_mb = 100
            router = create_document_routes(
                rag=mock_rag,
                doc_manager=mock_doc_manager,
                api_key=None,
                s3_client=s3_client,
            )
            app.include_router(router)

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url='http://test',
            ) as test_client:
                response = await test_client.post(
                    '/documents/upload',
                    files={'file': ('Q1 Research Deck.pdf', b'%PDF-1.7', 'application/pdf')},
                )

        assert response.status_code == 200
        uploaded_key = s3_client.upload_object.await_args.kwargs['key']
        assert uploaded_key.startswith('default/doc_')
        assert uploaded_key.endswith('/Q1 Research Deck.pdf')
        assert '/original.' not in uploaded_key
        process_mock.assert_awaited_once()
        assert process_mock.await_args.args[2] == 'Q1 Research Deck.pdf'
        assert process_mock.await_args.args[6] == uploaded_key


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
        with (
            patch(
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
            ),
            patch(
                'yar.kg.shared_storage.get_namespace_lock',
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            ),
            patch(
                'yar.kg.shared_storage.get_all_update_flags_status',
                new_callable=AsyncMock,
                return_value={},
            ),
        ):
            response = await client.get('/documents/pipeline_status')

        assert response.status_code == 200
        data = response.json()
        assert 'autoscanned' in data
        assert 'busy' in data
        assert isinstance(data['busy'], bool)


@pytest.mark.offline
class TestDeleteDocumentEndpoint:
    """Tests for POST /documents/delete_document endpoint."""

    @pytest.mark.asyncio
    async def test_delete_document_deletes_error_entry_while_pipeline_busy(self, client, mock_rag):
        with (
            patch(
                'yar.kg.shared_storage.get_namespace_data',
                new_callable=AsyncMock,
                return_value={'busy': True},
            ),
            patch(
                'yar.kg.shared_storage.get_namespace_lock',
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            ),
        ):
            response = await client.post('/documents/delete_document', json={'doc_ids': ['error-123']})

        assert response.status_code == 200
        assert response.json() == {
            'status': 'success',
            'message': 'Deleted 1 orphaned document entries.',
            'doc_id': 'error-123',
        }
        mock_rag.doc_status.delete.assert_awaited_once_with(['error-123'])
        mock_rag.full_docs.get_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_document_deletes_failed_entry_without_content_while_pipeline_busy(self, client, mock_rag):
        from yar.base import DocStatus

        mock_rag.doc_status.get_by_id.return_value = {'status': DocStatus.FAILED, 'file_path': 'failed.txt'}
        mock_rag.full_docs.get_by_id.return_value = None

        with (
            patch(
                'yar.kg.shared_storage.get_namespace_data',
                new_callable=AsyncMock,
                return_value={'busy': True},
            ),
            patch(
                'yar.kg.shared_storage.get_namespace_lock',
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            ),
        ):
            response = await client.post('/documents/delete_document', json={'doc_ids': ['failed-123']})

        assert response.status_code == 200
        assert response.json() == {
            'status': 'success',
            'message': 'Deleted 1 orphaned document entries.',
            'doc_id': 'failed-123',
        }
        mock_rag.doc_status.delete.assert_awaited_once_with(['failed-123'])
        mock_rag.full_docs.get_by_id.assert_awaited_once_with('failed-123')

    @pytest.mark.asyncio
    async def test_delete_document_only_blocks_remaining_heavy_docs_when_pipeline_busy(self, client, mock_rag):
        from yar.base import DocStatus

        mock_rag.doc_status.get_by_id.return_value = {'status': DocStatus.FAILED, 'file_path': 'failed.txt'}
        mock_rag.full_docs.get_by_id.return_value = {'content': 'still present'}

        with (
            patch(
                'yar.kg.shared_storage.get_namespace_data',
                new_callable=AsyncMock,
                return_value={'busy': True},
            ),
            patch(
                'yar.kg.shared_storage.get_namespace_lock',
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock()),
            ),
        ):
            response = await client.post(
                '/documents/delete_document',
                json={'doc_ids': ['error-123', 'failed-123']},
            )

        assert response.status_code == 200
        assert response.json() == {
            'status': 'busy',
            'message': 'Cannot delete documents while pipeline is busy',
            'doc_id': 'failed-123',
        }
        mock_rag.doc_status.delete.assert_awaited_once_with(['error-123'])
        mock_rag.full_docs.get_by_id.assert_awaited_once_with('failed-123')


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


@pytest.mark.offline
class TestDocumentExtractionDispatch:
    """Tests for shared document extraction dispatch."""

    def test_document_manager_supports_vision_documents(self, tmp_path):
        from yar.api.routers.document_routes import DocumentManager

        manager = DocumentManager(str(tmp_path))
        assert manager.is_supported_file('scan.png') is True
        assert manager.is_supported_file('slides.docx') is True
        assert manager.is_supported_file('deck.PPSX') is True
        assert manager.is_supported_file('notes.odt') is True
        assert manager.is_supported_file('budget.xlsx') is True
        assert manager.is_supported_file('legacy.xls') is False
        assert manager.is_supported_file('open.ods') is False
        assert manager.is_supported_file('archive.zip') is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('binding_overrides', 'expected_base_url', 'expected_api_key'),
        [
            pytest.param(
                {
                    'vision_binding_host': 'http://vision-host/v1',
                    'vision_binding_api_key': 'vision-key',
                },
                'http://vision-host/v1',
                'vision-key',
                id='vision-binding',
            ),
            pytest.param({}, 'http://llm-host/v1', 'llm-key', id='llm-binding-fallback'),
        ],
    )
    async def test_extract_document_with_vision_uses_expected_binding_credentials(
        self,
        binding_overrides,
        expected_base_url,
        expected_api_key,
    ):
        from yar.api.routers import document_routes as routes

        vision_response = SimpleNamespace(
            content='vision markdown',
            pre_chunks=['chunk'],
            tables=[{'name': 'table'}],
            metadata={'pages': 1},
        )
        global_args = {
            'vision_model': 'salmon',
            'llm_binding_host': 'http://llm-host/v1',
            'llm_binding_api_key': 'llm-key',
            'pdf_decrypt_password': 'secret',
            'chunk_size': 1200,
        }
        global_args.update(binding_overrides)

        tokenizer = object()
        with (
            patch.object(routes, 'global_args', SimpleNamespace(**global_args)),
            patch.object(
                routes,
                'extract_document_with_vision',
                AsyncMock(return_value=vision_response),
            ) as extract_mock,
        ):
            result = await routes._extract_document_with_vision_bytes(
                b'%PDF-1.7',
                filename='report.pdf',
                mime_type='application/pdf',
                tokenizer=tokenizer,
            )

        extract_mock.assert_awaited_once_with(
            b'%PDF-1.7',
            filename='report.pdf',
            mime_type='application/pdf',
            model='salmon',
            base_url=expected_base_url,
            api_key=expected_api_key,
            pdf_password='secret',
            chunk_token_size=1200,
            tokenizer=tokenizer,
        )
        assert result.content == 'vision markdown'
        assert result.pre_chunks == ['chunk']
        assert result.tables == [{'name': 'table'}]
        assert result.extractor == 'vision'
        assert result.metadata == {'pages': 1}

    @pytest.mark.asyncio
    async def test_dispatch_routes_xlsx_to_vision(self):
        from yar.api.routers import document_routes as routes

        vision_result = routes.DocumentExtractionResult(content='spreadsheet markdown', extractor='vision')
        with (
            patch.object(
                routes,
                '_extract_document_with_vision_bytes',
                AsyncMock(return_value=vision_result),
            ) as vision_mock,
            patch.object(routes, '_decode_text_document_bytes') as decode_mock,
        ):
            tokenizer = object()
            result = await routes._dispatch_document_extraction(
                file_content=b'xlsx-bytes',
                filename='budget.XLSX',
                mime_type='application/octet-stream',
                error_prefix='[Bytes Extraction]',
                tokenizer=tokenizer,
            )

        assert result is vision_result
        vision_mock.assert_awaited_once_with(
            b'xlsx-bytes',
            filename='budget.XLSX',
            mime_type='application/octet-stream',
            tokenizer=tokenizer,
        )
        decode_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_routes_pdf_to_vision(self):
        from yar.api.routers import document_routes as routes

        vision_result = routes.DocumentExtractionResult(content='vision markdown', extractor='vision')
        with (
            patch.object(
                routes, '_extract_document_with_vision_bytes', AsyncMock(return_value=vision_result)
            ) as vision_mock,
            patch.object(routes, '_decode_text_document_bytes') as decode_mock,
        ):
            result = await routes._dispatch_document_extraction(
                file_content=b'%PDF-1.7',
                filename='report.pdf',
                mime_type='application/pdf',
                error_prefix='[Bytes Extraction]',
            )

        assert result is vision_result
        vision_mock.assert_awaited_once()
        decode_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_routes_text_to_utf8_decode(self):
        from yar.api.routers import document_routes as routes

        text_result = routes.DocumentExtractionResult(content='plain text', extractor='text')
        with (
            patch.object(routes, '_extract_document_with_vision_bytes', AsyncMock()) as vision_mock,
            patch.object(routes, '_decode_text_document_bytes', return_value=text_result) as decode_mock,
        ):
            result = await routes._dispatch_document_extraction(
                file_content=b'hello',
                filename='notes.txt',
                mime_type='text/plain',
                error_prefix='[Bytes Extraction]',
            )

        assert result is text_result
        decode_mock.assert_called_once_with(
            b'hello',
            filename='notes.txt',
            error_prefix='[Bytes Extraction]',
        )
        vision_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_routes_office_docs_to_vision(self):
        from yar.api.routers import document_routes as routes

        vision_result = routes.DocumentExtractionResult(content='office markdown', extractor='vision')
        with (
            patch.object(
                routes, '_extract_document_with_vision_bytes', AsyncMock(return_value=vision_result)
            ) as vision_mock,
            patch.object(routes, '_decode_text_document_bytes') as decode_mock,
        ):
            result = await routes._dispatch_document_extraction(
                file_content=b'docx-bytes',
                filename='slides.DOCX',
                mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                error_prefix='[Bytes Extraction]',
            )

        assert result is vision_result
        vision_mock.assert_awaited_once_with(
            b'docx-bytes',
            filename='slides.DOCX',
            mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        )
        decode_mock.assert_not_called()
