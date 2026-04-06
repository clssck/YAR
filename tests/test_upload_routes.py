"""Tests for upload routes in yar/api/routers/upload_routes.py.

This module tests the S3 document staging endpoints using httpx AsyncClient
and FastAPI's TestClient pattern with mocked S3Client and YAR.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from yar.api.routers.upload_routes import _process_s3_document, create_upload_routes
from yar.base import DocStatus

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_s3_client():
    """Create a mock S3Client."""
    client = MagicMock()
    client.upload_to_staging = AsyncMock()
    client.list_staging = AsyncMock()
    client.get_presigned_url = AsyncMock()
    client.object_exists = AsyncMock()
    client.delete_object = AsyncMock()
    client.get_object = AsyncMock()
    client.move_to_archive = AsyncMock()
    client.get_s3_url = MagicMock()
    return client


@pytest.fixture
def mock_rag():
    """Create a mock YAR instance."""
    rag = MagicMock()
    rag.ainsert = AsyncMock()
    rag.text_chunks = MagicMock()
    rag.text_chunks.update_s3_key_by_doc_id = AsyncMock()
    rag.doc_status = MagicMock()
    rag.doc_status.update_s3_key = AsyncMock()
    rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    rag.doc_status.get_by_id = AsyncMock(return_value=None)
    rag.doc_status.upsert = AsyncMock()
    return rag


@pytest.fixture
def app(mock_rag, mock_s3_client):
    """Create FastAPI app with upload routes."""
    app = FastAPI()
    router = create_upload_routes(
        rag=mock_rag,
        s3_client=mock_s3_client,
        api_key=None,
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


# ============================================================================
# Upload Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestUploadEndpoint:
    """Tests for POST /upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_creates_staging_key(self, client, mock_s3_client):
        """Test that upload creates correct S3 staging key."""
        mock_s3_client.upload_to_staging.return_value = 'staging/default/doc_abc123/test.txt'
        mock_s3_client.get_s3_url.return_value = 's3://bucket/staging/default/doc_abc123/test.txt'

        files = {'file': ('test.txt', b'Hello, World!', 'text/plain')}
        data = {'workspace': 'default', 'doc_id': 'doc_abc123'}

        response = await client.post('/upload', files=files, data=data)

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'uploaded'
        assert data['doc_id'] == 'doc_abc123'
        assert data['s3_key'] == 'staging/default/doc_abc123/test.txt'

        mock_s3_client.upload_to_staging.assert_called_once()
        call_args = mock_s3_client.upload_to_staging.call_args
        assert call_args.kwargs['workspace'] == 'default'
        assert call_args.kwargs['doc_id'] == 'doc_abc123'
        assert call_args.kwargs['content'] == b'Hello, World!'

    @pytest.mark.asyncio
    async def test_upload_auto_generates_doc_id(self, client, mock_s3_client):
        """Test that doc_id is auto-generated if not provided."""
        mock_s3_client.upload_to_staging.return_value = 'staging/default/doc_auto/test.txt'
        mock_s3_client.get_s3_url.return_value = 's3://bucket/staging/default/doc_auto/test.txt'

        files = {'file': ('test.txt', b'Test content', 'text/plain')}
        data = {'workspace': 'default'}

        response = await client.post('/upload', files=files, data=data)

        assert response.status_code == 200
        data = response.json()
        assert data['doc_id'].startswith('doc_')

    @pytest.mark.asyncio
    async def test_upload_empty_file_rejected(self, client, mock_s3_client):
        """Test that empty files are rejected."""
        files = {'file': ('empty.txt', b'', 'text/plain')}

        response = await client.post('/upload', files=files)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_returns_s3_url(self, client, mock_s3_client):
        """Test that upload returns S3 URL."""
        mock_s3_client.upload_to_staging.return_value = 'staging/default/doc_xyz/file.pdf'
        mock_s3_client.get_s3_url.return_value = 's3://mybucket/staging/default/doc_xyz/file.pdf'

        files = {'file': ('file.pdf', b'PDF content', 'application/pdf')}

        response = await client.post('/upload', files=files)

        assert response.status_code == 200
        assert response.json()['s3_url'] == 's3://mybucket/staging/default/doc_xyz/file.pdf'

    @pytest.mark.asyncio
    async def test_upload_handles_s3_error(self, client, mock_s3_client):
        """Test that S3 errors are handled."""
        mock_s3_client.upload_to_staging.side_effect = Exception('S3 connection failed')

        files = {'file': ('test.txt', b'Content', 'text/plain')}

        response = await client.post('/upload', files=files)

        assert response.status_code == 500


# ============================================================================
# List Staged Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestListStagedEndpoint:
    """Tests for GET /upload/staged endpoint."""

    @pytest.mark.asyncio
    async def test_list_staged_returns_documents(self, client, mock_s3_client):
        """Test that list returns staged documents."""
        mock_s3_client.list_staging.return_value = [
            {
                'key': 'staging/default/doc1/file.pdf',
                'size': 1024,
                'last_modified': '2024-01-01T00:00:00Z',
            },
            {
                'key': 'staging/default/doc2/report.docx',
                'size': 2048,
                'last_modified': '2024-01-02T00:00:00Z',
            },
        ]

        response = await client.get('/upload/staged')

        assert response.status_code == 200
        data = response.json()
        assert data['workspace'] == 'default'
        assert data['count'] == 2
        assert len(data['documents']) == 2
        assert data['documents'][0]['key'] == 'staging/default/doc1/file.pdf'

    @pytest.mark.asyncio
    async def test_list_staged_empty(self, client, mock_s3_client):
        """Test that empty staging returns empty list."""
        mock_s3_client.list_staging.return_value = []

        response = await client.get('/upload/staged')

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 0
        assert data['documents'] == []

    @pytest.mark.asyncio
    async def test_list_staged_custom_workspace(self, client, mock_s3_client):
        """Test listing documents in custom workspace."""
        mock_s3_client.list_staging.return_value = []

        response = await client.get('/upload/staged', params={'workspace': 'custom'})

        assert response.status_code == 200
        assert response.json()['workspace'] == 'custom'
        mock_s3_client.list_staging.assert_called_once_with('custom')


# ============================================================================
# Presigned URL Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestPresignedUrlEndpoint:
    """Tests for GET /upload/presigned-url endpoint."""

    @pytest.mark.asyncio
    async def test_presigned_url_returns_url(self, client, mock_s3_client):
        """Test that presigned URL is returned."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_presigned_url.return_value = 'https://s3.example.com/signed-url?token=xyz'

        response = await client.get(
            '/upload/presigned-url',
            params={'s3_key': 'staging/default/doc1/file.pdf'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data['s3_key'] == 'staging/default/doc1/file.pdf'
        assert data['presigned_url'] == 'https://s3.example.com/signed-url?token=xyz'
        assert data['expiry_seconds'] == 3600

    @pytest.mark.asyncio
    async def test_presigned_url_custom_expiry(self, client, mock_s3_client):
        """Test custom expiry time."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_presigned_url.return_value = 'https://signed-url'

        response = await client.get(
            '/upload/presigned-url',
            params={'s3_key': 'staging/default/doc1/file.pdf', 'expiry': 7200},
        )

        assert response.status_code == 200
        assert response.json()['expiry_seconds'] == 7200
        mock_s3_client.get_presigned_url.assert_called_once_with(
            'staging/default/doc1/file.pdf', expiry=7200
        )

    @pytest.mark.asyncio
    async def test_presigned_url_not_found(self, client, mock_s3_client):
        """Test 404 for non-existent object."""
        mock_s3_client.object_exists.return_value = False

        response = await client.get(
            '/upload/presigned-url',
            params={'s3_key': 'staging/default/nonexistent/key'},
        )

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()


# ============================================================================
# Delete Staged Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestDeleteStagedEndpoint:
    """Tests for DELETE /upload/staged/{doc_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_removes_document(self, client, mock_s3_client):
        """Test that delete removes the document."""
        mock_s3_client.list_staging.return_value = [
            {'key': 'staging/default/doc123/file.pdf', 'size': 1024, 'last_modified': '2024-01-01'},
        ]

        response = await client.delete('/upload/staged/doc123')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'deleted'
        assert data['doc_id'] == 'doc123'
        assert data['deleted_count'] == '1'

        mock_s3_client.delete_object.assert_called_once_with('staging/default/doc123/file.pdf')

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client, mock_s3_client):
        """Test 404 when document not found."""
        mock_s3_client.list_staging.return_value = []

        response = await client.delete('/upload/staged/nonexistent')

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_delete_multiple_objects(self, client, mock_s3_client):
        """Test deleting document with multiple S3 objects."""
        mock_s3_client.list_staging.return_value = [
            {'key': 'staging/default/doc456/part1.pdf', 'size': 1024, 'last_modified': '2024-01-01'},
            {'key': 'staging/default/doc456/part2.pdf', 'size': 2048, 'last_modified': '2024-01-01'},
        ]

        response = await client.delete('/upload/staged/doc456')

        assert response.status_code == 200
        assert response.json()['deleted_count'] == '2'
        assert mock_s3_client.delete_object.call_count == 2


# ============================================================================
# Process S3 Endpoint Tests
# ============================================================================


@pytest.mark.offline
class TestProcessS3BackgroundTask:
    """Tests for the background helper used by POST /upload/process."""

    @pytest.mark.asyncio
    async def test_background_helper_passes_track_id_to_ainsert(self, mock_s3_client, mock_rag):
        """Test that the generated track_id is preserved during ingestion."""
        await _process_s3_document(
            rag=mock_rag,
            text_content='Document content here',
            doc_id='doc1',
            track_id='s3_track_123',
            s3_url='s3://bucket/staging/default/doc1/file.txt',
            s3_key='staging/default/doc1/file.txt',
            archive_after_processing=False,
            s3_client=mock_s3_client,
        )

        mock_rag.ainsert.assert_awaited_once_with(
            input='Document content here',
            ids='doc1',
            file_paths='s3://bucket/staging/default/doc1/file.txt',
            track_id='s3_track_123',
        )

    @pytest.mark.asyncio
    async def test_background_helper_records_failed_status_with_track_id(self, mock_s3_client, mock_rag):
        """Test that background failures are stored against the same track_id."""
        mock_rag.ainsert.side_effect = RuntimeError('ingestion exploded')

        await _process_s3_document(
            rag=mock_rag,
            text_content='Document content here',
            doc_id='doc1',
            track_id='s3_track_123',
            s3_url='s3://bucket/staging/default/doc1/file.txt',
            s3_key='staging/default/doc1/file.txt',
            archive_after_processing=False,
            s3_client=mock_s3_client,
        )

        mock_rag.doc_status.get_by_id.assert_awaited_once_with('doc1')
        mock_rag.doc_status.upsert.assert_awaited_once()
        failure_payload = mock_rag.doc_status.upsert.await_args.args[0]['doc1']
        assert failure_payload['status'] == DocStatus.FAILED
        assert failure_payload['track_id'] == 's3_track_123'
        assert failure_payload['error_msg'] == 'ingestion exploded'
        assert failure_payload['file_path'] == 's3://bucket/staging/default/doc1/file.txt'

    @pytest.mark.asyncio
    async def test_background_helper_skips_duplicate_failed_status(self, mock_s3_client, mock_rag):
        """Test that helper does not double-record a failure already stored by ingestion."""
        mock_rag.ainsert.side_effect = RuntimeError('ingestion exploded')
        mock_rag.doc_status.get_by_id.return_value = {
            'status': DocStatus.FAILED,
            'track_id': 's3_track_123',
        }

        await _process_s3_document(
            rag=mock_rag,
            text_content='Document content here',
            doc_id='doc1',
            track_id='s3_track_123',
            s3_url='s3://bucket/staging/default/doc1/file.txt',
            s3_key='staging/default/doc1/file.txt',
            archive_after_processing=False,
            s3_client=mock_s3_client,
        )

        mock_rag.doc_status.upsert.assert_not_awaited()


@pytest.mark.offline
class TestProcessS3Endpoint:
    """Tests for POST /upload/process endpoint."""

    @pytest.mark.asyncio
    async def test_process_returns_generated_track_id(self, client, mock_s3_client, mock_rag):
        """Test that process returns the generated track_id while background work uses it."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Document content here', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/staging/default/doc1/file.txt'

        with patch('yar.api.routers.upload_routes.generate_track_id', return_value='s3_track_123'):
            response = await client.post(
                '/upload/process',
                json={'s3_key': 'staging/default/doc1/file.txt'},
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'processing_started'
        assert data['doc_id'] == 'doc1'
        assert data['s3_key'] == 'staging/default/doc1/file.txt'
        assert data['track_id'] == 's3_track_123'
        mock_rag.ainsert.assert_awaited_once_with(
            input='Document content here',
            ids='doc1',
            file_paths='s3://bucket/staging/default/doc1/file.txt',
            track_id='s3_track_123',
        )
    @pytest.mark.asyncio
    async def test_process_without_archive(self, client, mock_s3_client, mock_rag):
        """Test processing without archiving."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'

        response = await client.post(
            '/upload/process',
            json={
                's3_key': 'staging/default/doc1/file.txt',
                'archive_after_processing': False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'processing_started'
        # archive_key is always None in immediate response (archiving is background)
        assert data['archive_key'] is None

    @pytest.mark.asyncio
    async def test_process_extracts_doc_id_from_key(self, client, mock_s3_client, mock_rag):
        """Test that doc_id is extracted from s3_key."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/workspace1/extracted_doc_id/file.txt'},
        )

        assert response.status_code == 200
        assert response.json()['doc_id'] == 'extracted_doc_id'

    @pytest.mark.asyncio
    async def test_process_not_found(self, client, mock_s3_client, mock_rag):
        """Test 404 when S3 object not found."""
        mock_s3_client.object_exists.return_value = False

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/missing/file.txt'},
        )

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_process_empty_content_rejected(self, client, mock_s3_client, mock_rag):
        """Test that empty content is rejected."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'   \n  ', {'content_type': 'text/plain'})

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/doc1/file.txt'},
        )

        assert response.status_code == 400
        assert 'empty' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_process_binary_content_rejected(self, client, mock_s3_client, mock_rag):
        """Test that binary content that can't be decoded is rejected."""
        mock_s3_client.object_exists.return_value = True
        # Invalid UTF-8 bytes
        mock_s3_client.get_object.return_value = (b'\x80\x81\x82\x83', {'content_type': 'application/pdf'})

        response = await client.post(
            '/upload/process',
            json={'s3_key': 'staging/default/doc1/file.pdf'},
        )

        assert response.status_code == 400
        assert 'binary' in response.json()['detail'].lower()

    @pytest.mark.asyncio
    async def test_process_skip_archive(self, client, mock_s3_client, mock_rag):
        """Test that archiving can be skipped."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'

        response = await client.post(
            '/upload/process',
            json={
                's3_key': 'staging/default/doc1/file.txt',
                'archive_after_processing': False,
            },
        )

        assert response.status_code == 200
        assert response.json()['archive_key'] is None

    @pytest.mark.asyncio
    async def test_process_uses_provided_doc_id(self, client, mock_s3_client, mock_rag):
        """Test that provided doc_id is used."""
        mock_s3_client.object_exists.return_value = True
        mock_s3_client.get_object.return_value = (b'Content', {'content_type': 'text/plain'})
        mock_s3_client.get_s3_url.return_value = 's3://bucket/key'

        response = await client.post(
            '/upload/process',
            json={
                's3_key': 'staging/default/doc1/file.txt',
                'doc_id': 'custom_doc_id',
            },
        )

        assert response.status_code == 200
        assert response.json()['doc_id'] == 'custom_doc_id'
