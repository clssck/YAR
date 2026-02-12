"""Tests for upload size validation in yar/api/utils_api.py.

This module tests the streaming upload validation function that prevents
memory exhaustion from oversized file uploads.
"""

import io

import pytest
from fastapi import HTTPException

from yar.api.utils_api import validate_text_payload_size, validate_upload_size


class MockUploadFile:
    """Mock FastAPI UploadFile for testing."""

    def __init__(self, content: bytes, size: int | None = None, content_type: str = 'application/octet-stream'):
        self._content = content
        self._stream = io.BytesIO(content)
        self.size = size if size is not None else len(content)
        self.content_type = content_type
        self.filename = 'test_file.bin'

    async def read(self, size: int = -1) -> bytes:
        """Read from the mock file stream."""
        if size == -1:
            return self._stream.read()
        return self._stream.read(size)


@pytest.mark.offline
class TestValidateUploadSize:
    """Test cases for validate_upload_size function."""

    @pytest.mark.asyncio
    async def test_disabled_when_limit_is_none(self):
        """Size limit is disabled when max_size_mb is None."""
        content = b'x' * 100_000_000  # 100MB
        mock_file = MockUploadFile(content)

        result = await validate_upload_size(mock_file, None)

        assert result == content

    @pytest.mark.asyncio
    async def test_disabled_when_limit_is_zero(self):
        """Size limit is disabled when max_size_mb is 0."""
        content = b'x' * 100_000_000  # 100MB
        mock_file = MockUploadFile(content)

        result = await validate_upload_size(mock_file, 0)

        assert result == content

    @pytest.mark.asyncio
    async def test_accepts_file_under_limit(self):
        """File under the size limit is accepted."""
        content = b'x' * 500_000  # 500KB
        mock_file = MockUploadFile(content)

        result = await validate_upload_size(mock_file, 1)  # 1MB limit

        assert result == content

    @pytest.mark.asyncio
    async def test_accepts_file_at_exact_limit(self):
        """File exactly at the size limit is accepted."""
        content = b'x' * (1024 * 1024)  # Exactly 1MB
        mock_file = MockUploadFile(content)

        result = await validate_upload_size(mock_file, 1)  # 1MB limit

        assert result == content

    @pytest.mark.asyncio
    async def test_rejects_file_over_limit_via_header(self):
        """File over limit is rejected immediately via Content-Length header."""
        content = b'x' * 1000  # Small content, but header says 2MB
        mock_file = MockUploadFile(content, size=2 * 1024 * 1024)  # Header says 2MB

        with pytest.raises(HTTPException) as exc_info:
            await validate_upload_size(mock_file, 1)  # 1MB limit

        assert exc_info.value.status_code == 413
        assert 'File too large' in exc_info.value.detail
        assert '2.0MB' in exc_info.value.detail
        assert '1MB' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rejects_file_over_limit_via_streaming(self):
        """File over limit is rejected during streaming (no Content-Length header)."""
        content = b'x' * (2 * 1024 * 1024)  # 2MB content
        mock_file = MockUploadFile(content, size=None)  # No header size

        with pytest.raises(HTTPException) as exc_info:
            await validate_upload_size(mock_file, 1)  # 1MB limit

        assert exc_info.value.status_code == 413
        assert 'File too large' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_error_message_format(self):
        """Error message includes actual size and limit in readable format."""
        content = b'x' * int(2.5 * 1024 * 1024)  # 2.5MB
        mock_file = MockUploadFile(content)

        with pytest.raises(HTTPException) as exc_info:
            await validate_upload_size(mock_file, 1)

        detail = exc_info.value.detail
        # Should show formatted sizes like "2.5MB exceeds limit of 1MB"
        assert 'exceeds limit of' in detail

    @pytest.mark.asyncio
    async def test_streaming_reads_in_chunks(self):
        """Streaming validation reads file in chunks to prevent memory spikes."""
        # 5MB file, 2MB limit - should fail during second chunk
        content = b'x' * (5 * 1024 * 1024)
        mock_file = MockUploadFile(content, size=None)  # No header to force streaming

        with pytest.raises(HTTPException) as exc_info:
            await validate_upload_size(mock_file, 2)

        # Should fail when reading exceeds 2MB
        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_empty_file_passes_validation(self):
        """Empty files pass size validation (emptiness handled elsewhere)."""
        content = b''
        mock_file = MockUploadFile(content)

        result = await validate_upload_size(mock_file, 1)

        assert result == b''

    @pytest.mark.asyncio
    async def test_returns_bytes_for_drop_in_replacement(self):
        """Returns bytes object for drop-in replacement of file.read()."""
        content = b'Hello, World!'
        mock_file = MockUploadFile(content)

        result = await validate_upload_size(mock_file, 1)

        assert isinstance(result, bytes)
        assert result == content


@pytest.mark.offline
class TestValidateTextPayloadSize:
    """Test cases for validate_text_payload_size function."""

    def test_disabled_when_limit_is_none(self):
        validate_text_payload_size(['a' * 10_000], None)

    def test_disabled_when_limit_is_zero(self):
        validate_text_payload_size(['a' * 10_000], 0)

    def test_accepts_payload_under_limit(self):
        validate_text_payload_size(['a' * 1000], 1)

    def test_rejects_payload_over_limit(self):
        oversized = 'a' * (2 * 1024 * 1024)  # 2MB
        with pytest.raises(HTTPException) as exc_info:
            validate_text_payload_size([oversized], 1)

        assert exc_info.value.status_code == 413
        assert 'Text payload too large' in exc_info.value.detail
        assert '1MB' in exc_info.value.detail

    def test_counts_utf8_bytes_not_characters(self):
        # U+4F60 ('你') uses 3 bytes in UTF-8
        text = '你' * 400_000  # ~1.14MB
        with pytest.raises(HTTPException) as exc_info:
            validate_text_payload_size([text], 1)

        assert exc_info.value.status_code == 413
