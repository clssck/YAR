from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import yar.document.vision_adapter as vision
from yar.document.vision_adapter import _get_pdf_page_count, _render_pdf_page_range


def _make_chat_response(content: str, finish_reason: str = 'stop') -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)]
    )


@pytest.mark.offline
class TestVisionAdapter:
    @pytest.mark.asyncio
    async def test_extract_document_with_vision_builds_requests_and_stitches_pages(self):
        pages = [
            vision.RenderedPage(page_number=1, total_pages=2, media_type='image/png', image_bytes=b'page-one'),
            vision.RenderedPage(page_number=2, total_pages=2, media_type='image/png', image_bytes=b'page-two'),
        ]
        # The new batched extraction sends both pages in one call with PAGE markers.
        batched_response = '<!-- PAGE 1 -->\n\n# First page\n\nHello world\n\n<!-- PAGE 2 -->\n\n[NO_TEXT_DETECTED]'
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_chat_response(batched_response),
        )
        client.close = AsyncMock()

        with (
            patch.object(vision.asyncio, 'to_thread', AsyncMock(return_value=pages)) as to_thread_mock,
            patch.object(vision, 'create_openai_async_client', return_value=client) as client_factory,
        ):
            result = await vision.extract_document_with_vision(
                b'%PDF-1.7',
                filename='report.pdf',
                mime_type='application/pdf',
                model='salmon',
                base_url='http://litellm.example/v1',
                api_key='test-key',
            )

        to_thread_mock.assert_awaited_once()
        client_factory.assert_called_once_with(api_key='test-key', base_url='http://litellm.example/v1')
        # Batched: both pages sent in single call
        assert client.chat.completions.create.await_count == 1
        first_call = client.chat.completions.create.await_args_list[0].kwargs
        assert first_call['model'] == 'salmon'
        # Batch message is a single user message (no system message)
        messages = first_call['messages']
        user_content = messages[0]['content']
        assert user_content[0]['type'] == 'text'
        # Two image_url parts (one per page)
        image_parts = [part for part in user_content if part['type'] == 'image_url']
        assert len(image_parts) == 2
        assert image_parts[0]['image_url']['url'].startswith('data:image/png;base64,')
        # Content uses <!-- PAGE N --> markers
        assert '<!-- PAGE 1 -->' in result.content
        assert '# First page' in result.content
        # Page 2 was blank/sentinel so should be filtered out
        assert '<!-- PAGE 2 -->' not in result.content
        # Metadata reflects new format
        assert result.metadata['extractor'] == 'vision'
        assert result.metadata['model'] == 'salmon'
        assert result.metadata['page_count'] == 2
        # Pre-chunks should be populated by the semantic chunker
        assert result.pre_chunks is not None
        assert len(result.pre_chunks) >= 1
        assert result.pre_chunks[0]['content']
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_document_with_vision_rejects_all_blank_pages(self):
        pages = [vision.RenderedPage(page_number=1, total_pages=1, media_type='image/png', image_bytes=b'page-one')]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_make_chat_response(vision.NO_TEXT_DETECTED_SENTINEL))
        client.close = AsyncMock()

        with (
            patch.object(vision.asyncio, 'to_thread', AsyncMock(return_value=pages)),
            patch.object(vision, 'create_openai_async_client', return_value=client),
        ):
            with pytest.raises(vision.VisionExtractionError, match='no extractable text'):
                await vision.extract_document_with_vision(
                    b'fake-image',
                    filename='scan.png',
                    mime_type='image/png',
                )

        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_document_with_vision_surfaces_non_retryable_provider_failure(self):
        class ProviderError(RuntimeError):
            def __init__(self, message: str, *, status_code: int):
                super().__init__(message)
                self.status_code = status_code

        pages = [vision.RenderedPage(page_number=1, total_pages=1, media_type='image/png', image_bytes=b'page-one')]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=ProviderError('Key limit exceeded', status_code=403)
        )
        client.close = AsyncMock()

        with (
            patch.object(vision.asyncio, 'to_thread', AsyncMock(return_value=pages)),
            patch.object(vision, 'create_openai_async_client', return_value=client),
            patch.object(vision.asyncio, 'sleep', AsyncMock()) as sleep_mock,
        ):
            with pytest.raises(vision.VisionExtractionError, match='Key limit exceeded'):
                await vision.extract_document_with_vision(
                    b'fake-image',
                    filename='scan.png',
                    mime_type='image/png',
                )

        assert client.chat.completions.create.await_count == 1
        sleep_mock.assert_not_awaited()
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_document_with_vision_retries_transient_failures_then_succeeds(self):
        pages = [vision.RenderedPage(page_number=1, total_pages=1, media_type='image/png', image_bytes=b'page-one')]
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=[
                RuntimeError('temporary upstream timeout'),
                _make_chat_response('# Page 1\n\nRecovered text'),
            ]
        )
        client.close = AsyncMock()

        with (
            patch.object(vision.asyncio, 'to_thread', AsyncMock(return_value=pages)),
            patch.object(vision, 'create_openai_async_client', return_value=client),
            patch.object(vision.random, 'uniform', return_value=0.0),
            patch.object(vision.asyncio, 'sleep', AsyncMock()) as sleep_mock,
        ):
            result = await vision.extract_document_with_vision(
                b'fake-image',
                filename='scan.png',
                mime_type='image/png',
            )

        assert client.chat.completions.create.await_count == 2
        sleep_mock.assert_awaited_once_with(vision.RETRY_DELAY_SECONDS)
        assert 'Recovered text' in result.content
        assert result.warnings == []
        client.close.assert_awaited_once()

    def test_is_vision_document_recognizes_office_extensions_and_mime_types(self):
        assert vision.is_vision_document(filename='slides.PPTX') is True
        assert vision.is_vision_document(filename='notes.rtf') is True
        assert (
            vision.is_vision_document(
                mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
            is True
        )
        assert vision.is_vision_document(mime_type='application/vnd.ms-powerpoint') is True
        assert vision.is_vision_document(filename='archive.zip', mime_type='application/zip') is False

    def test_load_document_pages_converts_office_docs_to_pdf_before_rendering(self):
        expected_pages = [
            vision.RenderedPage(page_number=1, total_pages=1, media_type='image/jpeg', image_bytes=b'jpg-one')
        ]

        def fake_run(command, **_kwargs):
            _ = _kwargs
            assert command[0] == '/usr/bin/soffice'
            assert '--headless' in command
            assert '-env:UserInstallation=file://' in ' '.join(command)
            convert_idx = command.index('--convert-to')
            assert command[convert_idx + 1] == 'pdf'
            outdir_idx = command.index('--outdir')
            outdir = Path(command[outdir_idx + 1])
            input_path = Path(command[-1])
            assert input_path.suffix == '.docx'
            assert input_path.read_bytes() == b'office-bytes'
            (outdir / f'{input_path.stem}.pdf').write_bytes(b'%PDF-converted')
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        with (
            patch.dict(vision.os.environ, {vision.SOFFICE_COMMAND_ENV: '/usr/bin/soffice'}, clear=False),
            patch.object(vision.subprocess, 'run', side_effect=fake_run) as run_mock,
            patch.object(vision, '_render_pdf_pages', return_value=expected_pages) as render_mock,
        ):
            pages = vision._load_document_pages(
                b'office-bytes',
                filename='slides.DOCX',
                mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                pdf_password='ignored',
            )

        run_mock.assert_called_once()
        render_mock.assert_called_once_with(b'%PDF-converted')
        assert pages is expected_pages

    def test_load_document_pages_raises_truthful_error_when_soffice_is_missing(self):
        with (
            patch.dict(vision.os.environ, {vision.SOFFICE_COMMAND_ENV: '/usr/bin/soffice'}, clear=False),
            patch.object(vision.subprocess, 'run', side_effect=FileNotFoundError),
        ):
            with pytest.raises(
                vision.VisionExtractionError,
                match=r"Office document vision extraction requires '/usr/bin/soffice' to be installed",
            ):
                vision._load_document_pages(
                    b'office-bytes',
                    filename='deck.pptx',
                    mime_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    pdf_password=None,
                )

    def test_load_document_pages_raises_when_office_conversion_produces_no_pdf(self):
        with (
            patch.dict(vision.os.environ, {vision.SOFFICE_COMMAND_ENV: '/usr/bin/soffice'}, clear=False),
            patch.object(vision.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='ok', stderr='')),
        ):
            with pytest.raises(
                vision.VisionExtractionError,
                match='Office document conversion reported success but produced no PDF output',
            ):
                vision._load_document_pages(
                    b'office-bytes',
                    filename='notes.odt',
                    mime_type='application/vnd.oasis.opendocument.text',
                    pdf_password=None,
                )

    def test_render_pdf_pages_uses_pdftoppm_output(self):
        def fake_run(command, **_kwargs):
            _ = _kwargs
            assert command[0] == '/usr/bin/pdftoppm'
            output_prefix = Path(command[-1])
            (output_prefix.parent / 'page-1.jpg').write_bytes(b'jpg-one')
            (output_prefix.parent / 'page-2.jpg').write_bytes(b'jpg-two')
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        with (
            patch.dict(vision.os.environ, {vision.PDFTOPPM_COMMAND_ENV: '/usr/bin/pdftoppm'}, clear=False),
            patch.object(vision.subprocess, 'run', side_effect=fake_run) as run_mock,
        ):
            pages = vision._render_pdf_pages(b'%PDF-1.7')

        run_mock.assert_called_once()
        assert [page.page_number for page in pages] == [1, 2]
        assert all(page.media_type == 'image/jpeg' for page in pages)
        assert [page.image_bytes for page in pages] == [b'jpg-one', b'jpg-two']

    def test_content_with_context_prepends_heading_hierarchy(self):
        """heading_context must be prepended to chunk content for richer embeddings."""
        from types import SimpleNamespace

        chunk_with_ctx = SimpleNamespace(
            content='### IMPACTS:\n\n1. Wrong logo',
            heading_context='Topic 5: financial flow',
        )
        chunk_without_ctx = SimpleNamespace(
            content='# Top level heading\n\nSome body text.',
            heading_context=None,
        )

        result_with = vision._content_with_context(chunk_with_ctx)
        assert result_with.startswith('Topic 5: financial flow')
        assert '### IMPACTS:' in result_with

        result_without = vision._content_with_context(chunk_without_ctx)
        assert result_without == chunk_without_ctx.content

@pytest.mark.offline
class TestStreamingPageRendering:
    def test_get_pdf_page_count_parses_pdfinfo_output(self):
        with patch.object(
            vision.subprocess,
            'run',
            return_value=SimpleNamespace(returncode=0, stdout='Pages:          28\n', stderr=''),
        ):
            assert _get_pdf_page_count(Path('report.pdf')) == 28

    def test_get_pdf_page_count_returns_0_when_pdfinfo_not_found(self):
        with patch.object(vision.subprocess, 'run', side_effect=FileNotFoundError):
            assert _get_pdf_page_count(Path('report.pdf')) == 0

    def test_get_pdf_page_count_returns_0_on_parse_failure(self):
        with patch.object(
            vision.subprocess,
            'run',
            return_value=SimpleNamespace(returncode=0, stdout='Title: Example PDF\n', stderr=''),
        ):
            assert _get_pdf_page_count(Path('report.pdf')) == 0

    def test_render_pdf_page_range_uses_f_and_l_flags(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / 'input.pdf'
        pdf_path.write_bytes(b'%PDF-1.7')
        monkeypatch.setenv(vision.PDFTOPPM_COMMAND_ENV, '/usr/bin/pdftoppm')

        def fake_run(command, **_kwargs):
            _ = _kwargs
            assert command[0] == '/usr/bin/pdftoppm'
            assert command[command.index('-f') + 1] == '2'
            assert command[command.index('-l') + 1] == '3'
            assert Path(command[-2]) == pdf_path
            output_prefix = Path(command[-1])
            (output_prefix.parent / 'page-2.jpg').write_bytes(b'jpg-two')
            (output_prefix.parent / 'page-3.jpg').write_bytes(b'jpg-three')
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        with patch.object(vision.subprocess, 'run', side_effect=fake_run) as run_mock:
            pages = _render_pdf_page_range(pdf_path, 2, 3, total_pages=8)

        run_mock.assert_called_once()
        assert [page.page_number for page in pages] == [2, 3]
        assert [page.total_pages for page in pages] == [8, 8]
        assert all(page.media_type == 'image/jpeg' for page in pages)
        assert [page.image_bytes for page in pages] == [b'jpg-two', b'jpg-three']

    def test_render_pdf_page_range_raises_on_no_output(self, monkeypatch, tmp_path):
        pdf_path = tmp_path / 'input.pdf'
        pdf_path.write_bytes(b'%PDF-1.7')
        monkeypatch.setenv(vision.PDFTOPPM_COMMAND_ENV, '/usr/bin/pdftoppm')

        with patch.object(
            vision.subprocess,
            'run',
            return_value=SimpleNamespace(returncode=0, stdout='', stderr=''),
        ):
            with pytest.raises(vision.VisionExtractionError, match='PDF produced no renderable pages'):
                _render_pdf_page_range(pdf_path, 4, 5, total_pages=9)


@pytest.mark.offline
class TestImageVariantTransforms:
    """Verify the image-variant chain produces structurally valid bypass candidates."""

    def _sample_jpeg(self, width: int = 800, height: int = 1200) -> bytes:
        from io import BytesIO

        from PIL import Image

        img = Image.new('RGB', (width, height), color=(220, 220, 220))
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()

    def _decode(self, image_bytes: bytes):
        from io import BytesIO

        from PIL import Image

        return Image.open(BytesIO(image_bytes))

    def test_chain_contains_lossless_variants_first(self):
        chain = vision.IMAGE_VARIANT_CHAIN
        assert chain[0] == 'rotated-90', 'rotated-90 must lead the chain (lossless, full-content)'
        assert 'bottom-half' in chain[:3], 'bottom-half must be in the early chain (empirically validated)'
        assert chain[-1] == 'quarter-res', 'quarter-res is heaviest-loss; keep it last'
        assert set(chain) == set(vision._IMAGE_VARIANT_TRANSFORMS), 'chain and transform map must agree'

    def test_rotated_90_swaps_dimensions(self):
        original = self._sample_jpeg(width=800, height=1200)
        out = vision._transform_image_rotated_90(original)
        img = self._decode(out)
        assert img.size == (1200, 800), 'rotated-90 must swap width/height'

    def test_bottom_half_keeps_lower_region(self):
        original = self._sample_jpeg(width=800, height=1200)
        out = vision._transform_image_bottom_half(original)
        img = self._decode(out)
        assert img.size == (800, 600), 'bottom-half preserves width and halves height'

    def test_top_half_keeps_upper_region(self):
        original = self._sample_jpeg(width=800, height=1200)
        out = vision._transform_image_top_half(original)
        img = self._decode(out)
        assert img.size == (800, 600), 'top-half preserves width and halves height'

    def test_top_and_bottom_halves_cover_full_height(self):
        """Top + bottom halves together must cover the entire original page height."""
        original = self._sample_jpeg(width=400, height=1000)
        top = self._decode(vision._transform_image_top_half(original))
        bottom = self._decode(vision._transform_image_bottom_half(original))
        assert top.size[1] + bottom.size[1] == 1000, 'halves must partition the original height with no gap or overlap'

    def test_quarter_res_downsamples(self):
        original = self._sample_jpeg(width=800, height=1200)
        out = vision._transform_image_quarter_res(original)
        img = self._decode(out)
        assert img.size == (200, 300), 'quarter-res must produce 1/4 of each dimension'

    def test_blurred_preserves_dimensions(self):
        original = self._sample_jpeg(width=600, height=800)
        out = vision._transform_image_blurred(original)
        img = self._decode(out)
        assert img.size == (600, 800), 'blur is applied in-place; dimensions must not change'
