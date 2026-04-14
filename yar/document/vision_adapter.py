from __future__ import annotations

import asyncio
import base64
import math
import mimetypes
import os
import random
import re
import shlex
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

from yar.llm.openai import create_openai_async_client
from yar.utils import logger

VISION_MODEL_DEFAULT = 'salmon'
PDF_MIME_TYPE = 'application/pdf'
PDFTOPPM_COMMAND_ENV = 'PDFTOPPM_COMMAND'
SOFFICE_COMMAND_ENV = 'SOFFICE_COMMAND'
NO_TEXT_DETECTED_SENTINEL = '[NO_TEXT_DETECTED]'
VISION_MAX_OUTPUT_TOKENS_DEFAULT = 65536
VISION_PAGES_PER_CALL_DEFAULT = 15
CONCURRENCY = 10
RETRY_DELAY_SECONDS = 2.0
VISION_CONCURRENCY_DEFAULT = 4
VISION_MAX_RETRIES = 4
_vision_semaphore: asyncio.Semaphore | None = None


def _get_vision_semaphore() -> asyncio.Semaphore:
    """Return a global semaphore limiting concurrent vision API calls across all documents."""
    global _vision_semaphore
    if _vision_semaphore is None:
        limit = int(os.environ.get('VISION_CONCURRENCY', str(VISION_CONCURRENCY_DEFAULT)))
        _vision_semaphore = asyncio.Semaphore(limit)
        logger.info('Vision extraction concurrency limit: %d', limit)
    return _vision_semaphore


REQUEST_TIMEOUT_SECONDS = 120.0
MAX_TOKENS_PER_PAGE = 4096


def _compute_pages_per_call() -> int:
    env_override = os.getenv('VISION_PAGES_PER_CALL')
    if env_override:
        pages_per_call = max(1, int(env_override))
        max_output_tokens = pages_per_call * MAX_TOKENS_PER_PAGE
    else:
        max_output_tokens = int(
            os.getenv('VISION_MAX_OUTPUT_TOKENS', str(VISION_MAX_OUTPUT_TOKENS_DEFAULT))
        )
        pages_per_call = min(
            max(1, max_output_tokens * 3 // 4 // MAX_TOKENS_PER_PAGE),
            VISION_PAGES_PER_CALL_DEFAULT,
        )
    logger.info(
        'Vision batch sizing: %d pages/call (max_output_tokens=%d)',
        pages_per_call,
        max_output_tokens,
    )
    return pages_per_call

PAGE_SPLIT_RE = re.compile(r'<!--\s*PAGE\s+(\d+)\s*-->')
BATCH_EXTRACTION_PROMPT = (
    'You will see document pages. Extract all text and data from EACH page as clean markdown.\n'
    "Separate each page's content with a <!-- PAGE {N} --> marker where N is the page number.\n"
    'For tables, use HTML <table> tags with proper <thead> and <tbody>.\n'
    'For lists, use markdown lists.\n'
    'For diagrams, timelines, flow charts, and infographics, describe all visible labels, '
    'numeric values, durations, arrows, and stage names as a structured markdown list.\n'
    'Preserve heading hierarchy with # marks.\n'
    'Be thorough and accurate.'
)
VISION_IMAGE_EXTENSIONS: tuple[str, ...] = (
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.bmp',
    '.tif',
    '.tiff',
    '.webp',
)
VISION_OFFICE_EXTENSIONS: tuple[str, ...] = (
    '.doc',
    '.docx',
    '.odt',
    '.rtf',
    '.ppt',
    '.pptx',
    '.ppsx',
    '.odp',
)
VISION_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({'.pdf', *VISION_IMAGE_EXTENSIONS, *VISION_OFFICE_EXTENSIONS})
VISION_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        'image/png',
        'image/jpeg',
        'image/jpg',
        'image/gif',
        'image/bmp',
        'image/tiff',
        'image/webp',
    }
)
VISION_OFFICE_MIME_TYPES: frozenset[str] = frozenset(
    {
        'application/msword',
        'application/rtf',
        'application/vnd.ms-powerpoint',
        'application/vnd.oasis.opendocument.presentation',
        'application/vnd.oasis.opendocument.text',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.openxmlformats-officedocument.presentationml.slideshow',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/rtf',
    }
)
_OFFICE_EXTENSION_BY_MIME_TYPE: dict[str, str] = {
    'application/msword': '.doc',
    'application/rtf': '.rtf',
    'application/vnd.ms-powerpoint': '.ppt',
    'application/vnd.oasis.opendocument.presentation': '.odp',
    'application/vnd.oasis.opendocument.text': '.odt',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.openxmlformats-officedocument.presentationml.slideshow': '.ppsx',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'text/rtf': '.rtf',
}


@dataclass(slots=True)
class RenderedPage:
    page_number: int
    total_pages: int
    media_type: str
    image_bytes: bytes


@dataclass(slots=True)
class ExtractionWarning:
    source: str
    message: str


@dataclass(slots=True)
class PageResult:
    page_number: int
    content: str
    warning: ExtractionWarning | None = None


@dataclass(slots=True)
class VisionExtractionResult:
    content: str
    page_count: int = 0
    warnings: list[ExtractionWarning] = field(default_factory=list)
    pre_chunks: list[dict[str, Any]] | None = None
    tables: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionExtractionError(RuntimeError):
    """Raised when document vision extraction cannot produce truthful output."""


def is_vision_document(*, filename: str | None = None, mime_type: str | None = None) -> bool:
    """Return True when the document should be handled by the vision extractor."""
    return (
        _is_pdf_document(filename=filename, mime_type=mime_type)
        or _is_office_document(filename=filename, mime_type=mime_type)
        or _is_image_document(filename=filename, mime_type=mime_type)
    )


def _content_with_context(chunk: Any) -> str:
    """Prepend heading hierarchy to chunk content for richer embeddings.

    Without this, a chunk like ``### IMPACTS: 1. Wrong logo ...`` has no
    semantic link to its parent section (e.g. *Topic 5: financial flow*).
    Prepending the breadcrumb ensures the embedding captures the full
    context.
    """
    if chunk.heading_context:
        return f'{chunk.heading_context}\n\n{chunk.content}'
    return chunk.content


async def extract_document_with_vision(
    document_bytes: bytes,
    *,
    filename: str | None = None,
    mime_type: str | None = None,
    model: str = VISION_MODEL_DEFAULT,
    base_url: str | None = None,
    api_key: str | None = None,
    pdf_password: str | None = None,
) -> VisionExtractionResult:
    """Extract document text by sending page images to an OpenAI-compatible vision model."""
    from yar.document.semantic_chunker import chunk_markdown

    is_pdf_document = _is_pdf_document(filename=filename, mime_type=mime_type)
    is_office_document = _is_office_document(filename=filename, mime_type=mime_type)

    if is_pdf_document or is_office_document:
        pdf_bytes = document_bytes
        if is_office_document:
            pdf_bytes = await asyncio.to_thread(
                partial(
                    _convert_office_document_to_pdf,
                    document_bytes,
                    filename=filename,
                    mime_type=mime_type,
                )
            )
        page_results, warnings, total_pages = await _extract_pdf_streaming(
            pdf_bytes,
            model=model,
            base_url=base_url,
            api_key=api_key,
            pdf_password=pdf_password,
            filename=filename,
        )
    else:
        pages = await asyncio.to_thread(
            partial(
                _load_document_pages,
                document_bytes,
                filename=filename,
                mime_type=mime_type,
                pdf_password=pdf_password,
            )
        )
        if not pages:
            raise VisionExtractionError('Document produced no renderable pages for vision extraction')

        total_pages = len(pages)
        logger.info(
            'Vision extraction starting for %s: %s pages via %s',
            filename or 'document',
            total_pages,
            model,
        )

        client = create_openai_async_client(api_key=api_key, base_url=base_url)
        try:
            page_results, warnings = await _process_all_pages(client, model, pages)
        finally:
            await client.close()

    content, collected_warnings = stitch_extracted_pages(page_results, warnings)
    if not content:
        raise VisionExtractionError('Vision extraction found no extractable text in the document')

    if collected_warnings:
        logger.warning(
            'Vision extraction completed with degraded quality for %s: %s',
            filename or 'document',
            [warning.message for warning in collected_warnings],
        )

    # Run page-aware semantic chunking on the extracted markdown.
    # The resulting pre_chunks bypass Kreuzberg's generic chunker in the pipeline.
    chunks = chunk_markdown(content)
    pre_chunks: list[dict[str, Any]] | None = None
    if chunks:
        pre_chunks = [
            {
                'tokens': math.ceil(len(_content_with_context(chunk)) / 4),
                'content': _content_with_context(chunk),
                'chunk_order_index': chunk.chunk_index,
                'page_number': chunk.page_number,
                'heading_context': chunk.heading_context,
            }
            for chunk in chunks
        ]
        logger.info(
            'Vision extraction produced %s semantic chunks for %s',
            len(pre_chunks),
            filename or 'document',
        )

    return VisionExtractionResult(
        content=content,
        page_count=total_pages,
        warnings=collected_warnings,
        pre_chunks=pre_chunks,
        metadata={
            'extractor': 'vision',
            'model': model,
            'page_count': total_pages,
            'warning_count': len(collected_warnings),
            'chunk_count': len(pre_chunks) if pre_chunks else 0,
        },
    )


async def _extract_pdf_streaming(
    pdf_bytes: bytes,
    *,
    model: str,
    base_url: str | None,
    api_key: str | None,
    pdf_password: str | None,
    filename: str | None,
) -> tuple[list[PageResult], list[ExtractionWarning], int]:
    with tempfile.TemporaryDirectory(prefix='yar-vision-pdf-stream-') as tmp_dir:
        pdf_path = Path(tmp_dir) / 'input.pdf'
        pdf_path.write_bytes(pdf_bytes)

        total_pages = _get_pdf_page_count(pdf_path, pdf_password=pdf_password)
        client = create_openai_async_client(api_key=api_key, base_url=base_url)
        try:
            if total_pages == 0:
                pages = await asyncio.to_thread(partial(_render_pdf_pages, pdf_bytes, pdf_password=pdf_password))
                total_pages = len(pages)
                logger.info(
                    'Vision extraction starting for %s: %s pages via %s',
                    filename or 'document',
                    total_pages,
                    model,
                )
                page_results, warnings = await _process_all_pages(client, model, pages)
                return page_results, warnings, total_pages

            pages_per_call = _compute_pages_per_call()
            wave_size = pages_per_call * CONCURRENCY
            logger.info(
                'Vision extraction starting for %s: %s pages via %s',
                filename or 'document',
                total_pages,
                model,
            )
            logger.info(
                'Vision streaming extraction: %d pages, wave_size=%d (%d pages/call × %d concurrent)',
                total_pages,
                wave_size,
                pages_per_call,
                CONCURRENCY,
            )

            all_results: list[PageResult] = []
            all_warnings: list[ExtractionWarning] = []
            for wave_start in range(1, total_pages + 1, wave_size):
                wave_end = min(wave_start + wave_size - 1, total_pages)
                wave_pages = await asyncio.to_thread(
                    partial(
                        _render_pdf_page_range,
                        pdf_path,
                        wave_start,
                        wave_end,
                        total_pages,
                        pdf_password=pdf_password,
                    )
                )
                wave_results, wave_warnings = await _process_all_pages(client, model, wave_pages)
                all_results.extend(wave_results)
                all_warnings.extend(wave_warnings)

            all_results.sort(key=lambda result: result.page_number)
            return all_results, all_warnings, total_pages
        finally:
            await client.close()


def split_batch_response(raw: str, expected_pages: list[int]) -> tuple[list[PageResult], str]:
    if not expected_pages:
        return [], 'none'

    parts = PAGE_SPLIT_RE.split(raw)
    if len(parts) == 1:
        content = raw.strip()
        if content == NO_TEXT_DETECTED_SENTINEL:
            content = ''
        page_results = [PageResult(page_number=expected_pages[0], content=content)]
        page_results.extend(PageResult(page_number=page_number, content='') for page_number in expected_pages[1:])
        return page_results, 'none'

    content_by_page = dict.fromkeys(expected_pages, '')
    matched_markers: set[int] = set()
    duplicate_marker_found = False
    unexpected_marker_found = False
    unmatched_sections: list[str] = []

    preamble = parts[0].strip()
    if preamble:
        first_page = expected_pages[0]
        content_by_page[first_page] = preamble

    for index in range(1, len(parts), 2):
        marker_page = int(parts[index])
        section_text = parts[index + 1].strip()
        if section_text == NO_TEXT_DETECTED_SENTINEL:
            section_text = ''
        if marker_page in content_by_page:
            if marker_page in matched_markers:
                duplicate_marker_found = True
            matched_markers.add(marker_page)
            content_by_page[marker_page] = _merge_page_content(content_by_page[marker_page], section_text)
        else:
            unexpected_marker_found = True
            unmatched_sections.append(section_text)

    remaining_pages = [page_number for page_number in expected_pages if page_number not in matched_markers]
    if remaining_pages and unmatched_sections:
        last_index = len(remaining_pages) - 1
        for index, section_text in enumerate(unmatched_sections):
            target_page = remaining_pages[min(index, last_index)]
            content_by_page[target_page] = _merge_page_content(content_by_page[target_page], section_text)

    marker_mode = 'full'
    if len(matched_markers) != len(expected_pages) or duplicate_marker_found or unexpected_marker_found:
        marker_mode = 'partial'

    page_results = [
        PageResult(page_number=page_number, content=content_by_page[page_number].strip())
        for page_number in expected_pages
    ]
    return page_results, marker_mode


def should_split_batch(
    expected_page_count: int,
    marker_mode: str,
    finish_reason: str | None,
) -> bool:
    return expected_page_count > 1 and (finish_reason == 'length' or marker_mode != 'full')


async def _extract_batch(
    client: Any,
    model: str,
    pages: list[RenderedPage],
) -> tuple[list[PageResult], list[ExtractionWarning], bool]:
    """Send a page batch to the vision model, gated by a global concurrency semaphore."""
    sem = _get_vision_semaphore()
    async with sem:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=cast(Any, _build_batch_vision_messages(pages)),
                temperature=0,
                max_tokens=MAX_TOKENS_PER_PAGE * len(pages),
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    raw_text = _extract_response_text(response).strip()

    choices = getattr(response, 'choices', None) or []
    finish_reason = getattr(choices[0], 'finish_reason', None) if choices else None
    if not raw_text:
        empty_results = [PageResult(page_number=page.page_number, content='') for page in pages]
        warnings = [
            ExtractionWarning(
                source='vision_empty_response',
                message=f'Vision model returned empty content for page {page.page_number}.',
            )
            for page in pages
        ]
        return empty_results, warnings, False

    page_results, marker_mode = split_batch_response(
        raw_text,
        [page.page_number for page in pages],
    )
    should_split = should_split_batch(len(pages), marker_mode, finish_reason)
    if len(pages) == 1 and finish_reason == 'length':
        should_split = True

    return page_results, [], should_split


async def _extract_batch_with_retry(
    client: Any,
    model: str,
    pages: list[RenderedPage],
) -> tuple[list[PageResult], list[ExtractionWarning], bool]:
    """Try up to VISION_MAX_RETRIES times with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(VISION_MAX_RETRIES):
        try:
            return await _extract_batch(client, model, pages)
        except Exception as exc:
            last_exc = exc
            if attempt < VISION_MAX_RETRIES - 1:
                delay = RETRY_DELAY_SECONDS * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    'Vision extraction retrying batch %s (attempt %d/%d, %.1fs backoff) after error: %s',
                    _page_range_label(pages),
                    attempt + 2,
                    VISION_MAX_RETRIES,
                    delay,
                    _format_exception_message(exc),
                )
                await asyncio.sleep(delay)

    detail = _format_exception_message(last_exc)
    logger.warning(
        'Vision extraction failed batch %s after %d attempts: %s',
        _page_range_label(pages),
        VISION_MAX_RETRIES,
        detail,
    )
    warnings = [
        ExtractionWarning(
            source='vision_batch_failure',
            message=f'Vision extraction failed for page {page.page_number}: {detail}',
        )
        for page in pages
    ]
    page_results = [PageResult(page_number=page.page_number, content='') for page in pages]
    return page_results, warnings, False


async def _extract_batch_adaptively(
    client: Any,
    model: str,
    pages: list[RenderedPage],
) -> tuple[list[PageResult], list[ExtractionWarning]]:
    page_results, warnings, should_split = await _extract_batch_with_retry(client, model, pages)
    if not should_split:
        return page_results, warnings

    if len(pages) == 1:
        page_number = pages[0].page_number
        warnings.append(
            ExtractionWarning(
                source='vision_truncation',
                message=f'Vision extraction may be truncated for page {page_number}.',
            )
        )
        return page_results, warnings

    midpoint = math.ceil(len(pages) / 2)
    logger.info(
        'Vision extraction adaptively splitting batch %s into %s and %s',
        _page_range_label(pages),
        _page_range_label(pages[:midpoint]),
        _page_range_label(pages[midpoint:]),
    )
    left_result, right_result = await asyncio.gather(
        _extract_batch_adaptively(client, model, pages[:midpoint]),
        _extract_batch_adaptively(client, model, pages[midpoint:]),
    )
    left_pages, left_warnings = left_result
    right_pages, right_warnings = right_result
    return left_pages + right_pages, warnings + left_warnings + right_warnings


async def _process_all_pages(
    client: Any,
    model: str,
    pages: list[RenderedPage],
) -> tuple[list[PageResult], list[ExtractionWarning]]:
    if not pages:
        return [], []

    pages_per_call = _compute_pages_per_call()
    batches = [pages[index : index + pages_per_call] for index in range(0, len(pages), pages_per_call)]
    logger.info(
        'Vision extraction: %d pages in %d batches (%d pages/call)',
        len(pages),
        len(batches),
        pages_per_call,
    )
    all_results: list[PageResult] = []
    all_warnings: list[ExtractionWarning] = []

    for index in range(0, len(batches), CONCURRENCY):
        batch_group = batches[index : index + CONCURRENCY]
        batch_outputs = await asyncio.gather(
            *[_extract_batch_adaptively(client, model, batch) for batch in batch_group]
        )
        for batch_results, batch_warnings in batch_outputs:
            all_results.extend(batch_results)
            all_warnings.extend(batch_warnings)

    all_results.sort(key=lambda result: result.page_number)
    return all_results, all_warnings


def _has_usable_content(content: str) -> bool:
    """Check whether extracted page content contains any semantic text.

    Strips all HTML/XML artifacts (page markers, comments, processing
    instructions, DOCTYPE, CDATA sections, markup declarations, tags,
    entities) and punctuation/symbols/whitespace/control characters,
    then returns True if anything remains.
    """
    normalized = PAGE_SPLIT_RE.sub(' ', content)
    normalized = re.sub(r'<!--[\s\S]*?-->', ' ', normalized)
    normalized = re.sub(r'<\?[\s\S]*?\?>', ' ', normalized)
    normalized = re.sub(r'<!DOCTYPE[\s\S]*?>', ' ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'<!\[CDATA\[([\s\S]*?)\]\]>', r'\1', normalized)
    normalized = re.sub(r'<!(?!--|\[CDATA\[|DOCTYPE\b)[\s\S]*?>', ' ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</?[\w:-]+\b[^>]*>', ' ', normalized)
    normalized = re.sub(r'&(?:#\d+|#x[\da-f]+|[\w][\w:-]*);', ' ', normalized, flags=re.IGNORECASE)
    # Semantic content = any letter or digit (Unicode categories L, N)
    return any(unicodedata.category(ch)[0] in ('L', 'N') for ch in normalized)


def _summarize_extraction_quality(
    results: list[PageResult],
    warnings: list[ExtractionWarning],
) -> ExtractionWarning | None:
    empty_pages = [result.page_number for result in results if not _has_usable_content(result.content)]
    truncation_count = sum(1 for warning in warnings if warning.source == 'vision_truncation')
    if not empty_pages and not truncation_count:
        return None

    summary_parts: list[str] = []
    if empty_pages:
        summary_parts.append(
            'Pages with no usable extracted content: ' + ', '.join(str(page_number) for page_number in empty_pages)
        )
    if truncation_count:
        summary_parts.append(f'Pages flagged as potentially truncated: {truncation_count}')

    return ExtractionWarning(source='vision_quality', message='; '.join(summary_parts))


def stitch_extracted_pages(
    results: list[PageResult],
    extra_warnings: list[ExtractionWarning] | None = None,
) -> tuple[str, list[ExtractionWarning]]:
    ordered_results = sorted(results, key=lambda result: result.page_number)
    warnings = list(extra_warnings or [])
    warnings.extend(result.warning for result in ordered_results if result.warning is not None)

    quality_warning = _summarize_extraction_quality(ordered_results, warnings)
    if quality_warning is not None:
        warnings.append(quality_warning)

    stitched_pages = [
        f'<!-- PAGE {result.page_number} -->\n\n{result.content.strip()}'
        for result in ordered_results
        if _has_usable_content(result.content)
    ]
    return '\n\n'.join(stitched_pages), warnings


def _load_document_pages(
    document_bytes: bytes,
    *,
    filename: str | None,
    mime_type: str | None,
    pdf_password: str | None,
) -> list[RenderedPage]:
    if _is_pdf_document(filename=filename, mime_type=mime_type):
        return _render_pdf_pages(document_bytes, pdf_password=pdf_password)
    if _is_office_document(filename=filename, mime_type=mime_type):
        converted_pdf = _convert_office_document_to_pdf(
            document_bytes,
            filename=filename,
            mime_type=mime_type,
        )
        return _render_pdf_pages(converted_pdf)
    if _is_image_document(filename=filename, mime_type=mime_type):
        return [
            RenderedPage(
                page_number=1,
                total_pages=1,
                media_type=_normalize_image_media_type(filename=filename, mime_type=mime_type),
                image_bytes=document_bytes,
            )
        ]
    raise VisionExtractionError(
        f'Vision extraction supports PDFs, office documents, and images only; got filename={filename!r}, mime_type={mime_type!r}'
    )


def _get_pdf_page_count(pdf_path: Path, pdf_password: str | None = None) -> int:
    command = ['pdfinfo']
    if pdf_password:
        command.extend(['-upw', pdf_password])
    command.append(str(pdf_path))

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0

    if completed.returncode != 0:
        return 0

    match = re.search(r'^Pages:\s+(\d+)\s*$', completed.stdout, flags=re.MULTILINE)
    if match is None:
        return 0

    page_count = int(match.group(1))
    return page_count if page_count > 0 else 0


def _render_pdf_page_range(
    pdf_path: Path,
    first_page: int,
    last_page: int,
    total_pages: int,
    *,
    pdf_password: str | None = None,
) -> list[RenderedPage]:
    pdftoppm_command = os.environ.get(PDFTOPPM_COMMAND_ENV, 'pdftoppm').strip() or 'pdftoppm'

    with tempfile.TemporaryDirectory(prefix='yar-vision-pdf-range-') as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_prefix = tmp_path / 'page'

        command = [
            *shlex.split(pdftoppm_command),
            '-jpeg',
            '-jpegopt',
            'quality=80',
            '-scale-to',
            '1500',
            '-f',
            str(first_page),
            '-l',
            str(last_page),
        ]
        if pdf_password:
            command.extend(['-upw', pdf_password])
        command.extend([str(pdf_path), str(output_prefix)])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise VisionExtractionError(f'PDF vision extraction requires {pdftoppm_command!r} to be installed') from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or f'exit status {completed.returncode}'
            raise VisionExtractionError(f'Failed to rasterize PDF for vision extraction: {stderr}')

        numbered_page_paths = sorted(
            (
                (int(page_path.stem.rsplit('-', maxsplit=1)[-1]), page_path)
                for page_path in tmp_path.glob('page-*.jpg')
            ),
            key=lambda item: item[0],
        )
        if not numbered_page_paths:
            raise VisionExtractionError('PDF produced no renderable pages')

        return [
            RenderedPage(
                page_number=page_number,
                total_pages=total_pages,
                media_type='image/jpeg',
                image_bytes=page_path.read_bytes(),
            )
            for page_number, page_path in numbered_page_paths
        ]


def _render_pdf_pages(document_bytes: bytes, *, pdf_password: str | None = None) -> list[RenderedPage]:
    pdftoppm_command = os.environ.get(PDFTOPPM_COMMAND_ENV, 'pdftoppm').strip() or 'pdftoppm'

    with tempfile.TemporaryDirectory(prefix='yar-vision-pdf-') as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / 'input.pdf'
        output_prefix = tmp_path / 'page'
        input_path.write_bytes(document_bytes)

        command = [
            *shlex.split(pdftoppm_command),
            '-jpeg',
            '-jpegopt',
            'quality=80',
            '-scale-to',
            '1500',
        ]
        if pdf_password:
            command.extend(['-upw', pdf_password])
        command.extend([str(input_path), str(output_prefix)])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise VisionExtractionError(f'PDF vision extraction requires {pdftoppm_command!r} to be installed') from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or f'exit status {completed.returncode}'
            raise VisionExtractionError(f'Failed to rasterize PDF for vision extraction: {stderr}')

        numbered_page_paths = sorted(
            (
                (int(page_path.stem.rsplit('-', maxsplit=1)[-1]), page_path)
                for page_path in tmp_path.glob('page-*.jpg')
            ),
            key=lambda item: item[0],
        )
        if not numbered_page_paths:
            raise VisionExtractionError('PDF produced no renderable pages')

        total_pages = len(numbered_page_paths)
        return [
            RenderedPage(
                page_number=page_number,
                total_pages=total_pages,
                media_type='image/jpeg',
                image_bytes=page_path.read_bytes(),
            )
            for page_number, page_path in numbered_page_paths
        ]


def _convert_office_document_to_pdf(
    document_bytes: bytes,
    *,
    filename: str | None,
    mime_type: str | None,
) -> bytes:
    soffice_command = os.environ.get(SOFFICE_COMMAND_ENV, 'soffice').strip() or 'soffice'

    with tempfile.TemporaryDirectory(prefix='yar-vision-office-') as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_extension = _office_input_extension(filename=filename, mime_type=mime_type)
        input_path = tmp_path / f'input{input_extension}'
        input_path.write_bytes(document_bytes)

        command = [
            *shlex.split(soffice_command),
            '--headless',
            f'-env:UserInstallation=file://{tmp_path}/user',
            '--convert-to',
            'pdf',
            '--outdir',
            str(tmp_path),
            str(input_path),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise VisionExtractionError(
                f'Office document vision extraction requires {soffice_command!r} to be installed'
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or f'exit status {completed.returncode}'
            raise VisionExtractionError(f'Failed to convert office document to PDF for vision extraction: {stderr}')

        pdf_path = tmp_path / f'{input_path.stem}.pdf'
        if not pdf_path.exists():
            pdf_paths = sorted(tmp_path.glob('*.pdf'))
            if len(pdf_paths) == 1:
                pdf_path = pdf_paths[0]
            else:
                raise VisionExtractionError('Office document conversion reported success but produced no PDF output')

        return pdf_path.read_bytes()


def _build_batch_vision_messages(pages: list[RenderedPage]) -> list[dict[str, Any]]:
    page_numbers = [page.page_number for page in pages]
    expected_markers = ', '.join(f'<!-- PAGE {page_number} -->' for page_number in page_numbers)
    prompt = (
        f'{BATCH_EXTRACTION_PROMPT}\n\n'
        f'This batch covers {_page_range_label(pages)} of {pages[0].total_pages}.\n'
        f'Return exactly one section for each requested page marker: {expected_markers}.\n'
        f'If a page has no readable text, return the marker followed by {NO_TEXT_DETECTED_SENTINEL}.'
    )
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
    content.extend(
        {
            'type': 'image_url',
            'image_url': {'url': _as_data_url(page.image_bytes, page.media_type)},
        }
        for page in pages
    )
    return [{'role': 'user', 'content': content}]


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, 'choices', None)
    if not choices:
        raise VisionExtractionError('Vision model returned no choices')
    message = getattr(choices[0], 'message', None)
    content = getattr(message, 'content', None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get('type')
                text = part.get('text')
            else:
                part_type = getattr(part, 'type', None)
                text = getattr(part, 'text', None)
            if part_type in (None, 'text', 'output_text') and isinstance(text, str) and text.strip():
                parts.append(text)
        return '\n'.join(parts)
    raise VisionExtractionError('Vision model returned no textual content')


def _normalize_image_media_type(*, filename: str | None, mime_type: str | None) -> str:
    normalized_mime = _normalize_mime_type(mime_type)
    if normalized_mime in VISION_IMAGE_MIME_TYPES:
        return normalized_mime
    guessed_mime, _ = mimetypes.guess_type(filename or '')
    normalized_guess = _normalize_mime_type(guessed_mime)
    if normalized_guess in VISION_IMAGE_MIME_TYPES:
        return normalized_guess
    ext = Path(filename or '').suffix.lower()
    mapping = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff',
        '.webp': 'image/webp',
    }
    if ext in mapping:
        return mapping[ext]
    raise VisionExtractionError(f'Unsupported image MIME type for vision extraction: {mime_type!r}')


def _is_pdf_document(*, filename: str | None, mime_type: str | None) -> bool:
    return Path(filename or '').suffix.lower() == '.pdf' or _normalize_mime_type(mime_type) == PDF_MIME_TYPE


def _is_image_document(*, filename: str | None, mime_type: str | None) -> bool:
    ext = Path(filename or '').suffix.lower()
    normalized_mime = _normalize_mime_type(mime_type)
    return ext in VISION_IMAGE_EXTENSIONS or normalized_mime in VISION_IMAGE_MIME_TYPES


def _is_office_document(*, filename: str | None, mime_type: str | None) -> bool:
    ext = Path(filename or '').suffix.lower()
    normalized_mime = _normalize_mime_type(mime_type)
    return ext in VISION_OFFICE_EXTENSIONS or normalized_mime in VISION_OFFICE_MIME_TYPES


def _office_input_extension(*, filename: str | None, mime_type: str | None) -> str:
    ext = Path(filename or '').suffix.lower()
    if ext in VISION_OFFICE_EXTENSIONS:
        return ext

    normalized_mime = _normalize_mime_type(mime_type)
    if normalized_mime in _OFFICE_EXTENSION_BY_MIME_TYPE:
        return _OFFICE_EXTENSION_BY_MIME_TYPE[normalized_mime]

    guessed_mime, _ = mimetypes.guess_type(filename or '')
    normalized_guess = _normalize_mime_type(guessed_mime)
    if normalized_guess in _OFFICE_EXTENSION_BY_MIME_TYPE:
        return _OFFICE_EXTENSION_BY_MIME_TYPE[normalized_guess]

    raise VisionExtractionError(f'Unsupported office MIME type for vision extraction: {mime_type!r}')


def _normalize_mime_type(mime_type: str | None) -> str:
    return (mime_type or '').split(';', 1)[0].strip().lower()


def _as_data_url(image_bytes: bytes, media_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode('ascii')
    return f'data:{media_type};base64,{encoded}'


def _format_exception_message(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _merge_page_content(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if not addition:
        return existing
    return f'{existing}\n\n{addition}'


def _page_range_label(pages: list[RenderedPage]) -> str:
    if len(pages) == 1:
        return f'page {pages[0].page_number}'
    return f'pages {pages[0].page_number}-{pages[-1].page_number}'


__all__ = [
    'NO_TEXT_DETECTED_SENTINEL',
    'PDF_MIME_TYPE',
    'SOFFICE_COMMAND_ENV',
    'VISION_IMAGE_EXTENSIONS',
    'VISION_IMAGE_MIME_TYPES',
    'VISION_MODEL_DEFAULT',
    'VISION_OFFICE_EXTENSIONS',
    'VISION_OFFICE_MIME_TYPES',
    'VISION_SUPPORTED_EXTENSIONS',
    'ExtractionWarning',
    'PageResult',
    'RenderedPage',
    'VisionExtractionError',
    'VisionExtractionResult',
    'extract_document_with_vision',
    'is_vision_document',
    'should_split_batch',
    'split_batch_response',
    'stitch_extracted_pages',
]
