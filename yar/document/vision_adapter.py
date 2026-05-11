from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import shlex
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from functools import partial
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast

from yar.llm.openai import create_openai_async_client
from yar.utils import logger

VISION_MODEL_DEFAULT = 'salmon'
VISION_PROMPT_VERSION = 'vision-batch-v1'
VISION_EXTRACTOR_VERSION = 'vision-adapter-v1'
PDF_MIME_TYPE = 'application/pdf'
PDFTOPPM_COMMAND_ENV = 'PDFTOPPM_COMMAND'
SOFFICE_COMMAND_ENV = 'SOFFICE_COMMAND'
NO_TEXT_DETECTED_SENTINEL = '[NO_TEXT_DETECTED]'
VISION_MAX_OUTPUT_TOKENS_DEFAULT = 65536
VISION_MAX_CONTEXT_TOKENS_DEFAULT = 200_000
VISION_PAGES_PER_CALL_DEFAULT = 15
RETRY_DELAY_SECONDS = 2.0
VISION_CONCURRENCY_DEFAULT = 4
VISION_INPUT_TOKENS_PER_PAGE_ESTIMATE = 8_000
VISION_PROMPT_TOKENS_ESTIMATE = 4_000
VISION_MAX_RETRIES = 4
TINY_PAGE_CHAR_THRESHOLD = 100
_vision_concurrency_limit: int | None = None
_vision_semaphore: asyncio.Semaphore | None = None
_PAGE_RESULT_CACHE: dict[str, dict[str, Any]] = {}


def _get_vision_concurrency_limit() -> int:
    global _vision_concurrency_limit
    if _vision_concurrency_limit is None:
        _vision_concurrency_limit = max(
            1,
            int(os.environ.get('VISION_CONCURRENCY', str(VISION_CONCURRENCY_DEFAULT))),
        )
    return _vision_concurrency_limit


def _get_vision_semaphore() -> asyncio.Semaphore:
    """Return a global semaphore limiting concurrent vision API calls across all documents."""
    global _vision_semaphore
    if _vision_semaphore is None:
        limit = _get_vision_concurrency_limit()
        _vision_semaphore = asyncio.Semaphore(limit)
        logger.info('Vision extraction concurrency limit: %d', limit)
    return _vision_semaphore


REQUEST_TIMEOUT_SECONDS_DEFAULT = 120.0


def _get_request_timeout() -> float:
    env_override = os.getenv('VISION_REQUEST_TIMEOUT')
    if not env_override:
        return REQUEST_TIMEOUT_SECONDS_DEFAULT
    try:
        value = float(env_override)
    except ValueError:
        logger.warning(
            'Invalid VISION_REQUEST_TIMEOUT=%r, falling back to %ss', env_override, REQUEST_TIMEOUT_SECONDS_DEFAULT
        )
        return REQUEST_TIMEOUT_SECONDS_DEFAULT
    if value <= 0:
        return REQUEST_TIMEOUT_SECONDS_DEFAULT
    return value


MAX_TOKENS_PER_PAGE = 4096


def _compute_pages_per_call() -> int:
    env_override = os.getenv('VISION_PAGES_PER_CALL')
    requested_pages_per_call = max(1, int(env_override)) if env_override else VISION_PAGES_PER_CALL_DEFAULT
    max_output_tokens = int(os.getenv('VISION_MAX_OUTPUT_TOKENS', str(VISION_MAX_OUTPUT_TOKENS_DEFAULT)))
    output_cap_pages = max(1, max_output_tokens // MAX_TOKENS_PER_PAGE)
    max_context_tokens = int(os.getenv('VISION_MAX_CONTEXT_TOKENS', str(VISION_MAX_CONTEXT_TOKENS_DEFAULT)))
    available_context_tokens = max(0, max_context_tokens - VISION_PROMPT_TOKENS_ESTIMATE)
    total_context_tokens_per_page = VISION_INPUT_TOKENS_PER_PAGE_ESTIMATE + MAX_TOKENS_PER_PAGE
    context_cap_pages = max(1, available_context_tokens // total_context_tokens_per_page)
    pages_per_call = min(requested_pages_per_call, output_cap_pages, context_cap_pages)
    logger.info(
        'Vision batch sizing: %d pages/call (requested=%d, output_cap=%d from %d tokens, context_cap=%d from %d total-context tokens with %d prompt + %d/page input + %d/page output estimate)',
        pages_per_call,
        requested_pages_per_call,
        output_cap_pages,
        max_output_tokens,
        context_cap_pages,
        max_context_tokens,
        VISION_PROMPT_TOKENS_ESTIMATE,
        VISION_INPUT_TOKENS_PER_PAGE_ESTIMATE,
        MAX_TOKENS_PER_PAGE,
    )
    return pages_per_call


ALTERNATIVE_BASE_PROMPTS: tuple[str, ...] = (
    # Plain transcription framing - avoids 'extract data' verbs that sometimes trip
    # content guardrails on regulated documents (pharma, medical, financial).
    (
        'Transcribe the visible text from each page exactly as it appears. '
        'Use HTML <table> tags for tabular content with proper <thead> and <tbody>. '
        'Preserve all numbers, dates, identifiers, and labels verbatim. '
        'For diagrams and infographics, list visible labels and numeric values as a markdown list.'
    ),
    # Description framing - softer phrasing that often survives strict guardrails.
    (
        'Describe each page in detail. '
        'Include all visible headings, paragraph text, table contents, captions, dates, '
        'identifiers, batch numbers, and numeric values. '
        'Format your description as clean markdown using HTML <table> tags for tables.'
    ),
)
ALT_PROMPT_LOW_CONTENT_THRESHOLD = 100  # chars; below this we treat a solo page as a failure

# Image variants tried when a SOLO page is blocked by an image-classifier guardrail.
# Empirically validated against Bedrock's content_filter on pharma formulation tables
# AND diagram-heavy process pages: rotation, half-cropping, blur, and downsample each
# evade the classifier on different page layouts. Order is LOSSLESS FIRST. Half-crops
# follow rotation because they recover bottom-of-page content the classifier blocks
# when the top heading region is in view (empirically: BLU-808 PDP page 33 jumped from
# 71->1110 chars and page 64 jumped from 454->1319 chars via bottom-half). Lossy variants last.
IMAGE_VARIANT_CHAIN: tuple[str, ...] = (
    'rotated-90',  # Rotate 90 degrees; LOSSLESS - every pixel preserved, just reoriented.
    'bottom-half',  # Crop bottom 50%; LOSSLESS for that region. Recovers body when top trips classifier.
    'top-half',  # Crop top 50%; LOSSLESS for that region. Complementary to bottom-half.
    'blurred',  # Gaussian blur radius 3; LOSSY but text usually legible. Last resort.
    'quarter-res',  # 25% resolution; HEAVILY LOSSY - small text/charts degraded. Last resort.
)


PAGE_SPLIT_RE = re.compile(r'<!--\s*PAGE\s+(\d+)\s*-->')
BATCH_EXTRACTION_PROMPT = (
    'You will see document pages. Extract canonical, page-scoped Markdown for EACH page.\n'
    "Preserve the page's reading order and visible heading hierarchy with Markdown headings.\n"
    'Use Markdown lists for lists and grouped facts.\n'
    'Use Markdown tables for simple tables; use HTML <table> only when merged cells, multiline cells, '
    'or layout cannot be represented accurately as a Markdown table.\n'
    'Strip repeated page footers, headers, and boilerplate that do not carry page-specific meaning.\n'
    'Represent diagrams, timelines, flow charts, and infographics as structured sections with labels, '
    'numeric values, durations, arrows, and stage names in reading order.\n'
    'Use [unclear] for unreadable text. Do not invent missing text, values, labels, or relationships.\n'
    'Separate each page with its <!-- PAGE {N} --> marker exactly as requested.'
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
    '.xlsx',
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
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
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
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
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
    image_sha256: str | None = None
    cache_key: str | None = None
    cached: bool = False


@dataclass(slots=True)
class PageExtractionRecord:
    page_number: int
    vision_content_raw: str
    vision_content_normalized: str
    retrieval_content: str
    canonical_content: str
    status: str
    warnings: list[str] = field(default_factory=list)
    char_count_raw: int = 0
    char_count_normalized: int = 0
    char_count_retrieval: int = 0
    used_in_retrieval: bool = False
    source_method: str = 'none'
    native_content: str | None = None
    image_sha256: str | None = None
    cache_key: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'page_number': self.page_number,
            'vision_content_raw': self.vision_content_raw,
            'vision_content_normalized': self.vision_content_normalized,
            'retrieval_content': self.retrieval_content,
            'canonical_content': self.canonical_content,
            'status': self.status,
            'warnings': list(self.warnings),
            'char_count_raw': self.char_count_raw,
            'char_count_normalized': self.char_count_normalized,
            'char_count_retrieval': self.char_count_retrieval,
            'used_in_retrieval': self.used_in_retrieval,
            'source_method': self.source_method,
            'native_content': self.native_content,
            'image_sha256': self.image_sha256,
            'cache_key': self.cache_key,
            'cached': self.cached,
        }


@dataclass(slots=True)
class ExtractionQualityReport:
    page_count: int
    pages_emitted_canonical: int
    pages_emitted_retrieval: int
    empty_pages: list[int] = field(default_factory=list)
    tiny_pages: list[int] = field(default_factory=list)
    unexplained_tiny_pages: list[int] = field(default_factory=list)
    boilerplate_only_pages: list[int] = field(default_factory=list)
    native_fallback_pages: list[int] = field(default_factory=list)
    dropped_retrieval_pages: list[int] = field(default_factory=list)
    warning_counts: dict[str, int] = field(default_factory=dict)
    table_counts: dict[str, Any] = field(default_factory=dict)
    pages_containing_tables: list[int] = field(default_factory=list)
    pages_containing_diagrams: list[int] = field(default_factory=list)
    prompt_version: str = VISION_PROMPT_VERSION
    model: str = VISION_MODEL_DEFAULT
    extractor_version: str = VISION_EXTRACTOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            'page_count': self.page_count,
            'pages_emitted_canonical': self.pages_emitted_canonical,
            'pages_emitted_retrieval': self.pages_emitted_retrieval,
            'empty_pages': list(self.empty_pages),
            'tiny_pages': list(self.tiny_pages),
            'unexplained_tiny_pages': list(self.unexplained_tiny_pages),
            'boilerplate_only_pages': list(self.boilerplate_only_pages),
            'native_fallback_pages': list(self.native_fallback_pages),
            'dropped_retrieval_pages': list(self.dropped_retrieval_pages),
            'warning_counts': dict(self.warning_counts),
            'table_counts': dict(self.table_counts),
            'pages_containing_tables': list(self.pages_containing_tables),
            'pages_containing_diagrams': list(self.pages_containing_diagrams),
            'prompt_version': self.prompt_version,
            'model': self.model,
            'extractor_version': self.extractor_version,
        }


@dataclass(slots=True)
class ExtractionArtifacts:
    retrieval_content: str
    canonical_content: str
    manifest: dict[str, Any]
    page_records: list[PageExtractionRecord]
    quality_report: ExtractionQualityReport
    warnings: list[ExtractionWarning]


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
    chunk_token_size: int | None = None,
    extraction_cache: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
) -> VisionExtractionResult:
    """Extract document text by sending page images to an OpenAI-compatible vision model."""
    from yar.document.semantic_chunker import chunk_markdown, count_tokens

    is_pdf_document = _is_pdf_document(filename=filename, mime_type=mime_type)
    is_office_document = _is_office_document(filename=filename, mime_type=mime_type)
    source_sha256 = _sha256_bytes(document_bytes)
    native_pdf_bytes: bytes | None = None
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
        native_pdf_bytes = pdf_bytes
        page_results, warnings, total_pages = await _extract_pdf_streaming(
            pdf_bytes,
            model=model,
            base_url=base_url,
            api_key=api_key,
            pdf_password=pdf_password,
            filename=filename,
            source_sha256=source_sha256,
            extraction_cache=extraction_cache,
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
            page_results, warnings = await _process_all_pages(
                client,
                model,
                pages,
                filename=filename,
                source_sha256=source_sha256,
                extraction_cache=extraction_cache,
            )
        finally:
            await client.close()

    _log_extraction_summary(filename, page_results, warnings)

    native_text_by_page: dict[int, str] = {}
    if native_pdf_bytes is not None:
        native_text_by_page = await asyncio.to_thread(
            partial(_extract_native_pdf_page_texts, native_pdf_bytes, pdf_password=pdf_password)
        )
    artifacts = build_extraction_artifacts(
        page_results,
        warnings,
        page_count=total_pages,
        model=model,
        source_sha256=source_sha256,
        native_text_by_page=native_text_by_page,
    )
    content = artifacts.retrieval_content
    collected_warnings = artifacts.warnings
    unrecovered_failure_pages = _unrecovered_vision_failure_pages(collected_warnings, artifacts.quality_report)
    if unrecovered_failure_pages:
        failed_pages = _format_page_number_list(unrecovered_failure_pages)
        raise VisionExtractionError(
            'Vision extraction left unrecovered provider-failed/refused pages '
            f'({failed_pages}); refusing to index an incomplete document'
        )
    if not content:
        failure_detail = _failure_detail_for_empty_extraction(collected_warnings)
        if failure_detail is not None:
            raise VisionExtractionError(failure_detail)
        raise VisionExtractionError('Vision extraction found no extractable text in the document')

    if collected_warnings:
        logger.warning(
            'Vision extraction completed with degraded quality for %s: %s',
            filename or 'document',
            [warning.message for warning in collected_warnings],
        )

    # Run page-aware semantic chunking on the extracted markdown.
    # The resulting pre_chunks bypass the generic pipeline chunker.
    # Use the configured chunk_token_size when provided so vision-extracted content matches the
    # rest of the corpus. Without it the chunker falls back to its 500-token join_threshold (1000-
    # token max), regardless of what the global pipeline expects.
    join_threshold = 500
    if chunk_token_size and chunk_token_size > 0:
        join_threshold = max(chunk_token_size // 2, 100)
    chunks = chunk_markdown(content, join_threshold=join_threshold, tokenizer=tokenizer)
    pre_chunks: list[dict[str, Any]] | None = None
    if chunks:
        pre_chunks = []
        for chunk in chunks:
            chunk_content = _content_with_context(chunk)
            pre_chunk = {
                'tokens': count_tokens(chunk_content, tokenizer=tokenizer),
                'content': chunk_content,
                'chunk_order_index': chunk.chunk_index,
                'page_number': chunk.page_number,
                'heading_context': chunk.heading_context,
            }
            if chunk.page_start is not None:
                pre_chunk['page_start'] = chunk.page_start
                pre_chunk['page_end'] = chunk.page_end
                pre_chunk['page_numbers'] = list(chunk.page_numbers)
            pre_chunks.append(pre_chunk)
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
            'extractor_version': VISION_EXTRACTOR_VERSION,
            'prompt_version': VISION_PROMPT_VERSION,
            'model': model,
            'page_count': total_pages,
            'warning_count': len(collected_warnings),
            'chunk_count': len(pre_chunks) if pre_chunks else 0,
            'canonical_content': artifacts.canonical_content,
            'extraction_manifest': artifacts.manifest,
            'quality_report': artifacts.quality_report.to_dict(),
            'page_records': [record.to_dict() for record in artifacts.page_records],
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
    source_sha256: str | None = None,
    extraction_cache: dict[str, Any] | None = None,
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
                page_results, warnings = await _process_all_pages(
                    client,
                    model,
                    pages,
                    filename=filename,
                    source_sha256=source_sha256,
                    extraction_cache=extraction_cache,
                )
                return page_results, warnings, total_pages

            pages_per_call = _compute_pages_per_call()
            concurrency_limit = _get_vision_concurrency_limit()
            wave_size = pages_per_call * concurrency_limit
            logger.info(
                'Vision extraction starting for %s: %s pages via %s',
                filename or 'document',
                total_pages,
                model,
            )
            logger.info(
                'Vision streaming extraction: %d pages, wave_size=%d (%d pages/call × %d concurrent requests)',
                total_pages,
                wave_size,
                pages_per_call,
                concurrency_limit,
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
                wave_results, wave_warnings = await _process_all_pages(
                    client,
                    model,
                    wave_pages,
                    filename=filename,
                    source_sha256=source_sha256,
                    extraction_cache=extraction_cache,
                )
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
    *,
    prompt_override: str | None = None,
) -> tuple[list[PageResult], list[ExtractionWarning], bool]:
    """Send a page batch to the vision model, gated by a global concurrency semaphore."""
    sem = _get_vision_semaphore()
    async with sem:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=cast(Any, _build_batch_vision_messages(pages, prompt_override=prompt_override)),
                temperature=0,
                max_tokens=MAX_TOKENS_PER_PAGE * len(pages),
            ),
            timeout=_get_request_timeout(),
        )
    raw_text = _extract_response_text(response).strip()

    choices = getattr(response, 'choices', None) or []
    finish_reason = getattr(choices[0], 'finish_reason', None) if choices else None
    if not raw_text:
        finish_label = finish_reason or 'unknown'
        empty_results = [PageResult(page_number=page.page_number, content='') for page in pages]
        # Multi-page empty batch: trigger adaptive split so failing pages get isolated.
        # Children may recover individual pages or emit per-page warnings.
        if len(pages) > 1:
            logger.warning(
                'Vision empty batch %s finish=%s - splitting to isolate offending pages',
                _page_range_label(pages),
                finish_label,
            )
            return empty_results, [], True
        # Solo page empty - log clearly so operators can trace why content is missing.
        if finish_label == 'content_filter':
            logger.warning(
                'Vision GUARDRAIL BLOCK page %d finish=content_filter completion_tokens=%s '
                '(model refused to extract; will try alt-prompts)',
                pages[0].page_number,
                getattr(getattr(response, 'usage', None), 'completion_tokens', '?'),
            )
        else:
            logger.warning(
                'Vision empty page %d finish=%s (will try alt-prompts)',
                pages[0].page_number,
                finish_label,
            )
        warnings = [
            ExtractionWarning(
                source='vision_empty_response',
                message=(
                    f'Vision model returned empty content for page {pages[0].page_number} '
                    f'(finish_reason={finish_label}).'
                ),
            )
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
    *,
    prompt_override: str | None = None,
) -> tuple[list[PageResult], list[ExtractionWarning], bool]:
    """Try up to VISION_MAX_RETRIES times with exponential backoff + jitter."""
    last_exc: Exception | None = None
    attempts_used = 0
    stopped_on_non_retryable_error = False
    for attempt in range(VISION_MAX_RETRIES):
        attempts_used = attempt + 1
        try:
            return await _extract_batch(client, model, pages, prompt_override=prompt_override)
        except Exception as exc:
            last_exc = exc
            stopped_on_non_retryable_error = _is_non_retryable_vision_error(exc)
            if stopped_on_non_retryable_error or attempt == VISION_MAX_RETRIES - 1:
                break

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

    if last_exc is None:
        raise VisionExtractionError('Vision extraction failed without an error detail')
    detail = _format_exception_message(last_exc)
    attempt_label = 'attempt' if attempts_used == 1 else 'attempts'
    retry_note = ' (non-retryable)' if stopped_on_non_retryable_error else ''
    logger.warning(
        'Vision extraction failed batch %s after %d %s%s: %s',
        _page_range_label(pages),
        attempts_used,
        attempt_label,
        retry_note,
        detail,
    )
    if stopped_on_non_retryable_error:
        raise VisionExtractionError(detail) from last_exc
    page_results = [PageResult(page_number=page.page_number, content='') for page in pages]
    if len(pages) > 1:
        return page_results, [], True
    warnings = [
        ExtractionWarning(
            source='vision_batch_failure',
            message=f'Vision extraction failed for page {pages[0].page_number}: {detail}',
        )
    ]
    return page_results, warnings, False


def _content_chars(results: list[PageResult]) -> int:
    return sum(len(result.content) for result in results)


def _transform_image_blurred(image_bytes: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageFilter

    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
    buf = BytesIO()
    blurred.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


def _transform_image_quarter_res(image_bytes: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    width, height = img.size
    resized = img.resize((max(1, width // 4), max(1, height // 4)))
    buf = BytesIO()
    resized.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


def _transform_image_rotated_90(image_bytes: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    rotated = img.rotate(90, expand=True)
    buf = BytesIO()
    rotated.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


def _transform_image_bottom_half(image_bytes: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    width, height = img.size
    cropped = img.crop((0, height // 2, width, height))
    buf = BytesIO()
    cropped.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


def _transform_image_top_half(image_bytes: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    width, height = img.size
    cropped = img.crop((0, 0, width, height // 2))
    buf = BytesIO()
    cropped.save(buf, format='JPEG', quality=80)
    return buf.getvalue()


_IMAGE_VARIANT_TRANSFORMS = {
    'blurred': _transform_image_blurred,
    'quarter-res': _transform_image_quarter_res,
    'rotated-90': _transform_image_rotated_90,
    'bottom-half': _transform_image_bottom_half,
    'top-half': _transform_image_top_half,
}


async def _try_image_variants(
    client: Any,
    model: str,
    page: RenderedPage,
    primary_results: list[PageResult],
    primary_warnings: list[ExtractionWarning],
) -> tuple[list[PageResult], list[ExtractionWarning]]:
    """Retry a single page with image transformations to bypass image-classifier guardrails.

    Empirically validated: blur, quarter-resolution, and rotation all defeat Bedrock's
    visual content_filter on pharma formulation tables, recovering full extraction.
    Returns the longest extraction across variants. Pillow is required; if unavailable,
    returns the original results unchanged.
    """
    try:
        import PIL  # noqa: F401
    except ImportError:
        logger.warning(
            'Vision image-variant retry skipped for page %d: Pillow not installed',
            page.page_number,
        )
        return primary_results, primary_warnings

    best_results = primary_results
    best_warnings = primary_warnings
    best_chars = _content_chars(primary_results)
    page_number = page.page_number

    for variant_name in IMAGE_VARIANT_CHAIN:
        transform_fn = _IMAGE_VARIANT_TRANSFORMS.get(variant_name)
        if transform_fn is None:
            continue
        try:
            transformed_bytes = await asyncio.to_thread(transform_fn, page.image_bytes)
        except Exception as exc:
            logger.warning(
                'Vision image-variant %s transform failed for page %d: %s',
                variant_name,
                page_number,
                _format_exception_message(exc),
            )
            continue

        variant_page = RenderedPage(
            page_number=page.page_number,
            total_pages=page.total_pages,
            media_type='image/jpeg',
            image_bytes=transformed_bytes,
        )
        try:
            attempt_results, attempt_warnings, _ = await _extract_batch_with_retry(client, model, [variant_page])
        except Exception as exc:
            logger.warning(
                'Vision image-variant %s extraction failed for page %d: %s',
                variant_name,
                page_number,
                _format_exception_message(exc),
            )
            continue

        attempt_chars = _content_chars(attempt_results)
        logger.info(
            'Vision image-variant %s for page %d: %d chars (best so far: %d)',
            variant_name,
            page_number,
            attempt_chars,
            best_chars,
        )
        if attempt_chars > best_chars:
            best_results = attempt_results
            # Tag recovered content with a provenance warning so consumers can flag
            # this page for review - it came from a transformed image, not the original.
            best_warnings = [
                *attempt_warnings,
                ExtractionWarning(
                    source='vision_image_variant_recovered',
                    message=(
                        f'Page {page_number} content recovered via {variant_name!r} '
                        f'image transform after content_filter on original. '
                        f'Review for accuracy.'
                    ),
                ),
            ]
            best_chars = attempt_chars
            if best_chars >= 1000:
                break
    return best_results, best_warnings


async def _try_alternative_prompts(
    client: Any,
    model: str,
    page: RenderedPage,
    primary_results: list[PageResult],
    primary_warnings: list[ExtractionWarning],
) -> tuple[list[PageResult], list[ExtractionWarning]]:
    """Retry a single page with alternative prompts when the default returned little/no content.

    Useful when the model's safety guardrail (e.g. Bedrock content_filter) trips on the
    default extraction prompt for regulated content. Different prompt framings often survive
    the guardrail and recover meaningful text. Returns the longest extraction across attempts.
    """
    best_results = primary_results
    best_warnings = primary_warnings
    best_chars = _content_chars(primary_results)
    page_number = page.page_number

    for alt_idx, alt_prompt in enumerate(ALTERNATIVE_BASE_PROMPTS, start=1):
        try:
            attempt_results, attempt_warnings, _ = await _extract_batch_with_retry(
                client, model, [page], prompt_override=alt_prompt
            )
        except Exception as exc:
            logger.warning(
                'Vision alt-prompt %d/%d failed for page %d: %s',
                alt_idx,
                len(ALTERNATIVE_BASE_PROMPTS),
                page_number,
                _format_exception_message(exc),
            )
            continue

        attempt_chars = _content_chars(attempt_results)
        logger.info(
            'Vision alt-prompt %d/%d for page %d: %d chars (best so far: %d)',
            alt_idx,
            len(ALTERNATIVE_BASE_PROMPTS),
            page_number,
            attempt_chars,
            best_chars,
        )
        if attempt_chars > best_chars:
            best_results = attempt_results
            best_warnings = attempt_warnings
            best_chars = attempt_chars
            if best_chars >= 1000:
                break
    return best_results, best_warnings


async def _extract_batch_adaptively(
    client: Any,
    model: str,
    pages: list[RenderedPage],
) -> tuple[list[PageResult], list[ExtractionWarning]]:
    page_results, warnings, should_split = await _extract_batch_with_retry(client, model, pages)
    # When a SOLO page came back fully empty, retry with alternative prompts to work around
    # model safety guardrails (e.g. Bedrock content_filter on regulated content). Different
    # prompt framings often survive the guardrail and recover meaningful text.
    # We trigger only on confirmed-empty model responses, not on hard provider failures
    # (auth/rate-limit) which won't recover from a different prompt.
    if len(pages) == 1:
        has_empty_response = any(w.source == 'vision_empty_response' for w in warnings)
        has_hard_failure = any(w.source == 'vision_batch_failure' for w in warnings)
        if has_empty_response and not has_hard_failure:
            page_results, warnings = await _try_alternative_prompts(client, model, pages[0], page_results, warnings)
            # If alt-prompts didn't recover content, the filter MAY be image-specific.
            # GATE STRICTLY: only try image transforms when the warning explicitly cites
            # finish_reason=content_filter. For genuinely blank pages (finish=stop) we
            # leave them blank rather than risk hallucination from transformed images.
            if _content_chars(page_results) < ALT_PROMPT_LOW_CONTENT_THRESHOLD:
                blocked_by_filter = any(
                    w.source == 'vision_empty_response' and 'finish_reason=content_filter' in w.message
                    for w in warnings
                )
                if blocked_by_filter:
                    page_results, warnings = await _try_image_variants(client, model, pages[0], page_results, warnings)
            if _content_chars(page_results) >= ALT_PROMPT_LOW_CONTENT_THRESHOLD:
                should_split = False

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
    *,
    filename: str | None = None,
    source_sha256: str | None = None,
    extraction_cache: dict[str, Any] | None = None,
) -> tuple[list[PageResult], list[ExtractionWarning]]:
    if not pages:
        return [], []

    label = filename or 'document'
    manifest_cache = _cache_lookup_from_manifest(extraction_cache)
    cached_results: list[PageResult] = []
    uncached_pages: list[RenderedPage] = []
    page_hashes = {page.page_number: _sha256_bytes(page.image_bytes) for page in pages}
    cache_keys: dict[int, str] = {}
    for page in pages:
        image_sha256 = page_hashes[page.page_number]
        cache_key = _page_cache_key(
            source_sha256=source_sha256,
            page_number=page.page_number,
            image_sha256=image_sha256,
            model=model,
        )
        if cache_key is None:
            uncached_pages.append(page)
            continue
        cache_keys[page.page_number] = cache_key
        cached_record = _PAGE_RESULT_CACHE.get(cache_key) or manifest_cache.get(cache_key)
        cached_result = _page_result_from_cache(
            page,
            cache_key=cache_key,
            image_sha256=image_sha256,
            cached_record=cached_record or {},
        )
        if cached_result is None:
            uncached_pages.append(page)
        else:
            cached_results.append(cached_result)

    if cached_results:
        logger.info(
            'Vision extraction cache hit for %s: %d/%d pages',
            label,
            len(cached_results),
            len(pages),
        )
    if not uncached_pages:
        cached_results.sort(key=lambda result: result.page_number)
        return cached_results, []

    pages_per_call = _compute_pages_per_call()
    concurrency_limit = _get_vision_concurrency_limit()
    batches = [
        uncached_pages[index : index + pages_per_call] for index in range(0, len(uncached_pages), pages_per_call)
    ]
    logger.info(
        'Vision extraction: %d pages in %d batches (%d pages/call, %d concurrent requests)',
        len(uncached_pages),
        len(batches),
        pages_per_call,
        concurrency_limit,
    )
    all_results: list[PageResult] = cached_results
    all_warnings: list[ExtractionWarning] = []
    batches_done = 0
    cumulative_chars = 0
    extraction_start = time.monotonic()

    for index in range(0, len(batches), concurrency_limit):
        batch_group = batches[index : index + concurrency_limit]
        group_start = time.monotonic()
        batch_outputs = await asyncio.gather(
            *[_extract_batch_adaptively(client, model, batch) for batch in batch_group]
        )
        group_elapsed = time.monotonic() - group_start
        group_pages = sum(len(batch) for batch in batch_group)
        group_chars = sum(len(result.content) for results, _ in batch_outputs for result in results)
        batches_done += len(batch_group)
        cumulative_chars += group_chars
        logger.info(
            'Vision batch %d/%d done for %s: %d pages, %d chars in %.1fs',
            batches_done,
            len(batches),
            label,
            group_pages,
            group_chars,
            group_elapsed,
        )
        for batch_results, batch_warnings in batch_outputs:
            for result in batch_results:
                result.image_sha256 = page_hashes.get(result.page_number)
                result.cache_key = cache_keys.get(result.page_number)
            all_results.extend(batch_results)
            all_warnings.extend(batch_warnings)

    logger.info(
        'Vision extraction processed %d pages for %s in %.1fs (%d chars total)',
        len(all_results),
        label,
        time.monotonic() - extraction_start,
        cumulative_chars,
    )
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


def _log_extraction_summary(
    filename: str | None,
    page_results: list[PageResult],
    warnings: list[ExtractionWarning],
) -> None:
    """Emit a single high-signal summary line per document so operators can see at a
    glance how extraction went without scrolling through hundreds of per-batch lines.
    """
    label = filename or 'document'
    total_pages = len(page_results)
    pages_with_content = sum(1 for r in page_results if r.content.strip())
    empty_pages = total_pages - pages_with_content
    total_chars = sum(len(r.content) for r in page_results)

    # Count warnings by source so guardrail-tripped pages stand out.
    warning_counts: dict[str, int] = {}
    finish_reason_counts: dict[str, int] = {}
    for warning in warnings:
        warning_counts[warning.source] = warning_counts.get(warning.source, 0) + 1
        if warning.source == 'vision_empty_response':
            match = re.search(r'finish_reason=([\w_-]+)', warning.message)
            reason = match.group(1) if match else 'unknown'
            finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1

    log_method = logger.info if empty_pages == 0 else logger.warning
    log_method(
        'Vision audit for %s: pages=%d ok=%d empty=%d chars=%d warnings=%s finish_reasons=%s',
        label,
        total_pages,
        pages_with_content,
        empty_pages,
        total_chars,
        warning_counts or {},
        finish_reason_counts or {},
    )


_SIMPLE_HTML_TABLE_RE = re.compile(r'<table\b[\s\S]*?</table>', flags=re.IGNORECASE)
_HTML_TABLE_PLACEHOLDER_RE = re.compile(r'\x00YAR_HTML_TABLE_(\d+)\x00')
_ALWAYS_DROP_BOILERPLATE_KEYS = frozenset({'sanofi'})
_REPEATED_BOILERPLATE_MIN_PAGES = 3
_REPEATED_BOILERPLATE_MIN_RATIO = 0.25


def _normalize_table_cell_text(raw: str) -> str:
    return re.sub(r'\s+', ' ', unescape(raw)).strip()


def _markdown_table_cell(value: str) -> str:
    return value.replace('|', r'\|')


class _SimpleHtmlTableParser(HTMLParser):
    """Parse simple rectangular HTML tables into rows without accepting layout semantics."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[list[str]] = []
        self.is_complex = False
        self._table_depth = 0
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == 'table':
            if self._table_depth > 0:
                self.is_complex = True
            self._table_depth += 1
            return

        if self._table_depth == 0:
            return

        if tag == 'tr':
            if self._current_row is not None:
                self.is_complex = True
            self._current_row = []
            return

        if tag in {'td', 'th'}:
            if self._current_row is None or self._current_cell is not None:
                self.is_complex = True
            attr_map = {name.lower(): value for name, value in attrs}
            for span_attr in ('colspan', 'rowspan'):
                span_value = attr_map.get(span_attr)
                if span_value not in (None, '', '1'):
                    self.is_complex = True
            self._current_cell = []
            return

        if tag == 'br' and self._current_cell is not None:
            self.is_complex = True
            self._current_cell.append('\n')

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {'td', 'th'} and self._current_cell is not None:
            if self._current_row is None:
                self.is_complex = True
            else:
                self._current_row.append(_normalize_table_cell_text(''.join(self._current_cell)))
            self._current_cell = None
            return

        if tag == 'tr' and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None
            return

        if tag == 'table' and self._table_depth > 0:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(f'&{name};')

    def handle_charref(self, name: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(f'&#{name};')


def _simple_html_table_to_markdown(html_table: str) -> str | None:
    parser = _SimpleHtmlTableParser()
    try:
        parser.feed(html_table)
        parser.close()
    except Exception:
        return None

    rows = [row for row in parser.rows if any(cell for cell in row)]
    if parser.is_complex or not rows:
        return None

    column_count = len(rows[0])
    if column_count == 0 or any(len(row) != column_count for row in rows):
        return None

    header = [_markdown_table_cell(cell) for cell in rows[0]]
    body = [[_markdown_table_cell(cell) for cell in row] for row in rows[1:]]
    markdown_lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join('---' for _ in range(column_count)) + ' |',
    ]
    markdown_lines.extend('| ' + ' | '.join(row) + ' |' for row in body)
    return '\n'.join(markdown_lines)


def _convert_simple_html_tables_to_markdown(content: str) -> str:
    def replace_table(match: re.Match[str]) -> str:
        markdown_table = _simple_html_table_to_markdown(match.group(0))
        if markdown_table is None:
            return match.group(0)
        return f'\n\n{markdown_table}\n\n'

    return _SIMPLE_HTML_TABLE_RE.sub(replace_table, content)


def _protect_html_tables(content: str) -> tuple[str, list[str]]:
    tables: list[str] = []

    def replace_table(match: re.Match[str]) -> str:
        tables.append(match.group(0))
        return f'\x00YAR_HTML_TABLE_{len(tables) - 1}\x00'

    return _SIMPLE_HTML_TABLE_RE.sub(replace_table, content), tables


def _restore_html_tables(content: str, tables: list[str]) -> str:
    def restore_table(match: re.Match[str]) -> str:
        table_index = int(match.group(1))
        if table_index >= len(tables):
            return match.group(0)
        return tables[table_index]

    return _HTML_TABLE_PLACEHOLDER_RE.sub(restore_table, content)


def _strip_simple_inline_html(raw: str) -> str:
    stripped = re.sub(r'<br\s*/?>', ' ', raw, flags=re.IGNORECASE)
    stripped = re.sub(r'</?[\w:-]+\b[^>]*>', '', stripped)
    return re.sub(r'\s+', ' ', unescape(stripped)).strip()


def _normalize_simple_html_blocks(content: str) -> str:
    protected_content, tables = _protect_html_tables(content)
    normalized = protected_content
    normalized = re.sub(r'<br\s*/?>', '\n', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</p>\s*<p\b[^>]*>', '\n\n', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'<p\b[^>]*>', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</p>', '\n\n', normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r'<li\b[^>]*>([\s\S]*?)</li>',
        lambda m: f'- {_strip_simple_inline_html(m.group(1))}',
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r'</?(?:ul|ol)\b[^>]*>', '\n', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'</?(?:div|span|strong|b|em|i)\b[^>]*>', '', normalized, flags=re.IGNORECASE)
    return _restore_html_tables(unescape(normalized), tables)


def _normalize_repeated_boilerplate_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    heading_text = re.sub(r'^#+\s*', '', stripped).strip()
    if heading_text and heading_text.lower() == 'sanofi':
        return 'sanofi'

    if stripped.startswith(('#', '|', '>', '-', '*', '```')):
        return None
    if re.match(r'^\d+\.\s+\S', stripped):
        return None

    normalized = re.sub(r'\s+', ' ', stripped).strip().lower()
    normalized = normalized.strip('·•—–-:;|')
    if not normalized:
        return None

    if re.fullmatch(r'(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?', normalized):
        return re.sub(r'\d+', '#', normalized)

    if normalized == 'sanofi':
        return normalized

    boilerplate_terms = (
        'confidential',
        'proprietary',
        'all rights reserved',
        'internal use only',
        'do not distribute',
    )
    if len(normalized) <= 120 and any(term in normalized for term in boilerplate_terms):
        return re.sub(r'\d+', '#', normalized)

    return None


def _remove_repeated_page_boilerplate(page_contents: list[str]) -> list[str]:
    repeated_keys = set(_ALWAYS_DROP_BOILERPLATE_KEYS)

    if len(page_contents) >= _REPEATED_BOILERPLATE_MIN_PAGES:
        key_counts: dict[str, int] = {}
        for content in page_contents:
            page_keys = {
                key for line in content.splitlines() if (key := _normalize_repeated_boilerplate_key(line)) is not None
            }
            for key in page_keys:
                key_counts[key] = key_counts.get(key, 0) + 1

        min_occurrences = max(
            _REPEATED_BOILERPLATE_MIN_PAGES,
            math.ceil(len(page_contents) * _REPEATED_BOILERPLATE_MIN_RATIO),
        )
        repeated_keys.update(key for key, count in key_counts.items() if count >= min_occurrences)

    normalized_pages = []
    for content in page_contents:
        kept_lines = [
            line for line in content.splitlines() if _normalize_repeated_boilerplate_key(line) not in repeated_keys
        ]
        normalized_pages.append('\n'.join(kept_lines).strip())
    return normalized_pages


def _normalize_extracted_page_content(page_contents: list[str]) -> list[str]:
    normalized_pages = []
    for content in page_contents:
        normalized = _convert_simple_html_tables_to_markdown(content)
        normalized = _normalize_simple_html_blocks(normalized)
        normalized_pages.append(re.sub(r'\n{3,}', '\n\n', normalized).strip())
    return _remove_repeated_page_boilerplate(normalized_pages)


def _normalize_page_content_for_canonical(page_contents: list[str]) -> list[str]:
    normalized_pages = []
    for content in page_contents:
        normalized = _convert_simple_html_tables_to_markdown(content)
        normalized = _normalize_simple_html_blocks(normalized)
        normalized_pages.append(re.sub(r'\n{3,}', '\n\n', normalized).strip())
    return normalized_pages


def _count_markdown_table_lines(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip().startswith('|') and line.strip().endswith('|'))


def _table_metrics_for_content(content: str) -> dict[str, int]:
    html_tables = _SIMPLE_HTML_TABLE_RE.findall(content)
    simple_html_tables = sum(1 for table in html_tables if _simple_html_table_to_markdown(table) is not None)
    complex_html_tables = len(html_tables) - simple_html_tables
    return {
        'html_tables_total': len(html_tables),
        'simple_html_tables': simple_html_tables,
        'complex_html_tables': complex_html_tables,
        'markdown_table_lines': _count_markdown_table_lines(content),
    }


def _contains_diagram_language(content: str) -> bool:
    normalized = content.lower()
    diagram_terms = ('diagram', 'timeline', 'flowchart', 'flow chart', 'arrow', 'infographic', 'process flow')
    return any(term in normalized for term in diagram_terms)


def _merge_table_counts(metrics: list[dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {
        'html_tables_total': 0,
        'simple_html_tables': 0,
        'complex_html_tables': 0,
        'markdown_table_lines': 0,
    }
    for page_metrics in metrics:
        for key in totals:
            totals[key] += page_metrics.get(key, 0)
    return totals


def _warning_counts(
    warnings: list[ExtractionWarning],
    page_records: list[PageExtractionRecord],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for warning in warnings:
        counts[warning.source] = counts.get(warning.source, 0) + 1
    for record in page_records:
        for warning in record.warnings:
            source = warning.split(':', 1)[0]
            counts[source] = counts.get(source, 0) + 1
    return counts


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_cache_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _page_cache_key(
    *,
    source_sha256: str | None,
    page_number: int,
    image_sha256: str | None,
    model: str,
) -> str | None:
    if not source_sha256 or not image_sha256:
        return None
    payload = {
        'source_sha256': source_sha256,
        'page_number': page_number,
        'image_sha256': image_sha256,
        'model': model,
        'prompt_version': VISION_PROMPT_VERSION,
        'extractor_version': VISION_EXTRACTOR_VERSION,
    }
    return _sha256_bytes(_json_cache_payload(payload).encode('utf-8'))


def _cache_lookup_from_manifest(extraction_cache: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not extraction_cache:
        return {}
    records = extraction_cache.get('page_records')
    if not isinstance(records, list):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        cache_key = record.get('cache_key')
        raw_content = record.get('vision_content_raw')
        if isinstance(cache_key, str) and isinstance(raw_content, str):
            lookup[cache_key] = record
    return lookup


def _page_result_from_cache(
    page: RenderedPage,
    *,
    cache_key: str,
    image_sha256: str,
    cached_record: dict[str, Any],
) -> PageResult | None:
    raw_content = cached_record.get('vision_content_raw')
    if not isinstance(raw_content, str):
        return None
    return PageResult(
        page_number=page.page_number,
        content=raw_content,
        image_sha256=image_sha256,
        cache_key=cache_key,
        cached=True,
    )


def _store_page_result_cache(records: list[PageExtractionRecord]) -> None:
    for record in records:
        if not record.cache_key:
            continue
        _PAGE_RESULT_CACHE[record.cache_key] = record.to_dict()


def _needs_native_fallback(content: str, warning: ExtractionWarning | None) -> bool:
    if not _has_usable_content(content):
        return True
    if len(content.strip()) < TINY_PAGE_CHAR_THRESHOLD:
        return True
    return warning is not None and warning.source in {'vision_truncation', 'vision_empty_response', 'vision_error'}


def _should_use_native_fallback(
    *,
    vision_content: str,
    native_content: str,
    warning: ExtractionWarning | None,
) -> bool:
    if not _has_usable_content(native_content):
        return False
    if not _needs_native_fallback(vision_content, warning):
        return False
    if not _has_usable_content(vision_content):
        return True
    return len(native_content.strip()) > len(vision_content.strip())


def _page_status(
    *,
    canonical_content: str,
    retrieval_content: str,
    used_native: bool,
    warning: ExtractionWarning | None,
) -> tuple[str, list[str]]:
    record_warnings: list[str] = []
    has_canonical_content = _has_usable_content(canonical_content)
    used_in_retrieval = _has_usable_content(retrieval_content)
    if not has_canonical_content:
        return 'empty', ['empty_page']
    if used_native:
        record_warnings.append('native_fallback')
        return 'native_fallback', record_warnings
    if warning is not None:
        record_warnings.append(f'{warning.source}: {warning.message}')
        return 'vision_warning', record_warnings
    if not used_in_retrieval:
        return 'boilerplate_only', ['retrieval_omitted_boilerplate_only']
    if len(retrieval_content.strip()) < TINY_PAGE_CHAR_THRESHOLD:
        return 'tiny', ['low_content_page']
    return 'ok', record_warnings


def _canonical_page_section(record: PageExtractionRecord) -> str:
    parts = [f'<!-- PAGE {record.page_number} -->']
    if record.status != 'ok' or not record.used_in_retrieval:
        parts.append(f'<!-- YAR_PAGE_STATUS: {record.status} -->')
    if record.canonical_content.strip():
        parts.append(record.canonical_content.strip())
    return '\n\n'.join(parts)


def build_extraction_artifacts(
    results: list[PageResult],
    extra_warnings: list[ExtractionWarning] | None = None,
    *,
    page_count: int | None = None,
    model: str = VISION_MODEL_DEFAULT,
    source_sha256: str | None = None,
    native_text_by_page: dict[int, str] | None = None,
) -> ExtractionArtifacts:
    ordered_results = sorted(results, key=lambda result: result.page_number)
    max_result_page = max((result.page_number for result in ordered_results), default=0)
    expected_page_count = max(page_count or 0, max_result_page)
    result_by_page = {result.page_number: result for result in ordered_results}
    complete_results = [
        result_by_page.get(page_number, PageResult(page_number=page_number, content=''))
        for page_number in range(1, expected_page_count + 1)
    ]

    warnings = list(extra_warnings or [])
    warnings.extend(result.warning for result in complete_results if result.warning is not None)
    canonical_normalized = _normalize_page_content_for_canonical([result.content for result in complete_results])
    native_pages = native_text_by_page or {}
    retrieval_base_contents: list[str] = []
    canonical_contents: list[str] = []
    used_native_pages: set[int] = set()

    for result, canonical_content in zip(complete_results, canonical_normalized, strict=True):
        native_content = _normalize_page_content_for_canonical([native_pages.get(result.page_number, '')])[0]
        use_native = _should_use_native_fallback(
            vision_content=canonical_content,
            native_content=native_content,
            warning=result.warning,
        )
        if use_native:
            used_native_pages.add(result.page_number)
            retrieval_base_contents.append(native_content)
            canonical_contents.append(native_content)
        else:
            retrieval_base_contents.append(result.content)
            canonical_contents.append(canonical_content)

    retrieval_normalized = _normalize_extracted_page_content(retrieval_base_contents)
    normalized_results = [
        PageResult(
            page_number=result.page_number,
            content=content,
            warning=result.warning,
            image_sha256=result.image_sha256,
            cache_key=result.cache_key,
            cached=result.cached,
        )
        for result, content in zip(complete_results, retrieval_normalized, strict=True)
    ]
    quality_warning = _summarize_extraction_quality(normalized_results, warnings)
    if quality_warning is not None:
        warnings.append(quality_warning)

    page_records: list[PageExtractionRecord] = []
    table_metrics: list[dict[str, int]] = []
    for result, canonical_content, retrieval_content in zip(
        complete_results,
        canonical_contents,
        retrieval_normalized,
        strict=True,
    ):
        page_number = result.page_number
        native_content = native_pages.get(page_number)
        used_native = page_number in used_native_pages
        status, record_warnings = _page_status(
            canonical_content=canonical_content,
            retrieval_content=retrieval_content,
            used_native=used_native,
            warning=result.warning,
        )
        used_in_retrieval = _has_usable_content(retrieval_content)
        source_method = 'none'
        if used_native:
            source_method = 'native' if not _has_usable_content(result.content) else 'vision_plus_native'
        elif _has_usable_content(result.content):
            source_method = 'vision'
        record = PageExtractionRecord(
            page_number=page_number,
            vision_content_raw=result.content,
            vision_content_normalized=canonical_normalized[page_number - 1]
            if page_number <= len(canonical_normalized)
            else '',
            retrieval_content=retrieval_content,
            canonical_content=canonical_content,
            status=status,
            warnings=record_warnings,
            char_count_raw=len(result.content),
            char_count_normalized=len(canonical_content),
            char_count_retrieval=len(retrieval_content),
            used_in_retrieval=used_in_retrieval,
            source_method=source_method,
            native_content=native_content if used_native else None,
            image_sha256=result.image_sha256,
            cache_key=result.cache_key,
            cached=result.cached,
        )
        page_records.append(record)
        table_metrics.append(_table_metrics_for_content(canonical_content))

    _store_page_result_cache(page_records)
    retrieval_pages = [
        f'<!-- PAGE {record.page_number} -->\n\n{record.retrieval_content.strip()}'
        for record in page_records
        if record.used_in_retrieval
    ]
    canonical_pages = [_canonical_page_section(record) for record in page_records]
    pages_containing_tables = [
        record.page_number for record, metrics in zip(page_records, table_metrics, strict=True) if any(metrics.values())
    ]
    pages_containing_diagrams = [
        record.page_number for record in page_records if _contains_diagram_language(record.canonical_content)
    ]
    tiny_pages = [
        record.page_number for record in page_records if 0 < record.char_count_retrieval < TINY_PAGE_CHAR_THRESHOLD
    ]
    quality_report = ExtractionQualityReport(
        page_count=expected_page_count,
        pages_emitted_canonical=len(canonical_pages),
        pages_emitted_retrieval=len(retrieval_pages),
        empty_pages=[record.page_number for record in page_records if record.status == 'empty'],
        tiny_pages=tiny_pages,
        unexplained_tiny_pages=[
            record.page_number for record in page_records if record.page_number in tiny_pages and record.status == 'ok'
        ],
        boilerplate_only_pages=[record.page_number for record in page_records if record.status == 'boilerplate_only'],
        native_fallback_pages=sorted(used_native_pages),
        dropped_retrieval_pages=[record.page_number for record in page_records if not record.used_in_retrieval],
        warning_counts=_warning_counts(warnings, page_records),
        table_counts=_merge_table_counts(table_metrics),
        pages_containing_tables=pages_containing_tables,
        pages_containing_diagrams=pages_containing_diagrams,
        model=model,
    )
    manifest = {
        'schema_version': 1,
        'extractor': 'vision',
        'extractor_version': VISION_EXTRACTOR_VERSION,
        'prompt_version': VISION_PROMPT_VERSION,
        'model': model,
        'source_sha256': source_sha256,
        'quality_report': quality_report.to_dict(),
        'page_records': [record.to_dict() for record in page_records],
    }
    return ExtractionArtifacts(
        retrieval_content='\n\n'.join(retrieval_pages),
        canonical_content='\n\n'.join(canonical_pages),
        manifest=manifest,
        page_records=page_records,
        quality_report=quality_report,
        warnings=warnings,
    )


def stitch_extracted_pages(
    results: list[PageResult],
    extra_warnings: list[ExtractionWarning] | None = None,
) -> tuple[str, list[ExtractionWarning]]:
    artifacts = build_extraction_artifacts(results, extra_warnings)
    return artifacts.retrieval_content, artifacts.warnings


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


def _extract_native_pdf_page_texts(pdf_bytes: bytes, *, pdf_password: str | None = None) -> dict[int, str]:
    """Extract embedded PDF text per page without OCR for source-backed fallback."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return {}

    try:
        document = pdfium.PdfDocument(pdf_bytes, password=pdf_password)
    except Exception as exc:
        logger.debug('Native PDF text fallback could not open document: %s', exc)
        return {}

    page_texts: dict[int, str] = {}
    try:
        for page_index in range(len(document)):
            page = None
            text_page = None
            try:
                page = document[page_index]
                text_page = page.get_textpage()
                text = re.sub(r'\n{3,}', '\n\n', text_page.get_text_range().strip())
            except Exception as exc:
                logger.debug('Native PDF text fallback failed for page %d: %s', page_index + 1, exc)
                continue
            finally:
                close_text = getattr(text_page, 'close', None)
                if callable(close_text):
                    close_text()
                close_page = getattr(page, 'close', None)
                if callable(close_page):
                    close_page()
            if _has_usable_content(text):
                page_texts[page_index + 1] = text
    finally:
        close_document = getattr(document, 'close', None)
        if callable(close_document):
            close_document()
    return page_texts


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
            ((int(page_path.stem.rsplit('-', maxsplit=1)[-1]), page_path) for page_path in tmp_path.glob('page-*.jpg')),
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
            ((int(page_path.stem.rsplit('-', maxsplit=1)[-1]), page_path) for page_path in tmp_path.glob('page-*.jpg')),
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


def _build_batch_vision_messages(
    pages: list[RenderedPage],
    *,
    prompt_override: str | None = None,
) -> list[dict[str, Any]]:
    page_numbers = [page.page_number for page in pages]
    expected_markers = ', '.join(f'<!-- PAGE {page_number} -->' for page_number in page_numbers)
    base_prompt = prompt_override if prompt_override is not None else BATCH_EXTRACTION_PROMPT
    prompt = (
        f'{base_prompt}\n\n'
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


_NON_RETRYABLE_VISION_STATUS_CODES = frozenset({401, 403})
_NON_RETRYABLE_VISION_ERROR_SNIPPETS = (
    'authentication',
    'forbidden',
    'insufficient quota',
    'insufficient_quota',
    'key limit exceeded',
    'permission denied',
    'quota exceeded',
    'quota exhausted',
    'unauthorized',
)


def _failure_detail_for_empty_extraction(warnings: list[ExtractionWarning]) -> str | None:
    details: list[str] = []
    for warning in warnings:
        if warning.source != 'vision_batch_failure':
            continue
        _, _, detail = warning.message.partition(': ')
        normalized_detail = (detail or warning.message).strip()
        if normalized_detail and normalized_detail not in details:
            details.append(normalized_detail)

    if not details:
        return None
    if len(details) == 1:
        return details[0]
    return 'Vision extraction failed: ' + '; '.join(details)


_WARNING_PAGE_RE = re.compile(r'\bpage\s+(\d+)\b', flags=re.IGNORECASE)


def _format_page_number_list(page_numbers: list[int], *, max_items: int = 20) -> str:
    displayed = page_numbers[:max_items]
    rendered = ', '.join(str(page_number) for page_number in displayed)
    remaining = len(page_numbers) - len(displayed)
    if remaining > 0:
        rendered = f'{rendered}, ... (+{remaining} more)'
    return f'pages {rendered}' if len(page_numbers) != 1 else f'page {rendered}'


def _unrecovered_vision_failure_pages(
    warnings: list[ExtractionWarning],
    quality_report: ExtractionQualityReport,
) -> list[int]:
    empty_pages = set(quality_report.empty_pages)
    if not empty_pages:
        return []

    failed_pages: set[int] = set()
    for warning in warnings:
        is_provider_failure = warning.source == 'vision_batch_failure' or (
            warning.source == 'vision_empty_response' and 'finish_reason=content_filter' in warning.message
        )
        if not is_provider_failure:
            continue

        match = _WARNING_PAGE_RE.search(warning.message)
        if match:
            failed_pages.add(int(match.group(1)))

    return sorted(empty_pages & failed_pages)


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, 'status_code', None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, 'response', None)
    response_status = getattr(response, 'status_code', None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_non_retryable_vision_error(exc: Exception) -> bool:
    status_code = _exception_status_code(exc)
    if status_code in _NON_RETRYABLE_VISION_STATUS_CODES:
        return True

    detail = _format_exception_message(exc).lower()
    return any(snippet in detail for snippet in _NON_RETRYABLE_VISION_ERROR_SNIPPETS)


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
    'VISION_EXTRACTOR_VERSION',
    'VISION_IMAGE_EXTENSIONS',
    'VISION_IMAGE_MIME_TYPES',
    'VISION_MODEL_DEFAULT',
    'VISION_OFFICE_EXTENSIONS',
    'VISION_OFFICE_MIME_TYPES',
    'VISION_PROMPT_VERSION',
    'VISION_SUPPORTED_EXTENSIONS',
    'ExtractionArtifacts',
    'ExtractionQualityReport',
    'ExtractionWarning',
    'PageExtractionRecord',
    'PageResult',
    'RenderedPage',
    'VisionExtractionError',
    'VisionExtractionResult',
    'build_extraction_artifacts',
    'extract_document_with_vision',
    'is_vision_document',
    'should_split_batch',
    'split_batch_response',
    'stitch_extracted_pages',
]
