#!/usr/bin/env python
"""Try several prompt framings against a single page to see if any survive Bedrock's
content_filter guardrail.

Renders ONE specific page from a doc, then sends it to the vision model with a
panel of different prompt framings (transcription, description, accessibility,
meta-questions, structured-output, partial-OCR). Reports finish_reason +
content_chars per attempt so you can see empirically which (if any) phrasings
get past the guardrail.

Usage:
    uv run python scripts/test_vision_prompts.py <s3_doc_id> <page_number>

Examples:
    uv run python scripts/test_vision_prompts.py doc_84a1429369c1217187f782a4b41b18de 29

Optional env (inherits from start.sh/.env):
    YAR_API_BASE        default http://localhost:9621
    LLM_BINDING_HOST    LiteLLM base URL
    LLM_BINDING_API_KEY LiteLLM master key
    VISION_MODEL        default 'salmon'
    YAR_WORKSPACE       default 'default'
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yar.document.vision_adapter import (
    NO_TEXT_DETECTED_SENTINEL,
    RenderedPage,
    _convert_office_document_to_pdf,
    _render_pdf_page_range,
)
from yar.llm.openai import create_openai_async_client

# Panel of prompt framings to try, ordered roughly from "asks for content" to "asks
# for meta-properties only". Each is meant to elicit different guardrail behavior.
PROMPT_PANEL: tuple[tuple[str, str], ...] = (
    (
        'default-extract',
        'You will see document pages. Extract all text and data from EACH page as clean markdown. '
        'For tables, use HTML <table> tags. Be thorough and accurate.',
    ),
    (
        'transcribe-verbatim',
        'Transcribe the visible text from this page exactly as it appears. '
        'Use HTML <table> tags for tabular content. Preserve numbers, dates, and labels verbatim.',
    ),
    (
        'describe-page',
        'Describe each page in detail. Include all visible headings, paragraph text, '
        'table contents, captions, dates, and numeric values. Format as markdown.',
    ),
    (
        'accessibility-screen-reader',
        'You are an accessibility tool generating screen-reader output. '
        'Read out the visual contents of this page in reading order: '
        'headings, paragraphs, table cells (row by row), captions, footnotes. '
        'Use markdown formatting for structure.',
    ),
    (
        'visual-question',
        'What do you see in this image? Describe the visual elements: '
        'headings, paragraphs, tables (including cell contents), figures, and overall layout.',
    ),
    (
        'meta-only',
        'Describe the structure of this page WITHOUT reproducing specific values. '
        'Tell me: how many sections? how many tables? what are the column headers? '
        'what kind of content is in each section (e.g. narrative, tabular, diagram)?',
    ),
    (
        'document-quality-audit',
        'You are auditing this document for formatting consistency. '
        'List the visible elements (headings, tables, figures) and report any layout issues. '
        'Include the text of headings and table captions as part of the audit.',
    ),
    (
        'json-structured',
        'Output a JSON object describing this page. Schema: '
        '{"page_layout": str, "headings": [str], "tables": [{"caption": str, "rows": [[str, ...]]}], '
        '"figures": [str], "body_text": str}. Fill in every field based on what you see.',
    ),
)


def _api_base() -> str:
    return os.environ.get('YAR_API_BASE', 'http://localhost:9621').rstrip('/')


def _workspace() -> str:
    return os.environ.get('YAR_WORKSPACE', 'default')


def _resolve_doc_path(doc_id: str) -> str:
    api = _api_base()
    workspace = _workspace()
    prefix = f'{workspace}/{doc_id}/'
    resp = httpx.get(f'{api}/s3/list', params={'prefix': prefix}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    objects = data.get('objects') or data.get('items') or []
    for entry in objects:
        key = entry.get('key') if isinstance(entry, dict) else entry
        if key and Path(key).name.startswith('original.'):
            return key
    raise SystemExit(f'No original.* object found under {prefix}')


def _download_doc(key: str) -> bytes:
    api = _api_base()
    resp = httpx.get(f'{api}/s3/content/{key}', timeout=120.0)
    resp.raise_for_status()
    return resp.content


def _render_one_page(doc_bytes: bytes, key: str, page_num: int) -> RenderedPage:
    suffix = Path(key).suffix.lower()
    is_office = suffix in {'.doc', '.docx', '.odt', '.rtf', '.ppt', '.pptx', '.ppsx', '.odp'}
    if is_office:
        print('[test] converting office document to PDF via soffice...')
        doc_bytes = _convert_office_document_to_pdf(doc_bytes, filename=Path(key).name, mime_type=None)

    with tempfile.TemporaryDirectory(prefix='probe-prompts-') as tmp_dir:
        pdf_path = Path(tmp_dir) / 'input.pdf'
        pdf_path.write_bytes(doc_bytes)
        rendered = _render_pdf_page_range(pdf_path, page_num, page_num, page_num)
        if not rendered:
            raise SystemExit(f'Could not render page {page_num}')
        return rendered[0]


def _build_payload(model: str, page: RenderedPage, prompt_text: str) -> dict[str, Any]:
    image_b64 = base64.b64encode(page.image_bytes).decode()
    full_prompt = (
        f'{prompt_text}\n\n'
        f'Wrap your output with the marker: <!-- PAGE {page.page_number} -->.\n'
        f'If the page has no readable text, return the marker followed by {NO_TEXT_DETECTED_SENTINEL}.'
    )
    return {
        'model': model,
        'temperature': 0,
        'max_tokens': 4096,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': full_prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{page.media_type};base64,{image_b64}'},
                    },
                ],
            }
        ],
    }


async def _try_prompt(client: Any, model: str, page: RenderedPage, label: str, prompt_text: str) -> None:
    try:
        response = await client.chat.completions.create(**_build_payload(model, page, prompt_text))
    except Exception as exc:
        print(f'  [{label:30s}] EXCEPTION: {type(exc).__name__}: {exc}')
        return

    choices = getattr(response, 'choices', None) or []
    if not choices:
        print(f'  [{label:30s}] no choices in response')
        return

    choice = choices[0]
    finish = getattr(choice, 'finish_reason', None)
    msg = getattr(choice, 'message', None)
    content = getattr(msg, 'content', None) if msg else None
    refusal = getattr(msg, 'refusal', None) if msg else None
    chars = len(content) if content else 0

    flag = 'OK' if chars > 100 else ('PARTIAL' if chars > 0 else 'BLOCKED')
    print(f'  [{label:30s}] {flag:8s} finish={finish!s:18s} chars={chars:>6d}'
          + (f' refusal={refusal[:50]!r}' if refusal else ''))


async def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    doc_id, page_str = sys.argv[1], sys.argv[2]
    page_num = int(page_str)

    base_url = os.environ.get('LLM_BINDING_HOST') or os.environ.get('VISION_BINDING_HOST')
    api_key = os.environ.get('LLM_BINDING_API_KEY') or os.environ.get('VISION_BINDING_API_KEY')
    model = os.environ.get('VISION_MODEL', 'salmon')

    print(f'[test] doc_id={doc_id} page={page_num} model={model}')
    key = _resolve_doc_path(doc_id)
    print(f'[test] key: {key}')

    doc_bytes = _download_doc(key)
    print(f'[test] downloaded {len(doc_bytes):,} bytes')

    page = _render_one_page(doc_bytes, key, page_num)
    print(f'[test] rendered page {page_num} ({len(page.image_bytes):,} bytes)\n')
    print(f'Trying {len(PROMPT_PANEL)} prompt framings against page {page_num}:')
    print('-' * 100)

    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    try:
        for label, prompt_text in PROMPT_PANEL:
            await _try_prompt(client, model, page, label, prompt_text)
    finally:
        await client.close()

    print('\nLegend: OK=>100 chars  PARTIAL=1-100 chars  BLOCKED=0 chars')


if __name__ == '__main__':
    asyncio.run(main())
