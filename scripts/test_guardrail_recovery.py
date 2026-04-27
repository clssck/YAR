#!/usr/bin/env python
"""End-to-end test of the guardrail recovery pipeline.

Renders a specific page through the SAME code path the production pipeline uses
(_extract_batch_adaptively), so you can verify that:

1. The page does indeed hit content_filter on the original image
2. The image-variant retry kicks in (look for vision_image_variant_recovered warning)
3. The recovered content is actually accurate (not hallucinated)

Prints per-stage outcomes:
  - PRIMARY: what _extract_batch_with_retry returns first
  - ALT-PROMPTS: results from alternative prompt framings (if triggered)
  - IMAGE-VARIANTS: results from rotation / blur / quarter-res (if triggered)
  - FINAL: stitched result with all warnings

Usage:
    uv run python scripts/test_guardrail_recovery.py <s3_doc_id> <page>

Example:
    uv run python scripts/test_guardrail_recovery.py doc_84a1429369c1217187f782a4b41b18de 29

Optional env (inherits from start.sh/.env):
    YAR_API_BASE        default http://localhost:9621
    LLM_BINDING_HOST    LiteLLM base URL
    LLM_BINDING_API_KEY LiteLLM master key
    VISION_MODEL        default 'salmon'
    YAR_WORKSPACE       default 'default'
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yar.document.vision_adapter import (
    _convert_office_document_to_pdf,
    _extract_batch_adaptively,
    _render_pdf_page_range,
)
from yar.llm.openai import create_openai_async_client


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


def _render_page(doc_bytes: bytes, key: str, page_num: int):
    suffix = Path(key).suffix.lower()
    is_office = suffix in {'.doc', '.docx', '.odt', '.rtf', '.ppt', '.pptx', '.ppsx', '.odp'}
    if is_office:
        print('[test] converting office document to PDF via soffice...')
        doc_bytes = _convert_office_document_to_pdf(doc_bytes, filename=Path(key).name, mime_type=None)

    with tempfile.TemporaryDirectory(prefix='probe-recovery-') as tmp_dir:
        pdf_path = Path(tmp_dir) / 'input.pdf'
        pdf_path.write_bytes(doc_bytes)
        rendered = _render_pdf_page_range(pdf_path, page_num, page_num, page_num)
        if not rendered:
            raise SystemExit(f'Could not render page {page_num}')
        return rendered[0]


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
    print(f'[test] resolved: {key}')

    doc_bytes = _download_doc(key)
    print(f'[test] downloaded {len(doc_bytes):,} bytes')

    page = _render_page(doc_bytes, key, page_num)
    print(f'[test] rendered page {page_num} ({len(page.image_bytes):,} bytes)\n')

    print('=' * 90)
    print('Running _extract_batch_adaptively (full production retry chain)')
    print('=' * 90)
    print('Watch for log lines:')
    print('  - "Vision GUARDRAIL BLOCK" -> page hit content_filter on original')
    print('  - "Vision alt-prompt N/2 for page X: N chars" -> alt-prompts attempted')
    print('  - "Vision image-variant rotated-90 for page X: N chars" -> rotation attempted')
    print('  - "vision_image_variant_recovered" warning -> recovered via transform')
    print()

    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    try:
        results, warnings = await _extract_batch_adaptively(client, model, [page])
    finally:
        await client.close()

    print()
    print('=' * 90)
    print('FINAL RESULT')
    print('=' * 90)
    if not results:
        print('(no results)')
        return

    page_result = results[0]
    chars = len(page_result.content)
    print(f'page_number: {page_result.page_number}')
    print(f'content_chars: {chars}')
    print(f'warnings: {len(warnings)}')
    for w in warnings:
        print(f'  - source={w.source}')
        print(f'    msg={w.message}')

    print()
    print('--- content (first 3000 chars) ---')
    if chars == 0:
        print('(empty - all recovery attempts failed)')
    else:
        preview = page_result.content[:3000]
        print(preview)
        if chars > 3000:
            print(f'... [truncated, total {chars} chars]')

    print()
    print('=' * 90)
    print('VERIFICATION CHECKLIST:')
    print('  1. Did you see "GUARDRAIL BLOCK" log? -> confirmed content_filter on original')
    print('  2. Did you see "image-variant rotated-90 for page X: N chars" -> retry triggered')
    print('  3. Is final content > 100 chars? -> recovery succeeded')
    print('  4. Does the content match what is ACTUALLY on the page? -> no hallucination')
    print('  5. vision_image_variant_recovered warning present? -> provenance tagged')


if __name__ == '__main__':
    asyncio.run(main())
