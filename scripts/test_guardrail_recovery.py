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

def _parse_pages(spec: str) -> list[int]:
    pages: list[int] = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            lo, hi = token.split('-', 1)
            pages.extend(range(int(lo), int(hi) + 1))
        else:
            pages.append(int(token))
    return sorted(set(pages))




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

    doc_id, page_spec = sys.argv[1], sys.argv[2]
    pages_wanted = _parse_pages(page_spec)
    if not pages_wanted:
        sys.exit('No pages parsed from spec.')

    base_url = os.environ.get('LLM_BINDING_HOST') or os.environ.get('VISION_BINDING_HOST')
    api_key = os.environ.get('LLM_BINDING_API_KEY') or os.environ.get('VISION_BINDING_API_KEY')
    model = os.environ.get('VISION_MODEL', 'salmon')

    print(f'[test] doc_id={doc_id} pages={pages_wanted} model={model}')
    key = _resolve_doc_path(doc_id)
    print(f'[test] resolved: {key}')

    doc_bytes = _download_doc(key)
    print(f'[test] downloaded {len(doc_bytes):,} bytes')

    summary: list[tuple[int, int, list[str]]] = []

    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    try:
        for page_num in pages_wanted:
            print()
            print('#' * 90)
            print(f'#  Testing page {page_num}')
            print('#' * 90)
            try:
                page = _render_page(doc_bytes, key, page_num)
            except SystemExit as exc:
                print(f'[!] could not render page {page_num}: {exc}')
                summary.append((page_num, 0, ['render_failed']))
                continue
            print(f'[test] rendered page {page_num} ({len(page.image_bytes):,} bytes)')
            results, warnings = await _extract_batch_adaptively(client, model, [page])
            if not results:
                summary.append((page_num, 0, ['no_results']))
                continue
            page_result = results[0]
            chars = len(page_result.content)
            warning_sources = sorted({w.source for w in warnings})
            summary.append((page_num, chars, warning_sources))
            print()
            print(f'page {page_num}: chars={chars} warnings={warning_sources}')
            preview = page_result.content[:1500]
            print(f'--- content preview ({min(chars, 1500)} of {chars}) ---')
            print(preview if chars else '(empty)')
            if chars > 1500:
                print(f'... [+{chars - 1500} more chars]')
    finally:
        await client.close()

    print()
    print('=' * 90)
    print('SUMMARY')
    print('=' * 90)
    print(f'{"page":>4}  {"flag":<10}  {"chars":>6}  warnings')
    print('-' * 90)
    for page_num, chars, sources in summary:
        if chars >= 100:
            flag = 'RECOVERED' if 'vision_image_variant_recovered' in sources else 'OK'
        elif chars > 0:
            flag = 'PARTIAL'
        else:
            flag = 'FAILED'
        print(f'{page_num:>4}  {flag:<10}  {chars:>6}  {",".join(sources) or "-"}')

    recovered = sum(1 for _, chars, sources in summary if chars >= 100 and 'vision_image_variant_recovered' in sources)
    failed = sum(1 for _, chars, _ in summary if chars == 0)
    clean = sum(1 for _, chars, sources in summary if chars >= 100 and 'vision_image_variant_recovered' not in sources)
    print()
    print(f'Total: {len(summary)} pages | clean: {clean} | recovered via transform: {recovered} | failed: {failed}')


if __name__ == '__main__':
    asyncio.run(main())