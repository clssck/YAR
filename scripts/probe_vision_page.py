#!/usr/bin/env python
"""Probe the vision pipeline for a single doc + specific pages.

Pulls the original document from RustFS via YAR's /s3/content endpoint,
renders the requested pages with the same pdftoppm settings the production
pipeline uses, sends them to the configured vision model through LiteLLM,
and dumps the FULL response — including finish_reason, refusal text, raw
content — so you can see exactly why the pipeline got an empty result.

Usage:
    uv run python scripts/probe_vision_page.py <s3_doc_id> <pages>

Examples:
    # one page
    uv run python scripts/probe_vision_page.py doc_745784206c6e7c9fce2e65eb9d87296a 9

    # multiple pages, comma or range
    uv run python scripts/probe_vision_page.py doc_745784206c6e7c9fce2e65eb9d87296a 9,10,25-28

Optional env overrides (otherwise read from start.sh/.env):
    YAR_API_BASE        default http://localhost:9621
    LLM_BINDING_HOST    LiteLLM base URL (e.g. http://172.28.0.1:4000/v1)
    LLM_BINDING_API_KEY LiteLLM master key
    VISION_MODEL        model alias (default: salmon)
    YAR_WORKSPACE       default 'default'
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

# Make sure we can import yar.document.* without spinning up the API stack.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yar.document.vision_adapter import (
    BATCH_EXTRACTION_PROMPT,
    NO_TEXT_DETECTED_SENTINEL,
    RenderedPage,
    _convert_office_document_to_pdf,
    _render_pdf_page_range,
    _render_pdf_pages,
)
from yar.llm.openai import create_openai_async_client


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


def _api_base() -> str:
    return os.environ.get('YAR_API_BASE', 'http://localhost:9621').rstrip('/')


def _workspace() -> str:
    return os.environ.get('YAR_WORKSPACE', 'default')


def _resolve_doc_path(doc_id: str) -> str:
    """Find the original.* object key under workspace/<doc_id>/ via /s3/list."""
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


def _render_pages(doc_bytes: bytes, key: str, pages: list[int]) -> list[RenderedPage]:
    """Render the requested pages using the SAME settings the pipeline uses."""
    suffix = Path(key).suffix.lower()
    is_office = suffix in {'.doc', '.docx', '.odt', '.rtf', '.ppt', '.pptx', '.ppsx', '.odp'}

    if is_office:
        print('[probe] converting office document to PDF via soffice...')
        doc_bytes = _convert_office_document_to_pdf(doc_bytes, filename=Path(key).name, mime_type=None)

    if not pages:
        return _render_pdf_pages(doc_bytes)

    with tempfile.TemporaryDirectory(prefix='probe-vision-') as tmp_dir:
        pdf_path = Path(tmp_dir) / 'input.pdf'
        pdf_path.write_bytes(doc_bytes)
        first = min(pages)
        last = max(pages)
        rendered = _render_pdf_page_range(pdf_path, first, last, last)
        wanted = set(pages)
        return [page for page in rendered if page.page_number in wanted]


def _build_payload(model: str, page: RenderedPage) -> dict[str, Any]:
    """Single-page version of the production prompt."""
    import base64

    image_b64 = base64.b64encode(page.image_bytes).decode()
    prompt = (
        f'{BATCH_EXTRACTION_PROMPT}\n\n'
        f'This batch covers page {page.page_number} of {page.total_pages}.\n'
        f'Return exactly one section for the marker: <!-- PAGE {page.page_number} -->.\n'
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
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{page.media_type};base64,{image_b64}'},
                    },
                ],
            }
        ],
    }


async def _probe_page(model: str, base_url: str | None, api_key: str | None, page: RenderedPage) -> None:
    print(f'\n=== Page {page.page_number} ===')
    print(f'image: {len(page.image_bytes):,} bytes ({page.media_type})')

    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    try:
        payload = _build_payload(model, page)
        # We bypass the SDK helpers and send the raw payload to surface every field.
        response = await client.chat.completions.create(**payload)
    finally:
        await client.close()

    choices = getattr(response, 'choices', None) or []
    if not choices:
        print('NO CHOICES IN RESPONSE')
        print(json.dumps(getattr(response, 'model_dump', lambda: {})(), indent=2, default=str))
        return

    choice = choices[0]
    finish = getattr(choice, 'finish_reason', None)
    msg = getattr(choice, 'message', None)
    content = getattr(msg, 'content', None) if msg else None
    refusal = getattr(msg, 'refusal', None) if msg else None
    usage = getattr(response, 'usage', None)

    print(f'finish_reason: {finish!r}')
    print(f'refusal:       {refusal!r}')
    if usage is not None:
        print(f'usage:         {usage}')
    print(f'content_chars: {len(content) if content else 0}')
    print('--- content ---')
    if content:
        preview = content if len(content) < 2000 else content[:2000] + f'\n...[truncated, total {len(content)} chars]'
        print(preview)
    else:
        print('(empty)')


async def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    doc_id, pages_spec = sys.argv[1], sys.argv[2]
    pages_wanted = _parse_pages(pages_spec)
    if not pages_wanted:
        sys.exit('No pages parsed from spec.')

    save_dir_env = os.environ.get('PROBE_SAVE_DIR')
    save_dir = Path(save_dir_env) if save_dir_env else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get('LLM_BINDING_HOST') or os.environ.get('VISION_BINDING_HOST')
    api_key = os.environ.get('LLM_BINDING_API_KEY') or os.environ.get('VISION_BINDING_API_KEY')
    model = os.environ.get('VISION_MODEL', 'salmon')

    print(f'[probe] doc_id={doc_id} pages={pages_wanted}')
    print(f'[probe] api_base={_api_base()} workspace={_workspace()}')
    print(f'[probe] model={model} base_url={base_url}')

    key = _resolve_doc_path(doc_id)
    print(f'[probe] resolved key: {key}')

    doc_bytes = _download_doc(key)
    print(f'[probe] downloaded {len(doc_bytes):,} bytes')

    rendered = _render_pages(doc_bytes, key, pages_wanted)
    found_pages = {page.page_number for page in rendered}
    missing = sorted(set(pages_wanted) - found_pages)
    if missing:
        print(f'[probe] WARN: pages not rendered: {missing}')

    if save_dir:
        for page in rendered:
            target = save_dir / f'{doc_id}_page{page.page_number:03d}.jpg'
            target.write_bytes(page.image_bytes)
        print(f'[probe] saved {len(rendered)} rendered pages to {save_dir}')

    for page in sorted(rendered, key=lambda p: p.page_number):
        try:
            await _probe_page(model, base_url, api_key, page)
        except Exception as exc:
            print(f'\n=== Page {page.page_number} ===')
            print(f'EXCEPTION: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    asyncio.run(main())
