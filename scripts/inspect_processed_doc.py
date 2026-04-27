#!/usr/bin/env python
"""Inspect what the vision pipeline actually extracted for a given doc.

Pulls processed.md from RustFS via YAR's /s3/content endpoint, splits on
the <!-- PAGE N --> markers, and prints a per-page report: char count,
truthy/empty flag, and a content preview. Lets you see at a glance which
pages got real content vs. which came back as guardrail-killed empties.

Usage:
    uv run python scripts/inspect_processed_doc.py <s3_doc_id>

Examples:
    uv run python scripts/inspect_processed_doc.py doc_745784206c6e7c9fce2e65eb9d87296a

Optional env overrides (otherwise inherits from start.sh/.env):
    YAR_API_BASE        default http://localhost:9621
    YAR_WORKSPACE       default 'default'
    PREVIEW_CHARS       per-page preview length (default 200)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx

PAGE_SPLIT_RE = re.compile(r'<!--\s*PAGE\s+(\d+)\s*-->')


def _api_base() -> str:
    return os.environ.get('YAR_API_BASE', 'http://localhost:9621').rstrip('/')


def _workspace() -> str:
    return os.environ.get('YAR_WORKSPACE', 'default')


def _preview_chars() -> int:
    try:
        return int(os.environ.get('PREVIEW_CHARS', '200'))
    except ValueError:
        return 200


def _resolve_processed_key(doc_id: str) -> str:
    """Find the processed.* object key under workspace/<doc_id>/ via /s3/list."""
    api = _api_base()
    workspace = _workspace()
    prefix = f'{workspace}/{doc_id}/'
    resp = httpx.get(f'{api}/s3/list', params={'prefix': prefix}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    objects = data.get('objects') or data.get('items') or []
    for entry in objects:
        key = entry.get('key') if isinstance(entry, dict) else entry
        if key and Path(key).name.startswith('processed.'):
            return key
    raise SystemExit(f'No processed.* object found under {prefix}. Has the doc finished extracting?')


def _download_processed(key: str) -> str:
    api = _api_base()
    resp = httpx.get(f'{api}/s3/content/{key}', timeout=120.0)
    resp.raise_for_status()
    return resp.text


def _split_pages(content: str) -> list[tuple[int, str]]:
    """Split processed.md into (page_number, body) tuples."""
    parts = PAGE_SPLIT_RE.split(content)
    if len(parts) == 1:
        # No markers found - treat as single blob
        return [(0, content)]

    # parts pattern: [preamble, page_num, body, page_num, body, ...]
    pages: list[tuple[int, str]] = []
    preamble = parts[0].strip()
    if preamble:
        pages.append((0, preamble))
    for index in range(1, len(parts), 2):
        page_num = int(parts[index])
        body = parts[index + 1].strip() if index + 1 < len(parts) else ''
        pages.append((page_num, body))
    return pages


def _classify(body: str) -> str:
    chars = len(body)
    if chars == 0:
        return 'EMPTY'
    if chars < 100:
        return 'TINY '
    if chars < 500:
        return 'SHORT'
    return 'OK   '


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    doc_id = sys.argv[1]
    preview = _preview_chars()

    print(f'[inspect] doc_id={doc_id} api_base={_api_base()} workspace={_workspace()}')
    key = _resolve_processed_key(doc_id)
    print(f'[inspect] key: {key}')

    content = _download_processed(key)
    print(f'[inspect] downloaded {len(content):,} chars\n')

    pages = _split_pages(content)
    if not pages:
        print('No pages found in processed.md')
        return

    # Aggregate stats
    total = len(pages)
    counts = {'OK   ': 0, 'SHORT': 0, 'TINY ': 0, 'EMPTY': 0}
    empty_pages: list[int] = []
    short_pages: list[int] = []
    tiny_pages: list[int] = []

    print(f'{"page":>4}  {"flag":<5}  {"chars":>6}  preview')
    print('-' * 80)
    for page_num, body in pages:
        flag = _classify(body)
        counts[flag] += 1
        chars = len(body)
        if flag == 'EMPTY':
            empty_pages.append(page_num)
        elif flag == 'TINY ':
            tiny_pages.append(page_num)
        elif flag == 'SHORT':
            short_pages.append(page_num)
        snippet = body[:preview].replace('\n', ' ').strip()
        if len(body) > preview:
            snippet += f' ... [+{len(body) - preview} chars]'
        print(f'{page_num:>4}  {flag}  {chars:>6}  {snippet or "(empty)"}')

    # Detect missing pages (gaps in the page-number sequence). These are pages the
    # vision model returned empty content for - the stitcher strips them entirely so
    # they don't appear in the listing above. The audit log captures them; this surfaces
    # them here too.
    seen_pages = sorted({n for n, _ in pages if n > 0})
    missing_pages: list[int] = []
    if seen_pages:
        highest = seen_pages[-1]
        missing_pages = [n for n in range(1, highest + 1) if n not in set(seen_pages)]

    print('\n' + '=' * 80)
    expected = (seen_pages[-1] if seen_pages else 0)
    print(
        f'Summary: {expected} expected | {total} present | OK {counts["OK   "]} | '
        f'SHORT {counts["SHORT"]} | TINY {counts["TINY "]} | MISSING {len(missing_pages)}'
    )
    if missing_pages:
        print(f'  Missing pages (stripped after empty extraction): {missing_pages}')
    if tiny_pages:
        print(f'  Tiny pages (<100 chars): {tiny_pages}')
    if short_pages:
        print(f'  Short pages (<500 chars): {short_pages}')

    if missing_pages:
        print('\nTip: probe the missing pages to confirm guardrail vs blank:')
        joined = ','.join(str(p) for p in missing_pages)
        print(f'  uv run python scripts/probe_vision_page.py {doc_id} {joined}')


if __name__ == '__main__':
    main()
