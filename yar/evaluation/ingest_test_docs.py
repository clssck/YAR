#!/usr/bin/env python3
"""
Ingest test documents into YAR for testing.

This script reads text files from a directory and batch-uploads them to
YAR via the /documents/texts API endpoint, then polls track status until
processing completes.

Usage:
    python yar/evaluation/ingest_test_docs.py
    python yar/evaluation/ingest_test_docs.py --input wiki_documents/ --rag-url http://localhost:9621
"""

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_RAG_URL = 'http://localhost:9621'


def _normalize_status_key(status: str) -> str:
    """Normalize document status values from API responses.

    The API may return raw enum names or plain status strings.
    """
    return str(status).split('.')[-1].lower()


def _summarize_track_status(payload: dict[str, object]) -> dict[str, int]:
    """Build normalized status counts from a track-status payload."""
    summary: dict[str, int] = {
        'pending': 0,
        'processing': 0,
        'processed': 0,
        'failed': 0,
    }
    raw_summary_obj = payload.get('status_summary')
    raw_summary = raw_summary_obj if isinstance(raw_summary_obj, dict) else {}
    for raw_status, count in raw_summary.items():
        summary[_normalize_status_key(str(raw_status))] = int(count)

    if not raw_summary:
        documents_obj = payload.get('documents')
        documents = documents_obj if isinstance(documents_obj, list) else []
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            status = _normalize_status_key(str(doc.get('status', '')))
            summary[status] = summary.get(status, 0) + 1

    return summary


async def ingest_documents(
    input_dir: Path,
    rag_url: str,
) -> dict[str, Any]:
    timeout = httpx.Timeout(120.0, connect=30.0)
    api_key = os.getenv('YAR_API_KEY')
    headers = {'X-API-Key': api_key} if api_key else {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Check health
        try:
            health = await client.get(f'{rag_url}/health')
            if health.status_code != 200:
                raise ConnectionError(f'YAR not healthy: {health.status_code}')
        except httpx.ConnectError as e:
            raise ConnectionError(f'Cannot connect to YAR at {rag_url}') from e

        print(f'✓ Connected to YAR at {rag_url}')

        files = list(input_dir.glob('*.txt')) + list(input_dir.glob('*.md'))
        if not files:
            print(f'✗ No .txt or .md files found in {input_dir}')
            return {'documents': 0, 'processed': 0, 'failed': 0, 'elapsed_seconds': 0.0}

        print(f'  Found {len(files)} documents to ingest')

        texts: list[str] = []
        sources: list[str] = []
        for file in sorted(files):
            content = file.read_text()
            texts.append(content)
            sources.append(file.name)
            word_count = len(content.split())
            print(f'    {file.name}: {word_count:,} words')

        print(f'\n  Uploading {len(texts)} documents...')
        start = time.time()

        response = await client.post(
            f'{rag_url}/documents/texts',
            json={'texts': texts, 'file_sources': sources},
            headers=headers,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()

        track_id = str(result.get('track_id', '')).strip()
        if not track_id:
            raise RuntimeError('Upload succeeded but no track_id was returned')
        print(f'  Track ID: {track_id}')

        track_status_url = f'{rag_url}/documents/track_status/{track_id}'
        print('  Waiting for processing to start...')
        await asyncio.sleep(2)

        last_status = ''
        initial_check = True
        processed_count = 0
        failed_count = 0
        max_polls = 360  # 30 minutes at 5s cadence

        for _ in range(max_polls):
            status_response = await client.get(track_status_url, headers=headers)
            status_response.raise_for_status()
            track_payload: dict[str, Any] = status_response.json()
            summary = _summarize_track_status(track_payload)

            pending = summary.get('pending', 0)
            processing = summary.get('processing', 0)
            processed = summary.get('processed', 0)
            failed = summary.get('failed', 0)
            total_seen = len(track_payload.get('documents', []))

            current_status = (
                f'Pending: {pending}, Processing: {processing}, '
                f'Processed: {processed}, Failed: {failed}, Seen: {total_seen}/{len(texts)}'
            )
            if current_status != last_status:
                print(f'    {current_status}')
                last_status = current_status
                processed_count = processed
                failed_count = failed

            if initial_check and total_seen > 0:
                initial_check = False
                print('  Processing started!')

            if not initial_check and pending == 0 and processing == 0 and (processed + failed) >= len(texts):
                break

            await asyncio.sleep(5)
        else:
            raise TimeoutError(f'Timed out waiting for track {track_id} to finish processing')

        elapsed = time.time() - start
        print(f'\n✓ Ingestion complete in {elapsed:.1f}s')
        print(f'  Documents processed: {processed_count}')
        if failed_count:
            print(f'  Documents failed: {failed_count}')
        print(f'  Average: {elapsed / len(texts):.1f}s per document')

        return {
            'documents': len(texts),
            'processed': processed_count,
            'failed': failed_count,
            'elapsed_seconds': elapsed,
            'track_id': track_id,
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description='Ingest test documents into YAR')
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='yar/evaluation/wiki_documents',
        help='Input directory with text files',
    )
    parser.add_argument(
        '--rag-url',
        '-r',
        type=str,
        default=None,
        help=f'YAR API URL (default: {DEFAULT_RAG_URL})',
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    rag_url = args.rag_url or os.getenv('YAR_API_URL', DEFAULT_RAG_URL)

    print('=== YAR Document Ingestion ===')
    print(f'Input: {input_dir}/')
    print(f'RAG URL: {rag_url}')
    print()

    if not input_dir.exists():
        print(f'✗ Input directory not found: {input_dir}')
        print('  Run download_wikipedia.py first:')
        print('    python yar/evaluation/download_wikipedia.py')
        return

    _ = await ingest_documents(input_dir, rag_url)


if __name__ == '__main__':
    asyncio.run(main())
