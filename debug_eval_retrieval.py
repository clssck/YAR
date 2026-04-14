#!/usr/bin/env python3
"""Debug retrieval for eval questions.

For each question in the eval CSV, runs two checks:
  1. /query/data  — full RAG retrieval (entities + chunks), no LLM generation
  2. /search      — BM25 keyword search using terms from the expected answer

Prints a per-question diagnostic showing whether retrieval found relevant
chunks and which source documents were hit.

Usage:
    uv run python debug_eval_retrieval.py
    uv run python debug_eval_retrieval.py --input-csv qa_eval_for_runner.csv --api-url http://localhost:9621
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import httpx

API_URL = 'http://localhost:9621'
INPUT_CSV = 'qa_eval_for_runner.csv'
TIMEOUT = 120.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Debug eval retrieval quality')
    p.add_argument('--input-csv', type=Path, default=INPUT_CSV)
    p.add_argument('--api-url', type=str, default=API_URL)
    p.add_argument('--api-key', type=str, default='')
    p.add_argument('--mode', type=str, default='mix')
    p.add_argument('--chunk-top-k', type=int, default=20)
    return p.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    header_idx = next(i for i, r in enumerate(rows) if r == ['question', 'expectedResponse'])
    out = []
    for row in rows[header_idx + 1:]:
        if len(row) >= 2 and row[0].strip():
            out.append({'question': row[0].strip(), 'expected': row[1].strip()})
    return out


def headers(api_key: str) -> dict[str, str]:
    h: dict[str, str] = {'Content-Type': 'application/json'}
    if api_key:
        h['X-API-Key'] = api_key
    return h


def extract_keywords(text: str, n: int = 6) -> list[str]:
    """Pull distinctive terms from expected answer for BM25 probe."""
    stop = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
        'should', 'may', 'might', 'must', 'can', 'could', 'of', 'in', 'to',
        'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into',
        'through', 'during', 'before', 'after', 'and', 'but', 'or', 'nor',
        'not', 'so', 'yet', 'both', 'either', 'neither', 'each', 'every',
        'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'only', 'own', 'same', 'than', 'too', 'very', 'that', 'this',
        'these', 'those', 'it', 'its', 'what', 'which', 'who', 'whom',
        'how', 'when', 'where', 'why', 'if', 'then', 'also', 'just',
    }
    words = re.findall(r'[A-Za-z0-9][\w\-]*', text)
    seen: set[str] = set()
    keywords: list[str] = []
    for w in words:
        low = w.lower()
        if low not in stop and low not in seen and len(w) > 2:
            seen.add(low)
            keywords.append(w)
        if len(keywords) >= n:
            break
    return keywords


def keyword_hit_check(chunks: list[dict], expected: str) -> dict[str, bool]:
    """Check whether key terms from expected answer appear in any chunk."""
    keywords = extract_keywords(expected, n=8)
    all_text = ' '.join(c.get('content', '') for c in chunks).lower()
    return {kw: kw.lower() in all_text for kw in keywords}


def truncate(s: str, n: int = 120) -> str:
    return s[:n] + '...' if len(s) > n else s


def print_separator():
    print('=' * 100)


def run_query_data(client: httpx.Client, api_url: str, api_key: str, question: str, mode: str, chunk_top_k: int) -> dict:
    """Hit /query/data to get raw retrieval without LLM generation."""
    resp = client.post(
        f'{api_url.rstrip("/")}/query/data',
        json={
            'query': question,
            'mode': mode,
            'chunk_top_k': chunk_top_k,
            'include_chunk_content': True,
            'include_references': True,
        },
        headers=headers(api_key),
    )
    resp.raise_for_status()
    return resp.json()


def run_bm25_search(client: httpx.Client, api_url: str, api_key: str, query: str, limit: int = 5) -> dict:
    """Hit /search for BM25 keyword lookup."""
    resp = client.get(
        f'{api_url.rstrip("/")}/search',
        params={'q': query, 'limit': limit},
        headers=headers(api_key),
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    args = parse_args()
    cases = load_csv(args.input_csv)
    print(f'Loaded {len(cases)} questions from {args.input_csv}')
    print(f'API: {args.api_url}  mode: {args.mode}  chunk_top_k: {args.chunk_top_k}')
    print()

    # Health check
    try:
        health = httpx.get(f'{args.api_url.rstrip("/")}/health', headers=headers(args.api_key), timeout=20).json()
        print(f'Server: {health.get("status")}  Model: {health.get("configuration", {}).get("llm_model")}')
    except Exception as e:
        print(f'WARNING: Health check failed: {e}')
    print()

    summary: list[dict] = []

    with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=30.0)) as client:
        for idx, case in enumerate(cases, 1):
            question = case['question']
            expected = case['expected']

            print_separator()
            print(f'[{idx}/{len(cases)}] {truncate(question, 95)}')
            print(f'  Expected: {truncate(expected, 95)}')
            print()

            # --- 1. RAG retrieval via /query/data ---
            try:
                result = run_query_data(client, args.api_url, args.api_key, question, args.mode, args.chunk_top_k)
                data = result.get('data', {})
                metadata = result.get('metadata', {})
                chunks = data.get('chunks', [])
                entities = data.get('entities', [])
                relations = data.get('relationships', [])
                references = data.get('references', [])

                print(f'  RAG Retrieval:')
                print(f'    Chunks: {len(chunks)}  Entities: {len(entities)}  Relations: {len(relations)}  Refs: {len(references)}')

                # Show keywords the system extracted
                hl = metadata.get('hl_keywords', [])
                ll = metadata.get('ll_keywords', [])
                if hl or ll:
                    print(f'    HL keywords: {hl}')
                    print(f'    LL keywords: {ll}')

                # Source files hit
                source_files: set[str] = set()
                for c in chunks:
                    fp = c.get('file_path') or c.get('metadata', {}).get('file_path', '')
                    if fp:
                        source_files.add(fp.split('/')[-1] if '/' in fp else fp)
                for ref in references:
                    fp = ref.get('file_path', '')
                    if fp:
                        source_files.add(fp.split('/')[-1] if '/' in fp else fp)
                if source_files:
                    print(f'    Source files: {sorted(source_files)}')
                else:
                    print(f'    Source files: NONE')

                # Keyword presence check
                hits = keyword_hit_check(chunks, expected)
                found = [k for k, v in hits.items() if v]
                missed = [k for k, v in hits.items() if not v]
                print(f'    Expected-answer keywords in chunks:')
                print(f'      Found:  {found}')
                print(f'      Missed: {missed}')

                # Top 3 chunk previews
                if chunks:
                    print(f'    Top chunks:')
                    for i, c in enumerate(chunks[:3]):
                        content = c.get('content', '')
                        print(f'      [{i+1}] {truncate(content, 100)}')

                retrieval_ok = len(chunks) > 0 and len(found) > len(missed)

            except Exception as e:
                print(f'  RAG Retrieval FAILED: {e}')
                chunks = []
                retrieval_ok = False
                found = []
                missed = []
                source_files = set()

            # --- 2. BM25 keyword probe ---
            keywords = extract_keywords(expected, n=6)
            bm25_query = ' '.join(keywords)
            print()
            print(f'  BM25 Probe: "{bm25_query}"')
            try:
                bm25 = run_bm25_search(client, args.api_url, args.api_key, bm25_query, limit=5)
                bm25_results = bm25.get('results', [])
                print(f'    Hits: {len(bm25_results)}')
                for i, r in enumerate(bm25_results[:3]):
                    score = r.get('score', 0)
                    content = r.get('content', '')
                    fp = r.get('file_path', '')
                    fname = fp.split('/')[-1] if '/' in fp else fp
                    print(f'      [{i+1}] score={score:.3f}  file={fname}')
                    print(f'          {truncate(content, 100)}')
                bm25_ok = len(bm25_results) > 0
            except Exception as e:
                print(f'    BM25 FAILED: {e}')
                bm25_ok = False
                bm25_results = []

            # --- Verdict ---
            if retrieval_ok:
                verdict = 'OK'
            elif bm25_ok and not retrieval_ok:
                verdict = 'BM25_ONLY — chunks exist but RAG missed them'
            elif not bm25_ok and len(chunks) > 0:
                verdict = 'WRONG_CHUNKS — retrieved chunks lack expected content'
            else:
                verdict = 'NOT_FOUND — content likely not ingested'

            print()
            print(f'  >> VERDICT: {verdict}')

            summary.append({
                'idx': idx,
                'question': truncate(question, 80),
                'rag_chunks': len(chunks),
                'rag_keyword_found': len(found),
                'rag_keyword_missed': len(missed),
                'bm25_hits': len(bm25_results) if bm25_ok else 0,
                'sources': sorted(source_files) if source_files else [],
                'verdict': verdict,
            })

    # --- Summary table ---
    print()
    print_separator()
    print('SUMMARY')
    print_separator()
    print(f'{"#":<3} {"Verdict":<50} {"Chunks":<8} {"KW Hit":<8} {"BM25":<6}  Question')
    print('-' * 100)
    for s in summary:
        print(f'{s["idx"]:<3} {s["verdict"]:<50} {s["rag_chunks"]:<8} {s["rag_keyword_found"]:<8} {s["bm25_hits"]:<6}  {s["question"]}')

    ok = sum(1 for s in summary if s['verdict'] == 'OK')
    print()
    print(f'Retrieval quality: {ok}/{len(summary)} questions have good chunk coverage')

    not_found = [s for s in summary if 'NOT_FOUND' in s['verdict']]
    if not_found:
        print(f'\nLikely missing documents ({len(not_found)}):')
        for s in not_found:
            print(f'  - Q{s["idx"]}: {s["question"]}')


if __name__ == '__main__':
    main()
