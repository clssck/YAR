"""Retrieval-only evaluation scaffold for YAR."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from yar.evaluation.qa_eval_common import (
    DEFAULT_YAR_API_KEY,
    DEFAULT_YAR_API_URL,
    EVAL_DIR,
    build_api_headers,
    fetch_yar_health,
    timestamped_results_path,
)

DEFAULT_RETRIEVAL_CSV = EVAL_DIR / 'qa_eval_with_gold.csv'
DEFAULT_MODES = 'hybrid,mix,naive'
OUTPUT_COLUMNS = [
    'question', 'mode', 'retrieved_count', 'hit@1', 'hit@5', 'hit@20',
    'mrr_chunks', 'recall@20', 'doc_hit@1', 'doc_hit@5', 'doc_mrr',
    'zero_hits', 'latency_ms', 'top1_file_path',
]
CHUNK_METRIC_COLUMNS = ['hit@1', 'hit@5', 'hit@20', 'mrr_chunks', 'recall@20']
DOC_METRIC_COLUMNS = ['doc_hit@1', 'doc_hit@5', 'doc_mrr']


@dataclass(slots=True)
class LabeledQATestCase:
    """One QA row with optional retrieval gold labels."""

    question: str
    expected_response: str = ''
    gold_chunk_ids: set[str] = field(default_factory=set)
    gold_doc_substrings: set[str] = field(default_factory=set)
    notes: str = ''


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _top_k_unique(values: list[str], k: int) -> list[str]:
    return [] if k <= 0 else _unique_in_order(values)[:k]


def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> bool:
    """Was at least one gold ID retrieved in the top-k? Non-positive k returns False; duplicates count first occurrence only."""
    return bool(gold_ids) and any(item in gold_ids for item in _top_k_unique(retrieved_ids, k))


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Fraction of gold IDs found in the top-k. 0.0 when gold_ids is empty; duplicates count first occurrence only."""
    if not gold_ids:
        return 0.0
    return len(set(_top_k_unique(retrieved_ids, k)) & gold_ids) / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Reciprocal rank of the first gold hit. 0.0 if no gold hit appears; duplicates count first occurrence only."""
    if not gold_ids:
        return 0.0
    for rank, item in enumerate(_unique_in_order(retrieved_ids), start=1):
        if item in gold_ids:
            return 1.0 / rank
    return 0.0


def doc_hit_at_k(retrieved_file_paths: list[str], gold_doc_substrings: set[str], k: int) -> bool:
    """Top-k document hit using substring match, e.g. 'alpha.md' matches 's3://bucket/docs/alpha.md'."""
    gold_substrings = {value for value in gold_doc_substrings if value}
    if not gold_substrings or k <= 0:
        return False
    return any(
        gold in file_path
        for file_path in _top_k_unique(retrieved_file_paths, k)
        for gold in gold_substrings
    )


def _doc_mrr(retrieved_file_paths: list[str], gold_doc_substrings: set[str]) -> float:
    gold_substrings = {value for value in gold_doc_substrings if value}
    if not gold_substrings:
        return 0.0
    for rank, file_path in enumerate(_unique_in_order(retrieved_file_paths), start=1):
        if any(gold in file_path for gold in gold_substrings):
            return 1.0 / rank
    return 0.0


def _split_semicolon_values(raw_value: str) -> set[str]:
    return {value.strip() for value in raw_value.split(';') if value.strip()}


def load_labeled_qa_csv(path: Path) -> list[LabeledQATestCase]:
    """Load a QA CSV with question plus optional expectedResponse, gold labels, and notes."""
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'No CSV header found in {path}')
        if 'question' not in {field.strip() for field in reader.fieldnames if field}:
            raise ValueError(f'CSV {path} must include a question column')
        test_cases = [_row_to_test_case(row) for row in reader]
    test_cases = [test_case for test_case in test_cases if test_case is not None]
    if not test_cases:
        raise ValueError(f'No test cases found in {path}')
    return test_cases


def _row_to_test_case(raw_row: dict[str, str] | dict[str, Any]) -> LabeledQATestCase | None:
    row = {str(key).strip(): str(value or '').strip() for key, value in raw_row.items() if key is not None}
    question = row.get('question', '')
    if not question or question.startswith('#'):
        return None
    return LabeledQATestCase(
        question=question,
        expected_response=row.get('expectedResponse', ''),
        gold_chunk_ids=_split_semicolon_values(row.get('gold_chunk_ids', '')),
        gold_doc_substrings={row.get('gold_doc_id', '')} - {''},
        notes=row.get('notes', ''),
    )


async def _query_yar_data(
    client: httpx.AsyncClient,
    *,
    api_url: str,
    api_key: str,
    question: str,
    mode: str,
    chunk_top_k: int,
    response_type: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        f'{api_url.rstrip("/")}/query/data',
        json={'query': question, 'mode': mode, 'chunk_top_k': chunk_top_k, 'response_type': response_type, 'stream': False, 'include_references': False},
        headers=build_api_headers(api_key, {'Content-Type': 'application/json'}),
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    status = payload.get('status')
    if status not in (None, 'success'):
        message = payload.get('message') or f'unexpected status {status!r}'
        raise RuntimeError(f'/query/data failed for mode {mode}: {message}')
    chunks = _payload_chunks(payload)
    chunk_ids = [str(chunk.get('chunk_id') or '').strip() for chunk in chunks]
    file_paths = [str(chunk.get('file_path') or '').strip() for chunk in chunks]
    return {
        'chunk_ids': [chunk_id for chunk_id in chunk_ids if chunk_id],
        'file_paths': [file_path for file_path in file_paths if file_path],
        'zero_hits': _payload_zero_hits(payload),
        'latency_ms': latency_ms,
    }


def _payload_chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get('data')
    chunks = data.get('chunks') if isinstance(data, dict) else None
    return [chunk for chunk in chunks if isinstance(chunk, dict)] if isinstance(chunks, list) else []


def _payload_zero_hits(payload: dict[str, Any]) -> bool | None:
    metadata = payload.get('metadata')
    processing_info = metadata.get('processing_info') if isinstance(metadata, dict) else None
    zero_hits = processing_info.get('zero_hits') if isinstance(processing_info, dict) else None
    return zero_hits if isinstance(zero_hits, bool) else None


def _build_result_row(test_case: LabeledQATestCase, mode: str, retrieval_data: dict[str, Any]) -> dict[str, Any]:
    chunk_ids: list[str] = retrieval_data['chunk_ids']
    file_paths: list[str] = retrieval_data['file_paths']
    row: dict[str, Any] = {
        'question': test_case.question,
        'mode': mode,
        'retrieved_count': len(chunk_ids),
        'zero_hits': retrieval_data['zero_hits'],
        'latency_ms': retrieval_data['latency_ms'],
        'top1_file_path': file_paths[0] if file_paths else '',
    }
    row.update(
        {
            'hit@1': hit_at_k(chunk_ids, test_case.gold_chunk_ids, 1),
            'hit@5': hit_at_k(chunk_ids, test_case.gold_chunk_ids, 5),
            'hit@20': hit_at_k(chunk_ids, test_case.gold_chunk_ids, 20),
            'mrr_chunks': mrr(chunk_ids, test_case.gold_chunk_ids),
            'recall@20': recall_at_k(chunk_ids, test_case.gold_chunk_ids, 20),
        }
        if test_case.gold_chunk_ids
        else dict.fromkeys(CHUNK_METRIC_COLUMNS)
    )
    row.update(
        {
            'doc_hit@1': doc_hit_at_k(file_paths, test_case.gold_doc_substrings, 1),
            'doc_hit@5': doc_hit_at_k(file_paths, test_case.gold_doc_substrings, 5),
            'doc_mrr': _doc_mrr(file_paths, test_case.gold_doc_substrings),
        }
        if test_case.gold_doc_substrings
        else dict.fromkeys(DOC_METRIC_COLUMNS)
    )
    return row


async def run_retrieval_eval(
    *,
    input_csv: Path,
    output_csv: Path,
    api_url: str,
    api_key: str,
    modes: list[str],
    chunk_top_k: int,
    timeout: float,
    response_type: str = 'Multiple Paragraphs',
) -> None:
    """Run retrieval-only evaluation and write one CSV row per question per mode."""
    test_cases = load_labeled_qa_csv(input_csv)
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30.0)) as client:
        for index, test_case in enumerate(test_cases, start=1):
            print(f'[{index}/{len(test_cases)}] {test_case.question[:90]}')
            tasks = [
                _query_yar_data(client, api_url=api_url, api_key=api_key, question=test_case.question, mode=mode, chunk_top_k=chunk_top_k, response_type=response_type)
                for mode in modes
            ]
            responses = await asyncio.gather(*tasks)
            rows.extend(_build_result_row(test_case, mode, response) for mode, response in zip(modes, responses, strict=True))
    _write_results_csv(output_csv, rows)
    print(f'Wrote retrieval CSV: {output_csv}')
    _print_summary(rows, modes)


def _write_results_csv(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: _csv_value(key, row.get(key)) for key in OUTPUT_COLUMNS} for row in rows)


def _csv_value(key: str, value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float):
        return str(round(value)) if key == 'latency_ms' else f'{value:.6f}'
    return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return _mean([float(row[key]) for row in rows if row.get(key) is not None])


def _print_summary(rows: list[dict[str, Any]], modes: list[str]) -> None:
    columns = [('mode', 10), ('n_qs', 5), ('avg_retrieved', 13), ('hit@1', 7), ('hit@5', 7), ('hit@20', 7), ('mrr', 7), ('recall@20', 10), ('doc_hit@1', 10), ('doc_hit@5', 10), ('zero_hits_rate', 15), ('avg_latency_ms', 14)]
    print(_format_table_row([name for name, _ in columns], columns))
    for mode in modes:
        mode_rows = [row for row in rows if row['mode'] == mode]
        values = [
            mode,
            str(len(mode_rows)),
            _summary_float(_metric_mean(mode_rows, 'retrieved_count')),
            _summary_float(_metric_mean(mode_rows, 'hit@1')),
            _summary_float(_metric_mean(mode_rows, 'hit@5')),
            _summary_float(_metric_mean(mode_rows, 'hit@20')),
            _summary_float(_metric_mean(mode_rows, 'mrr_chunks')),
            _summary_float(_metric_mean(mode_rows, 'recall@20')),
            _summary_float(_metric_mean(mode_rows, 'doc_hit@1')),
            _summary_float(_metric_mean(mode_rows, 'doc_hit@5')),
            _summary_float(_metric_mean(mode_rows, 'zero_hits')),
            _summary_int(_metric_mean(mode_rows, 'latency_ms')),
        ]
        print(_format_table_row(values, columns))


def _summary_float(value: float | None) -> str:
    return '' if value is None else f'{value:.2f}'


def _summary_int(value: float | None) -> str:
    return '' if value is None else str(round(value))


def _format_table_row(values: list[str], columns: list[tuple[str, int]]) -> str:
    return '  '.join(value.ljust(width) for value, (_, width) in zip(values, columns, strict=True)).rstrip()


def _parse_modes(raw_modes: str) -> list[str]:
    modes = [mode.strip() for mode in raw_modes.split(',') if mode.strip()]
    if not modes:
        raise argparse.ArgumentTypeError('at least one mode is required')
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run retrieval-only metrics over a labeled YAR QA CSV.', epilog='Typical usage: uv run python yar/evaluation/retrieval_eval.py --chunk-top-k 20')
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_RETRIEVAL_CSV, help='QA CSV with question plus optional gold_chunk_ids and gold_doc_id columns')
    parser.add_argument('--output-csv', type=Path, default=None, help='Output CSV path (default: yar/evaluation/results/retrieval_eval_<timestamp>.csv)')
    parser.add_argument('--api-url', default=DEFAULT_YAR_API_URL, help='YAR API base URL')
    parser.add_argument('--api-key', default=DEFAULT_YAR_API_KEY, help='Optional YAR API key')
    parser.add_argument('--modes', type=_parse_modes, default=_parse_modes(DEFAULT_MODES))
    parser.add_argument('--chunk-top-k', type=int, default=20, help='Chunks to request per query')
    parser.add_argument('--timeout', type=float, default=300.0, help='Per-question timeout')
    parser.add_argument('--response-type', default='Multiple Paragraphs', help='Response type (default: Multiple Paragraphs)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_top_k <= 0:
        raise SystemExit('--chunk-top-k must be greater than 0')
    if not args.input_csv.exists():
        example_path = DEFAULT_RETRIEVAL_CSV.with_suffix('.csv.example')
        raise SystemExit(f'Input CSV not found: {args.input_csv}. Create it from {example_path} or pass --input-csv.')
    try:
        health = fetch_yar_health(args.api_url, args.api_key)
    except Exception as exc:
        print(f'Could not connect to YAR health at {args.api_url.rstrip("/")}/health: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
    configuration = health.get('configuration') if isinstance(health, dict) else None
    model = configuration.get('llm_model', 'unknown') if isinstance(configuration, dict) else 'unknown'
    status = health.get('status', 'unknown') if isinstance(health, dict) else 'unknown'
    print(f'Connected to YAR: {args.api_url.rstrip("/")} ({status})')
    print(f'Model: {model}')
    asyncio.run(run_retrieval_eval(input_csv=args.input_csv, output_csv=args.output_csv or timestamped_results_path('retrieval_eval'), api_url=args.api_url, api_key=args.api_key, modes=args.modes, chunk_top_k=args.chunk_top_k, timeout=args.timeout, response_type=args.response_type))


if __name__ == '__main__':
    main()
