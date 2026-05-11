"""Auto-export successful YAR query traces to a Phoenix dataset.

Builds an offline eval baseline from production traces. Pulls recent
``app.query*`` chain spans that meet quality criteria (status:success,
non-empty references, optional minimum precision), turns each into a
dataset row with ``input``, ``output``, ``reference_documents``, and
metadata, then pushes to a named Phoenix dataset.

Why: every production query becomes potential eval data. Once you've
exported a dataset you can run experiments against it, regression-test
prompt variants, or feed it back into the LLM-as-judge runner.

Usage::

    python -m yar.evaluation.phoenix_dataset_export \\
        --project yar-app --name baseline-2026-05 --since 7d --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    project: str = 'yar-app'
    dataset_name: str = 'yar-traces'
    description: str | None = None
    limit: int = 500
    since: timedelta = timedelta(days=7)
    min_references: int = 1
    min_precision: float | None = None
    require_no_failure: bool = True
    require_not_cached: bool = False
    api_key: str | None = None
    base_url: str | None = None


def _require_phoenix_client() -> Any:
    try:
        from phoenix.client import Client
    except Exception as exc:
        raise RuntimeError('phoenix.client is required. Install with `pip install -e .[observability]`.') from exc
    return Client


def _build_client(config: ExportConfig) -> Any:
    Client = _require_phoenix_client()
    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs['base_url'] = config.base_url
    if config.api_key:
        kwargs['api_key'] = config.api_key
    return Client(**kwargs)


def _fetch_query_spans(*, config: ExportConfig, client: Any) -> Any:
    try:
        from phoenix.client.types.spans import SpanQuery
    except Exception as exc:
        raise RuntimeError('phoenix.client.types.spans is unavailable.') from exc

    end_time = datetime.now(timezone.utc)
    start_time = end_time - config.since

    query = SpanQuery().where("name in ('app.query', 'app.query_stream', 'app.query_data')")

    spans = client.spans.get_spans_dataframe(
        query=query,
        project_identifier=config.project,
        start_time=start_time,
        end_time=end_time,
        limit=config.limit,
    )
    return spans


def _attr(row: Any, key: str, *, namespace: str | None = None) -> Any:
    """Read a span attribute that may surface as a flat column or nested dict.

    Phoenix's spans dataframe sometimes returns ``attributes.<ns>.<name>`` as a
    flat column and sometimes as a nested dict under ``attributes.<ns>``. This
    helper checks both and returns ``None`` when neither is populated.
    """
    flat_key = f'attributes.{namespace}.{key}' if namespace else f'attributes.{key}'
    flat_value = row.get(flat_key) if hasattr(row, 'get') else None
    if flat_value is not None:
        try:
            import pandas as pd

            if isinstance(flat_value, float) and pd.isna(flat_value):
                flat_value = None
        except Exception:
            pass
    if flat_value is not None:
        return flat_value
    if namespace is None:
        return None
    nested = row.get(f'attributes.{namespace}') if hasattr(row, 'get') else None
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def _row_meets_criteria(row: Any, config: ExportConfig) -> bool:
    status = _attr(row, 'status', namespace='rag')
    if config.require_no_failure and status != 'success':
        return False
    ref_count = _attr(row, 'reference_count', namespace='rag')
    if isinstance(ref_count, (int, float)) and int(ref_count) < config.min_references:
        return False
    if config.min_precision is not None:
        precision = _attr(row, 'precision', namespace='retrieval')
        if precision is None:
            return False
        try:
            if float(precision) < config.min_precision:
                return False
        except (TypeError, ValueError):
            return False
    if config.require_not_cached:
        cached = _attr(row, 'cache_hit', namespace='llm')
        if isinstance(cached, bool) and cached:
            return False
    return True


def _build_examples(spans: Any, config: ExportConfig) -> list[dict[str, Any]]:
    """Convert eligible span rows into Phoenix dataset examples."""
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError('pandas is required for dataset export.') from exc

    examples: list[dict[str, Any]] = []
    if spans is None or spans.empty:
        return examples

    for _, row in spans.iterrows():
        if not _row_meets_criteria(row, config):
            continue

        # Reconstruct retrieval documents. Phoenix dataframe shape (current):
        # ``attributes.retrieval.documents`` is a list-of-dicts per cell with
        # ``document.content``/``document.id``/``document.score`` keys. Older
        # versions used flattened ``attributes.retrieval.documents.{idx}.*``
        # columns; we fall back to that layout when the list isn't present.
        documents: list[dict[str, Any]] = []
        docs_attr = row.get('attributes.retrieval.documents')
        if isinstance(docs_attr, list):
            for idx, entry in enumerate(docs_attr):
                if not isinstance(entry, dict):
                    continue
                content = entry.get('document.content') or entry.get('content')
                if not content:
                    continue
                documents.append(
                    {
                        'id': str(entry.get('document.id', idx)),
                        'content': str(content),
                        'score': entry.get('document.score'),
                    }
                )
        else:
            for idx in range(50):
                content = row.get(f'attributes.retrieval.documents.{idx}.document.content')
                if content is None or (isinstance(content, float) and pd.isna(content)):
                    break
                documents.append(
                    {
                        'id': str(row.get(f'attributes.retrieval.documents.{idx}.document.id', idx)),
                        'content': str(content),
                        'score': row.get(f'attributes.retrieval.documents.{idx}.document.score'),
                    }
                )

        input_value = row.get('attributes.input.value', '')
        output_value = row.get('attributes.output.value', '')

        example = {
            'input': {
                'query': str(input_value) if input_value is not None else '',
                'mode': str(_attr(row, 'query_mode', namespace='rag') or ''),
            },
            'output': {
                'answer': str(output_value) if output_value is not None else '',
            },
            'metadata': {
                'span_id': str(row.get('context.span_id', '')),
                'trace_id': str(row.get('context.trace_id', '')),
                'effective_mode': str(_attr(row, 'effective_query_mode', namespace='rag') or ''),
                'reference_count': int(_attr(row, 'reference_count', namespace='rag') or 0),
                'cited_count': int(_attr(row, 'cited_count', namespace='retrieval') or 0),
                'precision': float(_attr(row, 'precision', namespace='retrieval') or 0.0),
                'fingerprint': str(_attr(row, 'fingerprint', namespace='retrieval') or ''),
                'documents': documents,
            },
        }
        examples.append(example)

    return examples


def export_traces_to_dataset(
    *,
    project: str = 'yar-app',
    dataset_name: str = 'yar-traces',
    description: str | None = None,
    limit: int = 500,
    since: timedelta | None = None,
    min_references: int = 1,
    min_precision: float | None = None,
    require_no_failure: bool = True,
    require_not_cached: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Export filtered YAR query traces into a Phoenix dataset.

    Returns a summary dict with keys ``examples`` (count), ``dataset`` (the
    Phoenix dataset object when push succeeded), and ``output_path`` (when
    a local JSON dump was requested).
    """
    config = ExportConfig(
        project=project,
        dataset_name=dataset_name,
        description=description,
        limit=limit,
        since=since or timedelta(days=7),
        min_references=min_references,
        min_precision=min_precision,
        require_no_failure=require_no_failure,
        require_not_cached=require_not_cached,
        api_key=api_key,
        base_url=base_url,
    )

    client = _build_client(config)
    spans = _fetch_query_spans(config=config, client=client)
    examples = _build_examples(spans, config)
    summary: dict[str, Any] = {'examples': len(examples), 'dataset': None, 'output_path': None}

    if not examples:
        logger.warning('No eligible query traces matched the criteria.')
        return summary

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as fh:
            json.dump(examples, fh, ensure_ascii=False, indent=2, default=str)
        summary['output_path'] = output_path
        logger.info('Wrote %d examples to %s', len(examples), output_path)

    try:
        dataset = client.datasets.create_dataset(
            name=config.dataset_name,
            dataset_description=(
                config.description or f'YAR query traces from {config.project} ({len(examples)} examples)'
            ),
            inputs=[ex['input'] for ex in examples],
            outputs=[ex['output'] for ex in examples],
            metadata=[ex['metadata'] for ex in examples],
        )
        summary['dataset'] = dataset
        logger.info('Created Phoenix dataset %s with %d examples', config.dataset_name, len(examples))
    except Exception as exc:
        logger.warning('Failed to push dataset to Phoenix: %s', exc)

    return summary


def _parse_since(raw: str) -> timedelta:
    raw = raw.strip().lower()
    if raw.endswith('h'):
        return timedelta(hours=float(raw[:-1]))
    if raw.endswith('d'):
        return timedelta(days=float(raw[:-1]))
    if raw.endswith('m'):
        return timedelta(minutes=float(raw[:-1]))
    return timedelta(days=float(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Export YAR query traces to a Phoenix dataset.')
    parser.add_argument('--project', default=os.getenv('YAR_TRACE_PROJECT', 'yar-app'))
    parser.add_argument('--name', default='yar-traces', help='Phoenix dataset name.')
    parser.add_argument('--description', default=None)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--since', default='7d', help='Lookback window: 24h, 7d, etc.')
    parser.add_argument('--min-references', type=int, default=1)
    parser.add_argument('--min-precision', type=float, default=None)
    parser.add_argument('--allow-failures', action='store_true', help='Include traces that failed.')
    parser.add_argument('--exclude-cached', action='store_true', help='Skip traces served from LLM cache.')
    parser.add_argument('--output', default=None, help='Optional local JSON dump path.')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    summary = export_traces_to_dataset(
        project=args.project,
        dataset_name=args.name,
        description=args.description,
        limit=args.limit,
        since=_parse_since(args.since),
        min_references=args.min_references,
        min_precision=args.min_precision,
        require_no_failure=not args.allow_failures,
        require_not_cached=args.exclude_cached,
        output_path=args.output,
    )

    logger.info('Exported %d examples; dataset_pushed=%s', summary['examples'], summary['dataset'] is not None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
