from __future__ import annotations

"""Canonical YAR graph analysis CLI.

This tracked analyzer replaces the old ad-hoc /tmp/yar_graph_analysis.py helper,
which under-counted relation evidence because it only looked up one directed
relationship VDB id. This script reuses graph_evidence_coverage.py's
forward-and-reverse lookup logic as the source of truth.
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from typing import Any

from graph_evidence_coverage import (
    _clean_graph_label,
    _finalize_read_storages,
    _initialize_read_storages,
    _load_relation_vdb_records,
    _make_noop_embedding_func,
    _noop_llm_model,
    _predicate,
    _safe_float,
    _split_source_ids,
    collect_graph_evidence_coverage,
)

from yar.constants import GRAPH_FIELD_SEP
from yar.graph_model import normalize_relation_keyword_terms
from yar.yar import YAR

_COMPOUND_PREDICATE_MARKERS = (',', ' and ', ' then ')


def _split_keywords(value: Any) -> list[str]:
    raw_value = str(value or '').strip()
    if not raw_value:
        return []
    raw_segments = [part.strip() for part in raw_value.split(GRAPH_FIELD_SEP) if part.strip()]
    normalized_terms = normalize_relation_keyword_terms(raw_segments or raw_value, max_keywords=20)
    if normalized_terms:
        return list(normalized_terms)
    return raw_segments


def _primary_predicate(value: Any) -> str:
    keywords = _split_keywords(value)
    if keywords:
        return keywords[0]
    return str(value or '').strip()


def _is_compound_predicate(value: str) -> bool:
    normalized = f' {value.casefold()} '
    return any(marker in normalized for marker in _COMPOUND_PREDICATE_MARKERS)


def _record_has_search_hints(vdb_record: dict[str, Any] | None) -> bool:
    if not isinstance(vdb_record, dict):
        return False
    return 'search_hints:' in str(vdb_record.get('content') or '').casefold()


def _edge_label(edge: dict[str, Any]) -> tuple[str, str]:
    return _clean_graph_label(edge.get('source') or edge.get('src_id')), _clean_graph_label(
        edge.get('target') or edge.get('tgt_id')
    )


async def analyze_graph(args: argparse.Namespace) -> dict[str, Any]:
    yar = YAR(
        working_dir=args.working_dir,
        workspace=args.workspace,
        embedding_func=_make_noop_embedding_func(),
        llm_model_func=_noop_llm_model,
    )
    storages = (yar.relation_chunks, yar.relationships_vdb, yar.chunk_entity_relation_graph)
    await _initialize_read_storages(*storages)
    try:
        coverage = await collect_graph_evidence_coverage(
            yar.chunk_entity_relation_graph,
            yar.relation_chunks,
            yar.relationships_vdb,
            limit=args.limit,
            min_weight=args.min_weight,
        )
        edges = await yar.chunk_entity_relation_graph.get_all_edges()
        nodes = await yar.chunk_entity_relation_graph.get_all_nodes()
        vdb_records_by_pair = await _load_relation_vdb_records(edges, yar.relationships_vdb)

        source_count_histogram: Counter[str] = Counter()
        predicate_histogram: Counter[str] = Counter()
        degree_histogram: Counter[str] = Counter()
        compound_predicate_examples: list[dict[str, Any]] = []
        relation_vdb_search_hint_count = 0

        for edge in edges:
            src, tgt = _edge_label(edge)
            vdb_record = vdb_records_by_pair.get((src, tgt))
            predicate = _predicate(edge, vdb_record)
            primary_predicate = _primary_predicate(predicate)
            predicate_histogram[primary_predicate] += 1
            degree_histogram[src] += 1
            degree_histogram[tgt] += 1

            source_ids = _split_source_ids(edge.get('source_id') or (vdb_record or {}).get('source_id'))
            source_count_histogram[str(len(source_ids))] += 1

            first_keyword = primary_predicate
            if len(compound_predicate_examples) < args.limit and _is_compound_predicate(first_keyword):
                compound_predicate_examples.append(
                    {
                        'src': src,
                        'tgt': tgt,
                        'keywords': predicate,
                        'first_keyword': first_keyword,
                        'weight': _safe_float(edge.get('weight') or (vdb_record or {}).get('weight')),
                    }
                )

            if _record_has_search_hints(vdb_record):
                relation_vdb_search_hint_count += 1

        node_ids = {_clean_graph_label(node.get('entity_id') or node.get('id') or node.get('label')) for node in nodes}
        for node_id in node_ids:
            degree_histogram.setdefault(node_id, 0)

        top_predicates = [
            {'predicate': predicate, 'count': count} for predicate, count in predicate_histogram.most_common(15)
        ]
        top_degree_nodes = [
            {'node': node, 'degree': degree}
            for node, degree in sorted(degree_histogram.items(), key=lambda item: (-item[1], item[0]))[:15]
        ]

        return {
            **coverage,
            'nodes_total': len(nodes),
            'edges_total': len(edges),
            'source_count_histogram': dict(sorted(source_count_histogram.items(), key=lambda item: int(item[0]))),
            'predicate_histogram': dict(sorted(predicate_histogram.items())),
            'top_predicates': top_predicates,
            'top_degree_nodes': top_degree_nodes,
            'compound_predicate_examples': compound_predicate_examples,
            'high_weight_unsupported_examples': coverage.get('high_weight_unsupported', []),
            'relation_vdb_search_hint_coverage': {
                'with_search_hints': relation_vdb_search_hint_count,
                'relations_total': len(edges),
                'coverage_pct': round((relation_vdb_search_hint_count / len(edges) * 100.0) if edges else 0.0, 4),
            },
        }
    finally:
        await _finalize_read_storages(*reversed(storages))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Analyze YAR graph construction quality as JSON.')
    parser.add_argument('--working-dir', default='./rag_storage')
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', ''))
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--min-weight', type=float, default=0.0)
    return parser


def main() -> None:
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        summary = asyncio.run(analyze_graph(build_arg_parser().parse_args()))
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(devnull_fd)
        os.close(stderr_fd)
    try:
        os.write(stdout_fd, (json.dumps(summary, ensure_ascii=False, sort_keys=True) + '\n').encode())
    finally:
        os.close(stdout_fd)


if __name__ == '__main__':
    main()
