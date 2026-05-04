from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yar.constants import GRAPH_FIELD_SEP
from yar.operate import _extract_stored_relation_evidence
from yar.utils import EmbeddingFunc, compute_mdhash_id, make_relation_chunk_key
from yar.yar import YAR


def _clean_graph_label(value: Any) -> str:
    text = str(value or '').strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return text.strip('"')
        return str(loaded)
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_source_ids(raw_source_id: Any) -> list[str]:
    return [part for part in str(raw_source_id or '').split(GRAPH_FIELD_SEP) if part]


def _relation_pair(edge: dict[str, Any]) -> tuple[str, str]:
    return _clean_graph_label(edge.get('source') or edge.get('src_id')), _clean_graph_label(
        edge.get('target') or edge.get('tgt_id')
    )


def _predicate(edge: dict[str, Any], vdb_record: dict[str, Any] | None) -> str:
    value = edge.get('keywords') or edge.get('predicate')
    if not value and vdb_record:
        value = vdb_record.get('keywords') or vdb_record.get('predicate')
    predicate = str(value or 'related_to').strip()
    return predicate or 'related_to'


def _vdb_record_has_evidence(vdb_record: dict[str, Any] | None) -> bool:
    if not vdb_record:
        return False
    evidence_spans = vdb_record.get('evidence_spans')
    if isinstance(evidence_spans, list) and any(str(span).strip() for span in evidence_spans):
        return True
    content = str(vdb_record.get('content') or '')
    for line in content.splitlines():
        if line.casefold().startswith('evidence_spans:'):
            return bool(line.split(':', 1)[1].strip())
    return False


async def _load_by_ids(storage: Any, keys: list[str]) -> dict[str, Any]:
    if storage is None or not keys:
        return {}
    unique_keys = list(dict.fromkeys(keys))
    records_by_key: dict[str, Any] = {}
    get_by_ids = getattr(storage, 'get_by_ids', None)
    if callable(get_by_ids):
        try:
            records = await get_by_ids(unique_keys)
        except NotImplementedError:
            records = None
        else:
            if isinstance(records, dict):
                records_by_key.update(records)
            else:
                records_by_key.update(zip(unique_keys, records or [], strict=False))
    if records_by_key:
        return records_by_key

    get_by_id = getattr(storage, 'get_by_id', None)
    if not callable(get_by_id):
        return records_by_key
    for key in unique_keys:
        records_by_key[key] = await get_by_id(key)
    return records_by_key


async def _load_relation_vdb_records(
    edges: list[dict[str, Any]], relationships_vdb: Any
) -> dict[tuple[str, str], dict[str, Any]]:
    relation_ids: list[str] = []
    ids_by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    for edge in edges:
        src, tgt = _relation_pair(edge)
        forward_id = compute_mdhash_id(src + tgt, prefix='rel-')
        reverse_id = compute_mdhash_id(tgt + src, prefix='rel-')
        ids_by_pair[(src, tgt)] = (forward_id, reverse_id)
        relation_ids.extend((forward_id, reverse_id))

    records_by_id = await _load_by_ids(relationships_vdb, relation_ids)
    records_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, ids in ids_by_pair.items():
        for relation_id in ids:
            record = records_by_id.get(relation_id)
            if isinstance(record, dict):
                records_by_pair[pair] = record
                break
    return records_by_pair


async def collect_graph_evidence_coverage(
    graph_storage: Any,
    relation_chunks_storage: Any,
    relationships_vdb: Any,
    *,
    limit: int | None = 20,
    min_weight: float = 0.0,
) -> dict[str, Any]:
    edges = await graph_storage.get_all_edges()
    vdb_records_by_pair = await _load_relation_vdb_records(edges, relationships_vdb)
    relation_keys = [make_relation_chunk_key(*_relation_pair(edge)) for edge in edges]
    sidecar_records_by_key = await _load_by_ids(relation_chunks_storage, relation_keys)

    predicate_histogram: Counter[str] = Counter()
    relations_with_evidence = 0
    unsupported: list[dict[str, Any]] = []

    for edge in edges:
        src, tgt = _relation_pair(edge)
        pair = (src, tgt)
        vdb_record = vdb_records_by_pair.get(pair)
        predicate_histogram[_predicate(edge, vdb_record)] += 1

        source_ids = _split_source_ids(edge.get('source_id') or (vdb_record or {}).get('source_id'))
        sidecar_record = sidecar_records_by_key.get(make_relation_chunk_key(src, tgt))
        evidence_spans = _extract_stored_relation_evidence(
            sidecar_record if isinstance(sidecar_record, dict) else None,
            source_ids,
            max_spans=1,
        )
        has_evidence = bool(evidence_spans) or _vdb_record_has_evidence(vdb_record)
        if has_evidence:
            relations_with_evidence += 1
            continue

        weight = _safe_float(edge.get('weight') or (vdb_record or {}).get('weight'))
        if weight >= min_weight:
            unsupported.append({'src': src, 'tgt': tgt, 'weight': weight})

    unsupported.sort(key=lambda item: (-item['weight'], item['src'], item['tgt']))
    if limit is not None:
        unsupported = unsupported[: max(limit, 0)]

    relations_total = len(edges)
    evidence_coverage_pct = (relations_with_evidence / relations_total * 100.0) if relations_total else 0.0
    return {
        'relations_total': relations_total,
        'relations_with_evidence': relations_with_evidence,
        'evidence_coverage_pct': round(evidence_coverage_pct, 4),
        'high_weight_unsupported': unsupported,
        'predicate_histogram': dict(sorted(predicate_histogram.items())),
    }


async def _noop_llm_model(*_args: Any, **_kwargs: Any) -> str:
    return ''


def _make_noop_embedding_func() -> EmbeddingFunc:
    embedding_dim = int(os.getenv('EMBEDDING_DIM', '1536'))

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.0] * embedding_dim for _ in texts]

    return EmbeddingFunc(embedding_dim=embedding_dim, func=_embed)


async def _initialize_read_storages(*storages: Any) -> None:
    for storage in storages:
        if storage is not None:
            await storage.initialize()


async def _finalize_read_storages(*storages: Any) -> None:
    for storage in storages:
        finalize = getattr(storage, 'finalize', None)
        if callable(finalize):
            await finalize()


async def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    yar = YAR(
        working_dir=args.working_dir,
        workspace=args.workspace,
        embedding_func=_make_noop_embedding_func(),
        llm_model_func=_noop_llm_model,
    )
    storages = (yar.relation_chunks, yar.relationships_vdb, yar.chunk_entity_relation_graph)
    await _initialize_read_storages(*storages)
    try:
        return await collect_graph_evidence_coverage(
            yar.chunk_entity_relation_graph,
            yar.relation_chunks,
            yar.relationships_vdb,
            limit=args.limit,
            min_weight=args.min_weight,
        )
    finally:
        await _finalize_read_storages(*reversed(storages))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize graph relation evidence coverage as JSON.')
    parser.add_argument('--working-dir', default='./rag_storage')
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', ''))
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--min-weight', type=float, default=0.0)
    return parser


def main() -> None:
    stdout_fd = os.dup(1)
    try:
        summary = asyncio.run(run_cli(build_arg_parser().parse_args()))
        os.write(stdout_fd, (json.dumps(summary, ensure_ascii=False, sort_keys=True) + '\n').encode())
    finally:
        os.close(stdout_fd)


if __name__ == '__main__':
    main()
