from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI

from yar.constants import DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE
from yar.operate import _classify_malformed_relation_record, _process_extraction_result
from yar.prompt import PROMPTS

BARE_DATE_RE = re.compile(
    r'^\d{1,2}(?:st|nd|rd|th)?\s+'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
    r'January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+\d{4})?$',
    re.IGNORECASE,
)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == '':
        return default
    return int(value)


def _csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, '').split(',') if item.strip()]


def _relation_pairs_env(name: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_pair in os.getenv(name, '').split(';'):
        if not raw_pair.strip():
            continue
        if '->' not in raw_pair:
            raise AssertionError(f'{name} entries must use "source->target" format: {raw_pair!r}')
        source, target = raw_pair.split('->', 1)
        pairs.add((source.strip(), target.strip()))
    return pairs


def _workspace() -> str:
    return os.getenv('POSTGRES_WORKSPACE', os.getenv('WORKSPACE', 'default')) or 'default'


async def _connect_postgres() -> asyncpg.Connection:
    load_dotenv(dotenv_path='.env', override=False)
    config = {
        'host': os.getenv('POSTGRES_HOST', '127.0.0.1'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'yar'),
        'password': os.getenv('POSTGRES_PASSWORD', 'yar_pass'),
        'database': os.getenv('POSTGRES_DATABASE', 'yar'),
    }
    try:
        return await asyncpg.connect(**config, timeout=10)
    except (OSError, TimeoutError, asyncpg.PostgresError) as exc:
        pytest.skip(f'PostgreSQL is unavailable for live extraction test: {exc}')
        raise


async def _select_processed_chunks(conn: asyncpg.Connection) -> dict[str, Any]:
    workspace = _workspace()
    chunk_ids = _csv_env('YAR_LIVE_EXTRACTION_CHUNK_IDS')
    single_chunk_id = os.getenv('YAR_LIVE_EXTRACTION_CHUNK_ID', '').strip()
    if single_chunk_id:
        chunk_ids = [single_chunk_id, *[chunk_id for chunk_id in chunk_ids if chunk_id != single_chunk_id]]

    doc_id = os.getenv('YAR_LIVE_EXTRACTION_DOC_ID', '').strip()
    file_path = os.getenv('YAR_LIVE_EXTRACTION_FILE_PATH', '').strip()

    doc: asyncpg.Record | None = None
    chunks: list[asyncpg.Record] = []

    if chunk_ids:
        chunks = list(
            await conn.fetch(
                """
                SELECT id, full_doc_id, chunk_order_index, tokens, content, file_path
                FROM YAR_VDB_CHUNKS
                WHERE workspace=$1 AND id = ANY($2::text[])
                ORDER BY array_position($2::text[], id)
                """,
                workspace,
                chunk_ids,
            )
        )
        if chunks:
            doc_id = str(chunks[0]['full_doc_id'] or doc_id)

    if not chunks and doc_id:
        doc = await conn.fetchrow(
            """
            SELECT id, file_path, status, chunks_count, updated_at
            FROM YAR_DOC_STATUS
            WHERE workspace=$1 AND id=$2
            """,
            workspace,
            doc_id,
        )
        chunks = list(
            await conn.fetch(
                """
                SELECT id, full_doc_id, chunk_order_index, tokens, content, file_path
                FROM YAR_VDB_CHUNKS
                WHERE workspace=$1 AND full_doc_id=$2
                ORDER BY chunk_order_index NULLS LAST, id
                """,
                workspace,
                doc_id,
            )
        )

    if not chunks and file_path:
        doc = await conn.fetchrow(
            """
            SELECT id, file_path, status, chunks_count, updated_at
            FROM YAR_DOC_STATUS
            WHERE workspace=$1 AND file_path=$2
            ORDER BY updated_at DESC NULLS LAST, id
            LIMIT 1
            """,
            workspace,
            file_path,
        )
        chunks = list(
            await conn.fetch(
                """
                SELECT id, full_doc_id, chunk_order_index, tokens, content, file_path
                FROM YAR_VDB_CHUNKS
                WHERE workspace=$1 AND file_path=$2
                ORDER BY chunk_order_index NULLS LAST, id
                """,
                workspace,
                file_path,
            )
        )
        if doc is not None:
            doc_id = str(doc['id'])

    if not chunks:
        doc = await conn.fetchrow(
            """
            SELECT id, file_path, status, chunks_count, updated_at
            FROM YAR_DOC_STATUS
            WHERE workspace=$1 AND status='processed' AND chunks_count > 0
            ORDER BY updated_at DESC NULLS LAST, id
            LIMIT 1
            """,
            workspace,
        )
        if doc is not None:
            doc_id = str(doc['id'])
            chunks = list(
                await conn.fetch(
                    """
                    SELECT id, full_doc_id, chunk_order_index, tokens, content, file_path
                    FROM YAR_VDB_CHUNKS
                    WHERE workspace=$1 AND full_doc_id=$2
                    ORDER BY chunk_order_index NULLS LAST, id
                    """,
                    workspace,
                    doc_id,
                )
            )

    if chunks and doc is None:
        doc = await conn.fetchrow(
            """
            SELECT id, file_path, status, chunks_count, updated_at
            FROM YAR_DOC_STATUS
            WHERE workspace=$1 AND id=$2
            """,
            workspace,
            str(chunks[0]['full_doc_id']),
        )

    if not chunks:
        pytest.skip(
            'No processed chunks found. Set YAR_LIVE_EXTRACTION_DOC_ID, '
            'YAR_LIVE_EXTRACTION_FILE_PATH, or YAR_LIVE_EXTRACTION_CHUNK_ID(S) '
            'to point the live tuna extraction test at an ingested document.'
        )

    offset = max(_int_env('YAR_LIVE_EXTRACTION_CHUNK_OFFSET', 0), 0)
    limit = _int_env('YAR_LIVE_EXTRACTION_CHUNK_LIMIT', 0)
    selected = chunks[offset:]
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        pytest.skip(f'Chunk selection is empty after offset={offset}, limit={limit}.')

    return {
        'workspace': workspace,
        'doc': dict(doc) if doc is not None else None,
        'chunks': [
            {
                'id': row['id'],
                'full_doc_id': row['full_doc_id'],
                'chunk_order_index': row['chunk_order_index'],
                'tokens': row['tokens'],
                'content': row['content'],
                'file_path': row['file_path'],
            }
            for row in selected
        ],
        'selection': {
            'requested_doc_id': os.getenv('YAR_LIVE_EXTRACTION_DOC_ID', ''),
            'requested_file_path': os.getenv('YAR_LIVE_EXTRACTION_FILE_PATH', ''),
            'requested_chunk_ids': chunk_ids,
            'offset': offset,
            'limit': limit,
            'available_chunks': len(chunks),
            'selected_chunks': len(selected),
        },
    }


async def _fetch_stored_extraction(
    conn: asyncpg.Connection,
    workspace: str,
    chunk_ids: list[str],
) -> dict[str, Any]:
    entities = await conn.fetch(
        """
        SELECT entity_name, entity_type, content, chunk_ids, file_path
        FROM YAR_VDB_ENTITY
        WHERE workspace=$1 AND chunk_ids::text[] && $2::text[]
        ORDER BY entity_name
        """,
        workspace,
        chunk_ids,
    )
    relations = await conn.fetch(
        """
        SELECT source_id, target_id, content, chunk_ids, file_path
        FROM YAR_VDB_RELATION
        WHERE workspace=$1 AND chunk_ids::text[] && $2::text[]
        ORDER BY source_id, target_id
        """,
        workspace,
        chunk_ids,
    )
    entity_degrees = await conn.fetch(
        """
        WITH selected_entities AS (
            SELECT DISTINCT entity_name
            FROM YAR_VDB_ENTITY
            WHERE workspace=$1 AND chunk_ids::text[] && $2::text[]
        ),
        relation_endpoints AS (
            SELECT source_id AS entity_name
            FROM YAR_VDB_RELATION
            WHERE workspace=$1 AND chunk_ids::text[] && $2::text[]
            UNION ALL
            SELECT target_id AS entity_name
            FROM YAR_VDB_RELATION
            WHERE workspace=$1 AND chunk_ids::text[] && $2::text[]
        ),
        degree_counts AS (
            SELECT entity_name, COUNT(*) AS degree
            FROM relation_endpoints
            GROUP BY entity_name
        )
        SELECT selected_entities.entity_name, COALESCE(degree_counts.degree, 0) AS degree
        FROM selected_entities
        LEFT JOIN degree_counts USING (entity_name)
        ORDER BY degree, selected_entities.entity_name
        """,
        workspace,
        chunk_ids,
    )
    return {
        'entities': [dict(row) for row in entities],
        'relations': [dict(row) for row in relations],
        'entity_degrees': [dict(row) for row in entity_degrees],
    }


def _build_extraction_prompts(chunk_text: str) -> tuple[str, str]:
    entity_types = list(DEFAULT_ENTITY_TYPES)
    language = os.getenv('YAR_LIVE_EXTRACTION_LANGUAGE', DEFAULT_SUMMARY_LANGUAGE)

    example_context = {
        'tuple_delimiter': PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        'completion_delimiter': PROMPTS['DEFAULT_COMPLETION_DELIMITER'],
        'entity_types': ', '.join(entity_types),
        'language': language,
    }
    examples = '\n'.join(PROMPTS['entity_extraction_examples']).format(**example_context)

    context = {
        'tuple_delimiter': PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        'completion_delimiter': PROMPTS['DEFAULT_COMPLETION_DELIMITER'],
        'entity_types': ','.join(entity_types),
        'examples': examples,
        'language': language,
    }
    system_prompt = PROMPTS['entity_extraction_system_prompt'].format(**context)
    user_prompt = PROMPTS['entity_extraction_user_prompt'].format(**{**context, 'input_text': chunk_text})
    return system_prompt, user_prompt


def _build_tuna_client() -> tuple[OpenAI, str]:
    load_dotenv(dotenv_path='.env', override=False)
    client = OpenAI(
        base_url=os.getenv('YAR_LIVE_TUNA_BASE_URL', 'http://127.0.0.1:4000/v1'),
        api_key=os.getenv('LITELLM_MASTER_KEY', os.getenv('OPENAI_API_KEY', 'sk-litellm-master-key')),
        timeout=float(os.getenv('YAR_LIVE_TUNA_TIMEOUT', '180')),
        max_retries=0,
    )
    return client, os.getenv('YAR_LIVE_TUNA_MODEL', 'tuna')


def _call_tuna(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content if response.choices else None
    except (APIConnectionError, APITimeoutError) as exc:
        pytest.skip(f'LiteLLM/tuna endpoint is unavailable for live extraction test: {exc}')
        raise

    if not content:
        pytest.fail('LiteLLM/tuna returned an empty extraction response')
        raise AssertionError('LiteLLM/tuna returned an empty extraction response')
    return content


def _snapshot(
    nodes: dict[str, list[dict[str, Any]]], edges: dict[tuple[str, str], list[dict[str, Any]]]
) -> dict[str, Any]:
    return {
        'entities': [
            {
                'name': name,
                'type': records[0].get('entity_type') if records else None,
                'description': records[0].get('description') if records else None,
            }
            for name, records in sorted(nodes.items())
        ],
        'relations': [
            {
                'source': source,
                'target': target,
                'keywords': records[0].get('keywords') if records else None,
                'description': records[0].get('description') if records else None,
            }
            for (source, target), records in sorted(edges.items())
        ],
    }


def _malformed_relation_records(
    raw_output: str,
    chunk_id: str = 'live-output',
    file_path: str = 'unknown_source',
) -> list[dict[str, Any]]:
    malformed = []
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    for line in raw_output.splitlines():
        record = line.strip()
        if not record.startswith('relation'):
            continue
        fields = record.split(tuple_delimiter)
        diagnostic = _classify_malformed_relation_record(fields, chunk_id, file_path)
        if diagnostic is not None:
            malformed.append({'record': record, **diagnostic.to_dict()})
    return malformed


def _orphan_entity_names(
    entity_names: set[str],
    relation_pairs: set[tuple[str, str]],
) -> list[str]:
    relation_endpoints = {endpoint for pair in relation_pairs for endpoint in pair}
    return sorted(entity_names - relation_endpoints)


@pytest.mark.offline
def test_malformed_relation_records_classifies_action_verb_target_slot():
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    raw_output = (
        f'relation{tuple_delimiter}Obeya{tuple_delimiter}supports launch preparation'
        f'{tuple_delimiter}Obeya supports launch preparation.'
    )

    records = _malformed_relation_records(raw_output, 'chunk-obeya', 'source.pptx')

    assert len(records) == 1
    assert records[0]['field_count'] == 4
    assert {'wrong_field_count', 'action_verb_in_target_slot', 'missing_target'} <= set(records[0]['reasons'])


@pytest.mark.offline
def test_orphan_entity_names_reports_uncovered_entities():
    assert _orphan_entity_names({'A', 'B', 'C'}, {('A', 'B'), ('external', 'A')}) == ['C']


def _content_preview(content: str, limit: int = 320) -> str:
    collapsed = ' '.join(str(content).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3].rstrip() + '...'


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_live_tuna_extracts_entities_and_relations_from_processed_document():
    """Run live tuna extraction over processed chunks from an ingested document.

    This test is intentionally integration-gated and diagnostic-heavy. It lets us
    point tuna at exactly the processed text YAR stored, parse the raw extraction
    with production parser code, and inspect entity/relation mapping before
    reprocessing the graph.

    Run:
        uv run --extra dev python -m pytest tests/test_live_tuna_entity_extraction.py --run-integration -s

    Target selection, in priority order:
        YAR_LIVE_EXTRACTION_CHUNK_ID=chunk-...
        YAR_LIVE_EXTRACTION_CHUNK_IDS=chunk-a,chunk-b
        YAR_LIVE_EXTRACTION_DOC_ID=doc-...
        YAR_LIVE_EXTRACTION_FILE_PATH="document name.pdf"
        otherwise: latest processed document with chunks

    Useful knobs:
        YAR_LIVE_EXTRACTION_CHUNK_LIMIT=10     # cap selected document chunks; default all
        YAR_LIVE_EXTRACTION_CHUNK_OFFSET=0
        YAR_LIVE_EXTRACTION_EXPECTED_ENTITIES="Entity A,Entity B"
        YAR_LIVE_EXTRACTION_EXPECTED_RELATIONS="Source->Target;Other Source->Other Target"
        YAR_LIVE_EXTRACTION_REPORT_PATH=/tmp/live_extraction.json
        YAR_LIVE_EXTRACTION_FAIL_ON_MALFORMED=true
        YAR_LIVE_EXTRACTION_FAIL_ON_DANGLING_ENDPOINTS=true
        YAR_LIVE_EXTRACTION_FAIL_ON_BARE_DATE_RECORDS=true
        YAR_LIVE_EXTRACTION_FAIL_ON_ORPHAN_ENTITIES=true
        YAR_LIVE_EXTRACTION_REQUIRE_RELATIONS=false
    """
    conn = await _connect_postgres()
    try:
        selection = await _select_processed_chunks(conn)
        stored = await _fetch_stored_extraction(
            conn,
            str(selection['workspace']),
            [str(chunk['id']) for chunk in selection['chunks']],
        )
    finally:
        await conn.close()

    client, model = _build_tuna_client()
    chunk_reports: list[dict[str, Any]] = []
    aggregate_nodes: dict[str, list[dict[str, Any]]] = {}
    aggregate_edges: dict[tuple[str, str], list[dict[str, Any]]] = {}
    malformed_records: list[dict[str, Any]] = []

    for chunk in selection['chunks']:
        system_prompt, user_prompt = _build_extraction_prompts(str(chunk['content']))
        raw_output = _call_tuna(client, model, system_prompt, user_prompt)
        nodes, edges = await _process_extraction_result(
            raw_output,
            str(chunk['id']),
            int(time.time()),
            str(chunk.get('file_path') or 'unknown_source'),
            tuple_delimiter=PROMPTS['DEFAULT_TUPLE_DELIMITER'],
            completion_delimiter=PROMPTS['DEFAULT_COMPLETION_DELIMITER'],
        )

        chunk_malformed = _malformed_relation_records(
            raw_output,
            str(chunk['id']),
            str(chunk.get('file_path') or 'unknown_source'),
        )
        malformed_records.extend(chunk_malformed)

        for name, records in nodes.items():
            aggregate_nodes.setdefault(name, []).extend(records)
        for pair, records in edges.items():
            aggregate_edges.setdefault(pair, []).extend(records)

        chunk_reports.append(
            {
                'chunk_id': chunk['id'],
                'chunk_order_index': chunk.get('chunk_order_index'),
                'tokens': chunk.get('tokens'),
                'file_path': chunk.get('file_path'),
                'content_preview': _content_preview(str(chunk['content'])),
                'raw_output': raw_output,
                'parsed': _snapshot(nodes, edges),
                'malformed_relation_records': chunk_malformed,
            }
        )

    live_entity_names = set(aggregate_nodes)
    live_relation_pairs = set(aggregate_edges)
    stored_entity_names = {row['entity_name'] for row in stored['entities']}
    stored_relation_pairs = {(row['source_id'], row['target_id']) for row in stored['relations']}
    stored_entity_degrees = {row['entity_name']: int(row['degree']) for row in stored['entity_degrees']}
    live_orphan_entities = _orphan_entity_names(live_entity_names, live_relation_pairs)
    stored_orphan_entities = sorted(name for name, degree in stored_entity_degrees.items() if degree == 0)
    dangling_relation_endpoints = [
        {'source': source, 'target': target}
        for source, target in sorted(live_relation_pairs)
        if source not in live_entity_names or target not in live_entity_names
    ]
    bare_date_entities = sorted(name for name in live_entity_names if BARE_DATE_RE.match(name))
    bare_date_relations = [
        {'source': source, 'target': target}
        for source, target in sorted(live_relation_pairs)
        if BARE_DATE_RE.match(source) or BARE_DATE_RE.match(target)
    ]

    malformed_reason_counts = Counter(reason for record in malformed_records for reason in record.get('reasons', []))

    report = {
        'config': {
            'model': model,
            'base_url': os.getenv('YAR_LIVE_TUNA_BASE_URL', 'http://127.0.0.1:4000/v1'),
            'workspace': selection['workspace'],
            'strict': {
                'fail_on_malformed': _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_MALFORMED', False),
                'fail_on_dangling_endpoints': _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_DANGLING_ENDPOINTS', False),
                'fail_on_bare_date_records': _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_BARE_DATE_RECORDS', False),
                'fail_on_orphan_entities': _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_ORPHAN_ENTITIES', False),
                'require_relations': _bool_env('YAR_LIVE_EXTRACTION_REQUIRE_RELATIONS', True),
            },
        },
        'selection': selection['selection'],
        'document': selection['doc'],
        'live_totals': {
            'chunks': len(chunk_reports),
            'entities': len(live_entity_names),
            'relations': len(live_relation_pairs),
        },
        'stored_totals_for_selected_chunks': {
            'entities': len(stored_entity_names),
            'relations': len(stored_relation_pairs),
        },
        'live_vs_stored': {
            'entity_overlap': sorted(live_entity_names & stored_entity_names),
            'live_only_entities': sorted(live_entity_names - stored_entity_names),
            'stored_only_entities': sorted(stored_entity_names - live_entity_names),
            'relation_overlap': [list(pair) for pair in sorted(live_relation_pairs & stored_relation_pairs)],
            'live_only_relations': [list(pair) for pair in sorted(live_relation_pairs - stored_relation_pairs)],
            'stored_only_relations': [list(pair) for pair in sorted(stored_relation_pairs - live_relation_pairs)],
        },
        'graph_coverage': {
            'live_orphan_entities': live_orphan_entities,
            'stored_orphan_entities': stored_orphan_entities,
            'stored_entity_degrees': stored_entity_degrees,
        },
        'invariants': {
            'malformed_relation_records': malformed_records,
            'malformed_relation_reason_counts': dict(malformed_reason_counts),
            'dangling_relation_endpoints': dangling_relation_endpoints,
            'bare_date_entities': bare_date_entities,
            'bare_date_relations': bare_date_relations,
            'live_orphan_entities': live_orphan_entities,
        },
        'chunks': chunk_reports,
    }
    report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)

    report_path = os.getenv('YAR_LIVE_EXTRACTION_REPORT_PATH', '').strip()
    if report_path:
        Path(report_path).write_text(report_json + '\n', encoding='utf-8')

    print('\nLIVE_TUNA_ENTITY_EXTRACTION_DIAGNOSTIC=')
    print(report_json)

    expected_entities = set(_csv_env('YAR_LIVE_EXTRACTION_EXPECTED_ENTITIES'))
    expected_relations = _relation_pairs_env('YAR_LIVE_EXTRACTION_EXPECTED_RELATIONS')

    assert live_entity_names, report_json
    if _bool_env('YAR_LIVE_EXTRACTION_REQUIRE_RELATIONS', True):
        assert live_relation_pairs, report_json
    if expected_entities:
        assert expected_entities <= live_entity_names, report_json
    if expected_relations:
        assert expected_relations <= live_relation_pairs, report_json
    if _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_MALFORMED', False):
        assert malformed_records == [], report_json
    if _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_DANGLING_ENDPOINTS', False):
        assert dangling_relation_endpoints == [], report_json
    if _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_BARE_DATE_RECORDS', False):
        assert bare_date_entities == [], report_json
        assert bare_date_relations == [], report_json
    if _bool_env('YAR_LIVE_EXTRACTION_FAIL_ON_ORPHAN_ENTITIES', False):
        assert live_orphan_entities == [], report_json
        assert stored_orphan_entities == [], report_json
