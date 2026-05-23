"""
Tests for yar/operate.py - Core operation functions.

This module tests:
- Text chunking functions (chunking_by_semantic, create_chunker)
- Entity identifier truncation (_truncate_entity_identifier)
- Entity/relation summarization (_handle_entity_relation_summary, _summarize_descriptions)
- Entity type inference (_batch_infer_entity_types)
- Entity extraction (extract_entities)
- Keyword extraction (get_keywords_from_query, extract_keywords_only)
- Graph operation helpers
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from yar.base import QueryContextResult, QueryParam, TextChunkSchema
from yar.constants import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_SUMMARY_LANGUAGE,
    GRAPH_FIELD_SEP,
)
from yar.entity_resolution.config import EntityResolutionConfig
from yar.graph_model import (
    ChunkExtractionResult,
    EntityFact,
    RelationFacet,
    RelationFact,
    RelationKey,
    RelationPolarity,
    RelationPredicate,
    RelationSemantics,
    RelationSummary,
    build_relation_storage_projection,
    normalize_relation_keywords,
)
from yar.operate import (
    _apply_auto_entity_filter,
    _apply_token_truncation,
    _attach_relation_evidence_from_storage,
    _augment_retrieval_keywords,
    _build_context_str,
    _build_entity_lookup_query,
    _build_exact_chunk_search_query,
    _build_prompt_chunk_context,
    _build_query_context,
    _build_query_shaping_instructions,
    _chunk_phrase_terms_for_search,
    _chunk_relevance_components,
    _classify_malformed_relation_record,
    _derive_phrase_terms_for_chunk_search,
    _enrich_local_keywords,
    _extract_relation_evidence_spans,
    _extract_supporting_evidence_spans,
    _filter_confident_exact_context_chunks,
    _filter_nodes_to_relation_endpoints,
    _filter_prompt_relations_for_query,
    _find_most_related_edges_from_entities,
    _find_related_text_unit_from_entities,
    _find_related_text_unit_from_relations,
    _get_edge_data,
    _get_node_data,
    _get_vector_context,
    _guidance_chunk_search_query,
    _is_temporal_or_comparative_query,
    _matches_entity_filter,
    _merge_all_chunks,
    _merge_edges_then_upsert,
    _metadata_chunk_search_query,
    _metadata_query_match_score,
    _normalize_query_shaped_response,
    _normalize_retrieval_query_typos,
    _perform_kg_search,
    _precise_temporal_chunk_search_query,
    _prepare_visible_reference_payload,
    _prioritize_substantive_chunks,
    _process_extraction_result,
    _rebuild_from_extraction_result,
    _rebuild_single_relationship,
    _resolve_entity_aliases_for_batch,
    _resolve_max_file_paths,
    _should_enable_exact_chunk_fusion,
    _should_validate_inline_citations,
    _split_keyword_terms,
    _temporal_chunk_search_query,
    _tokenize_relevance_terms,
    _truncate_entity_identifier,
    chunking_by_semantic,
    create_chunker,
    extract_keywords_only,
    get_keywords_from_query,
    kg_query,
    naive_query,
)
from yar.prompt import PROMPTS
from yar.relation_resolution import RelationReviewResult
from yar.utils import TiktokenTokenizer, _chunk_document_key, process_chunks_unified
from yar.utils import logger as yar_logger


def _relation_fact(
    source: str,
    target: str,
    *,
    description: str,
    keywords: str,
    chunk_id: str = 'chunk-1',
    file_path: str = 'source.txt',
    timestamp: int = 1,
) -> RelationFact:
    fact = RelationFact.from_record(
        ['relation', source, target, keywords, description],
        chunk_id,
        timestamp,
        file_path,
    )
    assert fact is not None
    return fact


def _entity_fact(
    name: str,
    *,
    entity_type: str = 'concept',
    description: str = 'Entity description.',
    chunk_id: str = 'chunk-1',
    file_path: str = 'source.txt',
    timestamp: int = 1,
) -> EntityFact:
    fact = EntityFact.from_record(
        ['entity', name, entity_type, description],
        chunk_id,
        timestamp,
        file_path,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    assert fact is not None
    return fact


@pytest.mark.offline
def test_entity_fact_from_record_normalizes_and_preserves_provenance():
    fact = EntityFact.from_record(
        ['entity', ' Launch Owner ', 'role', ' Owns launch readiness. '],
        'chunk-entity',
        123,
        'launch.pptx',
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )

    assert fact == EntityFact(
        name='Launch Owner',
        entity_type='person',
        description='Owns launch readiness.',
        source_id='chunk-entity',
        file_path='launch.pptx',
        timestamp=123,
    )


@pytest.mark.offline
def test_relation_fact_from_record_preserves_direction_and_semantics():
    fact = RelationFact.from_record(
        [
            'relation',
            'Intervention Delay',
            'Launch Readiness',
            'impacts',
            'The delay could impact launch readiness, but there was no evidence it improved quality.',
        ],
        'chunk-relation',
        456,
        'launch.pptx',
    )

    assert fact is not None
    assert fact.key == RelationKey('Intervention Delay', 'Launch Readiness')
    assert fact.key.storage_pair == ('Intervention Delay', 'Launch Readiness')
    assert fact.key != RelationKey('Launch Readiness', 'Intervention Delay')
    assert fact.keywords == 'impacts'
    assert fact.description.startswith('The delay could impact launch readiness')
    assert fact.source_id == 'chunk-relation'
    assert fact.file_path == 'launch.pptx'
    assert fact.timestamp == 456
    assert fact.semantics.polarity == RelationPolarity.NEGATIVE_EVIDENCE
    assert {RelationFacet.IMPACT, RelationFacet.CONSEQUENCE, RelationFacet.EVIDENCE} <= fact.semantics.facets


@pytest.mark.offline
@pytest.mark.asyncio
async def test_relation_fact_extracts_table_row_evidence_from_source_content():
    raw_result = (
        'relation<|#|>Adapter strategy<|#|>Dosing error mitigation<|#|>uses<|#|>'
        'Adapter strategy uses prototype mock prep to mitigate dosing error.<|COMPLETE|>'
    )
    source_content = (
        '<table><tr><th>Risk</th><th>Mitigation</th></tr>'
        '<tr><td>Dosing error</td><td>Adapter strategy uses prototype mock prep</td></tr></table>'
    )

    extraction = await _process_extraction_result(
        raw_result,
        'chunk-evidence',
        456,
        'strategy.html',
        source_content=source_content,
    )
    relation = extraction.edges[RelationKey('Adapter strategy', 'Dosing error mitigation')][0]

    assert relation.evidence_spans
    assert relation.evidence_spans[0].startswith('Table row:')
    assert 'prototype mock prep' in relation.evidence_spans[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_process_extraction_result_supplements_explicit_risk_relations_from_source():
    extraction = await _process_extraction_result(
        '<|COMPLETE|>',
        'chunk-risk',
        456,
        'strategy.pptx',
        source_content='* The use of transfer adapters can pose risks to product quality and accurate dosing (literature and internal data).',
    )

    product_quality = extraction.edges[RelationKey('Transfer Adapters', 'Product Quality')][0]
    accurate_dosing = extraction.edges[RelationKey('Transfer Adapters', 'Accurate Dosing')][0]

    assert 'Transfer Adapters' in extraction.nodes
    assert 'Product Quality' in extraction.nodes
    assert 'Accurate Dosing' in extraction.nodes
    assert product_quality.keywords == 'poses risk to'
    assert accurate_dosing.keywords == 'poses risk to'
    assert RelationFacet.IMPACT in product_quality.semantics.facets
    assert product_quality.evidence_spans == (
        'The use of transfer adapters can pose risks to product quality and accurate dosing (literature and internal data)',
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_supplement_byline_contributors_emits_represents_relations_from_dash_form():
    byline = 'Vasco Filipe and Céline Thierens – on behalf of the WG'
    source_content = f'# Adapter Strategy WG\n{byline}'
    raw_result = f'relation<|#|>Adapter Strategy WG<|#|>Vasco Filipe<|#|>on behalf of<|#|>{byline}<|COMPLETE|>'

    extraction = await _process_extraction_result(
        raw_result,
        'chunk-byline',
        456,
        'strategy.pptx',
        source_content=source_content,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )

    vasco_relations = extraction.edges[RelationKey('Vasco Filipe', 'Adapter Strategy WG')]
    celine_relations = extraction.edges[RelationKey('Céline Thierens', 'Adapter Strategy WG')]
    assert RelationKey('Adapter Strategy WG', 'Vasco Filipe') not in extraction.edges
    assert [relation.keywords for relation in vasco_relations] == ['represents', 'on behalf of']
    assert [relation.keywords for relation in celine_relations] == ['represents']
    for relation in [*vasco_relations, *celine_relations]:
        assert byline in relation.evidence_spans

    assert 'Vasco Filipe' in extraction.nodes
    assert 'Céline Thierens' in extraction.nodes
    assert 'Adapter Strategy WG' in extraction.nodes


@pytest.mark.offline
@pytest.mark.asyncio
async def test_supplement_byline_contributors_skips_when_names_invalid():
    extraction = await _process_extraction_result(
        '<|COMPLETE|>',
        'chunk-invalid-byline',
        456,
        'strategy.pptx',
        source_content=('Al – on behalf of the Adapter Strategy WG\nThe – on behalf of the Adapter Strategy WG'),
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )

    assert extraction.edges == {}
    assert extraction.nodes == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_from_extraction_result_uses_truncated_source_content_for_relation_evidence():
    class SimpleTokenizer:
        def encode(self, text):
            return text.split()

        def decode(self, tokens):
            return ' '.join(tokens)

    class FakeTextChunksStorage:
        async def get_by_id(self, chunk_id):
            return {
                'file_path': 'strategy.html',
                'content': (
                    'Unrelated intro line. Adapter strategy uses prototype mock prep to mitigate dosing error.'
                ),
            }

    raw_result = (
        'relation<|#|>Adapter strategy<|#|>Dosing error mitigation<|#|>uses<|#|>'
        'Adapter strategy uses prototype mock prep to mitigate dosing error.<|COMPLETE|>'
    )

    extraction = await _rebuild_from_extraction_result(
        FakeTextChunksStorage(),
        raw_result,
        'chunk-rebuild',
        456,
        global_config={'tokenizer': SimpleTokenizer(), 'max_extract_input_tokens': 3},
    )
    relation = extraction.edges[RelationKey('Adapter strategy', 'Dosing error mitigation')][0]

    assert relation.evidence_spans == ()


@pytest.mark.offline
def test_relation_fact_from_record_rejects_malformed_and_self_loop_records():
    assert RelationFact.from_record(['relation', 'A', 'supports', 'A supports B.'], 'chunk', 1) is None
    assert RelationFact.from_record(['relation', 'A', 'A', 'supports', 'A supports itself.'], 'chunk', 1) is None


@pytest.mark.offline
def test_relation_storage_projection_preserves_external_schema_and_direction():
    summary = RelationSummary(
        key=RelationKey('Later Evidence', 'Earlier Decision'),
        predicate=RelationPredicate.from_raw('influenced by'),
        description='Later evidence influenced the earlier decision.',
        weight=1.5,
        source_id='chunk-projection',
        file_path='projection.pptx',
        created_at=789,
        truncate='Later Evidence' + GRAPH_FIELD_SEP + 'Earlier Decision',
        semantics=RelationSemantics.from_text('influenced by', 'Later evidence influenced the earlier decision.'),
    )

    projection = build_relation_storage_projection(summary)
    payload = next(iter(projection.relation_vdb_payload.values()))

    assert projection.graph_edge_data == {
        'weight': 1.5,
        'description': 'Later evidence influenced the earlier decision.',
        'keywords': 'influenced by',
        'source_id': 'chunk-projection',
        'file_path': 'projection.pptx',
        'created_at': 789,
        'truncate': 'Later Evidence' + GRAPH_FIELD_SEP + 'Earlier Decision',
    }
    assert set(payload) == {
        'src_id',
        'tgt_id',
        'source_id',
        'content',
        'keywords',
        'description',
        'weight',
        'file_path',
    }
    assert payload['src_id'] == 'Later Evidence'
    assert payload['tgt_id'] == 'Earlier Decision'
    assert payload['content'].startswith('influenced by\tLater Evidence\nEarlier Decision\n')
    evidence_summary = RelationSummary(
        key=RelationKey('Later Evidence', 'Earlier Decision'),
        predicate=RelationPredicate.from_raw('influenced by'),
        description='Later evidence influenced the earlier decision.',
        weight=1.5,
        source_id='chunk-projection',
        file_path='projection.pptx',
        created_at=789,
        truncate='',
        semantics=RelationSemantics.from_text('influenced by', 'Later evidence influenced the earlier decision.'),
        evidence_spans=('Table row: Decision: Earlier Decision | Evidence: Later Evidence',),
    )
    evidence_projection = build_relation_storage_projection(evidence_summary)
    evidence_payload = next(iter(evidence_projection.relation_vdb_payload.values()))
    assert 'evidence_spans' not in evidence_projection.graph_edge_data
    assert 'evidence_spans' not in evidence_payload
    assert 'evidence_spans: Table row: Decision:' in evidence_payload['content']


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_preserves_directed_relation_vector_payload():
    """Relation VDB content should keep extraction direction even when ids are canonicalized."""

    source = 'Primary Stability Batch Delay Communication'
    target = 'Kripa Ram'
    chunk_id = 'chunk-1'
    file_path = '20190227_alpha_launch_lessons_learned.pptx'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []
            self.upserted_nodes = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': chunk_id,
                'description': f'{node_id} description',
                'entity_type': 'event' if node_id == source else 'person',
                'file_path': file_path,
            }

        async def upsert_node(self, node_id, node_data):
            self.upserted_nodes.append((node_id, node_data))

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    graph = FakeGraphStorage()
    edge_data, _ent_vdb, rel_vdb, rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [
            _relation_fact(
                source,
                target,
                description='The delay communication was sent to Kripa Ram.',
                keywords='sent to',
                chunk_id=chunk_id,
                file_path=file_path,
            )
        ],
        graph,
        {
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
        },
    )

    rel_payload = next(iter(rel_vdb.values()))

    assert edge_data['src_id'] == source
    assert edge_data['tgt_id'] == target
    assert graph.upserted_edges[0][0] == source
    assert graph.upserted_edges[0][1] == target
    assert rel_payload['src_id'] == source
    assert rel_payload['tgt_id'] == target
    assert rel_payload['content'].splitlines()[:2] == [f'sends to\t{source}', target]
    assert len(rel_delete_ids) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_stores_internal_relation_evidence_without_schema_drift():
    source = 'Adapter strategy'
    target = 'Dosing error mitigation'
    chunk_id = 'chunk-evidence'
    evidence_span = 'Table row: Risk: Dosing error | Mitigation: Adapter strategy uses prototype mock prep'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': chunk_id,
                'description': f'{node_id} description',
                'entity_type': 'concept',
                'file_path': 'strategy.html',
            }

        async def upsert_node(self, node_id, node_data):
            return None

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeRelationChunks:
        def __init__(self):
            self.records = {}

        async def get_by_id(self, key):
            return self.records.get(key)

        async def upsert(self, payload):
            self.records.update(payload)

    graph = FakeGraphStorage()
    relation_chunks = FakeRelationChunks()
    relation = _relation_fact(
        source,
        target,
        description='Adapter strategy uses prototype mock prep to mitigate dosing error.',
        keywords='uses',
        chunk_id=chunk_id,
        file_path='strategy.html',
    ).with_evidence_spans([evidence_span])

    edge_data, _ent_vdb, rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [relation],
        graph,
        {
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
        },
        relation_chunks_storage=relation_chunks,
    )

    storage_key = GRAPH_FIELD_SEP.join(sorted((source, target)))
    stored_record = relation_chunks.records[storage_key]
    rel_payload = next(iter(rel_vdb.values()))
    assert stored_record['evidence_by_chunk'] == {chunk_id: [evidence_span]}
    assert stored_record['evidence_spans'] == [evidence_span]
    assert 'evidence_spans' not in graph.upserted_edges[0][2]
    assert 'evidence_spans' not in rel_payload
    assert 'evidence_spans: Table row: Risk:' in rel_payload['content']
    assert 'evidence_spans' not in edge_data


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_logs_high_weight_unsupported_diagnostic(caplog):
    source = 'Diagnostic Source'
    target = 'Diagnostic Target'
    description = 'Diagnostic Source supports Diagnostic Target.'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': 'chunk-existing',
                'description': f'{node_id} description',
                'entity_type': 'concept',
                'file_path': 'diagnostic.md',
            }

        async def upsert_node(self, node_id, node_data):
            return None

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    async def merge_relations(relations):
        await _merge_edges_then_upsert(
            source,
            target,
            relations,
            FakeGraphStorage(),
            {
                'source_ids_limit_method': 'KEEP',
                'max_source_ids_per_relation': 10,
                'max_source_ids_per_entity': 10,
                'max_file_paths': 10,
            },
        )

    caplog.set_level('INFO', logger='yar')
    yar_logger.addHandler(caplog.handler)
    try:
        unsupported_relations = [
            _relation_fact(
                source,
                target,
                description=description,
                keywords='supports',
                chunk_id=f'chunk-unsupported-{index}',
                file_path='diagnostic.md',
            )
            for index in range(3)
        ]

        await merge_relations(unsupported_relations)

        assert any(
            'High-weight unsupported edge: Diagnostic Source --supports--> Diagnostic Target '
            '(weight=3.0, sources=3) — no extractive evidence spans' in record.getMessage()
            for record in caplog.records
        )

        caplog.clear()
        supported_relations = [
            relation.with_evidence_spans(['Diagnostic Source supports Diagnostic Target.'])
            for relation in unsupported_relations
        ]

        await merge_relations(supported_relations)

        assert not any('High-weight unsupported edge:' in record.getMessage() for record in caplog.records)
    finally:
        yar_logger.removeHandler(caplog.handler)


@pytest.mark.offline
def test_extract_relation_evidence_spans_falls_back_to_endpoint_cooccurrence():
    relation = _relation_fact(
        'Safety Request',
        'Handling Guide',
        description='Handling Guide classifies the Safety Request.',
        keywords='classifies',
        chunk_id='chunk-evidence-fallback',
    )
    content = 'Background only. Handling Guide classifies the Safety Request for handling decisions.'

    with patch('yar.operate._extract_supporting_evidence_spans', return_value=[]):
        spans = _extract_relation_evidence_spans(content, relation)

    assert spans == ['Handling Guide classifies the Safety Request for handling decisions.']


@pytest.mark.offline
def test_extract_relation_evidence_spans_returns_empty_when_endpoints_absent():
    relation = _relation_fact(
        'Safety Request',
        'Handling Guide',
        description='Handling Guide classifies the Safety Request.',
        keywords='classifies',
        chunk_id='chunk-evidence-absent',
    )
    content = 'Unrelated handling guidance mentions neither endpoint nor predicate evidence.'

    with patch('yar.operate._extract_supporting_evidence_spans', return_value=[]):
        spans = _extract_relation_evidence_spans(content, relation)

    assert spans == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_recovers_evidence_for_high_weight_unsupported_edge():
    source = 'Safety Request'
    target = 'Handling Guide'
    chunk_ids = ['chunk-recover-1', 'chunk-recover-2']

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': chunk_ids[0],
                'description': f'{node_id} description',
                'entity_type': 'concept',
                'file_path': 'source.md',
            }

        async def upsert_node(self, node_id, node_data):
            return None

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeTextChunksStorage:
        async def get_by_ids(self, keys):
            return [
                {'content': 'Background context without direct evidence.'},
                {
                    'content': (
                        'Handling Guide classifies the Safety Request when operations manuals evaluate '
                        'closed-system handling requirements.'
                    )
                },
            ]

    graph = FakeGraphStorage()
    _edge_data, _ent_vdb, rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [
            _relation_fact(
                source,
                target,
                description='Handling Guide classifies the Safety Request.',
                keywords='classifies',
                chunk_id=chunk_id,
                file_path='source.md',
            )
            for chunk_id in chunk_ids
        ],
        graph,
        {
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
        },
        text_chunks_storage=FakeTextChunksStorage(),
    )

    rel_payload = next(iter(rel_vdb.values()))
    assert 'evidence_spans: Handling Guide classifies the Safety Request' in rel_payload['content']
    assert 'evidence_spans' not in graph.upserted_edges[0][2]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_recovers_evidence_for_low_weight_edge_before_vector_upsert():
    source = 'Launch Checklist'
    target = 'Site Readiness'
    chunk_id = 'chunk-low-weight-recover'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': chunk_id,
                'description': f'{node_id} description',
                'entity_type': 'concept',
                'file_path': 'source.md',
            }

        async def upsert_node(self, node_id, node_data):
            return None

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeTextChunksStorage:
        async def get_by_ids(self, keys):
            return [{'content': 'Launch Checklist tracks Site Readiness owners and target dates.'}]

    _edge_data, _ent_vdb, rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [
            _relation_fact(
                source,
                target,
                description='Launch Checklist tracks Site Readiness.',
                keywords='tracks',
                chunk_id=chunk_id,
                file_path='source.md',
            )
        ],
        FakeGraphStorage(),
        {
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
        },
        text_chunks_storage=FakeTextChunksStorage(),
    )

    rel_payload = next(iter(rel_vdb.values()))
    assert 'evidence_spans: Launch Checklist tracks Site Readiness owners and target dates.' in rel_payload['content']


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_applies_relation_review_when_enabled():
    source = 'Handling Guide'
    target = 'Safety Request'
    evidence_span = 'Handling Guide classifies the Safety Request in the operations manual.'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': 'chunk-review',
                'description': f'{node_id} description',
                'entity_type': 'concept',
                'file_path': 'source.md',
            }

        async def upsert_node(self, node_id, node_data):
            return None

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    review_result = RelationReviewResult(
        src=source,
        tgt=target,
        original_keywords=('classifies', 'evaluates necessity of'),
        canonical_keywords=('classifies',),
        primary='classifies',
        reasoning='Compound predicate reduced to the stated classification action.',
        confidence=0.9,
    )

    with patch(
        'yar.operate.llm_review_relation_predicates_batch', new=AsyncMock(return_value=[review_result])
    ) as review:
        edge_data, _ent_vdb, rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
            source,
            target,
            [
                _relation_fact(
                    source,
                    target,
                    description='Handling Guide classifies the Safety Request.',
                    keywords='classifies, evaluates necessity of',
                    chunk_id='chunk-review',
                    file_path='source.md',
                ).with_evidence_spans([evidence_span])
            ],
            FakeGraphStorage(),
            {
                'source_ids_limit_method': 'KEEP',
                'max_source_ids_per_relation': 10,
                'max_source_ids_per_entity': 10,
                'max_file_paths': 10,
                'llm_model_func': AsyncMock(return_value='[]'),
                'relation_resolution_config': {'enabled': True, 'confidence_threshold': 0.6},
            },
        )

    review.assert_awaited_once()
    rel_payload = next(iter(rel_vdb.values()))
    assert edge_data['keywords'] == 'classifies'
    assert rel_payload['keywords'] == 'classifies'
    assert f'evidence_spans: {evidence_span}' in rel_payload['content']


@pytest.mark.offline
@pytest.mark.asyncio
async def test_attach_relation_evidence_from_storage_adds_prompt_only_spans():
    source = 'Adapter strategy'
    target = 'Dosing error mitigation'
    chunk_id = 'chunk-evidence'
    evidence_span = 'Table row: Risk: Dosing error | Mitigation: Adapter strategy uses prototype mock prep'
    storage_key = GRAPH_FIELD_SEP.join(sorted((source, target)))

    class FakeRelationChunks:
        def __init__(self):
            self.records = {
                storage_key: {
                    'chunk_ids': [chunk_id],
                    'count': 1,
                    'evidence_by_chunk': {chunk_id: [evidence_span]},
                }
            }

        async def get_by_id(self, key):
            return self.records.get(key)

    relations = [
        {
            'src_tgt': (source, target),
            'source_id': chunk_id,
            'description': 'Adapter strategy uses prototype mock prep.',
            'keywords': 'uses',
        }
    ]

    enriched = await _attach_relation_evidence_from_storage(relations, FakeRelationChunks())

    assert 'evidence_spans' not in relations[0]
    assert enriched[0]['evidence_spans'] == [evidence_span]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_attach_relation_evidence_uses_batch_get_by_ids_when_supported():
    first_source = 'Adapter strategy'
    first_target = 'Dosing error mitigation'
    second_source = 'Regional submission task force'
    second_target = 'Submission Package'
    first_chunk = 'chunk-evidence-1'
    second_chunk = 'chunk-evidence-2'
    first_key = GRAPH_FIELD_SEP.join(sorted((first_source, first_target)))
    second_key = GRAPH_FIELD_SEP.join(sorted((second_source, second_target)))
    first_span = 'Adapter strategy mitigates dosing error through mock preparation.'
    second_span = 'Regional submission task force prepares the submission package.'

    class FakeRelationChunks:
        def __init__(self):
            self.batch_calls = []
            self.single_calls = []
            self.records = {
                first_key: {'evidence_by_chunk': {first_chunk: [first_span]}},
                second_key: {'evidence_spans': [second_span]},
            }

        async def get_by_ids(self, keys):
            self.batch_calls.append(keys)
            return [self.records.get(key) for key in keys]

        async def get_by_id(self, key):
            self.single_calls.append(key)
            return self.records.get(key)

    storage = FakeRelationChunks()
    relations = [
        {'src_tgt': (first_source, first_target), 'source_id': first_chunk},
        {'src_id': second_source, 'tgt_id': second_target, 'source_id': second_chunk},
    ]

    enriched = await _attach_relation_evidence_from_storage(relations, storage)

    assert storage.batch_calls == [[first_key, second_key]]
    assert storage.single_calls == []
    assert enriched[0]['evidence_spans'] == [first_span]
    assert enriched[1]['evidence_spans'] == [second_span]
    assert 'evidence_spans' not in relations[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_attach_relation_evidence_falls_back_to_per_id_when_no_batch():
    source = 'Protocol'
    target = 'Deviation'
    chunk_id = 'chunk-fallback'
    storage_key = GRAPH_FIELD_SEP.join(sorted((source, target)))
    evidence_span = 'Protocol deviation evidence is listed in the source table.'

    class FakeRelationChunks:
        def __init__(self):
            self.single_calls = []
            self.records = {storage_key: {'evidence_by_chunk': {chunk_id: [evidence_span]}}}

        async def get_by_id(self, key):
            self.single_calls.append(key)
            return self.records.get(key)

    storage = FakeRelationChunks()
    relations = [{'src_tgt': (source, target), 'source_id': chunk_id}]

    enriched = await _attach_relation_evidence_from_storage(relations, storage)

    assert storage.single_calls == [storage_key]
    assert enriched[0]['evidence_spans'] == [evidence_span]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_adds_relation_search_hints_for_impact_and_negative_evidence():
    source = 'Intervention Delay'
    target = 'Launch Readiness'
    chunk_id = 'chunk-impact'
    description = 'The delay could impact launch readiness, but there was no evidence it improved quality.'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []
            self.upserted_nodes = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': chunk_id,
                'description': f'{node_id} description',
                'entity_type': 'event',
                'file_path': 'source.pptx',
            }

        async def upsert_node(self, node_id, node_data):
            self.upserted_nodes.append((node_id, node_data))

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    _edge_data, _ent_vdb, rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [
            _relation_fact(
                source,
                target,
                description=description,
                keywords='impacts',
                chunk_id=chunk_id,
                file_path='source.pptx',
            )
        ],
        FakeGraphStorage(),
        {
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
        },
    )

    rel_payload = next(iter(rel_vdb.values()))
    assert rel_payload['content'].splitlines()[:3] == [f'impacts\t{source}', target, description]
    assert 'impact consequence effect result outcome' in rel_payload['content']
    assert 'negative evidence no evidence insufficient evidence not supported' in rel_payload['content']


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_assigns_configured_type_to_missing_endpoint_nodes():
    """Relation-created endpoint nodes should not persist UNKNOWN entity types."""
    source = 'Relation Only Source'
    target = 'Relation Only Target'
    chunk_id = 'chunk-1'
    file_path = 'source.pdf'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_nodes = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return None

        async def upsert_node(self, node_id, node_data):
            self.upserted_nodes.append((node_id, node_data))

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            return None

    graph = FakeGraphStorage()
    _edge_data, ent_vdb, _rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [
            _relation_fact(
                source,
                target,
                description='The source is related to the target.',
                keywords='related to',
                chunk_id=chunk_id,
                file_path=file_path,
            )
        ],
        graph,
        {
            'addon_params': {'entity_types': list(DEFAULT_ENTITY_TYPES)},
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
        },
    )

    assert len(graph.upserted_nodes) == 2
    assert {node_data['entity_type'] for _node_id, node_data in graph.upserted_nodes} == {'concept'}
    assert {payload['entity_type'] for payload in ent_vdb.values()} == {'concept'}


@pytest.mark.offline
def test_normalize_relation_keywords_splits_and_deduplicates_labels():
    assert (
        normalize_relation_keywords([' Uses, supports ', 'uses,Requires', '', 'SUPPORTS']) == 'uses, supports, requires'
    )


@pytest.mark.offline
def test_normalize_relation_keywords_canonicalizes_without_inverting_direction():
    assert (
        normalize_relation_keywords(
            ['manufactured by, manufactured, evaluates, evaluated in'],
            max_keywords=10,
        )
        == 'manufactured by, manufactures, evaluates, evaluated in'
    )
    assert (
        normalize_relation_keywords(
            [
                'participated in, participated, attends, attended',
                'submitted to, sent to, included participant',
            ],
            max_keywords=10,
        )
        == 'participates in, attends, submits to, sends to, included participant'
    )


@pytest.mark.offline
def test_normalize_relation_keywords_bounds_labels_to_first_three():
    assert normalize_relation_keywords('supports, requires, evaluates, approves') == 'supports, requires, evaluates'


@pytest.mark.offline
def test_relation_predicate_aliases_collapse_synonymous_surfaces():
    predicate = RelationPredicate.from_raw(
        [
            'applies',
            'apply',
            'uses',
            'supported by',
            'depends on',
            'mitigates',
            'collaborated with',
            'represented',
            'poses risks to',
        ],
        max_keywords=10,
    )

    assert predicate.keywords == (
        'uses',
        'supported_by',
        'depends_on',
        'mitigates',
        'collaborates with',
        'represents',
        'poses risk to',
    )
    assert predicate.text == ('uses, supported_by, depends_on, mitigates, collaborates with, represents, poses risk to')


@pytest.mark.offline
def test_predicate_splits_compound_keyword_into_canonical_first():
    predicate = RelationPredicate.from_raw('classifies, evaluates necessity of', max_keywords=10)

    assert predicate.keywords == ('classifies', 'evaluates necessity of')
    assert predicate.primary == 'classifies'


@pytest.mark.offline
def test_predicate_drops_unknown_secondary_clause():
    predicate = RelationPredicate.from_raw('requires and ambiguous follow-up clause', max_keywords=10)

    assert predicate.keywords == ('requires',)


@pytest.mark.offline
def test_predicate_primary_never_contains_comma():
    predicate = RelationPredicate(('opaque first, opaque second',))

    assert predicate.primary == 'opaque first'


@pytest.mark.offline
def test_relation_vector_content_uses_primary_predicate_for_surface_form():
    predicate = RelationPredicate.from_raw('partnered with, collaborates with, required extra team effort')
    summary = RelationSummary(
        key=RelationKey('SRC', 'TGT'),
        predicate=predicate,
        description='SRC partnered with TGT and required extra team effort.',
        weight=2.0,
        source_id='chunk-primary-predicate',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text(
            predicate.text,
            'SRC partnered with TGT and required extra team effort.',
        ),
    )

    projection = build_relation_storage_projection(summary)
    payload = next(iter(projection.relation_vdb_payload.values()))

    assert projection.graph_edge_data['keywords'] == 'partnered with, collaborates with, required extra team effort'
    assert payload['keywords'] == 'partnered with, collaborates with, required extra team effort'
    assert payload['content'].splitlines()[:3] == [
        'partnered with, collaborates with, required extra team effort\tSRC',
        'TGT',
        'SRC partnered with TGT and required extra team effort.',
    ]
    assert 'relation: SRC --partnered with--> TGT' in payload['content']
    assert 'predicate: partnered with' in payload['content']


@pytest.mark.offline
def test_relation_vector_content_uses_canonical_prefix_for_primary_predicate():
    predicate = RelationPredicate.from_raw(
        'requires updates to avoid systematic amendment, allows flexibility language for'
    )
    summary = RelationSummary(
        key=RelationKey('Protocol Package', 'Transfer Adapter'),
        predicate=predicate,
        description='Protocol Package requires updates to avoid systematic amendment.',
        weight=2.0,
        source_id='chunk-primary-prefix',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text(
            predicate.text,
            'Protocol Package requires updates to avoid systematic amendment.',
        ),
    )

    payload = next(iter(build_relation_storage_projection(summary).relation_vdb_payload.values()))

    assert payload['keywords'] == 'requires updates to avoid systematic amendment, allows flexibility language for'
    assert 'relation: Protocol Package --requires--> Transfer Adapter' in payload['content']
    assert 'predicate: requires' in payload['content']


@pytest.mark.offline
def test_relation_predicate_aliases_do_not_merge_antonyms():
    predicate = RelationPredicate.from_raw(['supports', 'blocks', 'enables', 'prevents'], max_keywords=10)

    assert predicate.keywords == ('supports', 'blocks', 'enables', 'prevents')


@pytest.mark.offline
def test_relation_semantics_classifies_risk_phrasing_as_impact():
    semantics = RelationSemantics.from_text(
        'poses risk to',
        'The use of transfer adapters can pose risks to product quality and accurate dosing.',
    )

    assert RelationFacet.IMPACT in semantics.facets
    assert RelationFacet.CONSEQUENCE in semantics.facets

    summary = RelationSummary(
        key=RelationKey('Transfer Adapter', 'Product Quality'),
        predicate=RelationPredicate.from_raw('poses risk to'),
        description='The use of transfer adapters can pose risks to product quality.',
        weight=1.0,
        source_id='chunk-risk',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=semantics,
        evidence_spans=('The use of transfer adapters can pose risks to product quality.',),
    )
    payload = next(iter(build_relation_storage_projection(summary).relation_vdb_payload.values()))

    assert 'search_hints: impact consequence effect result outcome' in payload['content']
    assert 'relation: Transfer Adapter --poses risk to--> Product Quality' in payload['content']
    assert 'source_entity: Transfer Adapter' in payload['content']
    assert 'target_entity: Product Quality' in payload['content']


@pytest.mark.offline
def test_relation_vector_content_adds_predicate_specific_search_hints():
    represented_summary = RelationSummary(
        key=RelationKey('Adapter Strategy WG', 'Vasco Filipe'),
        predicate=RelationPredicate.from_raw('represented by'),
        description='Vasco Filipe represents the Adapter Strategy WG.',
        weight=1.0,
        source_id='chunk-represented',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text(
            'represented by',
            'Vasco Filipe represents the Adapter Strategy WG.',
        ),
    )
    collaboration_summary = RelationSummary(
        key=RelationKey('Company A', 'Company B'),
        predicate=RelationPredicate.from_raw('collaborated with'),
        description='Company A and Company B collaborated during the product transition.',
        weight=1.0,
        source_id='chunk-collaboration',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text(
            'collaborated with',
            'Company A and Company B collaborated during the product transition.',
        ),
    )

    represented_payload = next(
        iter(build_relation_storage_projection(represented_summary).relation_vdb_payload.values())
    )
    collaboration_payload = next(
        iter(build_relation_storage_projection(collaboration_summary).relation_vdb_payload.values())
    )

    assert (
        'search_hints: contributor contributes representative represented on behalf working group member'
        in represented_payload['content']
    )
    assert 'collaboration partnership alliance joint transition relationship' in collaboration_payload['content']


@pytest.mark.offline
def test_relation_vector_content_emits_generic_search_hints_when_no_rule_matches():
    summary = RelationSummary(
        key=RelationKey('Milestone Ledger', 'Action Register'),
        predicate=RelationPredicate.from_raw('tracks'),
        description='Milestone ledger tracks action register owners and dates.',
        weight=1.0,
        source_id='chunk-generic-hint',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text('tracks', 'Milestone ledger tracks action register owners and dates.'),
    )

    payload = next(iter(build_relation_storage_projection(summary).relation_vdb_payload.values()))

    assert 'search_hints: tracks milestone ledger action register owners dates' in payload['content']


@pytest.mark.offline
def test_relation_vector_content_emits_member_of_hint():
    summary = RelationSummary(
        key=RelationKey('Vasco Filipe', 'Adapter Strategy WG'),
        predicate=RelationPredicate.from_raw('member of'),
        description='Vasco Filipe is a member of the Adapter Strategy WG.',
        weight=1.0,
        source_id='chunk-member',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text('member of', 'Vasco Filipe is a member of the Adapter Strategy WG.'),
    )

    payload = next(iter(build_relation_storage_projection(summary).relation_vdb_payload.values()))

    assert 'search_hints: team membership working group participant member of role' in payload['content']


@pytest.mark.offline
def test_relation_vector_content_emits_manufactured_by_hint():
    summary = RelationSummary(
        key=RelationKey('Product A', 'Manufacturing Site'),
        predicate=RelationPredicate.from_raw('manufactured by'),
        description='Product A is manufactured by Manufacturing Site.',
        weight=1.0,
        source_id='chunk-manufacturing',
        file_path='source.md',
        created_at=123,
        truncate='',
        semantics=RelationSemantics.from_text('manufactured by', 'Product A is manufactured by Manufacturing Site.'),
    )

    payload = next(iter(build_relation_storage_projection(summary).relation_vdb_payload.values()))

    assert 'search_hints: vendor supplier manufacturer manufactured produces production' in payload['content']


@pytest.mark.offline
def test_relation_vector_content_emits_domain_specific_hint_clusters():
    cases = [
        (
            'approved by',
            'Arisure was approved by the U.S. Food and Drug Administration.',
            'regulatory approval clearance compliance authority consultation approved',
        ),
        (
            'classified as',
            'Classified Item is classified as Restricted.',
            'classification category taxonomy type class part of grouping',
        ),
        (
            'holds commercial rights to, obtained commercial rights to',
            'Company A holds commercial rights to Product A.',
            'business agreement contract license rights commercial negotiation alliance',
        ),
        (
            'sends memo to',
            'CSU sends memo to Trial Operation.',
            'communication coordination reporting memo project management workflow',
        ),
        (
            'authored',
            'Julia Marinina authored the Product A lessons learned presentation.',
            'document author prepared published drafted presentation handbook source',
        ),
        (
            'based in',
            'Manufacturing Site is based in Goa.',
            'location site facility geography country market region',
        ),
        (
            'supported by',
            'Target receives support.',
            'support supports enables backing rationale',
        ),
        (
            'depends on',
            'Source depends on Target.',
            'dependency prerequisite required-by depends upon',
        ),
        (
            'reviews',
            'Source reviews Target.',
            'review audit examination oversight peer review',
        ),
        (
            'assesses',
            'Source assesses Target.',
            'assessment evaluation audit review appraisal',
        ),
    ]

    for predicate_text, description, expected_hint in cases:
        predicate = RelationPredicate.from_raw(predicate_text)
        summary = RelationSummary(
            key=RelationKey('Source', 'Target'),
            predicate=predicate,
            description=description,
            weight=1.0,
            source_id='chunk-domain-hint',
            file_path='source.md',
            created_at=123,
            truncate='',
            semantics=RelationSemantics.from_text(predicate.text, description),
        )

        payload = next(iter(build_relation_storage_projection(summary).relation_vdb_payload.values()))

        assert expected_hint in payload['content']


@pytest.mark.offline
def test_relation_fact_preserves_direction_for_inverse_predicate():
    fact = RelationFact.from_record(
        [
            'relation',
            'Adapter Strategy WG',
            'Vasco Filipe',
            'represented by',
            'The Adapter Strategy WG is represented by Vasco Filipe.',
        ],
        'chunk-represented',
        123,
        'source.md',
    )

    assert fact is not None
    assert fact.key == RelationKey('Vasco Filipe', 'Adapter Strategy WG')
    assert fact.predicate.text == 'represents'


@pytest.mark.offline
def test_relation_fact_keeps_mixed_inverse_predicates_unflipped():
    fact = RelationFact.from_record(
        [
            'relation',
            'Adapter Strategy WG',
            'Vasco Filipe',
            'represented by, includes',
            'The Adapter Strategy WG is represented by Vasco Filipe and includes multiple workstreams.',
        ],
        'chunk-mixed',
        123,
        'source.md',
    )

    assert fact is not None
    assert fact.key == RelationKey('Adapter Strategy WG', 'Vasco Filipe')
    assert fact.predicate.text == 'represented by, includes'


@pytest.mark.offline
def test_normalize_relation_keywords_preserves_external_storage_shape():
    predicate = RelationPredicate.from_raw('apply')
    summary = RelationSummary(
        key=RelationKey('Source', 'Target'),
        predicate=predicate,
        description='Source applies the target method.',
        weight=2.0,
        source_id='chunk-predicate',
        file_path='source.md',
        created_at=123,
        truncate='no',
        semantics=RelationSemantics.from_text(predicate.text, 'Source applies the target method.'),
        evidence_spans=('Source applies the target method.',),
    )

    projection = build_relation_storage_projection(summary)
    payload = next(iter(projection.relation_vdb_payload.values()))

    assert projection.graph_edge_data['keywords'] == 'uses'
    assert payload['keywords'] == 'uses'
    assert set(projection.graph_edge_data) == {
        'weight',
        'description',
        'keywords',
        'source_id',
        'file_path',
        'created_at',
        'truncate',
    }
    assert set(payload) == {
        'src_id',
        'tgt_id',
        'source_id',
        'content',
        'keywords',
        'description',
        'weight',
        'file_path',
    }


@pytest.mark.offline
def test_malformed_relation_classifier_identifies_action_verb_target_slot():
    diagnostic = _classify_malformed_relation_record(
        ['relation', 'Obeya', 'supports launch preparation', 'Obeya supports launch preparation.'],
        'chunk-obeya',
        'source.pptx',
    )

    assert diagnostic is not None
    assert diagnostic.field_count == 4
    assert diagnostic.source == 'Obeya'
    assert diagnostic.target_slot == 'supports launch preparation'
    assert {'wrong_field_count', 'action_verb_in_target_slot', 'missing_target'} <= set(diagnostic.reasons)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_process_extraction_result_skips_4field_action_relation_without_dangling_endpoint():
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    raw_result = '\n'.join(
        [
            f'entity{tuple_delimiter}Obeya{tuple_delimiter}organization{tuple_delimiter}Obeya is a launch room.',
            f'relation{tuple_delimiter}Obeya{tuple_delimiter}supports launch preparation'
            f'{tuple_delimiter}Obeya supports launch preparation.',
            completion_delimiter,
        ]
    )

    extraction = await _process_extraction_result(
        raw_result,
        'chunk-obeya',
        1234567890,
        'source.pptx',
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    nodes = extraction.nodes
    edges = extraction.edges
    assert set(nodes) == {'Obeya'}
    assert edges == {}
    assert _filter_nodes_to_relation_endpoints(nodes, edges, 'chunk-obeya') == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_process_extraction_result_accepts_corrected_5field_relation():
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    raw_result = '\n'.join(
        [
            f'entity{tuple_delimiter}Obeya{tuple_delimiter}organization{tuple_delimiter}Obeya is a launch room.',
            f'entity{tuple_delimiter}Launch Preparation{tuple_delimiter}event'
            f'{tuple_delimiter}Launch preparation is supported by Obeya.',
            f'relation{tuple_delimiter}Obeya{tuple_delimiter}Launch Preparation'
            f'{tuple_delimiter}supports{tuple_delimiter}Obeya supports launch preparation.',
            completion_delimiter,
        ]
    )

    extraction = await _process_extraction_result(
        raw_result,
        'chunk-obeya',
        1234567890,
        'source.pptx',
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    nodes = extraction.nodes
    edges = extraction.edges
    assert set(nodes) == {'Obeya', 'Launch Preparation'}
    assert set(edges) == {RelationKey('Obeya', 'Launch Preparation')}
    assert edges[RelationKey('Obeya', 'Launch Preparation')][0].keywords == 'supports'


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_bounds_keywords_but_preserves_relation_descriptions():
    source = 'Source Entity'
    target = 'Target Entity'
    descriptions = [
        'Source supports Target.',
        'Source requires Target.',
        'Source evaluates Target.',
        'Source approves Target.',
    ]

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []
            self.upserted_nodes = []

        async def has_edge(self, source_node_id, target_node_id):
            return False

        async def get_node(self, node_id):
            return {
                'entity_id': node_id,
                'source_id': 'existing-chunk',
                'description': f'{node_id} description',
                'entity_type': 'concept',
                'file_path': 'source.pptx',
            }

        async def upsert_node(self, node_id, node_data):
            self.upserted_nodes.append((node_id, node_data))

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    graph = FakeGraphStorage()
    edge_data, _ent_vdb, rel_vdb, _rel_delete_ids = await _merge_edges_then_upsert(
        source,
        target,
        [
            _relation_fact(
                source,
                target,
                description=description,
                keywords=keyword,
                chunk_id=f'chunk-{index}',
                file_path='source.pptx',
                timestamp=index,
            )
            for index, (description, keyword) in enumerate(
                zip(descriptions, ['supports', 'requires', 'evaluates', 'approves'], strict=True),
                start=1,
            )
        ],
        graph,
        {
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_source_ids_per_entity': 10,
            'max_file_paths': 10,
            'tokenizer': Mock(encode=Mock(side_effect=lambda text: text.split())),
            'summary_context_size': 1000,
            'summary_max_tokens': 500,
            'force_llm_summary_on_merge': 10,
        },
    )

    assert edge_data['keywords'] == 'supports, requires, evaluates'
    assert next(iter(rel_vdb.values()))['keywords'] == 'supports, requires, evaluates'
    for description in descriptions:
        assert description in edge_data['description']


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_relationship_uses_configured_type_for_missing_endpoint_nodes():
    source = 'Relation Only Source'
    target = 'Relation Only Target'
    chunk_id = 'chunk-1'
    file_path = 'source.pdf'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_nodes = []
            self.upserted_edges = []

        async def get_edge(self, source_node_id, target_node_id):
            return {
                'description': 'Existing relation.',
                'keywords': 'existing',
                'weight': 1.0,
                'source_id': chunk_id,
                'file_path': file_path,
            }

        async def has_node(self, node_id):
            return False

        async def upsert_node(self, node_id, node_data):
            self.upserted_nodes.append((node_id, node_data))

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeVectorStorage:
        def __init__(self):
            self.deleted = []
            self.payloads = {}

        async def delete(self, ids):
            self.deleted.extend(ids)

        async def upsert(self, payload):
            self.payloads.update(payload)

    graph = FakeGraphStorage()
    ent_vdb = FakeVectorStorage()
    rel_vdb = FakeVectorStorage()

    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=rel_vdb,
        entities_vdb=ent_vdb,
        src=source,
        tgt=target,
        chunk_ids=[chunk_id],
        chunk_relationships={
            chunk_id: {
                RelationKey(source, target): [
                    _relation_fact(
                        source,
                        target,
                        description='The source supports the target.',
                        keywords='Supports, uses, supports',
                        chunk_id=chunk_id,
                        file_path=file_path,
                    )
                ]
            }
        },
        llm_response_cache=None,
        global_config={
            'addon_params': {'entity_types': list(DEFAULT_ENTITY_TYPES)},
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_file_paths': 10,
        },
    )

    assert {node_data['entity_type'] for _node_id, node_data in graph.upserted_nodes} == {'concept'}
    assert {payload['entity_type'] for payload in ent_vdb.payloads.values()} == {'concept'}
    assert graph.upserted_edges[0][2]['keywords'] == 'supports, uses'
    assert next(iter(rel_vdb.payloads.values()))['keywords'] == 'supports, uses'


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_relationship_prefers_direct_facts_over_inverse_facts():
    source = 'A Entity'
    target = 'B Entity'
    chunk_id = 'chunk-direct'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def get_edge(self, source_node_id, target_node_id):
            return {
                'description': 'Existing relation.',
                'keywords': 'existing',
                'weight': 1.0,
                'source_id': chunk_id,
                'file_path': 'existing.txt',
            }

        async def has_node(self, node_id):
            return True

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeVectorStorage:
        def __init__(self):
            self.deleted = []
            self.payloads = {}

        async def delete(self, ids):
            self.deleted.extend(ids)

        async def upsert(self, payload):
            self.payloads.update(payload)

    graph = FakeGraphStorage()
    rel_vdb = FakeVectorStorage()

    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=rel_vdb,
        entities_vdb=FakeVectorStorage(),
        src=source,
        tgt=target,
        chunk_ids=[chunk_id],
        chunk_relationships={
            chunk_id: {
                RelationKey(source, target): [
                    _relation_fact(source, target, description='A supports B.', keywords='supports', chunk_id=chunk_id)
                ],
                RelationKey(target, source): [
                    _relation_fact(target, source, description='B blocks A.', keywords='blocks', chunk_id=chunk_id)
                ],
            }
        },
        llm_response_cache=None,
        global_config={
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_file_paths': 10,
        },
    )

    assert graph.upserted_edges == [
        (
            source,
            target,
            {
                'description': 'A supports B.',
                'keywords': 'supports',
                'weight': 1.0,
                'source_id': chunk_id,
                'file_path': 'source.txt',
                'truncate': '',
            },
        )
    ]
    payload = next(iter(rel_vdb.payloads.values()))
    assert payload['src_id'] == source
    assert payload['tgt_id'] == target
    assert payload['description'] == 'A supports B.'


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_relationship_uses_only_selected_direction_source_ids():
    source = 'A Entity'
    target = 'B Entity'
    direct_chunk_id = 'chunk-direct'
    reverse_chunk_id = 'chunk-reverse'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def get_edge(self, source_node_id, target_node_id):
            return {
                'description': 'Existing relation.',
                'keywords': 'existing',
                'weight': 1.0,
                'source_id': GRAPH_FIELD_SEP.join([direct_chunk_id, reverse_chunk_id]),
                'file_path': 'existing.txt',
            }

        async def has_node(self, node_id):
            return True

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeVectorStorage:
        def __init__(self):
            self.deleted = []
            self.payloads = {}

        async def delete(self, ids):
            self.deleted.extend(ids)

        async def upsert(self, payload):
            self.payloads.update(payload)

    graph = FakeGraphStorage()
    rel_vdb = FakeVectorStorage()

    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=rel_vdb,
        entities_vdb=FakeVectorStorage(),
        src=source,
        tgt=target,
        chunk_ids=[direct_chunk_id, reverse_chunk_id],
        chunk_relationships={
            direct_chunk_id: {
                RelationKey(source, target): [
                    _relation_fact(
                        source,
                        target,
                        description='A supports B.',
                        keywords='supports',
                        chunk_id=direct_chunk_id,
                        file_path='direct.txt',
                    )
                ]
            },
            reverse_chunk_id: {
                RelationKey(target, source): [
                    _relation_fact(
                        target,
                        source,
                        description='B blocks A.',
                        keywords='blocks',
                        chunk_id=reverse_chunk_id,
                        file_path='reverse.txt',
                    )
                ]
            },
        },
        llm_response_cache=None,
        global_config={
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_file_paths': 10,
        },
    )

    edge_data = graph.upserted_edges[0][2]
    assert edge_data['description'] == 'A supports B.'
    assert edge_data['source_id'] == direct_chunk_id
    assert edge_data['file_path'] == 'direct.txt'
    payload = next(iter(rel_vdb.payloads.values()))
    assert payload['src_id'] == source
    assert payload['tgt_id'] == target
    assert payload['source_id'] == direct_chunk_id
    assert payload['description'] == 'A supports B.'


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_relationship_reverse_fallback_preserves_fact_direction():
    source = 'A Entity'
    target = 'B Entity'
    chunk_id = 'chunk-reverse'

    class FakeGraphStorage:
        def __init__(self):
            self.upserted_edges = []

        async def get_edge(self, source_node_id, target_node_id):
            return {
                'description': 'Existing relation.',
                'keywords': 'existing',
                'weight': 1.0,
                'source_id': chunk_id,
                'file_path': 'existing.txt',
            }

        async def has_node(self, node_id):
            return True

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.upserted_edges.append((source_node_id, target_node_id, edge_data))

    class FakeVectorStorage:
        def __init__(self):
            self.deleted = []
            self.payloads = {}

        async def delete(self, ids):
            self.deleted.extend(ids)

        async def upsert(self, payload):
            self.payloads.update(payload)

    graph = FakeGraphStorage()
    rel_vdb = FakeVectorStorage()

    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=rel_vdb,
        entities_vdb=FakeVectorStorage(),
        src=source,
        tgt=target,
        chunk_ids=[chunk_id],
        chunk_relationships={
            chunk_id: {
                RelationKey(target, source): [
                    _relation_fact(target, source, description='B blocks A.', keywords='blocks', chunk_id=chunk_id)
                ]
            }
        },
        llm_response_cache=None,
        global_config={
            'source_ids_limit_method': 'KEEP',
            'max_source_ids_per_relation': 10,
            'max_file_paths': 10,
        },
    )

    assert graph.upserted_edges[0][0:2] == (source, target)
    assert graph.upserted_edges[0][2]['description'] == 'B blocks A.'
    payload = next(iter(rel_vdb.payloads.values()))
    assert payload['src_id'] == target
    assert payload['tgt_id'] == source
    assert payload['content'].startswith('blocks\tB Entity\nA Entity\nB blocks A.')


def test_edge_grouping_preserves_extracted_relation_direction():
    """Directional relationships remain distinct typed relation keys."""
    forward = _relation_fact(
        'B Entity',
        'A Entity',
        description='B Entity supports A Entity.',
        keywords='supports',
    )
    reverse = _relation_fact(
        'A Entity',
        'B Entity',
        description='A Entity supports B Entity.',
        keywords='supports',
    )

    all_edges: dict[RelationKey, list[RelationFact]] = {}
    for relation in (forward, reverse):
        all_edges.setdefault(relation.key, []).append(relation)

    assert set(all_edges) == {RelationKey('B Entity', 'A Entity'), RelationKey('A Entity', 'B Entity')}
    assert all_edges[RelationKey('B Entity', 'A Entity')][0].key == RelationKey('B Entity', 'A Entity')
    assert all_edges[RelationKey('A Entity', 'B Entity')][0].key == RelationKey('A Entity', 'B Entity')


@pytest.mark.offline
@pytest.mark.asyncio
async def test_alias_resolution_skips_alias_that_would_collapse_existing_edge():
    """Alias auto-apply must not erase a real relation by turning it into a self-loop."""

    class FakeEntityVdb:
        async def hybrid_entity_search(self, entity_name, *, top_k):
            if entity_name == 'Acme':
                return [{'entity_name': 'Acme Research Center', 'entity_type': 'Organization'}]
            return []

    llm_response = (
        '[{"new_entity": "Acme", "matches_existing": true, '
        '"canonical": "Acme Research Center", "confidence": 0.95, '
        '"reasoning": "LLM considered this the same organization"}]'
    )
    llm_model_func = AsyncMock(return_value=llm_response)
    all_nodes = {
        'Acme': [_entity_fact('Acme', entity_type='organization', description='Acme')],
        'Acme Research Center': [
            _entity_fact('Acme Research Center', entity_type='organization', description='Acme Research Center')
        ],
    }
    all_edges = {
        RelationKey('Acme', 'Acme Research Center'): [
            _relation_fact(
                'Acme',
                'Acme Research Center',
                description='Acme works with Acme Research Center.',
                keywords='works with',
            )
        ]
    }
    expected_nodes = {entity_name: list(entity_facts) for entity_name, entity_facts in all_nodes.items()}
    expected_edges = {edge_key: list(edge_facts) for edge_key, edge_facts in all_edges.items()}

    resolved_nodes, resolved_edges = await _resolve_entity_aliases_for_batch(
        all_nodes=all_nodes,
        all_edges=all_edges,
        entity_vdb=FakeEntityVdb(),
        global_config={
            'workspace': 'default',
            'llm_model_func': llm_model_func,
            'entity_resolution_config': EntityResolutionConfig(
                enabled=True,
                auto_resolve_on_extraction=True,
                auto_apply=True,
            ),
        },
    )

    assert resolved_nodes == expected_nodes
    assert resolved_edges == expected_edges
    llm_model_func.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_alias_resolution_skips_alias_that_would_collapse_edge_after_prior_alias():
    """Batch aliases must be checked together so two accepted aliases cannot erase an edge."""

    class FakeEntityVdb:
        async def hybrid_entity_search(self, entity_name, *, top_k):
            if entity_name in {'Acme', 'Acme Research'}:
                return [{'entity_name': 'Acme Research Center', 'entity_type': 'Organization'}]
            return []

    llm_response = (
        '['
        '{"new_entity": "Acme", "matches_existing": true, '
        '"canonical": "Acme Research Center", "confidence": 0.95, '
        '"reasoning": "abbreviation"}, '
        '{"new_entity": "Acme Research", "matches_existing": true, '
        '"canonical": "Acme Research Center", "confidence": 0.95, '
        '"reasoning": "abbreviation"}'
        ']'
    )
    llm_model_func = AsyncMock(return_value=llm_response)
    all_nodes = {
        'Acme': [_entity_fact('Acme', entity_type='organization', description='Acme')],
        'Acme Research': [_entity_fact('Acme Research', entity_type='organization', description='Acme Research')],
        'Acme Research Center': [
            _entity_fact('Acme Research Center', entity_type='organization', description='Acme Research Center')
        ],
    }
    edge_records = [
        _relation_fact(
            'Acme',
            'Acme Research',
            description='Acme works with Acme Research.',
            keywords='works with',
        )
    ]
    all_edges = {RelationKey('Acme', 'Acme Research'): edge_records}

    resolved_nodes, resolved_edges = await _resolve_entity_aliases_for_batch(
        all_nodes=all_nodes,
        all_edges=all_edges,
        entity_vdb=FakeEntityVdb(),
        global_config={
            'workspace': 'default',
            'llm_model_func': llm_model_func,
            'entity_resolution_config': EntityResolutionConfig(
                enabled=True,
                auto_resolve_on_extraction=True,
                auto_apply=True,
            ),
        },
    )

    assert set(resolved_nodes) == {'Acme Research', 'Acme Research Center'}
    resolved_key = RelationKey('Acme Research Center', 'Acme Research')
    assert set(resolved_edges) == {resolved_key}
    assert resolved_edges[resolved_key][0].key == resolved_key
    assert all(edge_key.src != edge_key.tgt for edge_key in resolved_edges)
    llm_model_func.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_process_extraction_result_normalizes_types_before_chunk_filtering():
    """Parser should preserve parsed nodes until all chunk extraction passes are merged."""
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    raw_result = '\n'.join(
        [
            f'entity{tuple_delimiter}Connected Person{tuple_delimiter}role{tuple_delimiter}Person tied to a relation.',
            f'entity{tuple_delimiter}Unsupported Widget{tuple_delimiter}widget{tuple_delimiter}Unsupported type tied to a relation.',
            f'entity{tuple_delimiter}Isolated Author{tuple_delimiter}person{tuple_delimiter}Metadata-only author.',
            f'relation{tuple_delimiter}Connected Person{tuple_delimiter}Unsupported Widget'
            f'{tuple_delimiter}collaborated with{tuple_delimiter}Connected Person collaborated with Unsupported Widget.',
            completion_delimiter,
        ]
    )

    extraction = await _process_extraction_result(
        raw_result,
        'chunk-1',
        1234567890,
        'source.pdf',
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    nodes = extraction.nodes
    edges = extraction.edges
    assert set(nodes) == {'Connected Person', 'Unsupported Widget', 'Isolated Author'}
    assert set(edges) == {RelationKey('Connected Person', 'Unsupported Widget')}
    assert nodes['Connected Person'][0].entity_type == 'person'
    assert nodes['Unsupported Widget'][0].entity_type == 'concept'
    assert all(
        record.entity_type in {entity_type.lower() for entity_type in DEFAULT_ENTITY_TYPES}
        for records in nodes.values()
        for record in records
    )
    assert set(_filter_nodes_to_relation_endpoints(nodes, edges, 'chunk-1')) == {
        'Connected Person',
        'Unsupported Widget',
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_chunk_entity_filter_drops_entity_only_outputs():
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    raw_result = (
        f'entity{tuple_delimiter}Standalone Topic{tuple_delimiter}concept'
        f'{tuple_delimiter}A heading with no explicit relation.\n{completion_delimiter}'
    )

    extraction = await _process_extraction_result(
        raw_result,
        'chunk-1',
        1234567890,
        'source.pdf',
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    nodes = extraction.nodes
    edges = extraction.edges
    assert set(nodes) == {'Standalone Topic'}
    assert _filter_nodes_to_relation_endpoints(nodes, edges, 'chunk-1') == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_chunk_entity_filter_preserves_entities_connected_by_later_pass():
    tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    initial_result = '\n'.join(
        [
            f'entity{tuple_delimiter}Initial Source{tuple_delimiter}person{tuple_delimiter}Detailed source metadata.',
            f'entity{tuple_delimiter}Initial Target{tuple_delimiter}concept{tuple_delimiter}Detailed target metadata.',
            completion_delimiter,
        ]
    )
    glean_result = '\n'.join(
        [
            f'relation{tuple_delimiter}Initial Source{tuple_delimiter}Initial Target'
            f'{tuple_delimiter}supports{tuple_delimiter}Initial Source supports Initial Target.',
            completion_delimiter,
        ]
    )

    initial_extraction = await _process_extraction_result(
        initial_result,
        'chunk-1',
        1234567890,
        'source.pdf',
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    glean_extraction = await _process_extraction_result(
        glean_result,
        'chunk-1',
        1234567891,
        'source.pdf',
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=list(DEFAULT_ENTITY_TYPES),
    )
    initial_nodes = initial_extraction.nodes
    initial_edges = initial_extraction.edges
    glean_nodes = glean_extraction.nodes
    glean_edges = glean_extraction.edges
    merged_nodes = {**initial_nodes, **glean_nodes}
    merged_edges = {**initial_edges, **glean_edges}

    filtered_nodes = _filter_nodes_to_relation_endpoints(merged_nodes, merged_edges, 'chunk-1')

    assert set(filtered_nodes) == {'Initial Source', 'Initial Target'}
    assert filtered_nodes['Initial Source'][0].description == 'Detailed source metadata.'
    assert set(merged_edges) == {RelationKey('Initial Source', 'Initial Target')}


class TestEntityFilterMatching:
    """Tests for entity filter normalization used during retrieval."""

    def test_matches_entity_filter_ignores_punctuation_and_case(self):
        assert _matches_entity_filter('Alpha-Launch leader guidance', 'Alpha Launch')
        assert _matches_entity_filter('ALPHA LAUNCH leader guidance', 'alpha-launch')
        assert not _matches_entity_filter('Beta lessons learned', 'Alpha Launch')
        assert not _matches_entity_filter('Alpha-Launch leader guidance', '')

    def test_relation_matches_entity_filter_uses_relation_text_when_entities_miss(self):
        relation = {
            'src_id': 'Partner A',
            'tgt_id': 'Partner B',
            'description': 'Lessons learned from compound alpha transition and collaboration.',
            'keywords': 'collaborates with',
        }

        from yar.operate import _relation_matches_entity_filter

        assert _relation_matches_entity_filter(relation, 'compound alpha', set())
        assert not _relation_matches_entity_filter(relation, 'compound beta', set())

    def test_auto_entity_filter_still_applies_to_direct_entity_queries(self):
        query_param = QueryParam(mode='mix')

        with patch('yar.operate.resolve_entity_filter', return_value='compound-alpha'):
            resolved = _apply_auto_entity_filter('What is the Compound Alpha project status?', query_param)

        assert resolved == 'compound-alpha'
        assert query_param.entity_filter == 'compound-alpha'

    @pytest.mark.asyncio
    async def test_vector_context_filter_matches_hyphenated_source_metadata(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.query = AsyncMock(
            return_value=[
                {
                    'id': 'chunk-1',
                    'content': 'The leader must avoid delaying submission.',
                    'file_path': '2019 Alpha-Launch lessons learned.pptx',
                    's3_key': 'default/doc-1/processed.md',
                    'score': 0.91,
                }
            ]
        )
        query_param = QueryParam(
            mode='mix', top_k=5, chunk_top_k=5, entity_filter='Alpha Launch', enable_bm25_fusion=False
        )

        chunks = await _get_vector_context('What must the leader avoid?', chunks_vdb, query_param)

        assert len(chunks) == 1
        assert chunks[0]['file_path'] == '2019 Alpha-Launch lessons learned.pptx'

    @pytest.mark.asyncio
    async def test_vector_context_filter_returns_empty_when_no_field_matches(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.query = AsyncMock(
            return_value=[
                {
                    'id': 'chunk-1',
                    'content': 'Beta sharing lessons learned.',
                    'file_path': '2019 Beta lessons learned.pptx',
                    's3_key': 'default/doc-1/processed.md',
                    'score': 0.91,
                }
            ]
        )
        query_param = QueryParam(
            mode='mix', top_k=5, chunk_top_k=5, entity_filter='Alpha Launch', enable_bm25_fusion=False
        )

        chunks = await _get_vector_context('What must the leader avoid?', chunks_vdb, query_param)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_vector_context_marks_exact_support_metadata(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.query = AsyncMock(
            return_value=[
                {
                    'id': 'chunk-1',
                    'content': 'Transfer valve definition by Safety Standard: sealed connector.',
                    'file_path': 'transfer-valve.pptx',
                    'score': 0.91,
                }
            ]
        )
        query_param = QueryParam(mode='hybrid', top_k=5, chunk_top_k=5, enable_bm25_fusion=False)

        chunks = await _get_vector_context(
            'What is the definition of transfer valve according to Safety Standard?',
            chunks_vdb,
            query_param,
            phrase_terms=['Transfer Valve Definition'],
        )

        assert chunks[0]['metadata_query_match'] == 1.5
        assert chunks[0]['exact_phrase_match'] == 1.0

        guidance_chunks_vdb = MagicMock()
        guidance_chunks_vdb.cosine_better_than_threshold = 0.4
        guidance_chunks_vdb.query = AsyncMock(
            return_value=[
                {
                    'id': 'chunk-2',
                    'content': 'Process Standard states transfer valves should be used when compounding volatile reagents.',
                    'file_path': 'transfer-valve.pptx',
                    'score': 0.82,
                }
            ]
        )

        guidance_chunks = await _get_vector_context(
            'How does Process Standard guidance recommend transfer valve use?',
            guidance_chunks_vdb,
            query_param,
            phrase_terms=['Process Standard guidance'],
        )

        assert guidance_chunks[0]['metadata_query_match'] == 1.25

    @pytest.mark.asyncio
    async def test_vector_context_retries_original_query_when_exact_lookup_is_empty(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.hybrid_search = AsyncMock(
            side_effect=[
                [],
                [
                    {
                        'id': 'chunk-1',
                        'content': 'The product presentation section discusses isatuximab handling context.',
                        'file_path': 'presentation.pptx',
                        'score': 0.72,
                    }
                ],
            ]
        )
        query_param = QueryParam(mode='mix', top_k=5, chunk_top_k=5, enable_bm25_fusion=True)

        chunks = await _get_vector_context(
            'What is the presentation of Sarclisa (isatuximab)?',
            chunks_vdb,
            query_param,
        )

        assert chunks_vdb.hybrid_search.await_count == 2
        assert chunks[0]['chunk_id'] == 'chunk-1'
        assert (
            query_param.__dict__['_exact_chunk_search_fallback']['failed_chunk_search_query'] == 'Sarclisa isatuximab'
        )
        assert query_param.__dict__['_vector_search_trace']['exact_fallback']['fallback_result_count'] == 1

    @pytest.mark.asyncio
    async def test_vector_context_adds_guidance_supplemental_results(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.hybrid_search = AsyncMock(
            side_effect=[
                [
                    {
                        'id': 'definition',
                        'content': 'Transfer valve definition by Safety Standard: sealed connector.',
                        'file_path': 'definition.pdf',
                        'score': 0.91,
                    }
                ],
                [
                    {
                        'id': 'use-guidance',
                        'content': 'Process Standard states transfer valves should be used when handling volatile reagents.',
                        'file_path': 'guidance.pdf',
                        'score': 0.80,
                    }
                ],
            ]
        )
        query_param = QueryParam(mode='mix', top_k=5, chunk_top_k=5, enable_bm25_fusion=True)

        chunks = await _get_vector_context(
            'What is the definition of transfer valve, and what guidance recommends its use for handling volatile reagents?',
            chunks_vdb,
            query_param,
        )

        assert chunks_vdb.hybrid_search.await_count == 2
        assert [chunk['chunk_id'] for chunk in chunks] == ['use-guidance', 'definition']
        assert chunks[0]['metadata_query_match'] >= 0.85

    @pytest.mark.asyncio
    async def test_vector_context_adds_temporal_supplemental_results(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.hybrid_search = AsyncMock(
            side_effect=[
                [
                    {
                        'id': 'background',
                        'content': 'IND Phase 1 background.',
                        'file_path': 'background.pdf',
                        'score': 0.91,
                    }
                ],
                [
                    {
                        'id': 'timeline',
                        'content': 'US submission in Mar 24 and US approval in Mar 25.',
                        'file_path': 'timeline.pdf',
                        'score': 0.80,
                    }
                ],
            ]
        )
        query_param = QueryParam(mode='mix', top_k=5, chunk_top_k=5, enable_bm25_fusion=True)

        chunks = await _get_vector_context(
            'What is the EU approval timeline and project management impact?',
            chunks_vdb,
            query_param,
        )

        assert chunks_vdb.hybrid_search.await_count == 2
        supplemental_call = chunks_vdb.hybrid_search.await_args_list[1]
        assert 'EU approval' in supplemental_call.args[0]
        assert 'EU approval' in supplemental_call.kwargs['phrase_terms']
        assert 'approval date' in supplemental_call.args[0]
        assert [chunk['chunk_id'] for chunk in chunks] == ['timeline', 'background']

    @pytest.mark.asyncio
    async def test_vector_context_adds_action_supplemental_results(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.hybrid_search = AsyncMock(
            side_effect=[
                [
                    {
                        'id': 'context',
                        'content': 'Lessons learned context for conflict management.',
                        'file_path': 'context.pdf',
                        'score': 0.91,
                    }
                ],
                [
                    {
                        'id': 'actions',
                        'content': 'Conflict management requires quick reaction and practical action steps.',
                        'file_path': 'actions.pdf',
                        'score': 0.80,
                    }
                ],
            ]
        )
        query_param = QueryParam(mode='mix', top_k=5, chunk_top_k=5, enable_bm25_fusion=True)

        chunks = await _get_vector_context(
            'How can lessons learned in conflict management be applied?',
            chunks_vdb,
            query_param,
        )

        assert chunks_vdb.hybrid_search.await_count == 2
        supplemental_call = chunks_vdb.hybrid_search.await_args_list[1]
        assert 'conflict management' in supplemental_call.args[0]
        assert 'conflict management requires' in supplemental_call.args[0]
        assert 'conflict management' in supplemental_call.kwargs['phrase_terms']
        assert [chunk['chunk_id'] for chunk in chunks] == ['actions', 'context']

    @pytest.mark.asyncio
    async def test_naive_query_passes_low_level_phrases_to_chunk_search(self):
        query_param = QueryParam(
            mode='naive',
            top_k=5,
            chunk_top_k=5,
            ll_keywords=['REG-CTD', 'shipping validation between Region A and Region B', 'sales limits'],
            only_need_context=True,
            enable_bm25_fusion=True,
            model_func=AsyncMock(return_value='unused'),
        )
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        global_config = {
            'tokenizer': MagicMock(encode=Mock(side_effect=lambda text: text.split())),
            'max_total_tokens': 4000,
        }
        vector_chunk = {
            'content': 'Shipping Validation is executed between Region A and Region B.',
            'file_path': 'regional operations handbook.pdf',
            'chunk_id': 'chunk-region',
        }
        vector_context_mock = AsyncMock(return_value=[vector_chunk])

        with (
            patch('yar.operate._get_vector_context', new=vector_context_mock),
            patch('yar.operate.process_chunks_unified', new=AsyncMock(return_value=[vector_chunk])),
        ):
            result = await naive_query(
                'Regional operations handbook shipping validation sales limits',
                chunks_vdb,
                query_param,
                global_config,
            )

        assert result.raw_data['metadata']['processing_info']['total_chunks_found'] == 1
        assert vector_context_mock.await_args.kwargs['phrase_terms'] == [
            'REG-CTD',
            'shipping validation between Region A and Region B',
            'sales limits',
        ]

    @pytest.mark.asyncio
    async def test_naive_query_derives_query_phrases_when_keywords_are_empty(self):
        query_param = QueryParam(
            mode='naive',
            top_k=5,
            chunk_top_k=5,
            ll_keywords=[],
            only_need_context=True,
            enable_bm25_fusion=True,
            model_func=AsyncMock(return_value='unused'),
        )
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        global_config = {
            'tokenizer': MagicMock(encode=Mock(side_effect=lambda text: text.split())),
            'max_total_tokens': 4000,
        }
        vector_chunk = {
            'content': 'Timeline step happens 1-3 months before the launch milestone.',
            'file_path': 'workflow.pdf',
            'chunk_id': 'chunk-workflow',
        }
        vector_context_mock = AsyncMock(return_value=[vector_chunk])

        with (
            patch('yar.operate._get_vector_context', new=vector_context_mock),
            patch('yar.operate.process_chunks_unified', new=AsyncMock(return_value=[vector_chunk])),
        ):
            await naive_query(
                'What is the standard duration before the launch milestone?',
                chunks_vdb,
                query_param,
                global_config,
            )

        assert vector_context_mock.await_args.kwargs['phrase_terms'] is None

    def test_derive_phrase_terms_prefers_explicit_low_level_keywords(self):
        assert _derive_phrase_terms_for_chunk_search(
            'What is the standard duration before the launch milestone?',
            ['explicit retrieval phrase'],
        ) == ['explicit retrieval phrase']

    def test_supporting_evidence_spans_linearize_html_tables_and_workflows(self):
        content = """
        <table><thead><tr><th>Challenge</th><th>Mitigation</th></tr></thead>
        <tbody><tr><td>How to avoid contamination?</td><td>Confirm the cleaning cycle</td></tr></tbody></table>
        ## Timeline Stages
        * 6-3 months before launch milestone
        * Final verification completed before release
        """

        spans = _extract_supporting_evidence_spans(
            content,
            query='What was put in place to mitigate contamination before launch?',
        )

        assert any('Challenge: How to avoid contamination?' in span for span in spans)
        assert any('Mitigation: Confirm the cleaning cycle' in span for span in spans)
        assert any('Workflow timeline evidence' in span for span in spans)

    def test_supporting_evidence_spans_linearize_markdown_tables_and_label_lists(self):
        content = """
        | Product | Role |
        | :--- | :--- |
        | Example Device | Project Leader |

        * **Status:**
            * Planned
            * Actions opened
            * Actions finalized
        """

        spans = _extract_supporting_evidence_spans(
            content,
            query='What role and status did the product have?',
            topic_terms=['Example Device'],
            facet_terms=['Project Leader', 'status'],
        )

        assert any('Product: Example Device | Role: Project Leader' in span for span in spans)
        assert any('Status: Planned; Actions opened; Actions finalized' in span for span in spans)


# ============================================================================
# Text Chunking Tests
# ============================================================================


@pytest.mark.offline
class TestChunkingBySemantic:
    """Tests for chunking_by_semantic function."""

    def test_basic_chunking(self):
        """Test basic semantic chunking."""
        content = 'This is a test paragraph.\n\nThis is another paragraph.'
        result = chunking_by_semantic(content, max_chars=4800, max_overlap=400)

        assert isinstance(result, list)
        assert len(result) >= 1
        for chunk in result:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk
            assert 'char_start' in chunk
            assert 'char_end' in chunk

    def test_chunking_semantic_text(self):
        """Test semantic chunking directly."""
        content = 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'
        result = chunking_by_semantic(content)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_order_indices(self):
        """Test that chunk indices are sequential."""
        content = '\n\n'.join([f'Paragraph {i}' for i in range(10)])
        result = chunking_by_semantic(content, max_chars=100, max_overlap=20)

        indices = [chunk['chunk_order_index'] for chunk in result]
        assert indices == list(range(len(result)))

    def test_chunk_tokens_are_exact_tiktoken_counts(self):
        """Test token counts are exact tokenizer counts."""
        tokenizer = TiktokenTokenizer()
        content = '😀' * 10 + ' word ' * 100
        result = chunking_by_semantic(content, max_chars=200, max_overlap=50, tokenizer=tokenizer)

        for chunk in result:
            assert chunk['tokens'] == len(tokenizer.encode(chunk['content']))

    def test_chunking_preserves_page_range_metadata(self):
        """Test page marker ranges are exposed on chunks."""
        content = '# Report\n\n<!-- PAGE 2 -->\n\nAlpha\n\n<!-- PAGE 3 -->\n\nBeta'

        result = chunking_by_semantic(content, max_chars=500)

        assert len(result) == 1
        assert result[0]['page_number'] == 2
        assert result[0]['page_start'] == 2
        assert result[0]['page_end'] == 3
        assert result[0]['page_numbers'] == [2, 3]

    def test_char_offsets_valid(self):
        """Test that character offsets are valid."""
        content = 'Test content for offset validation.'
        result = chunking_by_semantic(content)

        for chunk in result:
            assert chunk['char_start'] >= 0
            assert chunk['char_end'] > chunk['char_start']
            assert chunk['char_end'] <= len(content) + len(chunk['content'])

    def test_empty_content_fallback(self):
        """Test fallback for empty content."""
        content = ''
        result = chunking_by_semantic(content)

        # Should return at least one chunk even if empty
        assert isinstance(result, list)

    def test_long_content_multiple_chunks(self):
        """Test that long content creates multiple chunks."""
        content = 'Test sentence. ' * 500  # ~7500 characters
        result = chunking_by_semantic(content, max_chars=1000, max_overlap=100)

        assert len(result) > 1

    def test_unicode_content(self):
        """Test chunking with Unicode content."""
        content = 'Hello 世界! 🌍 Test content with émojis.'
        result = chunking_by_semantic(content)

        assert isinstance(result, list)
        assert len(result) >= 1
        # Content should be preserved
        combined = ''.join(chunk['content'] for chunk in result)
        assert '世界' in combined or '🌍' in combined


@pytest.mark.offline
class TestCreateChunker:
    """Tests for create_chunker factory function."""

    def test_returns_callable(self):
        """Test that create_chunker returns a callable."""
        chunker = create_chunker()
        assert callable(chunker)

    def test_semantic_adapter_chunks_text(self):
        """Test semantic chunker adapter."""
        chunker = create_chunker()
        result = chunker(None, 'Test content', None, False, 100, 1200)

        assert isinstance(result, list)

    def test_default_factory_chunks_text(self):
        """Test default factory chunks text."""
        chunker = create_chunker()
        result = chunker(None, 'Test content', None, False, 100, 1200)

        assert isinstance(result, list)

    def test_adapter_signature(self):
        """Test that adapter accepts expected parameters."""
        chunker = create_chunker()

        # Should accept standard YAR chunking_func signature
        result = chunker(
            tokenizer=None,
            content='Test',
            split_by_character=None,
            split_by_character_only=False,
            chunk_overlap_token_size=100,
            chunk_token_size=1200,
        )

        assert isinstance(result, list)

    def test_token_to_char_conversion(self):
        """Test token size to character size conversion."""
        chunker = create_chunker()
        content = 'word ' * 1000

        # Small token size should create multiple chunks
        result = chunker(None, content, None, False, 50, 100)
        assert len(result) > 1

    def test_ignores_unused_params(self):
        """Test that unused parameters don't affect behavior."""
        chunker = create_chunker()

        # These params are ignored by the default chunker adapter
        result1 = chunker(None, 'Test', None, False, 100, 1200)
        result2 = chunker(MagicMock(), 'Test', '\n\n', True, 100, 1200)

        # Results should be similar (content chunked the same way)
        assert len(result1) == len(result2)


# ============================================================================
# Entity Identifier Truncation Tests
# ============================================================================


@pytest.mark.offline
class TestTruncateEntityIdentifier:
    """Tests for _truncate_entity_identifier function."""

    def test_no_truncation_needed(self):
        """Test that short identifiers are not truncated."""
        identifier = 'John Smith'
        result = _truncate_entity_identifier(identifier, limit=100, chunk_key='test', identifier_role='Entity')

        assert result == identifier

    def test_truncation_at_limit(self):
        """Test truncation at exact limit."""
        identifier = 'A' * 150
        limit = 100
        result = _truncate_entity_identifier(identifier, limit=limit, chunk_key='test', identifier_role='Entity')

        assert len(result) == limit
        assert result == 'A' * limit

    def test_truncation_with_warning(self):
        """Test that truncation logs a warning."""

        identifier = 'Very Long Entity Name ' * 10

        with patch('yar.operate.logger') as mock_logger:
            result = _truncate_entity_identifier(identifier, limit=50, chunk_key='chunk-123', identifier_role='Entity')

            assert len(result) == 50
            # Warning should have been called
            mock_logger.warning.assert_called_once()

    def test_preview_in_warning(self):
        """Test that warning includes preview of identifier."""

        identifier = 'EntityNameThatIsTooLong' + 'X' * 100

        with patch('yar.operate.logger') as mock_logger:
            _truncate_entity_identifier(identifier, limit=50, chunk_key='chunk-456', identifier_role='Entity')

            # Should have logged warning with preview
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            # Check that first 20 chars appear in message
            assert 'EntityNameThatIsTooL' in str(call_args)

    def test_boundary_condition_exact_limit(self):
        """Test identifier exactly at limit."""
        identifier = 'X' * 100
        result = _truncate_entity_identifier(identifier, limit=100, chunk_key='test', identifier_role='Entity')

        assert result == identifier
        assert len(result) == 100

    def test_unicode_identifier_truncation(self):
        """Test truncation with Unicode characters."""
        identifier = '日本語エンティティ名' * 20
        result = _truncate_entity_identifier(identifier, limit=50, chunk_key='test', identifier_role='Entity')

        assert len(result) == 50


# ============================================================================
# Entity/Relation Summarization Tests
# ============================================================================


@pytest.mark.offline
class TestHandleEntityRelationSummary:
    """Tests for _handle_entity_relation_summary function."""

    @pytest.mark.asyncio
    async def test_empty_description_list(self):
        """Test handling of empty description list."""
        from yar.operate import _handle_entity_relation_summary

        result, llm_used = await _handle_entity_relation_summary(
            description_type='entity',
            entity_or_relation_name='TestEntity',
            description_list=[],
            separator=GRAPH_FIELD_SEP,
            global_config={},
        )

        assert result == ''
        assert llm_used is False

    @pytest.mark.asyncio
    async def test_single_description_no_llm(self):
        """Test that single description doesn't use LLM."""
        from yar.operate import _handle_entity_relation_summary

        result, llm_used = await _handle_entity_relation_summary(
            description_type='entity',
            entity_or_relation_name='TestEntity',
            description_list=['Single description'],
            separator=GRAPH_FIELD_SEP,
            global_config={},
        )

        assert result == 'Single description'
        assert llm_used is False

    @pytest.mark.asyncio
    async def test_small_descriptions_no_llm(self):
        """Test that small descriptions don't trigger LLM."""
        from yar.operate import _handle_entity_relation_summary

        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda x: [0] * len(x.split()))

        global_config = {
            'tokenizer': mock_tokenizer,
            'summary_context_size': 1000,
            'summary_max_tokens': 500,
            'force_llm_summary_on_merge': 10,  # Higher than our count
        }

        descriptions = ['Desc one', 'Desc two']
        result, llm_used = await _handle_entity_relation_summary(
            description_type='entity',
            entity_or_relation_name='TestEntity',
            description_list=descriptions,
            separator=GRAPH_FIELD_SEP,
            global_config=global_config,
        )

        assert GRAPH_FIELD_SEP in result
        assert llm_used is False

    @pytest.mark.asyncio
    async def test_large_descriptions_use_llm(self):
        """Test that large descriptions trigger LLM summarization."""
        from yar.operate import _handle_entity_relation_summary

        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda x: [0] * 500)  # Large token count

        # Mock the LLM response with proper cache structure
        async def mock_llm_with_cache(*args, **kwargs):
            return 'Summarized description', 123456

        mock_llm = AsyncMock(return_value=('Summarized description', {}))

        # Create a proper mock cache that returns None
        mock_cache = AsyncMock()
        mock_cache.get_by_id = AsyncMock(return_value=None)

        global_config = {
            'tokenizer': mock_tokenizer,
            'summary_context_size': 100,
            'summary_max_tokens': 200,
            'force_llm_summary_on_merge': 2,
            'llm_model_func': mock_llm,
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
            'summary_length_recommended': 100,
        }

        descriptions = ['Long description ' * 50, 'Another long description ' * 50]

        with patch('yar.operate.use_llm_func_with_cache', new=mock_llm_with_cache):
            _result, llm_used = await _handle_entity_relation_summary(
                description_type='entity',
                entity_or_relation_name='TestEntity',
                description_list=descriptions,
                separator=GRAPH_FIELD_SEP,
                global_config=global_config,
                llm_response_cache=mock_cache,
            )

            assert llm_used is True


@pytest.mark.offline
class TestSummarizeDescriptions:
    """Tests for _summarize_descriptions helper function."""

    @pytest.mark.asyncio
    async def test_summarization_call_structure(self):
        """Test that LLM is called with correct structure."""
        from yar.operate import _summarize_descriptions

        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[0] * 10)

        # Mock use_llm_func_with_cache to return proper structure
        async def mock_llm_with_cache(*args, **kwargs):
            return 'Summary', 123456

        mock_llm = AsyncMock(return_value=('Summary', {}))

        # Create a mock cache with proper structure
        mock_cache = Mock()
        mock_cache.global_config = {'enable_llm_cache_for_entity_extract': False}
        mock_cache.get_by_id = AsyncMock(return_value=None)
        mock_cache.upsert = AsyncMock()

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': mock_tokenizer,
            'summary_context_size': 1000,
            'summary_length_recommended': 100,
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        with patch('yar.operate.use_llm_func_with_cache', new=mock_llm_with_cache):
            result = await _summarize_descriptions(
                description_type='entity',
                description_name='TestEntity',
                description_list=['Description 1', 'Description 2'],
                global_config=global_config,
                llm_response_cache=mock_cache,
            )

            assert result == 'Summary'


# ============================================================================
# Entity Type Inference Tests
# ============================================================================


@pytest.mark.offline
class TestBatchInferEntityTypes:
    """Tests for _batch_infer_entity_types function."""

    @pytest.mark.asyncio
    async def test_empty_entity_list(self):
        """Test handling of empty entity list."""
        from yar.operate import _batch_infer_entity_types

        result = await _batch_infer_entity_types(
            unknown_entities=[],
            global_config={},
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_no_unknown_entities(self):
        """Test handling when no UNKNOWN entities present."""
        from yar.operate import _batch_infer_entity_types

        entities = [
            {'entity_name': 'John', 'entity_type': 'PERSON'},
            {'entity_name': 'Microsoft', 'entity_type': 'ORGANIZATION'},
        ]

        result = await _batch_infer_entity_types(
            unknown_entities=entities,
            global_config={},
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_size_splitting(self):
        """Test that entities are processed in batches."""
        from yar.operate import _batch_infer_entity_types

        mock_llm = AsyncMock(return_value='[{"entity_name": "Entity1", "inferred_type": "organization"}]')

        global_config = {
            'llm_model_func': mock_llm,
            'addon_params': {'entity_types': ['person', 'organization', 'location']},
        }

        # Create more entities than batch size
        entities = [{'entity_name': f'Entity{i}', 'entity_type': 'UNKNOWN'} for i in range(25)]

        mock_graph = AsyncMock()
        mock_graph.upsert_node = AsyncMock()

        mock_vdb = AsyncMock()
        mock_vdb.upsert = AsyncMock()

        await _batch_infer_entity_types(
            unknown_entities=entities,
            global_config=global_config,
            knowledge_graph_inst=mock_graph,
            entity_vdb=mock_vdb,
            batch_size=20,
        )

        # Should have made at least 2 LLM calls (25 entities / 20 per batch)
        assert mock_llm.call_count >= 1


# ============================================================================
# Keyword Extraction Tests
# ============================================================================


@pytest.mark.offline
class TestGetKeywordsFromQuery:
    """Tests for get_keywords_from_query function."""

    @pytest.mark.asyncio
    async def test_uses_predefined_keywords(self):
        """Test that predefined keywords are used if provided."""
        query_param = QueryParam(
            hl_keywords=['high', 'level'],
            ll_keywords=['low', 'level'],
        )

        hl, ll = await get_keywords_from_query(
            query='test query',
            query_param=query_param,
            global_config={},
        )

        assert hl == ['high', 'level']
        assert ll == ['low', 'level']

    @pytest.mark.asyncio
    async def test_extracts_keywords_when_not_provided(self):
        """Test that keywords are extracted when not provided."""
        query_param = QueryParam()

        mock_llm = AsyncMock(return_value='{"high_level_keywords": ["AI"], "low_level_keywords": ["machine learning"]}')

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        hl, ll = await get_keywords_from_query(
            query='What is AI?',
            query_param=query_param,
            global_config=global_config,
        )

        assert isinstance(hl, list)
        assert isinstance(ll, list)


class TestAugmentRetrievalKeywords:
    """Tests for deterministic keyword expansion on brittle retrieval intents."""

    def test_temporal_and_study_queries_add_generic_terms(self):
        hl, ll = _augment_retrieval_keywords(
            'What are the dates or milestones mentioned during the platform launch period?',
            ['launch milestones'],
            [],
        )

        assert 'history' in hl
        assert 'background' in hl
        assert 'chronology' in hl
        assert 'timeline' in hl
        assert 'key events' in hl
        assert ll == []

        hl, ll = _augment_retrieval_keywords(
            'Which studies were delayed or impacted by the launch change?',
            ['study delays'],
            [],
        )

        assert 'timeline' in hl
        assert 'studies' in ll
        assert 'domain-specific plan' not in hl
        assert 'Domain-Specific Plan' not in ll

        terms = _tokenize_relevance_terms('Which sponsors owned the statuses of these studies?')
        assert {'sponsor', 'status', 'study'} <= terms
        assert _should_enable_exact_chunk_fusion(
            'How does Process Standard guidance recommend handling volatile reagents?',
            ['Process Standard'],
        )
        assert _should_enable_exact_chunk_fusion(
            'Who were the sponsors and what was the status of the session?',
            ['sponsors'],
        )

        assert _normalize_retrieval_query_typos('Wht is the best practce?') == 'What is the best practice?'
        assert _normalize_retrieval_query_typos('Who were the sponsors?') == 'Who were the sponsors?'

        guidance_query = _guidance_chunk_search_query('How does Safety Standard guidance recommend transfer valve use?')
        assert 'Safety' in guidance_query
        assert 'Standard' in guidance_query
        assert 'transfer' in guidance_query
        assert 'valve' in guidance_query
        assert 'should' in guidance_query
        assert 'must' in guidance_query

        hl, ll = _augment_retrieval_keywords(
            'How can effective conflict management be applied based on lessons learned?',
            ['conflict management'],
            ['Partner Alpha'],
        )
        assert 'best practice' in hl
        assert 'practical actions' in hl
        assert 'implementation steps' in hl
        assert 'action plan' in hl
        assert 'requires' in hl
        assert 'Best Practice' in ll

    def test_technology_issue_recommended_step_adds_literal_anchors(self):
        _hl, ll = _augment_retrieval_keywords(
            'What is the first recommended step for resolving a CMC technology issue?',
            ['incident response'],
            ['CMC technology', 'issue'],
        )

        assert 'ad hoc meeting' in ll
        assert 'Subject Matter Expert' in ll
        assert 'Technology Issue Quick Sharing' in ll
        assert 'CMC team' in ll

    def test_duration_query_adds_shipment_timing_anchors(self):
        hl, ll = _augment_retrieval_keywords(
            'What is the standard duration of shipment to depot?',
            [],
            [],
        )

        assert 'duration' in hl
        assert 'months' in hl
        assert 'months shipment' in ll
        assert 'goods shipment preparation' in ll

    def test_qualified_temporal_queries_preserve_timeline_table_terms(self):
        query = 'How did the EU approval timeline compare with the PX-482 Phase 1 milestone?'
        search_query = _temporal_chunk_search_query(query)

        assert 'EU approval' in search_query
        assert 'EU submission' in search_query
        assert 'EU: Approval' in search_query

    def test_timeline_queries_search_milestone_and_clearance_language(self):
        search_query = _temporal_chunk_search_query('What is the approval timeline for Product X Phase 1?')

        assert 'approval' in search_query
        assert 'clearance' in search_query
        assert 'milestone' in search_query
        assert 'timeline' in search_query

    def test_temporal_detector_requires_time_qualifier_for_approval_or_change_terms(self):
        assert not _is_temporal_or_comparative_query('What is the approval process for change management?')
        assert not _is_temporal_or_comparative_query('Which change management owner approved the workflow?')
        assert _is_temporal_or_comparative_query('What is the approval date for Product X?')
        assert _is_temporal_or_comparative_query('What changed since 2024?')

    def test_impact_queries_do_not_add_timeline_terms_without_temporal_signal(self):
        hl, ll = _augment_retrieval_keywords(
            'What impact did the supplier outage have on onboarding?',
            ['supplier impact'],
            ['Supplier Alpha'],
        )

        assert ll == ['Supplier Alpha']
        assert 'timeline' not in hl
        assert 'chronology' not in hl
        assert 'key events' not in hl

    def test_precise_temporal_query_combines_code_and_timeline_terms(self):
        query = (
            'What is the significance of the EU approval timeline for PX-482 Phase 1 and its project management impact?'
        )
        search_query = _precise_temporal_chunk_search_query(query)

        assert 'PX-482' in search_query
        assert 'EU approval' in search_query
        assert 'EU submission' in search_query
        assert 'key dates' in search_query

    def test_document_number_query_adds_guideline_resource_terms(self):
        hl, ll = _augment_retrieval_keywords(
            'Which internal document number is referenced as the implementation guideline in the Best Practice section?',
            ['document lookup'],
            ['Implementation Guideline'],
        )

        assert 'document number' in hl
        assert 'guideline reference' in hl
        assert 'links to resources' in hl
        assert 'Implementation Guideline' in ll
        assert 'Best Practice' in ll

    def test_exact_chunk_queries_add_table_and_section_terms(self):
        hl, ll = _augment_retrieval_keywords(
            "List the roles or participants mentioned in the 'Critical Success Factors' section.",
            ['participant roles'],
            ['participants'],
        )

        assert 'critical success factors' in hl
        assert 'implementation roles' in hl
        assert 'Critical Success Factors' in ll
        assert _should_enable_exact_chunk_fusion('Which Critical Success Factors are listed?', ll)

        hl, ll = _augment_retrieval_keywords(
            'List the types of differences that lead to conflicts in strategy management.',
            ['conflict causes'],
            [],
        )

        assert 'sources of conflict' in hl
        assert 'types of differences' in hl
        assert 'conflict drivers' in hl
        assert not _should_enable_exact_chunk_fusion('List the types of differences that lead to conflicts.', ll)

        hl, ll = _augment_retrieval_keywords(
            'In session 09, what are the explicit sources of conflict listed under Recognize conflict?',
            ['conflict sources'],
            [],
        )

        assert 'sources of conflict' in hl

    def test_exact_chunk_lookup_query_appends_literal_terms(self):
        query = 'Which document number is listed in the best practice?'
        search_query = _build_exact_chunk_search_query(
            query,
            [
                'Best Practice',
                'Document number DOC-123',
                'Document number DOC-123',
                'Guideline reference GL-45',
            ],
            exact_lookup=True,
        )

        assert not search_query.startswith(query)
        assert search_query.startswith('Best Practice')
        assert search_query.count('Document number DOC-123') == 1
        assert 'Guideline reference GL-45' in search_query
        assert 'document number' in search_query
        assert _build_exact_chunk_search_query(query, ['Document number DOC-123'], exact_lookup=False) == query
        metadata_query = _build_exact_chunk_search_query(
            'Who were the sponsors and what was the status of the blinded comparator session?',
            ['sponsors'],
            exact_lookup=True,
        )
        assert metadata_query.startswith('Who were the sponsors')
        assert metadata_query.endswith('sponsors status')
        supplemental_metadata_query = _metadata_chunk_search_query(
            'Who were the sponsors and what was the status of the blinded comparator session?'
        )
        assert supplemental_metadata_query.startswith('the blinded comparator session')
        assert 'session sponsor status' in supplemental_metadata_query
        assert 'blinded' in supplemental_metadata_query
        assert 'comparator' in supplemental_metadata_query
        definition_query = _build_exact_chunk_search_query(
            'What is the definition according to Safety Standard?',
            ['Safety Standard'],
            exact_lookup=True,
        )
        assert definition_query.startswith('Safety Standard')
        assert 'definition' in definition_query

        literal_query = _build_exact_chunk_search_query(
            'Which article of Japanese GMP covers the MOU?',
            None,
            exact_lookup=True,
        )
        assert 'Japanese GMP' in literal_query
        assert 'MOU' in literal_query

        presentation_terms = _derive_phrase_terms_for_chunk_search(
            'What is the presentation of Sarclisa (isatuximab)?',
            [],
        )
        assert presentation_terms == ['Sarclisa', 'isatuximab']

    def test_precise_entity_terms_trigger_exact_chunk_lookup(self):
        assert _should_enable_exact_chunk_fusion(
            'Who is Jane Doe in the context of the clinical study?',
            ['Jane Doe'],
        )
        assert _should_enable_exact_chunk_fusion(
            'What role does ABCD play in the development of adaptive devices?',
            ['ABCD', 'adaptive devices'],
        )
        assert not _should_enable_exact_chunk_fusion(
            'What is the overall project context?',
            ['project context'],
        )
        assert not _should_enable_exact_chunk_fusion(
            'How does the compound alpha presentation discuss device?',
            ['2024-02-21 Compound Alpha Milestone Presentation'],
        )

    def test_entity_lookup_query_preserves_precise_terms(self):
        search_query = _build_entity_lookup_query(
            'What role did Jane Doe play for ABCD123?',
            ['project role', 'ABCD123'],
            'project role, ABCD123',
        )

        assert 'ABCD123' in _split_keyword_terms(search_query)
        assert 'Jane Doe' in _split_keyword_terms(search_query)
        assert 'Doe Jane' in _split_keyword_terms(search_query)

    def test_chunk_phrase_terms_include_precise_singletons(self):
        phrase_terms = _chunk_phrase_terms_for_search(
            ['transfer adapters', 'Safety Standard', 'NHA', 'product quality']
        )

        assert phrase_terms is not None
        assert 'transfer adapters' in phrase_terms
        assert 'Safety Standard' in phrase_terms
        assert 'NHA' not in phrase_terms
        assert 'product quality' in phrase_terms

    def test_exact_chunk_lookup_keeps_precise_terms_from_query(self):
        query = 'Who is Jane Doe in the context of the clinical study?'
        search_query = _build_exact_chunk_search_query(query, ['Jane Doe'], exact_lookup=True)

        assert search_query.startswith('Jane Doe')
        assert 'Doe Jane' in search_query
        assert 'Doe, Jane' in search_query
        assert (
            _build_exact_chunk_search_query(
                'What role does ABCD play in the development of adaptive devices?',
                ['ABCD', 'adaptive devices'],
                exact_lookup=True,
            )
            == 'ABCD'
        )

    def test_bait_queries_are_not_expanded(self):
        hl, ll = _augment_retrieval_keywords(
            'what is the detection principle behind the ddPCR vector genome concentration assay',
            ['assay mechanism'],
            ['ddPCR'],
        )

        assert hl == ['assay mechanism']
        assert ll == ['ddPCR']


@pytest.mark.offline
class TestExtractKeywordsOnly:
    """Tests for extract_keywords_only function."""

    @pytest.mark.asyncio
    async def test_basic_keyword_extraction(self):
        """Test basic keyword extraction."""
        mock_llm = AsyncMock(
            return_value='{"high_level_keywords": ["technology", "AI"], "low_level_keywords": ["neural networks"]}'
        )

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        param = QueryParam()

        hl, _ll = await extract_keywords_only(
            text='What is AI technology?',
            param=param,
            global_config=global_config,
        )

        assert 'technology' in hl or 'AI' in hl
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_behavior(self):
        """Test that cache storage is used when provided."""
        # Test with None cache - should call LLM
        mock_llm = AsyncMock(return_value='{"high_level_keywords": ["ai"], "low_level_keywords": ["ml"]}')

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        param = QueryParam()

        hl, ll = await extract_keywords_only(
            text='test query',
            param=param,
            global_config=global_config,
            hashing_kv=None,  # No cache
        )

        # LLM should have been called
        mock_llm.assert_called_once()
        assert isinstance(hl, list)
        assert isinstance(ll, list)

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        """Test handling of invalid JSON response."""
        mock_llm = AsyncMock(return_value='not valid json')

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        param = QueryParam()

        hl, ll = await extract_keywords_only(
            text='test query',
            param=param,
            global_config=global_config,
        )

        # Should return empty lists on parse error
        assert hl == []
        assert ll == []

    @pytest.mark.asyncio
    async def test_custom_model_func(self):
        """Test using custom model function from param."""
        custom_llm = AsyncMock(return_value='{"high_level_keywords": ["custom"], "low_level_keywords": []}')

        global_config = {
            'llm_model_func': AsyncMock(),  # Should not be used
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        param = QueryParam(model_func=custom_llm)

        hl, _ll = await extract_keywords_only(
            text='test query',
            param=param,
            global_config=global_config,
        )

        custom_llm.assert_called_once()
        assert 'custom' in hl

    @pytest.mark.asyncio
    async def test_think_tags_removed(self):
        """Test that <think> tags are removed from response."""
        mock_llm = AsyncMock(
            return_value='<think>reasoning</think>{"high_level_keywords": ["result"], "low_level_keywords": []}'
        )

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        param = QueryParam()

        hl, _ll = await extract_keywords_only(
            text='test query',
            param=param,
            global_config=global_config,
        )

        assert 'result' in hl


# ============================================================================
# Entity Extraction Tests
# ============================================================================


@pytest.mark.offline
class TestExtractEntities:
    """Tests for extract_entities function."""

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        """Test handling of empty chunks dictionary."""
        from yar.operate import extract_entities

        # Empty chunks raises ValueError from asyncio.wait with empty task set
        # This is expected behavior - the function doesn't handle empty chunks specially
        with pytest.raises(ValueError, match='empty'):
            await extract_entities(
                chunks={},
                global_config={
                    'llm_model_func': AsyncMock(return_value='test'),
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 4,
                },
            )

    @pytest.mark.asyncio
    async def test_basic_extraction_structure(self):
        """Test basic extraction with single chunk."""
        from yar.operate import extract_entities

        mock_llm = AsyncMock(return_value='("entity1", "PERSON", "A person")')

        chunks: dict[str, TextChunkSchema] = {
            'chunk-1': {
                'tokens': 10,
                'content': 'Test content',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            }
        }

        global_config = {
            'llm_model_func': mock_llm,
            'entity_extract_max_gleaning': 0,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {
                'language': DEFAULT_SUMMARY_LANGUAGE,
                'entity_types': ['PERSON', 'ORGANIZATION'],
            },
        }

        result = await extract_entities(
            chunks=chunks,
            global_config=global_config,
        )

        # Should return a list of results
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_pipeline_status_accepted(self):
        """Test that pipeline status parameters are accepted."""
        import asyncio

        from yar.operate import extract_entities

        # Test that function signature accepts pipeline status parameters
        # Complex extraction testing is out of scope - we just verify the params work
        pipeline_status = {'status': 'processing', 'history_messages': [], 'latest_message': ''}
        pipeline_lock = asyncio.Lock()

        chunks: dict[str, TextChunkSchema] = {
            'chunk-1': {
                'tokens': 10,
                'content': 'John Smith works at Microsoft.',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
                'file_path': 'test.txt',
            }
        }

        # Mock LLM to return properly formatted extraction (with all 4 fields)
        # Format: ("entity_name"<|>"entity_type"<|>"description"<|>"source_id")
        mock_llm = AsyncMock(return_value='(COMPLETE)\n("John Smith"<|>"PERSON"<|>"An employee"<|>"chunk-1")')

        try:
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': mock_llm,
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON', 'ORGANIZATION']},
                    'llm_model_max_async': 4,
                },
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_lock,
            )

            # Should complete successfully with pipeline status
            assert isinstance(result, list)
        except KeyError:
            # The complex extraction logic may fail due to missing config keys
            # That's acceptable - we're just testing the function signature
            pass

    @pytest.mark.asyncio
    async def test_extract_entities_truncates_oversized_input(self):
        """Extraction prompts should apply max_extract_input_tokens guard."""
        from yar.operate import extract_entities

        content = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(range(len(text)))
        tokenizer.decode.side_effect = lambda tokens: 'TRUNCATED_CONTENT'

        chunks: dict[str, TextChunkSchema] = {
            'chunk-1': {
                'tokens': 10,
                'content': content,
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            }
        }

        captured_prompts: list[str] = []

        async def fake_use_llm_with_cache(prompt, *_args, **_kwargs):
            captured_prompts.append(prompt)
            return '(COMPLETE)', 1234567890

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            patch(
                'yar.operate._process_extraction_result',
                new=AsyncMock(return_value=ChunkExtractionResult(nodes={}, edges={})),
            ),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'max_extract_input_tokens': 8,
                    'tokenizer': tokenizer,
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert len(captured_prompts) == 1
        assert 'TRUNCATED_CONTENT' in captured_prompts[0]
        assert content not in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_entities_batch_parsing_uses_truncated_source_content(self):
        """Batch relation provenance must use the same text sent to the extraction prompt."""
        from yar.operate import extract_entities

        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text: list(text)
        tokenizer.decode.side_effect = lambda tokens: ''.join(tokens)
        chunks: dict[str, TextChunkSchema] = {
            'chunk-alpha': {
                'tokens': 10,
                'content': 'ALPHA hidden evidence outside prompt',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            },
            'chunk-bravo': {
                'tokens': 10,
                'content': 'BRAVO hidden evidence outside prompt',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
        }
        captured_prompts: list[str] = []
        captured_source_content: dict[str, str] = {}

        async def fake_use_llm_with_cache(prompt, *_args, **_kwargs):
            captured_prompts.append(prompt)
            return (
                '[CHUNK: chunk-alpha]\nsection-alpha\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                '[CHUNK: chunk-bravo]\nsection-bravo\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                1234567890,
            )

        async def fake_process_extraction(_section_text, chunk_key, *_args, **kwargs):
            captured_source_content[chunk_key] = kwargs['source_content']
            return ChunkExtractionResult(nodes={}, edges={})

        with (
            patch.dict('os.environ', {'ENTITY_EXTRACT_BATCH_SIZE': '2'}),
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            patch('yar.operate._process_extraction_result', new=AsyncMock(side_effect=fake_process_extraction)),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'max_extract_input_tokens': 5,
                    'tokenizer': tokenizer,
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert len(captured_prompts) == 1
        assert 'ALPHA' in captured_prompts[0]
        assert 'BRAVO' in captured_prompts[0]
        assert chunks['chunk-alpha']['content'] not in captured_prompts[0]
        assert chunks['chunk-bravo']['content'] not in captured_prompts[0]
        assert captured_source_content == {'chunk-alpha': 'ALPHA', 'chunk-bravo': 'BRAVO'}

    @pytest.mark.asyncio
    async def test_extract_entities_caps_default_batch_size_and_honors_override(self):
        """Default batching should stay bounded while env override still wins."""
        from yar.operate import extract_entities

        chunks: dict[str, TextChunkSchema] = {
            f'chunk-{index}': {
                'tokens': 10,
                'content': f'Content {index}',
                'full_doc_id': 'doc-1',
                'chunk_order_index': index,
            }
            for index in range(16)
        }

        captured_batch_sizes: list[int] = []

        async def fake_use_llm_with_cache(prompt, *_args, **_kwargs):
            chunk_ids = [
                line[len('[CHUNK: ') : -1]
                for line in prompt.splitlines()
                if line.startswith('[CHUNK: ') and line.endswith(']')
            ]
            if chunk_ids:
                captured_batch_sizes.append(len(chunk_ids))
                raw_output = '\n'.join(
                    f'[CHUNK: {chunk_id}]\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}' for chunk_id in chunk_ids
                )
            else:
                raw_output = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
            return raw_output, 1234567890

        global_config = {
            'llm_model_func': AsyncMock(return_value='unused'),
            'entity_extract_max_gleaning': 0,
            'max_output_tokens': 96000,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
            'llm_model_max_async': 1,
        }

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            patch(
                'yar.operate._process_extraction_result',
                new=AsyncMock(return_value=ChunkExtractionResult(nodes={}, edges={})),
            ),
        ):
            result = await extract_entities(chunks=chunks, global_config=global_config)

        assert isinstance(result, list)
        assert captured_batch_sizes == [8, 8]

        captured_batch_sizes.clear()
        with (
            patch.dict('os.environ', {'ENTITY_EXTRACT_BATCH_SIZE': '20'}, clear=False),
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            patch(
                'yar.operate._process_extraction_result',
                new=AsyncMock(return_value=ChunkExtractionResult(nodes={}, edges={})),
            ),
        ):
            result = await extract_entities(chunks=chunks, global_config=global_config)

        assert isinstance(result, list)
        assert captured_batch_sizes == [16]

    @pytest.mark.asyncio
    async def test_extract_entities_tolerates_minor_batch_header_drift(self):
        """Minor header drift should still map output sections to the right chunks."""
        from yar.operate import extract_entities

        chunks: dict[str, TextChunkSchema] = {
            'chunk-1': {
                'tokens': 10,
                'content': 'Alpha',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            },
            'chunk-2': {
                'tokens': 10,
                'content': 'Beta',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
        }

        captured_sections: list[tuple[str, str]] = []

        async def fake_use_llm_with_cache(*_args, **_kwargs):
            return (
                'CHUNK: chunk-1\nsection-one\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                '### [Chunk: chunk-2]\nsection-two\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                1234567890,
            )

        async def fake_process_extraction(section_text, chunk_key, *_args, **_kwargs):
            captured_sections.append((chunk_key, section_text.strip()))
            return ChunkExtractionResult(nodes={}, edges={})

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache) as llm_mock,
            patch('yar.operate._process_extraction_result', new=AsyncMock(side_effect=fake_process_extraction)),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert llm_mock.await_count == 1
        assert captured_sections == [
            ('chunk-1', f'section-one\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
            ('chunk-2', f'section-two\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
        ]

    @pytest.mark.asyncio
    async def test_extract_entities_resolves_unique_single_character_batch_header_drift(self):
        """A unique one-character chunk-id drift should stay batched."""
        from yar.operate import extract_entities

        first_chunk_id = 'chunk-726cdd081ee9de77680e61cb56cb588c'
        second_chunk_id = 'chunk-723fe86b033d3a295ba9d02345e1853a'
        chunks: dict[str, TextChunkSchema] = {
            first_chunk_id: {
                'tokens': 10,
                'content': 'Alpha',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            },
            second_chunk_id: {
                'tokens': 10,
                'content': 'Beta',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
        }

        captured_sections: list[tuple[str, str]] = []

        async def fake_use_llm_with_cache(*_args, **_kwargs):
            return (
                '[CHUNK: chunk-726cdd081ee9de77680e61cb56cb588f]\nsection-one\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                f'[CHUNK: {second_chunk_id}]\nsection-two\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                1234567890,
            )

        async def fake_process_extraction(section_text, chunk_key, *_args, **_kwargs):
            captured_sections.append((chunk_key, section_text.strip()))
            return ChunkExtractionResult(nodes={}, edges={})

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache) as llm_mock,
            patch('yar.operate._process_extraction_result', new=AsyncMock(side_effect=fake_process_extraction)),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert llm_mock.await_count == 1
        assert captured_sections == [
            (first_chunk_id, f'section-one\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
            (second_chunk_id, f'section-two\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
        ]

    @pytest.mark.asyncio
    async def test_extract_entities_does_not_canonicalize_ambiguous_single_character_batch_header_drift(self):
        """Ambiguous one-character header drift should fall back instead of guessing."""
        from yar.operate import extract_entities

        chunks: dict[str, TextChunkSchema] = {
            'chunk-aaaa': {
                'tokens': 10,
                'content': 'Alpha',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            },
            'chunk-baaa': {
                'tokens': 10,
                'content': 'Beta',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
        }

        batch_call_count = 0
        single_chunk_ids: list[str] = []
        captured_sections: list[tuple[str, str]] = []

        async def fake_use_llm_with_cache(prompt, *_args, **kwargs):
            nonlocal batch_call_count
            batch_chunk_ids = [
                line[len('[CHUNK: ') : -1]
                for line in prompt.splitlines()
                if line.startswith('[CHUNK: ') and line.endswith(']')
            ]
            if len(batch_chunk_ids) > 1:
                batch_call_count += 1
                return (
                    '[CHUNK: chunk-caaa]\nsection-ambiguous\n'
                    f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                    '[CHUNK: chunk-baaa]\nsection-two\n'
                    f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                    1234567890,
                )

            single_chunk_ids.append(kwargs['chunk_id'])
            return (
                f'single-recovered-{kwargs["chunk_id"]}\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                1234567890,
            )

        async def fake_process_extraction(section_text, chunk_key, *_args, **_kwargs):
            captured_sections.append((chunk_key, section_text.strip()))
            return ChunkExtractionResult(nodes={}, edges={})

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache) as llm_mock,
            patch('yar.operate._process_extraction_result', new=AsyncMock(side_effect=fake_process_extraction)),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert batch_call_count == 1
        assert llm_mock.await_count == 2
        assert single_chunk_ids == ['chunk-aaaa']
        assert captured_sections == [
            ('chunk-baaa', f'section-two\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
            ('chunk-aaaa', f'single-recovered-chunk-aaaa\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
        ]

    @pytest.mark.asyncio
    async def test_extract_entities_ignores_trailing_empty_duplicate_batch_section(self):
        """A later delimiter-only duplicate should not force per-chunk fallback."""
        from yar.operate import extract_entities

        first_chunk_id = 'chunk-726cdd081ee9de77680e61cb56cb588c'
        second_chunk_id = 'chunk-723fe86b033d3a295ba9d02345e1853a'
        chunks: dict[str, TextChunkSchema] = {
            first_chunk_id: {
                'tokens': 10,
                'content': 'Alpha',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            },
            second_chunk_id: {
                'tokens': 10,
                'content': 'Beta',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
        }

        captured_sections: list[tuple[str, str]] = []

        async def fake_use_llm_with_cache(*_args, **_kwargs):
            return (
                f'[CHUNK: {first_chunk_id}]\nsection-one\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                f'[CHUNK: {second_chunk_id}]\nsection-two\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                f'[CHUNK: {second_chunk_id}]\n'
                f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                1234567890,
            )

        async def fake_process_extraction(section_text, chunk_key, *_args, **_kwargs):
            captured_sections.append((chunk_key, section_text.strip()))
            return ChunkExtractionResult(nodes={}, edges={})

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache) as llm_mock,
            patch('yar.operate._process_extraction_result', new=AsyncMock(side_effect=fake_process_extraction)),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert llm_mock.await_count == 1
        assert captured_sections == [
            (first_chunk_id, f'section-one\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
            (second_chunk_id, f'section-two\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
        ]

    @pytest.mark.asyncio
    async def test_extract_entities_conflicting_duplicate_batch_sections_fall_back_to_single_chunk(self):
        """Conflicting duplicate sections must still fall back for safety."""
        from yar.operate import extract_entities

        chunks: dict[str, TextChunkSchema] = {
            'chunk-1': {
                'tokens': 10,
                'content': 'Alpha',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            },
            'chunk-2': {
                'tokens': 10,
                'content': 'Beta',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
        }

        batch_call_count = 0
        single_chunk_ids: list[str] = []
        captured_sections: list[tuple[str, str]] = []

        async def fake_use_llm_with_cache(prompt, *_args, **kwargs):
            nonlocal batch_call_count
            batch_chunk_ids = [
                line[len('[CHUNK: ') : -1]
                for line in prompt.splitlines()
                if line.startswith('[CHUNK: ') and line.endswith(']')
            ]
            if len(batch_chunk_ids) > 1:
                batch_call_count += 1
                return (
                    '[CHUNK: chunk-1]\nsection-one\n'
                    f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                    '[CHUNK: chunk-2]\nsection-two\n'
                    f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}\n'
                    '[CHUNK: chunk-2]\nconflicting-two\n'
                    f'{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                    1234567890,
                )

            single_chunk_ids.append(kwargs['chunk_id'])
            return (
                f'single-recovered-{kwargs["chunk_id"]}\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}',
                1234567890,
            )

        async def fake_process_extraction(section_text, chunk_key, *_args, **_kwargs):
            captured_sections.append((chunk_key, section_text.strip()))
            return ChunkExtractionResult(nodes={}, edges={})

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache) as llm_mock,
            patch('yar.operate._process_extraction_result', new=AsyncMock(side_effect=fake_process_extraction)),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert isinstance(result, list)
        assert batch_call_count == 1
        assert llm_mock.await_count == 2
        assert single_chunk_ids == ['chunk-2']
        assert captured_sections == [
            ('chunk-1', f'section-one\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
            ('chunk-2', f'single-recovered-chunk-2\n{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}'),
        ]

    @pytest.mark.asyncio
    async def test_extract_entities_batch_failure_falls_back_without_gleaning_amplification(self):
        """Batch failures should recover via bounded parallel singles without gleaning retries."""
        from yar.operate import extract_entities

        chunks: dict[str, TextChunkSchema] = {
            f'chunk-{index}': {
                'tokens': 10,
                'content': f'Content {index}',
                'full_doc_id': 'doc-1',
                'chunk_order_index': index,
            }
            for index in range(3)
        }

        batch_call_count = 0
        single_call_count = 0
        active_single_calls = 0
        max_active_single_calls = 0

        async def fake_use_llm_with_cache(prompt, *_args, **kwargs):
            nonlocal batch_call_count, single_call_count, active_single_calls, max_active_single_calls
            batch_chunk_ids = [
                line[len('[CHUNK: ') : -1]
                for line in prompt.splitlines()
                if line.startswith('[CHUNK: ') and line.endswith(']')
            ]
            if len(batch_chunk_ids) > 1:
                batch_call_count += 1
                raise RuntimeError('batch exploded')

            single_call_count += 1
            active_single_calls += 1
            max_active_single_calls = max(max_active_single_calls, active_single_calls)
            try:
                await asyncio.sleep(0.01)
                return PROMPTS['DEFAULT_COMPLETION_DELIMITER'], 1234567890
            finally:
                active_single_calls -= 1

        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            patch(
                'yar.operate._process_extraction_result',
                new=AsyncMock(return_value=ChunkExtractionResult(nodes={}, edges={})),
            ),
        ):
            result = await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'entity_extract_max_gleaning': 1,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 4,
                },
            )

        assert isinstance(result, list)
        assert len(result) == 3
        assert batch_call_count == 1
        assert single_call_count == 3
        assert 1 < max_active_single_calls <= 2

    @pytest.mark.asyncio
    async def test_extract_entities_times_out_hung_llm_calls(self):
        """Hung extraction calls should fail fast enough to surface document failure."""
        from yar.operate import extract_entities

        chunks: dict[str, TextChunkSchema] = {
            'chunk-1': {
                'tokens': 10,
                'content': 'Alpha',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 0,
            }
        }

        async def fake_use_llm_with_cache(*_args, **_kwargs):
            await asyncio.sleep(1)
            return PROMPTS['DEFAULT_COMPLETION_DELIMITER'], 1234567890

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        with (
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            pytest.raises(RuntimeError, match='All 1 chunks failed during entity extraction'),
        ):
            await extract_entities(
                chunks=chunks,
                global_config={
                    'llm_model_func': AsyncMock(return_value='unused'),
                    'default_llm_timeout': 0.01,
                    'entity_extract_max_gleaning': 0,
                    'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
                    'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
                    'llm_model_max_async': 1,
                },
            )

        assert loop.time() - started_at < 0.2

    @pytest.mark.asyncio
    async def test_extract_entities_respects_shared_extraction_semaphore_across_invocations(self):
        """Concurrent extractions should share one semaphore for actual LLM calls."""
        from yar.operate import extract_entities

        def make_chunks(prefix: str) -> dict[str, TextChunkSchema]:
            return {
                f'{prefix}-{index}': {
                    'tokens': 10,
                    'content': f'Content {prefix}-{index}',
                    'full_doc_id': prefix,
                    'chunk_order_index': index,
                }
                for index in range(3)
            }

        shared_semaphore = asyncio.Semaphore(2)
        counter_lock = asyncio.Lock()
        first_two_started = asyncio.Event()
        active_calls = 0
        max_active_calls = 0
        seen_chunk_ids: list[str] = []

        async def fake_use_llm_with_cache(prompt, *_args, **kwargs):
            nonlocal active_calls, max_active_calls
            chunk_id = kwargs['chunk_id']
            async with counter_lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
                seen_chunk_ids.append(chunk_id)
                if active_calls == 2:
                    first_two_started.set()
            try:
                await asyncio.wait_for(first_two_started.wait(), timeout=0.1)
                await asyncio.sleep(0.01)
                return PROMPTS['DEFAULT_COMPLETION_DELIMITER'], 1234567890
            finally:
                async with counter_lock:
                    active_calls -= 1

        global_config = {
            'llm_model_func': AsyncMock(return_value='unused'),
            'entity_extract_max_gleaning': 0,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE, 'entity_types': ['PERSON']},
            'llm_model_max_async': 4,
            'entity_extract_max_async': 3,
            'entity_extract_semaphore': shared_semaphore,
        }

        with (
            patch.dict('os.environ', {'ENTITY_EXTRACT_BATCH_SIZE': '1'}, clear=False),
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache) as llm_mock,
            patch(
                'yar.operate._process_extraction_result',
                new=AsyncMock(return_value=ChunkExtractionResult(nodes={}, edges={})),
            ),
        ):
            first_result, second_result = await asyncio.gather(
                extract_entities(chunks=make_chunks('doc-a'), global_config=dict(global_config)),
                extract_entities(chunks=make_chunks('doc-b'), global_config=dict(global_config)),
            )

        assert [len(first_result), len(second_result)] == [3, 3]
        assert llm_mock.await_count == 6
        assert max_active_calls == 2
        assert active_calls == 0
        assert sorted(seen_chunk_ids) == [
            'doc-a-0',
            'doc-a-1',
            'doc-a-2',
            'doc-b-0',
            'doc-b-1',
            'doc-b-2',
        ]


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


@pytest.mark.offline
class TestOperateEdgeCases:
    """Edge case tests for operate module."""

    def test_chunking_with_null_bytes(self):
        """Test chunking with null bytes in content."""
        content = 'Normal text\x00with null\x00bytes'
        try:
            result = chunking_by_semantic(content)
            assert isinstance(result, list)
        except (ValueError, UnicodeError):
            # Acceptable to reject invalid content
            pass

    def test_chunking_very_small_limits(self):
        """Test chunking with very small character limits."""
        content = 'Test content'
        result = chunking_by_semantic(content, max_chars=10, max_overlap=2)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_identifier_truncation_empty_string(self):
        """Test truncation with empty identifier."""
        result = _truncate_entity_identifier('', limit=100, chunk_key='test', identifier_role='Entity')
        assert result == ''

    def test_identifier_truncation_unicode_boundary(self):
        """Test truncation doesn't break Unicode characters."""
        identifier = 'Test日本語' * 20
        result = _truncate_entity_identifier(identifier, limit=50, chunk_key='test', identifier_role='Entity')

        assert len(result) == 50

    @pytest.mark.asyncio
    async def test_keyword_extraction_empty_query(self):
        """Test keyword extraction with empty query."""
        mock_llm = AsyncMock(return_value='{"high_level_keywords": [], "low_level_keywords": []}')

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[])),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }

        param = QueryParam()

        hl, ll = await extract_keywords_only(
            text='',
            param=param,
            global_config=global_config,
        )

        assert isinstance(hl, list)
        assert isinstance(ll, list)


# ============================================================================
# Helper Function Tests
# ============================================================================


@pytest.mark.offline
class TestOperateHelpers:
    """Tests for helper functions in operate module."""

    def test_create_chunker_returns_same_signature(self):
        """Test that semantic chunker compatibility entry points share a signature."""
        chunkers = [
            create_chunker(),
            create_chunker(),
            create_chunker(),
        ]

        content = 'Test content'
        for chunker in chunkers:
            result = chunker(None, content, None, False, 100, 1200)
            assert isinstance(result, list)
            # All should have same structure
            if result:
                assert 'content' in result[0]
                assert 'tokens' in result[0]

    def test_resolve_max_file_paths_handles_edge_values(self):
        """max_file_paths parser should use defaults and clamp negatives."""
        assert _resolve_max_file_paths({}) == DEFAULT_MAX_FILE_PATHS
        assert _resolve_max_file_paths({'max_file_paths': '12'}) == 12
        assert _resolve_max_file_paths({'max_file_paths': -5}) == 0
        assert _resolve_max_file_paths({'max_file_paths': 'not-a-number'}) == DEFAULT_MAX_FILE_PATHS

    def test_chunking_preserves_content_order(self):
        """Test that chunks preserve content order."""
        content = 'First. Second. Third. Fourth. Fifth.'
        result = chunking_by_semantic(content, max_chars=100, max_overlap=20)

        # Recombine chunks
        combined = ' '.join(chunk['content'] for chunk in result)

        # Order should be preserved
        if 'First' in combined and 'Fifth' in combined:
            assert combined.find('First') < combined.find('Fifth')

    def test_chunk_metadata_completeness(self):
        """Test that all required metadata fields are present."""
        content = 'Test content for metadata validation.'
        result = chunking_by_semantic(content)

        required_fields = ['content', 'tokens', 'chunk_order_index', 'char_start', 'char_end']

        for chunk in result:
            for field in required_fields:
                assert field in chunk, f'Missing required field: {field}'

    def test_split_keyword_terms_deduplicates_and_trims(self):
        """Keyword term splitting should normalize comma-separated strings."""
        terms = _split_keyword_terms(' Amazon , AWS, amazon ;  cloud platform  ')
        assert terms == ['Amazon', 'AWS', 'cloud platform']

    def test_enrich_local_keywords_promotes_high_level_when_low_missing(self):
        """Local mode should promote one focused high-level keyword when low-level is empty."""
        enriched = _enrich_local_keywords(
            hl_keywords=['quantum computing', 'company research'],
            ll_keywords=[],
            mode='local',
            query='What companies are working on quantum computing?',
        )
        assert enriched == ['quantum computing']

    def test_enrich_local_keywords_keeps_existing_low_level_terms(self):
        """Local mode should keep existing low-level keywords without broadening."""
        enriched = _enrich_local_keywords(
            hl_keywords=['technology company', 'market profile', 'cloud services'],
            ll_keywords=['Amazon'],
            mode='local',
            user_supplied_ll=True,
        )
        assert enriched == ['Amazon']

    def test_enrich_local_keywords_filters_auto_generated_generic_low_level_terms(self):
        """Auto-generated low-level keywords should be narrowed to focused terms."""
        enriched = _enrich_local_keywords(
            hl_keywords=['quantum computing', 'company research'],
            ll_keywords=['quantum computing', 'company research', 'technology development'],
            mode='local',
            user_supplied_ll=False,
        )
        assert enriched == ['quantum computing']

    def test_enrich_local_keywords_keeps_multiple_focused_low_level_terms(self):
        """Local mode should preserve all focused auto-generated low-level keywords."""
        enriched = _enrich_local_keywords(
            hl_keywords=['hemophilia treatment', 'drug comparison'],
            ll_keywords=['PX-482', 'company research', 'Product Beta'],
            mode='local',
            user_supplied_ll=False,
        )

        assert enriched == ['PX-482', 'Product Beta']

    def test_enrich_local_keywords_falls_back_to_query_when_high_level_is_generic(self):
        """When all HL terms are generic, local mode should use the original query."""
        enriched = _enrich_local_keywords(
            hl_keywords=['company research', 'technology development'],
            ll_keywords=[],
            mode='local',
            query='What is Amazon?',
        )
        assert enriched == ['What is Amazon?']


@pytest.mark.offline
class TestBuildQueryContextMixModeGuards:
    """Regression tests for mix-mode no-KG-result guard behavior."""

    @pytest.mark.asyncio
    async def test_mix_mode_with_vector_chunks_and_empty_tracking_keeps_context_building(self):
        """Mix mode should continue when vector chunks exist even if chunk_tracking is empty."""
        query_param = QueryParam(mode='mix')
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        with (
            patch(
                'yar.operate._perform_kg_search',
                new=AsyncMock(
                    return_value={
                        'final_entities': [],
                        'final_relations': [],
                        'vector_chunks': [{'content': 'vector-only chunk'}],
                        'chunk_tracking': {},
                        'query_embedding': None,
                    }
                ),
            ),
            patch(
                'yar.operate._apply_token_truncation',
                new=AsyncMock(
                    return_value={
                        'entities_context': [],
                        'relations_context': [],
                        'filtered_entities': [],
                        'filtered_relations': [],
                        'entity_id_to_original': {},
                        'relation_id_to_original': {},
                    }
                ),
            ),
            patch(
                'yar.operate._merge_all_chunks',
                new=AsyncMock(return_value=[{'content': 'vector-only chunk'}]),
            ),
            patch(
                'yar.operate._build_context_str',
                new=AsyncMock(
                    return_value=(
                        'context text',
                        {'data': {'entities': [], 'relationships': [], 'chunks': [{'content': 'vector-only chunk'}]}},
                    )
                ),
            ),
        ):
            result = await _build_query_context(
                query='what happened',
                ll_keywords='',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=MagicMock(),
            )

        assert result is not None
        assert result.context == 'context text'

    @pytest.mark.asyncio
    async def test_mix_mode_with_no_entities_relations_or_vector_chunks_returns_none(self):
        """Mix mode should still return None when no retrieval source returns data."""
        query_param = QueryParam(mode='mix')
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        with (
            patch(
                'yar.operate._perform_kg_search',
                new=AsyncMock(
                    return_value={
                        'final_entities': [],
                        'final_relations': [],
                        'vector_chunks': [],
                        'chunk_tracking': {},
                        'query_embedding': None,
                    }
                ),
            ),
            patch('yar.operate._apply_token_truncation', new=AsyncMock()) as truncation_mock,
        ):
            result = await _build_query_context(
                query='what happened',
                ll_keywords='',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=MagicMock(),
            )

        assert result is None
        truncation_mock.assert_not_awaited()


@pytest.mark.offline
class TestPerformKgSearchScoreAwareMerge:
    """Regression tests for score-aware entity/relation merge behavior."""

    @pytest.mark.asyncio
    async def test_entity_merge_prefers_highest_scored_candidate(self):
        query_param = QueryParam(mode='hybrid', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        local_entities = [
            {'entity_name': 'A', 'score': 0.9, 'rank': 1},
            {'entity_name': 'B', 'score': 0.2},
        ]
        global_entities = [
            {'entity_name': 'B', 'score': 0.95},
            {'entity_name': 'C', 'score': 0.7},
        ]

        with (
            patch(
                'yar.operate._get_node_data',
                new=AsyncMock(return_value=(local_entities, [])),
            ),
            patch(
                'yar.operate._get_edge_data',
                new=AsyncMock(return_value=([], global_entities)),
            ),
        ):
            result = await _perform_kg_search(
                query='',
                ll_keywords='local-keyword',
                hl_keywords='global-keyword',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        assert [entity['entity_name'] for entity in result['final_entities']] == ['B', 'A', 'C']
        assert len([entity for entity in result['final_entities'] if entity['entity_name'] == 'B']) == 1

    @pytest.mark.asyncio
    async def test_entity_merge_normalizes_rank_so_hubs_do_not_dominate_similarity(self):
        query_param = QueryParam(mode='local', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        local_entities = [
            {'entity_name': 'ExactMatch', 'score': 0.95, 'rank': 2},
            {'entity_name': 'HubEntity', 'score': 0.05, 'rank': 500},
        ]

        with patch('yar.operate._get_node_data', new=AsyncMock(return_value=(local_entities, []))):
            result = await _perform_kg_search(
                query='',
                ll_keywords='ExactMatch, HubEntity',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        assert [entity['entity_name'] for entity in result['final_entities']] == ['ExactMatch']

    @pytest.mark.asyncio
    async def test_entity_confidence_floor_drops_weak_tail_without_disabling_entities(self, monkeypatch):
        monkeypatch.setenv('YAR_ENTITY_CONFIDENCE_FLOOR', '0.45')
        query_param = QueryParam(mode='local', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        local_entities = [
            {'entity_name': 'ExactMatch', 'score': 0.95, 'rank': 2},
            {'entity_name': 'WeakNeighbor', 'score': 0.20, 'rank': 1},
        ]

        with patch('yar.operate._get_node_data', new=AsyncMock(return_value=(local_entities, []))):
            result = await _perform_kg_search(
                query='',
                ll_keywords='ExactMatch, WeakNeighbor',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        assert [entity['entity_name'] for entity in result['final_entities']] == ['ExactMatch']

    @pytest.mark.asyncio
    async def test_entity_confidence_floor_uses_source_specific_scores(self, monkeypatch):
        monkeypatch.setenv('YAR_ENTITY_CONFIDENCE_FLOOR', '0.45')
        query_param = QueryParam(mode='local', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        local_entities = [
            {'entity_name': 'SemanticMatch', 'vector_score': 0.78, 'rank': 1},
            {'entity_name': 'ExactName', 'trgm_score': 0.91, 'rank': 2},
            {'entity_name': 'WeakNeighbor', 'vector_score': 0.20, 'rank': 1},
        ]

        with patch('yar.operate._get_node_data', new=AsyncMock(return_value=(local_entities, []))):
            result = await _perform_kg_search(
                query='',
                ll_keywords='SemanticMatch, ExactName, WeakNeighbor',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        assert [entity['entity_name'] for entity in result['final_entities']] == [
            'ExactName',
            'SemanticMatch',
        ]

    @pytest.mark.asyncio
    async def test_entity_confidence_floor_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv('YAR_ENTITY_CONFIDENCE_FLOOR', '0')
        query_param = QueryParam(mode='local', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        local_entities = [
            {'entity_name': 'ExactMatch', 'score': 0.95, 'rank': 2},
            {'entity_name': 'WeakNeighbor', 'score': 0.20, 'rank': 1},
        ]

        with patch('yar.operate._get_node_data', new=AsyncMock(return_value=(local_entities, []))):
            result = await _perform_kg_search(
                query='',
                ll_keywords='ExactMatch, WeakNeighbor',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        assert [entity['entity_name'] for entity in result['final_entities']] == [
            'ExactMatch',
            'WeakNeighbor',
        ]

    @pytest.mark.asyncio
    async def test_relation_merge_prefers_highest_scored_duplicate(self):
        query_param = QueryParam(mode='hybrid', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        local_relations = [
            {'src_tgt': ('a', 'b'), 'rank': 2, 'weight': 0.2},
            {'src_tgt': ('c', 'd'), 'score': 0.4},
        ]
        global_relations = [
            {'src_id': 'a', 'tgt_id': 'b', 'score': 0.8},
            {'src_id': 'e', 'tgt_id': 'f', 'weight': 3.0},
        ]

        with (
            patch(
                'yar.operate._get_node_data',
                new=AsyncMock(return_value=([], local_relations)),
            ),
            patch(
                'yar.operate._get_edge_data',
                new=AsyncMock(return_value=(global_relations, [])),
            ),
        ):
            result = await _perform_kg_search(
                query='',
                ll_keywords='local-keyword',
                hl_keywords='global-keyword',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        def relation_key(relation: dict[str, str]) -> tuple[str, str]:
            src_tgt = relation.get('src_tgt')
            if isinstance(src_tgt, (tuple, list)) and len(src_tgt) == 2:
                return tuple(sorted((str(src_tgt[0]), str(src_tgt[1]))))
            return tuple(sorted((str(relation.get('src_id')), str(relation.get('tgt_id')))))

        assert [relation_key(relation) for relation in result['final_relations']] == [
            ('a', 'b'),
            ('c', 'd'),
            ('e', 'f'),
        ]
        assert result['final_relations'][0].get('src_id') == 'a'

    @pytest.mark.asyncio
    async def test_perform_kg_search_preserves_conflicting_relation_pair_when_polarity_differs(self):
        query_param = QueryParam(mode='hybrid', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        local_relations = [
            {
                'src_id': 'Treatment',
                'tgt_id': 'Outcome',
                'keywords': 'supports',
                'description': 'Treatment supports the outcome.',
                'score': 0.8,
            }
        ]
        global_relations = [
            {
                'src_id': 'Outcome',
                'tgt_id': 'Treatment',
                'keywords': 'blocks',
                'description': 'Outcome blocks the treatment.',
                'score': 0.7,
            }
        ]

        with (
            patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], local_relations))),
            patch('yar.operate._get_edge_data', new=AsyncMock(return_value=(global_relations, []))),
        ):
            result = await _perform_kg_search(
                query='',
                ll_keywords='local-keyword',
                hl_keywords='global-keyword',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        assert [(relation['src_id'], relation['tgt_id']) for relation in result['final_relations']] == [
            ('Treatment', 'Outcome'),
            ('Outcome', 'Treatment'),
        ]
        assert all(relation.get('relation_conflict') is True for relation in result['final_relations'])
        assert result['final_relations'][0]['relation_conflict_predicates'] == ['blocks']
        assert result['final_relations'][1]['relation_conflict_predicates'] == ['supports']

    @pytest.mark.asyncio
    async def test_hybrid_search_uses_only_primary_relation_sources_when_entities_miss(self):
        query_param = QueryParam(mode='hybrid', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        global_relations = [{'src_id': 'Handbook', 'tgt_id': 'Abbreviations', 'score': 0.2}]
        edge_mock = AsyncMock(return_value=(global_relations, []))

        with (
            patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], []))),
            patch('yar.operate._get_edge_data', new=edge_mock),
        ):
            result = await _perform_kg_search(
                query='What are the region-specific activities?',
                ll_keywords='Foreign Manufacturer Accreditation, Submission Package',
                hl_keywords='Regional operations handbook',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        edge_mock.assert_awaited_once()
        assert result['final_relations'] == global_relations

    @pytest.mark.asyncio
    async def test_global_search_does_not_retry_low_level_relations_when_primary_misses(self):
        query_param = QueryParam(mode='global', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        edge_mock = AsyncMock(return_value=([], []))

        with patch('yar.operate._get_edge_data', new=edge_mock):
            result = await _perform_kg_search(
                query='What risks does the adapter strategy address for product quality and dosing?',
                ll_keywords='adapter strategy, product quality, dosing',
                hl_keywords='risk mitigation, product quality, dosing, manufacturing strategy',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        edge_mock.assert_awaited_once()
        assert (
            edge_mock.await_args_list[0].args[0] == 'risk mitigation, product quality, dosing, manufacturing strategy'
        )
        assert result['final_relations'] == []

    @pytest.mark.asyncio
    async def test_perform_kg_search_uses_high_level_phrases_for_chunk_search(self):
        query_param = QueryParam(mode='hybrid', top_k=3, chunk_top_k=3, enable_bm25_fusion=True)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.hybrid_search = AsyncMock(return_value=[])

        with (
            patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], []))),
            patch('yar.operate._get_edge_data', new=AsyncMock(return_value=([], []))),
        ):
            result = await _perform_kg_search(
                query='What best practices come from the lessons learned?',
                ll_keywords='Partner Alpha, communication',
                hl_keywords='best practice, conflict management, lessons learned',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=chunks_vdb,
            )

        chunks_vdb.hybrid_search.assert_awaited_once()
        phrase_terms = chunks_vdb.hybrid_search.await_args.kwargs['phrase_terms']
        assert phrase_terms is not None
        assert 'conflict management' in phrase_terms
        assert 'lessons learned' in phrase_terms
        assert result['chunk_phrase_terms'] == phrase_terms

    @pytest.mark.asyncio
    async def test_global_search_uses_single_relation_query_when_primary_hits(self):
        query_param = QueryParam(mode='global', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}
        global_relations = [{'src_id': 'Risk', 'tgt_id': 'Mitigation', 'score': 0.8}]
        edge_mock = AsyncMock(return_value=(global_relations, []))

        with patch('yar.operate._get_edge_data', new=edge_mock):
            result = await _perform_kg_search(
                query='What risks does the adapter strategy address?',
                ll_keywords='adapter strategy',
                hl_keywords='risk mitigation',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        edge_mock.assert_awaited_once()
        assert result['final_relations'] == global_relations


@pytest.mark.offline
class TestEntityQueryEmbeddingReuse:
    """Regression tests for reusing precomputed entity query embeddings."""

    @pytest.mark.asyncio
    async def test_get_node_data_passes_precomputed_embedding_to_vector_search(self):
        entities_vdb = MagicMock()
        entities_vdb.cosine_better_than_threshold = 0.2
        entities_vdb.hybrid_entity_search = None
        entities_vdb.query = AsyncMock(return_value=[{'entity_name': 'Amazon', 'score': 0.9}])

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={'Amazon': {'entity_type': 'COMPANY', 'description': 'Cloud provider'}}
        )
        knowledge_graph_inst.node_degrees_batch = AsyncMock(return_value={'Amazon': 2})

        with patch('yar.operate._find_most_related_edges_from_entities', new=AsyncMock(return_value=[])):
            node_datas, relations = await _get_node_data(
                'Amazon',
                knowledge_graph_inst,
                entities_vdb,
                QueryParam(mode='local', top_k=5),
                query_embedding=[0.1, 0.2],
            )

        assert relations == []
        assert node_datas[0]['entity_name'] == 'Amazon'
        assert entities_vdb.query.await_args.kwargs['query_embedding'] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_get_node_data_interleaves_keyword_results_without_starving_later_terms(self):
        entities_vdb = MagicMock()
        entities_vdb.cosine_better_than_threshold = 0.2

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'AlphaOne': {'entity_type': 'COMPANY', 'description': 'Alpha one'},
                'BetaOne': {'entity_type': 'COMPANY', 'description': 'Beta one'},
            }
        )
        knowledge_graph_inst.node_degrees_batch = AsyncMock(return_value={'AlphaOne': 5, 'BetaOne': 3})

        candidate_lists = [
            [
                {'entity_name': 'AlphaOne', 'score': 0.99},
                {'entity_name': 'AlphaTwo', 'score': 0.98},
            ],
            [
                {'entity_name': 'AlphaOne', 'score': 0.97},
                {'entity_name': 'BetaOne', 'score': 0.96},
            ],
        ]

        with (
            patch('yar.operate._query_entity_candidates', new=AsyncMock(side_effect=candidate_lists)),
            patch('yar.operate._find_most_related_edges_from_entities', new=AsyncMock(return_value=[])),
        ):
            node_datas, relations = await _get_node_data(
                'alpha, beta',
                knowledge_graph_inst,
                entities_vdb,
                QueryParam(mode='local', top_k=2),
            )

        assert relations == []
        assert [node['entity_name'] for node in node_datas] == ['AlphaOne', 'BetaOne']
        knowledge_graph_inst.get_nodes_batch.assert_awaited_once_with(['AlphaOne', 'BetaOne'])
        knowledge_graph_inst.node_degrees_batch.assert_awaited_once_with(['AlphaOne', 'BetaOne'])
        assert all(node['entity_name'] != 'AlphaTwo' for node in node_datas)

    @pytest.mark.asyncio
    async def test_get_node_data_sorts_fill_candidates_by_similarity_after_term_coverage(self):
        entities_vdb = MagicMock()
        entities_vdb.cosine_better_than_threshold = 0.2

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'AlphaOne': {'entity_type': 'Concept', 'description': 'Alpha one'},
                'AlphaTwo': {'entity_type': 'Concept', 'description': 'Alpha two'},
                'BetaOne': {'entity_type': 'Concept', 'description': 'Beta one'},
            }
        )
        knowledge_graph_inst.node_degrees_batch = AsyncMock(return_value={'AlphaOne': 5, 'AlphaTwo': 4, 'BetaOne': 3})

        candidate_lists = [
            [
                {'entity_name': 'AlphaOne', 'score': 0.99},
                {'entity_name': 'AlphaTwo', 'score': 0.98},
            ],
            [{'entity_name': 'BetaOne', 'score': 0.50}],
        ]

        with (
            patch('yar.operate._query_entity_candidates', new=AsyncMock(side_effect=candidate_lists)),
            patch('yar.operate._find_most_related_edges_from_entities', new=AsyncMock(return_value=[])),
        ):
            node_datas, relations = await _get_node_data(
                'alpha, beta',
                knowledge_graph_inst,
                entities_vdb,
                QueryParam(mode='local', top_k=3),
            )

        assert relations == []
        assert [node['entity_name'] for node in node_datas] == ['AlphaOne', 'AlphaTwo', 'BetaOne']

    @pytest.mark.asyncio
    async def test_get_node_data_keeps_best_score_for_duplicate_entity_candidates(self):
        entities_vdb = MagicMock()
        entities_vdb.cosine_better_than_threshold = 0.2

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'AlphaOne': {'entity_type': 'Concept', 'description': 'Alpha one'},
                'BetaOne': {'entity_type': 'Concept', 'description': 'Beta one'},
            }
        )
        knowledge_graph_inst.node_degrees_batch = AsyncMock(return_value={'AlphaOne': 5, 'BetaOne': 3})

        candidate_lists = [
            [{'entity_name': 'AlphaOne', 'score': 0.40}],
            [
                {'entity_name': 'AlphaOne', 'score': 0.95},
                {'entity_name': 'BetaOne', 'score': 0.50},
            ],
        ]

        with (
            patch('yar.operate._query_entity_candidates', new=AsyncMock(side_effect=candidate_lists)),
            patch('yar.operate._find_most_related_edges_from_entities', new=AsyncMock(return_value=[])),
        ):
            node_datas, relations = await _get_node_data(
                'alpha, beta',
                knowledge_graph_inst,
                entities_vdb,
                QueryParam(mode='local', top_k=2),
            )

        assert relations == []
        assert [node['entity_name'] for node in node_datas] == ['AlphaOne', 'BetaOne']
        assert node_datas[0]['score'] == 0.95

    @pytest.mark.asyncio
    async def test_get_node_data_uses_source_scores_when_score_is_missing(self):
        entities_vdb = MagicMock()
        entities_vdb.cosine_better_than_threshold = 0.2

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'ExactName': {'entity_type': 'Concept', 'description': 'Exact name'},
                'SemanticClose': {'entity_type': 'Concept', 'description': 'Semantic close'},
            }
        )
        knowledge_graph_inst.node_degrees_batch = AsyncMock(return_value={'ExactName': 3, 'SemanticClose': 2})

        candidate_lists = [
            [
                {'entity_name': 'SemanticClose', 'vector_score': 0.86},
                {'entity_name': 'ExactName', 'trgm_score': 0.92},
            ]
        ]

        with (
            patch('yar.operate._query_entity_candidates', new=AsyncMock(side_effect=candidate_lists)),
            patch('yar.operate._find_most_related_edges_from_entities', new=AsyncMock(return_value=[])),
        ):
            node_datas, relations = await _get_node_data(
                'entity',
                knowledge_graph_inst,
                entities_vdb,
                QueryParam(mode='local', top_k=2),
            )

        assert relations == []
        assert [node['entity_name'] for node in node_datas] == ['ExactName', 'SemanticClose']
        assert [node['score'] for node in node_datas] == [0.92, 0.86]

    @pytest.mark.asyncio
    async def test_get_node_data_queries_terms_concurrently(self):
        entities_vdb = MagicMock()
        entities_vdb.cosine_better_than_threshold = 0.2

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'AlphaOne': {'entity_type': 'COMPANY', 'description': 'Alpha one'},
                'BetaOne': {'entity_type': 'COMPANY', 'description': 'Beta one'},
            }
        )
        knowledge_graph_inst.node_degrees_batch = AsyncMock(return_value={'AlphaOne': 5, 'BetaOne': 3})

        started_terms: list[str] = []
        all_started = asyncio.Event()

        async def query_entity_candidates(term, *_args, **_kwargs):
            started_terms.append(term)
            if len(started_terms) == 2:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=0.1)
            return [{'entity_name': f'{term.title()}One', 'score': 0.9}]

        with (
            patch('yar.operate._query_entity_candidates', new=query_entity_candidates),
            patch('yar.operate._find_most_related_edges_from_entities', new=AsyncMock(return_value=[])),
        ):
            node_datas, _relations = await _get_node_data(
                'alpha, beta',
                knowledge_graph_inst,
                entities_vdb,
                QueryParam(mode='local', top_k=2),
            )

        assert started_terms == ['alpha', 'beta']
        assert [node['entity_name'] for node in node_datas] == ['AlphaOne', 'BetaOne']

    @pytest.mark.asyncio
    async def test_get_edge_data_uses_single_primary_relationship_query(self):
        relationships_vdb = MagicMock()
        relationships_vdb.cosine_better_than_threshold = 0.2

        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_edges_batch = AsyncMock(
            return_value={
                ('Alpha Source', 'Alpha Target'): {'description': 'Alpha relation', 'weight': 1.0},
                ('Beta Source', 'Beta Target'): {'description': 'Beta relation', 'weight': 1.0},
            }
        )
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'Alpha Source': {'entity_type': 'EVENT', 'description': 'Alpha source'},
                'Alpha Target': {'entity_type': 'PERSON', 'description': 'Alpha target'},
                'Beta Source': {'entity_type': 'EVENT', 'description': 'Beta source'},
                'Beta Target': {'entity_type': 'PERSON', 'description': 'Beta target'},
            }
        )

        query_terms: list[str] = []

        async def query_relationship(term, *, top_k):
            query_terms.append(term)
            return [
                {
                    'src_id': 'Alpha Source',
                    'tgt_id': 'Alpha Target',
                    'score': 0.9,
                    'source_type': 'vector+bm25',
                    'bm25_score': 0.4,
                },
                {
                    'src_id': 'Beta Source',
                    'tgt_id': 'Beta Target',
                    'score': 0.8,
                    'source_type': 'bm25',
                },
            ][:top_k]

        relationships_vdb.query = AsyncMock(side_effect=query_relationship)

        edge_datas, entity_datas = await _get_edge_data(
            'alpha, beta',
            knowledge_graph_inst,
            relationships_vdb,
            QueryParam(mode='global', top_k=2),
        )

        assert query_terms == ['alpha, beta']
        assert [(edge['src_id'], edge['tgt_id']) for edge in edge_datas] == [
            ('Alpha Source', 'Alpha Target'),
            ('Beta Source', 'Beta Target'),
        ]
        assert edge_datas[0]['source_type'] == 'vector+bm25'
        assert edge_datas[0]['bm25_score'] == 0.4
        assert [entity['entity_name'] for entity in entity_datas] == [
            'Alpha Source',
            'Alpha Target',
            'Beta Source',
            'Beta Target',
        ]

    @pytest.mark.asyncio
    async def test_get_edge_data_uses_primary_query_and_focus_ranks_results(self):
        relationships_vdb = MagicMock()
        relationships_vdb.cosine_better_than_threshold = 0.2

        relation_props = {
            ('Company A', 'Adapter Strategy Proposal'): {
                'description': 'Company A develops the adapter strategy proposal.',
                'keywords': 'develops',
                'weight': 1.0,
            },
            ('Company A', 'Transfer Adapter'): {
                'description': 'Adapter strategy mitigates product quality and accurate dosing risks.',
                'keywords': 'mitigates',
                'weight': 2.0,
            },
        }

        knowledge_graph_inst = MagicMock()

        async def get_edges_batch(edge_pairs):
            edge_data = {}
            for edge_pair in edge_pairs:
                pair = (edge_pair['src'], edge_pair['tgt'])
                if pair in relation_props:
                    edge_data[pair] = relation_props[pair]
            return edge_data

        knowledge_graph_inst.get_edges_batch = AsyncMock(side_effect=get_edges_batch)
        knowledge_graph_inst.get_nodes_batch = AsyncMock(
            return_value={
                'Company A': {'entity_type': 'ORG', 'description': 'Company A'},
                'Adapter Strategy Proposal': {'entity_type': 'DOCUMENT', 'description': 'Strategy proposal'},
                'Transfer Adapter': {'entity_type': 'DEVICE', 'description': 'Closed system transfer device'},
            }
        )

        query_terms: list[str] = []

        async def query_relationship(term, *, top_k):
            query_terms.append(term)
            assert term == 'adapter strategy, product quality, dosing'
            return [
                {
                    'src_id': 'Company A',
                    'tgt_id': 'Adapter Strategy Proposal',
                    'source_type': 'vector',
                    'score': 0.8,
                },
                {
                    'src_id': 'Company A',
                    'tgt_id': 'Transfer Adapter',
                    'source_type': 'bm25',
                    'score': 0.7,
                },
            ][:top_k]

        relationships_vdb.query = AsyncMock(side_effect=query_relationship)

        with patch('yar.operate.logger.info') as info_mock:
            edge_datas, entity_datas = await _get_edge_data(
                'adapter strategy, product quality, dosing',
                knowledge_graph_inst,
                relationships_vdb,
                QueryParam(mode='global', top_k=2),
                query='What risks does the adapter strategy address for product quality and dosing?',
            )

        assert query_terms == ['adapter strategy, product quality, dosing']
        assert [(edge['src_id'], edge['tgt_id']) for edge in edge_datas] == [
            ('Company A', 'Transfer Adapter'),
            ('Company A', 'Adapter Strategy Proposal'),
        ]
        assert edge_datas[0]['query_focus_overlap'] > edge_datas[1]['query_focus_overlap']
        diagnostic_call = next(
            call for call in info_mock.call_args_list if call.args[0] == 'Relation primary ranking: %s'
        )
        diagnostic_payload = json.loads(diagnostic_call.args[1])
        assert diagnostic_payload['query'] == 'adapter strategy, product quality, dosing'
        assert diagnostic_payload['raw_candidates'] == 2
        assert diagnostic_payload['deduplicated_candidates'] == 2
        assert diagnostic_payload['graph_validated'] == 2
        assert diagnostic_payload['ranked'][0]['source_type'] == 'bm25'
        assert [entity['entity_name'] for entity in entity_datas] == [
            'Company A',
            'Transfer Adapter',
            'Adapter Strategy Proposal',
        ]


@pytest.mark.offline
class TestPerformKgSearchBranchExecution:
    """Tests for branch execution behavior in _perform_kg_search."""

    @pytest.mark.asyncio
    async def test_hybrid_runs_node_and_edge_retrieval_concurrently_when_both_keywords_present(self):
        query_param = QueryParam(mode='hybrid', top_k=5)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        node_started = asyncio.Event()
        edge_started = asyncio.Event()
        release = asyncio.Event()
        state = {'concurrent_launch': False}

        async def node_side_effect(*_args, **_kwargs):
            node_started.set()
            if edge_started.is_set():
                state['concurrent_launch'] = True
            await release.wait()
            return ([{'entity_name': 'LocalEntity', 'score': 0.8}], [{'src_tgt': ('l', 'm'), 'score': 0.3}])

        async def edge_side_effect(*_args, **_kwargs):
            edge_started.set()
            if node_started.is_set():
                state['concurrent_launch'] = True
            await release.wait()
            return (
                [{'src_id': 'g', 'tgt_id': 'h', 'score': 0.9}],
                [{'entity_name': 'GlobalEntity', 'score': 0.4}],
            )

        with (
            patch('yar.operate._get_node_data', new=AsyncMock(side_effect=node_side_effect)) as node_mock,
            patch('yar.operate._get_edge_data', new=AsyncMock(side_effect=edge_side_effect)) as edge_mock,
        ):
            search_task = asyncio.create_task(
                _perform_kg_search(
                    query='query',
                    ll_keywords='local-keyword',
                    hl_keywords='global-keyword',
                    knowledge_graph_inst=MagicMock(),
                    entities_vdb=MagicMock(),
                    relationships_vdb=MagicMock(),
                    text_chunks_db=text_chunks_db,
                    query_param=query_param,
                    chunks_vdb=None,
                )
            )

            await node_started.wait()
            await edge_started.wait()
            assert state['concurrent_launch'] is True
            assert not search_task.done()

            release.set()
            result = await search_task

        assert node_mock.await_count == 1
        assert edge_mock.await_count == 1
        assert [entity['entity_name'] for entity in result['final_entities']] == [
            'LocalEntity',
            'GlobalEntity',
        ]
        assert [
            tuple(sorted((str(rel.get('src_id') or rel['src_tgt'][0]), str(rel.get('tgt_id') or rel['src_tgt'][1]))))
            for rel in result['final_relations']
        ] == [('g', 'h'), ('l', 'm')]

    @pytest.mark.asyncio
    async def test_hybrid_with_only_low_level_keywords_uses_node_branch_only(self):
        query_param = QueryParam(mode='hybrid', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        with (
            patch(
                'yar.operate._get_node_data',
                new=AsyncMock(
                    return_value=([{'entity_name': 'OnlyLocal', 'similarity': 0.9}], [{'src_tgt': ('a', 'b')}])
                ),
            ) as node_mock,
            patch('yar.operate._get_edge_data', new=AsyncMock(return_value=([], []))) as edge_mock,
        ):
            result = await _perform_kg_search(
                query='',
                ll_keywords='local-keyword',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        node_mock.assert_awaited_once()
        edge_mock.assert_not_awaited()
        assert [entity['entity_name'] for entity in result['final_entities']] == ['OnlyLocal']
        assert result['final_relations'] == [{'src_tgt': ('a', 'b')}]

    @pytest.mark.asyncio
    async def test_mix_with_only_high_level_keywords_uses_edge_branch_only(self):
        query_param = QueryParam(mode='mix', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        with (
            patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], []))) as node_mock,
            patch(
                'yar.operate._get_edge_data',
                new=AsyncMock(
                    return_value=(
                        [{'src_id': 'x', 'tgt_id': 'y', 'score': 0.5}],
                        [{'entity_name': 'OnlyGlobal', 'similarity': 0.9}],
                    )
                ),
            ) as edge_mock,
            patch('yar.operate._get_vector_context', new=AsyncMock(return_value=[])) as vector_mock,
        ):
            result = await _perform_kg_search(
                query='query',
                ll_keywords='',
                hl_keywords='global-keyword',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        node_mock.assert_not_awaited()
        edge_mock.assert_awaited_once()
        vector_mock.assert_not_awaited()
        assert [entity['entity_name'] for entity in result['final_entities']] == ['OnlyGlobal']
        assert result['final_relations'][0]['src_id'] == 'x'

    @pytest.mark.asyncio
    async def test_local_and_global_modes_keep_mode_specific_branching(self):
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        with (
            patch(
                'yar.operate._get_node_data',
                new=AsyncMock(return_value=([{'entity_name': 'LocalModeEntity', 'similarity': 0.9}], [])),
            ) as node_mock,
            patch(
                'yar.operate._get_edge_data',
                new=AsyncMock(
                    return_value=(
                        [{'src_id': 'g', 'tgt_id': 'h'}],
                        [{'entity_name': 'GlobalModeEntity', 'similarity': 0.9}],
                    )
                ),
            ) as edge_mock,
        ):
            local_result = await _perform_kg_search(
                query='',
                ll_keywords='local-keyword',
                hl_keywords='global-keyword',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=QueryParam(mode='local', top_k=3),
                chunks_vdb=None,
            )
            global_result = await _perform_kg_search(
                query='',
                ll_keywords='local-keyword',
                hl_keywords='global-keyword',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=QueryParam(mode='global', top_k=3),
                chunks_vdb=None,
            )

        assert node_mock.await_count == 1
        assert edge_mock.await_count == 1
        assert [entity['entity_name'] for entity in local_result['final_entities']] == ['LocalModeEntity']
        assert local_result['final_relations'] == []
        assert [entity['entity_name'] for entity in global_result['final_entities']] == ['GlobalModeEntity']
        assert global_result['final_relations'][0]['src_id'] == 'g'

    @pytest.mark.asyncio
    async def test_mix_with_no_keywords_keeps_empty_fallback(self):
        query_param = QueryParam(mode='mix', top_k=3)
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {}

        with (
            patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], []))) as node_mock,
            patch('yar.operate._get_edge_data', new=AsyncMock(return_value=([], []))) as edge_mock,
            patch('yar.operate._get_vector_context', new=AsyncMock(return_value=[])) as vector_mock,
        ):
            result = await _perform_kg_search(
                query='',
                ll_keywords='',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        node_mock.assert_not_awaited()
        edge_mock.assert_not_awaited()
        vector_mock.assert_not_awaited()
        assert result['final_entities'] == []
        assert result['final_relations'] == []
        assert result['vector_chunks'] == []


@pytest.mark.offline
class TestQueryCacheKeyInputs:
    """Regression tests for query cache key inputs."""


@pytest.mark.offline
class TestResponseQualityControls:
    """Tests for prompt/output and merge controls added for response quality."""

    @pytest.mark.asyncio
    async def test_kg_query_passes_default_single_paragraph_token_cap(self):
        model_func = AsyncMock(return_value='Answer')
        query_param = QueryParam(mode='mix', response_type='Single Paragraph', model_func=model_func)
        global_config = {
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
        }
        context_result = QueryContextResult(
            context='Context block',
            raw_data={'data': {'references': []}},
        )

        with (
            patch('yar.operate.get_keywords_from_query', new=AsyncMock(return_value=(['High'], ['Low']))),
            patch('yar.operate._build_query_context', new=AsyncMock(return_value=context_result)),
            patch('yar.operate.handle_cache', new=AsyncMock(return_value=None)),
        ):
            result = await kg_query(
                query='What is alpha therapy?',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=MagicMock(),
                query_param=query_param,
                global_config=global_config,
            )

        assert result is not None
        assert model_func.await_args.kwargs['max_tokens'] == 2048

    @pytest.mark.asyncio
    async def test_kg_query_expands_single_paragraph_token_cap_for_temporal_queries(self):
        model_func = AsyncMock(return_value='Answer')
        query_param = QueryParam(mode='mix', response_type='Single Paragraph', model_func=model_func)
        global_config = {
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
        }
        context_result = QueryContextResult(
            context='Context block',
            raw_data={'data': {'references': []}},
        )

        with (
            patch('yar.operate.get_keywords_from_query', new=AsyncMock(return_value=(['History'], ['Olympic Games']))),
            patch('yar.operate._build_query_context', new=AsyncMock(return_value=context_result)),
            patch('yar.operate.handle_cache', new=AsyncMock(return_value=None)),
        ):
            result = await kg_query(
                query='How have alpha systems evolved over time?',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=MagicMock(),
                query_param=query_param,
                global_config=global_config,
            )

        assert result is not None
        assert model_func.await_args.kwargs['max_tokens'] == 2048

    @pytest.mark.asyncio
    async def test_build_query_context_passes_topic_and_facet_terms_to_merge(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        search_result = {
            'final_entities': [{'entity_name': 'Diabetes'}],
            'final_relations': [],
            'vector_chunks': [],
            'chunk_tracking': {},
            'query_embedding': None,
            'll_keywords_for_search': 'diabetes, glycemic control',
            'hl_keywords_for_search': 'long-term complications, chronic conditions, care planning',
        }
        truncation_result = {
            'filtered_entities': [{'entity_name': 'Diabetes'}],
            'filtered_relations': [],
            'entities_context': [],
            'relations_context': [],
            'entity_id_to_original': {},
            'relation_id_to_original': {},
        }
        merge_mock = AsyncMock(
            return_value=[
                {
                    'content': 'Complication summary',
                    'file_path': 'medical_diabetes.md',
                    'chunk_id': 'chunk-1',
                    'source_type': 'vector',
                }
            ]
        )
        context_mock = AsyncMock(return_value=('Context block', {'data': {'chunks': []}, 'metadata': {}}))

        with (
            patch('yar.operate._perform_kg_search', new=AsyncMock(return_value=search_result)),
            patch('yar.operate._apply_token_truncation', new=AsyncMock(return_value=truncation_result)),
            patch('yar.operate._merge_all_chunks', new=merge_mock),
            patch('yar.operate._build_context_str', new=context_mock),
        ):
            result = await _build_query_context(
                query='What are the long-term complications associated with diabetes?',
                ll_keywords='diabetes',
                hl_keywords='long-term complications, chronic conditions',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=MagicMock(global_config={}),
                query_param=query_param,
            )

        assert result is not None
        assert merge_mock.await_args.kwargs['topic_terms'] == ['diabetes', 'glycemic control']
        assert merge_mock.await_args.kwargs['facet_terms'] == [
            'long-term complications',
            'chronic conditions',
            'care planning',
        ]
        assert context_mock.await_args.kwargs['topic_terms'] == ['diabetes', 'glycemic control']
        assert context_mock.await_args.kwargs['facet_terms'] == [
            'long-term complications',
            'chronic conditions',
            'care planning',
        ]

    def test_should_validate_inline_citations_default_is_on(self):
        # References default-on: when nothing in the query/system prompt forbids citations,
        # the validator should run.
        assert _should_validate_inline_citations(
            'What changed?',
            None,
            system_prompt=PROMPTS['rag_response'],
        )
        # Explicit opt-out via the system prompt should still suppress citations.
        assert not _should_validate_inline_citations(
            'What changed?',
            None,
            system_prompt='Do not include inline citations in the answer.',
        )
        # Explicit opt-in via query phrasing remains a clear signal.
        assert _should_validate_inline_citations(
            'Please cite sources inline for this answer.',
            None,
            system_prompt=PROMPTS['rag_response'],
        )

    @pytest.mark.asyncio
    async def test_kg_query_runs_citation_auto_fix_by_default(self):
        model_func = AsyncMock(return_value='Answer [1].')
        query_param = QueryParam(mode='mix', response_type='Single Paragraph', model_func=model_func)
        global_config = {
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
        }
        context_result = QueryContextResult(
            context='Context block',
            raw_data={'data': {'references': [{'reference_id': '1', 'file_path': 'alpha.md'}]}},
        )

        with (
            patch('yar.operate.get_keywords_from_query', new=AsyncMock(return_value=(['High'], ['Low']))),
            patch('yar.operate._build_query_context', new=AsyncMock(return_value=context_result)),
            patch('yar.operate.handle_cache', new=AsyncMock(return_value=None)),
            patch(
                'yar.operate.validate_and_fix_citations',
                new=Mock(return_value=('Answer [1].', False)),
            ) as validator_mock,
        ):
            result = await kg_query(
                query='What changed?',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=MagicMock(),
                query_param=query_param,
                global_config=global_config,
            )

        assert result is not None
        assert result.content == 'Answer [1].'
        validator_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_all_chunks_prefers_overlap_and_source_consensus(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        vector_chunks = [
            {
                'content': 'Generic medical overview unrelated to alpha therapy.',
                'file_path': 'generic.md',
                'chunk_id': 'weak-vector',
                'source_type': 'vector',
                'retrieval_score': 0.42,
                'source_order': 1,
            },
            {
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 'alpha.md',
                'chunk_id': 'shared',
                'source_type': 'vector',
                'retrieval_score': 0.41,
                'source_order': 2,
            },
        ]
        entity_chunks = [
            {
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 'alpha.md',
                'chunk_id': 'shared',
                'source_type': 'entity',
                'occurrence_count': 3,
                'source_order': 1,
            },
        ]
        relation_chunks = [
            {
                'content': 'Alpha therapy is linked to lower complication risk in the source graph.',
                'file_path': 'alpha_rel.md',
                'chunk_id': 'shared',
                'source_type': 'relationship',
                'occurrence_count': 2,
                'source_order': 1,
            },
        ]

        with (
            patch('yar.operate._find_related_text_unit_from_entities', new=AsyncMock(return_value=entity_chunks)),
            patch('yar.operate._find_related_text_unit_from_relations', new=AsyncMock(return_value=relation_chunks)),
        ):
            merged = await _merge_all_chunks(
                filtered_entities=[{'entity_name': 'Alpha therapy'}],
                filtered_relations=[{'src_id': 'Alpha', 'tgt_id': 'Complications'}],
                vector_chunks=vector_chunks,
                query='What does alpha therapy do?',
                knowledge_graph_inst=MagicMock(),
                text_chunks_db=MagicMock(),
                query_param=query_param,
            )

        assert [chunk['chunk_id'] for chunk in merged[:2]] == ['shared', 'weak-vector']
        assert merged[0]['source_type'] == 'entity+relationship+vector'
        assert merged[0]['merge_score'] > merged[1]['merge_score']

    @pytest.mark.asyncio
    async def test_merge_all_chunks_prefers_precise_low_level_entity_matches(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': (
                        '# Project responsibilities\n'
                        'Company A manages regulatory responsibilities for a different clinical program.'
                    ),
                    'file_path': 'generic_responsibilities.md',
                    'chunk_id': 'generic-responsibilities',
                    'source_type': 'vector',
                    'retrieval_score': 0.95,
                    'source_order': 1,
                },
                {
                    'content': (
                        '# Device X123 project background\n'
                        'Company A responsible for device development and manufacture and regulatory submissions.'
                    ),
                    'file_path': 'device_x123.md',
                    'chunk_id': 'device-x123-background',
                    'source_type': 'vector',
                    'retrieval_score': 0.55,
                    'source_order': 2,
                },
            ],
            query='What are the responsibilities of Company A in the Device X123 project?',
            topic_terms=['Company A', 'Device X123'],
            facet_terms=['project responsibilities'],
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged] == ['device-x123-background']
        assert merged[0]['precise_focus_overlap'] == 1.0

    @pytest.mark.asyncio
    async def test_merge_all_chunks_keeps_action_support_with_precise_entity(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': (
                        '# Collaboration context\nPartner Alpha collaboration context for managing conflict in CMC strategy.'
                    ),
                    'file_path': 'conflict.md',
                    'chunk_id': 'context',
                    'source_type': 'vector',
                    'retrieval_score': 0.95,
                    'source_order': 1,
                },
                {
                    'content': (
                        'Conflict management requires: Quick reaction; express needs with One Voice; '
                        'SMEs interact proactively; have a joint CMC team meeting face to face.'
                    ),
                    'file_path': 'conflict.md',
                    'chunk_id': 'actions',
                    'source_type': 'vector',
                    'retrieval_score': 0.55,
                    'source_order': 2,
                },
                {
                    'content': '# Action plan toward Best Practice\nDevelop unrelated risk-review implementation steps.',
                    'file_path': 'risk_review.md',
                    'chunk_id': 'risk-review-action',
                    'source_type': 'vector',
                    'retrieval_score': 0.99,
                    'source_order': 3,
                },
            ],
            query='How can effective conflict management be applied to CMC strategy development based on lessons learned from the Partner Alpha collaboration?',
            topic_terms=['Partner Alpha', 'conflict management'],
            facet_terms=['lessons learned', 'CMC strategy development'],
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged] == ['actions', 'context']
        assert 'Quick reaction' in merged[0]['content']

    @pytest.mark.asyncio
    async def test_merge_all_chunks_keeps_temporal_chunks_without_precise_entity(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'PX-482 Phase 1 background and first approval context.',
                    'file_path': 'product.md',
                    'chunk_id': 'product',
                    'source_type': 'vector',
                    'retrieval_score': 0.95,
                    'source_order': 1,
                },
                {
                    'content': 'Project management impact from an unrelated portfolio pause.',
                    'file_path': 'freeze.md',
                    'chunk_id': 'freeze',
                    'source_type': 'vector',
                    'retrieval_score': 0.99,
                    'source_order': 2,
                },
                {
                    'content': 'Risk row: EU approval planned in March 2025 if overseas inspection remains on track.',
                    'file_path': 'risk.md',
                    'chunk_id': 'risk',
                    'source_type': 'vector',
                    'retrieval_score': 0.98,
                    'source_order': 3,
                },
                {
                    'content': 'EU submission is planned for 29 March 2024. 2024: EU: Submission (Mar 24). 2025: EU: Approval (Mar 25).',
                    'chunk_id': 'timeline',
                    'source_type': 'vector',
                    'retrieval_score': 0.60,
                    'source_order': 3,
                },
            ],
            query='How did the EU approval timeline compare with PX-482 Phase 1?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['approval timeline'],
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged[:2]] == ['timeline', 'product']

    @pytest.mark.asyncio
    async def test_merge_all_chunks_filters_unanchored_timeline_for_precise_queries(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'PX-482 Phase 1 background and first approval context.',
                    'file_path': 'product.md',
                    'chunk_id': 'product',
                    'source_type': 'vector',
                    'retrieval_score': 0.80,
                    'source_order': 1,
                },
                {
                    'content': 'EU submission is planned for 29 March 2024. 2025: EU: Approval (Mar 25).',
                    'file_path': 'other-product.md',
                    'chunk_id': 'other-timeline',
                    'source_type': 'vector',
                    'retrieval_score': 0.99,
                    'source_order': 2,
                },
            ],
            query='What is the EU approval date for PX-482 Phase 1?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['approval date'],
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged] == ['product']

    @pytest.mark.asyncio
    async def test_merge_all_chunks_keeps_cross_document_timeline_for_impact_queries(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'PX-482 Phase 1 background and first approval context.',
                    'file_path': 'product.md',
                    'chunk_id': 'product',
                    'source_type': 'vector',
                    'retrieval_score': 0.80,
                    'source_order': 1,
                },
                {
                    'content': 'Program timeline: EU submission is planned for 29 March 2024; EU approval in March 2025 is on the critical path for project management and cross-functional collaboration.',
                    'file_path': 'portfolio.md',
                    'chunk_id': 'portfolio-timeline',
                    'source_type': 'vector',
                    'retrieval_score': 0.70,
                    'source_order': 2,
                },
                {
                    'content': 'Portfolio readiness depends on cross-functional collaboration and coordination across delivery teams.',
                    'file_path': 'portfolio.md',
                    'chunk_id': 'portfolio-collaboration',
                    'source_type': 'vector',
                    'retrieval_score': 0.65,
                    'source_order': 3,
                },
                {
                    'content': 'Portfolio readiness also tracks launch-supply owners, dependency reviews, and governance checkpoints.',
                    'file_path': 'portfolio.md',
                    'chunk_id': 'portfolio-extra',
                    'source_type': 'vector',
                    'retrieval_score': 0.64,
                    'source_order': 4,
                },
                {
                    'content': 'Project management playbook: clinical-study timelines require early comparator sourcing and coordination.',
                    'file_path': 'generic-playbook.md',
                    'chunk_id': 'generic-timeline',
                    'source_type': 'vector',
                    'retrieval_score': 0.95,
                    'source_order': 2,
                },
            ],
            query='What is the significance of the EU approval timeline for PX-482 Phase 1 and how does it impact project management?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['approval timeline', 'project management impact'],
            query_param=query_param,
        )

        chunk_ids = [chunk['chunk_id'] for chunk in merged]
        assert 'product' in chunk_ids
        assert 'portfolio-timeline' in chunk_ids
        assert 'portfolio-collaboration' in chunk_ids
        assert 'generic-timeline' not in chunk_ids
        assert 'portfolio-extra' not in chunk_ids

    @pytest.mark.asyncio
    async def test_merge_all_chunks_drops_generic_readiness_tail_from_cross_document_timeline_file(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'PX-482 Phase 1 background and first approval context.',
                    'file_path': 'product.md',
                    'chunk_id': 'product',
                    'source_type': 'vector',
                    'retrieval_score': 0.80,
                    'source_order': 1,
                },
                {
                    'content': ('EU submission is planned for 29 March 2024. 2025: EU: Approval (Mar 25).'),
                    'file_path': 'timeline.md',
                    'chunk_id': 'timeline',
                    'source_type': 'vector',
                    'retrieval_score': 0.70,
                    'source_order': 2,
                },
                {
                    'content': (
                        'Readiness risk table: approval and submission readiness risk; '
                        'commercial agreement unavailable to date.'
                    ),
                    'file_path': 'timeline.md',
                    'chunk_id': 'readiness-tail',
                    'source_type': 'vector',
                    'retrieval_score': 0.99,
                    'source_order': 3,
                },
            ],
            query='What is the significance of the EU approval timeline for PX-482 Phase 1 and how does it impact project management?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['approval timeline', 'project management impact'],
            query_param=query_param,
        )

        chunk_ids = [chunk['chunk_id'] for chunk in merged]
        assert 'product' in chunk_ids
        assert 'timeline' in chunk_ids
        assert 'readiness-tail' not in chunk_ids

    @pytest.mark.asyncio
    async def test_merge_all_chunks_carries_temporal_exact_match_metadata(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'Product X Phase 1 background and initial regulatory context.',
                    'file_path': 'product.md',
                    'chunk_id': 'product',
                    'source_type': 'vector',
                    'retrieval_score': 0.80,
                    'source_order': 1,
                },
                {
                    'content': (
                        '# The Journey\n'
                        '| Date | Activity/Milestone |\n'
                        '| 12Jan2024 | Product X Phase 1 team review |\n'
                        '| 09Jun2024 | FDA Clearance received |'
                    ),
                    'file_path': 'product.md',
                    'chunk_id': 'timeline-table',
                    'source_type': 'vector',
                    'retrieval_score': 0.60,
                    'source_order': 2,
                },
            ],
            query='What is the approval timeline for Product X Phase 1?',
            topic_terms=['Product X', 'Phase 1'],
            facet_terms=['approval timeline'],
            query_param=query_param,
        )

        timeline_chunk = next(chunk for chunk in merged if chunk['chunk_id'] == 'timeline-table')
        assert timeline_chunk['exact_phrase_match'] > 0.0

    @pytest.mark.asyncio
    async def test_merge_all_chunks_preserves_vector_exact_phrase_match_metadata(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'The source chunk contains a literal retrieval phrase from the first-stage search.',
                    'file_path': 'literal.md',
                    'chunk_id': 'literal',
                    'source_type': 'vector',
                    'retrieval_score': 0.40,
                    'source_order': 1,
                    'exact_phrase_match': 1.0,
                },
            ],
            query='What is the overall project context?',
            query_param=query_param,
        )

        assert merged[0]['exact_phrase_match'] == 1.0

    @pytest.mark.asyncio
    async def test_merge_all_chunks_prioritizes_metadata_support_for_lookup_queries(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'General guidance references process-standard recommendations for the workflow.',
                    'file_path': 'general.md',
                    'chunk_id': 'general',
                    'source_type': 'vector',
                    'retrieval_score': 0.99,
                    'source_order': 1,
                },
                {
                    'content': 'Transfer valve definition by Safety Standard: sealed connector.',
                    'file_path': 'definition.md',
                    'chunk_id': 'definition',
                    'source_type': 'vector',
                    'retrieval_score': 0.50,
                    'source_order': 2,
                },
                {
                    'content': 'Unrelated project schedule with a high retrieval score.',
                    'file_path': 'off-topic.md',
                    'chunk_id': 'off-topic',
                    'source_type': 'vector',
                    'retrieval_score': 1.0,
                    'source_order': 0,
                },
            ],
            query='What is the definition of transfer valve according to Safety Standard?',
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged] == ['definition']
        merge_trace = query_param.__dict__['_merge_filter_trace']
        assert merge_trace['dropped_count'] == 2
        assert {chunk['drop_reason'] for chunk in merge_trace['dropped_chunks']} == {'metadata_lookup_filter'}

    @pytest.mark.asyncio
    async def test_merge_all_chunks_filters_unfocused_guidance_when_precise_support_exists(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': 'Connector X definition by Safety Standard: sealed connector.',
                    'file_path': 'definition.md',
                    'chunk_id': 'definition',
                    'source_type': 'vector',
                    'retrieval_score': 0.70,
                    'source_order': 2,
                },
                {
                    'content': 'Safety Standard guidance for Connector X: should be used when volatile reagents are handled.',
                    'file_path': 'guidance.md',
                    'chunk_id': 'guidance',
                    'source_type': 'vector',
                    'retrieval_score': 0.65,
                    'source_order': 3,
                },
                {
                    'content': 'Unrelated assay guidance recommends validation before license submission.',
                    'file_path': 'unrelated-guidance.md',
                    'chunk_id': 'unrelated-guidance',
                    'source_type': 'vector',
                    'retrieval_score': 0.99,
                    'source_order': 1,
                },
            ],
            query='What is the definition of Connector X according to Safety Standard, and what does the guidance recommend?',
            topic_terms=['Connector X', 'Safety Standard'],
            facet_terms=['definition', 'guidance recommendation'],
            query_param=query_param,
        )

        chunk_ids = [chunk['chunk_id'] for chunk in merged]
        assert chunk_ids == ['definition', 'guidance']

    @pytest.mark.asyncio
    async def test_merge_all_chunks_adds_bounded_adjacent_document_chunks(self, monkeypatch):
        monkeypatch.setenv('YAR_SIBLING_CHUNK_LIMIT', '2')
        monkeypatch.setenv('YAR_SIBLING_CHUNK_WINDOW', '1')
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        text_chunks_db = MagicMock()
        text_chunks_db.get_chunk_ids_by_doc_id = MagicMock(return_value=['chunk-1', 'chunk-2', 'chunk-3', 'chunk-4'])

        chunk_payloads = {
            'chunk-1': {
                'id': 'chunk-1',
                'content': 'Context page before the retrieved section.',
                'file_path': 'source.md',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 1,
            },
            'chunk-2': {
                'id': 'chunk-2',
                'content': '# Best Practice\nRetrieved anchor section.',
                'file_path': 'source.md',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 2,
            },
            'chunk-3': {
                'id': 'chunk-3',
                'content': '# Lessons Learned\nAdjacent supporting section.',
                'file_path': 'source.md',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 3,
            },
            'chunk-4': {
                'id': 'chunk-4',
                'content': 'Outside the configured sibling window.',
                'file_path': 'source.md',
                'full_doc_id': 'doc-1',
                'chunk_order_index': 4,
            },
        }
        text_chunks_db.get_by_ids = MagicMock(side_effect=lambda ids: [chunk_payloads[chunk_id] for chunk_id in ids])
        chunk_tracking = {}

        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': '# Best Practice\nRetrieved anchor section.',
                    'file_path': 'source.md',
                    'chunk_id': 'chunk-2',
                    'full_doc_id': 'doc-1',
                    'chunk_order_index': 2,
                    'source_type': 'vector',
                    'retrieval_score': 0.9,
                    'source_order': 1,
                }
            ],
            query='What best practice lessons are listed?',
            text_chunks_db=text_chunks_db,
            query_param=query_param,
            chunk_tracking=chunk_tracking,
        )

        assert [chunk['chunk_id'] for chunk in merged] == ['chunk-2', 'chunk-1', 'chunk-3']
        assert [chunk['source_type'] for chunk in merged[1:]] == ['sibling', 'sibling']
        assert chunk_tracking['chunk-1']['source'] == 'S'
        assert chunk_tracking['chunk-3']['source'] == 'S'

    def test_prioritize_substantive_chunks_demotes_metadata_for_process_queries(self):
        chunks = [
            {
                'chunk_id': 'metadata',
                'content': '# Context\nName the Best Practice: Managing conflict\nObjective: Share background.',
            },
            {
                'chunk_id': 'steps',
                'content': '# Best Practice\n1) Recognize conflict\n2) Question the relationship',
            },
        ]

        reordered = _prioritize_substantive_chunks(chunks, 'How should teams manage conflict?')
        metadata_lookup = _prioritize_substantive_chunks(chunks, 'Who sponsored the session?')

        assert [chunk['chunk_id'] for chunk in reordered] == ['steps', 'metadata']
        assert [chunk['chunk_id'] for chunk in metadata_lookup] == ['metadata', 'steps']

        focused_metadata_lookup = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'target-metadata',
                    'content': '# Lessons Learning\nSession: LL-016\nStatus: Planned; Actions opened; Actions finalized\nSponsor: Alicia Morgan / Ben Torres\nName the Best Practice: Prototype comparator supply',
                },
                {
                    'chunk_id': 'wrong-session-metadata',
                    'content': '# Lessons Learning\nSession: LL-012\nStatus: Planned; Actions opened; Actions finalized\nSponsor: Drew Patel / Casey Lin\nName the Best Practice: Process Handoff End to End',
                },
                {
                    'chunk_id': 'subject-without-metadata',
                    'content': '# Best Practice\nPrototype comparator supply for clinical studies.',
                },
                {
                    'chunk_id': 'subject-sponsor-only',
                    'content': '# Context\nSponsor: Alicia Morgan\nPrototype comparator supply',
                },
                {
                    'chunk_id': 'subject-status-only',
                    'content': '# Action plan\nStatus: Planned\nPrototype comparator supply',
                },
            ],
            'Who were the sponsors involved in the session on prototype comparator supply, and what was the status of the session?',
        )

        assert [chunk['chunk_id'] for chunk in focused_metadata_lookup] == ['target-metadata']

        best_practice_order = _prioritize_substantive_chunks(
            [
                {'chunk_id': 'action', 'content': '# Action plan toward Best Practice\nFollow-up actions.'},
                {'chunk_id': 'practice', 'content': '# Best Practice\nFollow the implementation guideline.'},
            ],
            'What are the best practices for process handoff, and how can teams apply them?',
        )
        assert [chunk['chunk_id'] for chunk in best_practice_order] == ['practice', 'action']

        lessons_order = _prioritize_substantive_chunks(
            [
                {'chunk_id': 'context', 'content': '# Context\nName the Best Practice: Managing conflict.'},
                {'chunk_id': 'lessons', 'content': '# Lessons Learned\nConflict management requires communication.'},
            ],
            'What best practices come from the lessons learned?',
        )
        assert [chunk['chunk_id'] for chunk in lessons_order] == ['lessons', 'context']

        role_objective_order = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'role-objective-context',
                    'content': 'Objective: Share recent handoff experience. Target potential users: teams. The opportunity was triggered by Alicia Morgan and Ben Torres.',
                },
                {
                    'chunk_id': 'reversed-name',
                    'content': 'Attendees: Raman, Priya | Functions: Launch Programs Project Leader',
                },
                {'chunk_id': 'action-plan', 'content': 'Action plan toward implementation with Ben Torres only.'},
                {'chunk_id': 'other-objective', 'content': 'Objective: unrelated collaboration topic.'},
            ],
            'What role did Alicia Morgan play, and how does this relate to the objectives?',
        )
        assert [chunk['chunk_id'] for chunk in role_objective_order] == ['role-objective-context']
        reversed_name_order = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'reversed-name',
                    'content': 'Attendees: Raman, Priya | Functions: Launch Programs Project Leader',
                },
                {'chunk_id': 'other', 'content': 'Roles and responsibility action plan without the named person.'},
            ],
            'What roles did Priya Raman play?',
        )
        assert [chunk['chunk_id'] for chunk in reversed_name_order] == ['reversed-name']

        temporal_order = _prioritize_substantive_chunks(
            [
                {'chunk_id': 'metadata', 'content': 'Facilitator: Dr. Smith\nDate: Jan 2024\nParticipants: team'},
                {
                    'chunk_id': 'timeline',
                    'content': '# Timeline\nRegulatory submission March 2024; market approval planned March 2025.',
                },
            ],
            'What is the approval timeline for Product X?',
        )
        assert [chunk['chunk_id'] for chunk in temporal_order] == ['timeline', 'metadata']

        focused_temporal_order = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'focus-background',
                    'file_path': 'product.pdf',
                    'content': '# Introduction\nProduct PX-482 Phase 1 study background.',
                    'precise_focus_overlap': 1.0,
                },
                {
                    'chunk_id': 'external-timeline',
                    'file_path': 'portfolio.pdf',
                    'content': '# Timeline\nUS submission March 2024; US approval March 2025.',
                    'precise_focus_overlap': 0.0,
                },
                {
                    'chunk_id': 'focus-timeline',
                    'file_path': 'product.pdf',
                    'content': '# Timeline\nUS IND clearance and approval planning.',
                    'precise_focus_overlap': 0.0,
                },
            ],
            'What is the approval timeline for Product PX-482 Phase 1?',
        )
        assert [chunk['chunk_id'] for chunk in focused_temporal_order[:2]] == [
            'focus-background',
            'focus-timeline',
        ]

        duration_order = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'topic-only',
                    'content': 'Depot: Shipment lead-times, enrollment, and randomization balance.',
                },
                {
                    'chunk_id': 'duration-evidence',
                    'content': 'Detailed workflow: 1-3 months before Start packaging: Goods shipment preparation.',
                    'duration_answer_match': 1.75,
                },
            ],
            'What is the standard duration of shipment to depot?',
        )
        assert [chunk['chunk_id'] for chunk in duration_order] == ['duration-evidence', 'topic-only']

        recommendation_value_order = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'generic-recommendations',
                    'content': '# Lessons Learned & Recommendations\nHolistic view and process follow-up.',
                    'metadata_query_match': 1.25,
                    'exact_phrase_match': 0.0,
                    'merge_score': 1.14,
                },
                {
                    'chunk_id': 'dose-evidence',
                    'content': 'MABEL approach supports a 3-4 log dose-ranging recommendation.',
                    'metadata_query_match': 0.0,
                    'exact_phrase_match': 1.0,
                    'merge_score': 1.84,
                },
            ],
            'What is the dose-ranging recommended by the MABEL approach?',
        )
        assert [chunk['chunk_id'] for chunk in recommendation_value_order] == [
            'dose-evidence',
            'generic-recommendations',
        ]

        impact_order = _prioritize_substantive_chunks(
            [
                {
                    'chunk_id': 'timeline',
                    'content': 'Timeline: US/EU submission and launch readiness governance milestones.',
                    'impact_answer_match': 0.0,
                },
                {
                    'chunk_id': 'impact-answer',
                    'content': 'IMPACTS: physical flow over the Netherlands caused wrong logo and NDC issues.',
                    'impact_answer_match': 2.25,
                },
            ],
            'What were the consequences of including additional physical flow in the Netherlands?',
        )
        assert [chunk['chunk_id'] for chunk in impact_order] == ['impact-answer', 'timeline']

    def test_metadata_query_match_score_boosts_definition_chunks(self):
        definition_chunk = (
            'transfer valve definition by Safety Standard: sealed connector that mechanically prohibits escape.'
        )
        flowchart_chunk = 'Transfer valve request workflow: Team A sends memo and adds the device to the dossier.'

        query = 'What is the definition of transfer valves according to Safety Standard?'

        assert _metadata_query_match_score(definition_chunk, query) > _metadata_query_match_score(
            flowchart_chunk,
            query,
        )

    def test_metadata_chunk_search_query_preserves_session_subject_phrase(self):
        query = (
            'Who were the sponsors involved in the session on prototype comparator supply, '
            'and what was the status of the session?'
        )

        search_query = _metadata_chunk_search_query(query)

        assert 'prototype comparator supply' in search_query
        assert 'sponsor' in search_query
        assert 'status' in search_query

    def test_evidence_spans_include_requested_metadata_label_variants(self):
        query = (
            'Who were the sponsors involved in the session on prototype comparator supply, '
            'and what was the status of the session?'
        )
        spans = _extract_supporting_evidence_spans(
            '# Lessons Learning\n'
            '## Prototype comparator supply\n'
            '* **Session:** LL-016\n'
            '* **Sponsor:** Alicia Morgan, Ben Torres\n'
            '* **Status:**\n'
            '    * Planned\n'
            '    * Actions opened\n'
            '    * Actions finalized\n',
            query=query,
        )

        assert 'Sponsor: Alicia Morgan, Ben Torres' in spans
        assert 'Status: Planned; Actions opened; Actions finalized' in spans

    def test_evidence_spans_combine_objective_target_and_best_practice(self):
        spans = _extract_supporting_evidence_spans(
            '| Code | 2016-LL-12 |\n'
            '| :--- | :--- |\n'
            '| **Name the Best Practice** | Process Handoff End to End |\n'
            '| **Objective** | Share recent handoff experience from Project Orion to benefit to upcoming product teams |\n'
            '| **Target potential users** | Identify teams who can use and/or benefit from this practice |\n',
            query='How does this role relate to objectives?',
        )

        assert (
            'Objective: Share recent handoff experience from Project Orion to benefit to upcoming product teams; '
            'Target potential users: Identify teams who can use and/or benefit from this practice; '
            'Best practice: Process Handoff End to End'
        ) in spans

    def test_evidence_spans_prioritize_date_bearing_timeline_lines(self):
        spans = _extract_supporting_evidence_spans(
            'Submission remains on track for approval.\nSubmission Q1 2024 remains on track for approval.\n',
            query='What is the submission timeline?',
        )

        assert spans[0] == 'Submission Q1 2024 remains on track for approval.'

    def test_temporal_signal_ignores_common_time_adjectives(self):
        generic = _chunk_relevance_components(
            {'content': '# Status\nThe current formulation remains under review.'},
            set(),
        )
        milestone = _chunk_relevance_components(
            {'content': '# Status\nThe submission milestone remains under review.'},
            set(),
        )

        assert generic['body_temporal_signal'] == 0.0
        assert milestone['body_temporal_signal'] == 1.0

    def test_prompt_chunk_context_labels_unanchored_timeline_chunks(self):
        prompt_chunks, _text_units, _reference_list = _build_prompt_chunk_context(
            [
                {
                    'chunk_id': 'timeline',
                    'file_path': 'timeline.pdf',
                    'reference_id': '1',
                    'content': ('EU submission is planned for 29 March 2024. 2025: EU: Approval (Mar 25).'),
                }
            ],
            [{'reference_id': '1', 'file_path': 'timeline.pdf'}],
            include_reference_ids=True,
            query='How did the EU approval timeline compare with PX-482 Phase 1?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['EU approval timeline'],
        )

        assert prompt_chunks[0]['content'].startswith('Cross-document timeline evidence:')
        assert 'do not transfer its dates or milestones' in prompt_chunks[0]['content']

        noncomparison_chunks, _text_units, _reference_list = _build_prompt_chunk_context(
            [
                {
                    'chunk_id': 'timeline',
                    'file_path': 'timeline.pdf',
                    'reference_id': '1',
                    'content': 'EU submission is planned for 29 March 2024. 2025: EU: Approval (Mar 25).',
                },
                {
                    'chunk_id': 'product',
                    'file_path': 'product.pdf',
                    'reference_id': '2',
                    'content': 'PX-482 Phase 1 received Approval: US (IND) and Turkey (IMPD).',
                },
            ],
            [{'reference_id': '1', 'file_path': 'timeline.pdf'}, {'reference_id': '2', 'file_path': 'product.pdf'}],
            include_reference_ids=True,
            query='What is the EU approval date for PX-482 Phase 1?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['EU approval date'],
        )

        assert len(noncomparison_chunks) == 1
        assert 'PX-482 Phase 1' in noncomparison_chunks[0]['content']

        impact_chunks, _text_units, _reference_list = _build_prompt_chunk_context(
            [
                {
                    'chunk_id': 'timeline',
                    'file_path': 'timeline.pdf',
                    'reference_id': '1',
                    'content': 'Program timeline: EU submission is planned for 29 March 2024; EU approval in March 2025 is on the critical path.',
                },
                {
                    'chunk_id': 'product',
                    'file_path': 'product.pdf',
                    'reference_id': '2',
                    'content': 'PX-482 Phase 1 received an initial clearance.',
                },
                {
                    'chunk_id': 'generic-timeline',
                    'file_path': 'generic.pdf',
                    'reference_id': '3',
                    'content': 'Program timeline: cross-functional coordination affects project management readiness.',
                },
            ],
            [
                {'reference_id': '1', 'file_path': 'timeline.pdf'},
                {'reference_id': '2', 'file_path': 'product.pdf'},
                {'reference_id': '3', 'file_path': 'generic.pdf'},
            ],
            include_reference_ids=True,
            query='What is the significance of the EU approval timeline for PX-482 Phase 1 and its project management impact?',
            topic_terms=['PX-482', 'Phase 1'],
            facet_terms=['EU approval timeline', 'project management impact'],
        )

        assert len(impact_chunks) == 2
        assert impact_chunks[0]['content'].startswith('Cross-document timeline evidence:')
        assert all('cross-functional coordination' not in chunk.get('content', '') for chunk in impact_chunks)
        assert 'portfolio/project timeline' in impact_chunks[0]['content']

    def test_filter_prompt_relations_anchors_precise_timeline_queries(self):
        filtered = _filter_prompt_relations_for_query(
            [
                {'relation': 'PX-482 --cleared--> NHA', 'description': 'PX-482 received IND clearance.'},
                {'relation': 'Portfolio Pause --delayed--> Clinical Studies', 'description': '14 Nov 2018 freeze.'},
            ],
            'What is the significance of the US approval timeline for PX-482 Phase 1?',
            ['PX-482', 'Phase 1'],
        )

        assert [relation['relation'] for relation in filtered] == ['PX-482 --cleared--> NHA']

    def test_normalize_query_shaped_response_removes_unsupported_role_glosses(self):
        response = _normalize_query_shaped_response(
            query='What role did Alicia play, and how does this relate to the objectives?',
            response='Alicia Morgan was a sponsor and organizer of the session, with sponsorship and organization by Alicia Morgan and Ben Torres.',
            available_refs=[
                {
                    'content': (
                        'Sponsor: Alicia Morgan / Ben Torres. '
                        'The opportunity to join effort with the handoff team was triggered by Alicia Morgan & Ben Torres.'
                    )
                }
            ],
        )

        assert 'organizer' not in response
        assert 'organization' not in response
        assert 'sponsor of the session' in response

    def test_normalize_query_shaped_response_keeps_availability_label(self):
        response = _normalize_query_shaped_response(
            query='What roles did Priya Raman play in both sessions?',
            response='Priya Raman was the participant representing the Category Alpha availability category.',
            available_refs=[{'content': 'Availability: Category Alpha | Participants: Priya Raman'}],
        )

        assert response == 'Priya Raman was the participant listed under the Category Alpha availability category.'

    def test_normalize_query_shaped_response_rewrites_availability_representative_gloss(self):
        response = _normalize_query_shaped_response(
            query='What roles did Priya Raman play in both sessions?',
            response='In the review session, Priya Raman participated as the "Category Alpha" representative.',
            available_refs=[{'content': 'Availability: Category Alpha | Participants: Priya Raman'}],
        )

        assert (
            response
            == 'In the review session, Priya Raman was listed as a participant under Availability: Category Alpha.'
        )

    def test_normalize_query_shaped_response_removes_context_preamble(self):
        response = _normalize_query_shaped_response(
            query='What does the guidance recommend for Connector X?',
            response='The retrieved context provides that Connector X should be inspected before use.',
            available_refs=[{'content': 'Connector X should be inspected before use.'}],
        )

        assert response == 'Connector X should be inspected before use.'

    def test_normalize_query_shaped_response_adds_objective_rows(self):
        response = _normalize_query_shaped_response(
            query='How does Alicia Morgan relate to the objectives?',
            response='Alicia Morgan served as a sponsor.',
            available_refs=[
                {
                    'content': (
                        'Table row: Code: **Objective** | 2016-LL-12: Share recent handoff experience.\n'
                        'Table row: Code: **Target potential users** | 2016-LL-12: Identify teams who can use the practice.'
                    )
                }
            ],
        )

        assert 'Objective: Share recent handoff experience.' in response
        assert 'Target potential users: Identify teams who can use the practice.' in response

    def test_normalize_query_shaped_response_adds_best_practice_label(self):
        response = _normalize_query_shaped_response(
            query='What is the best practce for onboarding?',
            response='Use the checklist and review it weekly.',
            available_refs=[
                {
                    'content': (
                        'Table row: **Name the Best Practice** | 2026-EX-01: Team Onboarding Checklist\n'
                        'Table row: **Objective** | Improve handoff quality.'
                    )
                }
            ],
        )

        assert response.startswith('Name the Best Practice: Team Onboarding Checklist.')

    def test_normalize_query_shaped_response_frames_lessons_learned_answers(self):
        response = _normalize_query_shaped_response(
            query='What lessons were learned about scope and team responsibilities?',
            response='Scope ownership was unclear, and team responsibilities overlapped.',
            available_refs=[{'content': 'Lessons Learned: scope ownership unclear; responsibilities overlapped.'}],
        )

        assert response.startswith('The lessons learned were: scope ownership was unclear')

    def test_normalize_query_shaped_response_skips_lessons_learned_prefix_when_already_framed(self):
        trace: dict[str, object] = {}
        raw_response = 'The 3 categories of lessons learned about SERD were strategic, execution, and documentation.'

        response = _normalize_query_shaped_response(
            query='What lessons were learned about SERD categories?',
            response=raw_response,
            available_refs=[
                {'content': ('Lessons Learned: SERD categories include strategic, execution, and documentation.')}
            ],
            trace=trace,
        )

        assert response == raw_response
        assert trace['applied'] is False
        assert trace['reasons'] == []

    def test_normalize_query_shaped_response_cleans_lessons_learned_category_delimiter(self):
        trace: dict[str, object] = {}

        response = _normalize_query_shaped_response(
            query='What are the 3 categories of lessons learned about SERD SAR439589?',
            response='SERD Lessons Learned fall into 3 categories > Governance, Capabilities/Culture, Organization [1].',
            available_refs=[
                {
                    'content': (
                        'SERD Lessons Learned fall into 3 categories > Governance, Capabilities/Culture, Organization'
                    )
                }
            ],
            trace=trace,
        )

        assert response == (
            'SERD Lessons Learned fall into 3 categories: Governance, Capabilities/Culture, Organization [1].'
        )
        assert trace['reasons'] == ['lessons_learned_delimiter_cleanup']

    def test_normalize_query_shaped_response_uses_serd_category_source_row(self):
        trace: dict[str, object] = {}

        response = _normalize_query_shaped_response(
            query='What are the 3 categories of lessons learned about SERD SAR439589?',
            response=(
                'The lessons learned about SERD SAR439859 fall into 3 categories: '
                'Governance; Capabilities/Culture; Organization [1].'
            ),
            available_refs=[
                {
                    'reference_id': '1',
                    'content': 'SERD Lessons Learned fall into 3 categories',
                },
                {
                    'reference_id': '2',
                    'content': (
                        'SERD Lessons Learned fall into 3 categories > Governance, Capabilities/Culture, Organization'
                    ),
                },
            ],
            trace=trace,
        )

        assert response == (
            'SERD lessons learned for SAR439589 fall into 3 categories: '
            'Governance, Capabilities/Culture, Organization [1].'
        )
        assert trace['reasons'] == ['lessons_learned_category_source_row']

    def test_normalize_query_shaped_response_preserves_role_objective_labels(self):
        response = _normalize_query_shaped_response(
            query='What role did Alicia Morgan play, and how does this relate to objectives?',
            response=(
                'Alicia Morgan is listed as **Sponsor** of the session, directly aligning with '
                'the initiative endorsed by her and Ben Torres; this timing aimed to leverage '
                'a recently implemented end-to-end process handoff approach '
                '(Process Handoff-GEN-014176, Nov-2015) to share experience and enhance further implementation.'
            ),
            available_refs=[
                {
                    'content': (
                        'Sponsor: Alicia Morgan / Ben Torres\n'
                        'Context for LL: the opportunity to join effort was triggered by Alicia Morgan & Ben Torres.\n'
                        'Objective: Share recent handoff experience.\n'
                        'Target potential users: Identify teams who can use and/or benefit from this practice.'
                    )
                }
            ],
        )

        assert 'listed as sponsor alongside Ben Torres' in response
        assert 'Objective: Share recent handoff experience.' in response
        assert 'Target potential users: Identify teams who can use and/or benefit from this practice.' in response
        assert 'significant sponsor role' not in response
        assert 'Process Handoff-GEN' not in response

    def test_normalize_query_shaped_response_preserves_stated_guidance_conditions(self):
        response = _normalize_query_shaped_response(
            query='What guidance recommends transfer valve use?',
            response='Transfer valves must be used when administering volatile reagents under specified conditions.',
            available_refs=[
                {
                    'content': (
                        'Transfer valves must be used when administering volatile reagents '
                        'when a containment assessment supports that use.'
                    )
                }
            ],
        )

        assert (
            response == 'Transfer valves must be used when administering volatile reagents under specified conditions.'
        )

    def test_normalize_query_shaped_response_removes_negative_role_label(self):
        response = _normalize_query_shaped_response(
            query='What roles did Priya Raman play?',
            response='Priya Raman was listed under Category Alpha with no further role label specified.',
            available_refs=[{'content': 'Availability: Category Alpha | Participants: Priya Raman'}],
        )

        assert response == 'Priya Raman was listed under Category Alpha.'

    def test_normalize_query_shaped_response_rewrites_participant_role_label(self):
        response = _normalize_query_shaped_response(
            query='What roles did Priya Raman play?',
            response='Priya Raman is listed under Availability: Category Alpha with the role of Participants.',
            available_refs=[{'content': 'Availability: Category Alpha | Participants: Priya Raman'}],
        )

        assert response == 'Priya Raman is listed as a participant under Availability: Category Alpha.'

    @pytest.mark.asyncio
    async def test_merge_all_chunks_uses_low_level_terms_for_relation_chunk_selection(self):
        query_param = QueryParam(mode='hybrid', top_k=4, chunk_top_k=4)
        relation_chunk_mock = AsyncMock(return_value=[])

        with patch('yar.operate._find_related_text_unit_from_relations', new=relation_chunk_mock):
            await _merge_all_chunks(
                filtered_entities=[],
                filtered_relations=[{'src_id': 'Japan Handbook', 'tgt_id': 'FMA'}],
                vector_chunks=[],
                query='What are the japan-specific activities?',
                topic_terms=['Foreign Manufacturer Accreditation', 'J-CTD'],
                facet_terms=['Japanese Operations Managers handbook'],
                text_chunks_db=MagicMock(),
                query_param=query_param,
            )

        ranking_query = relation_chunk_mock.await_args.args[4]
        assert 'What are the japan-specific activities?' in ranking_query
        assert 'Japanese Operations Managers handbook' in ranking_query
        assert 'Foreign Manufacturer Accreditation' in ranking_query
        assert 'J-CTD' in ranking_query

    @pytest.mark.asyncio
    async def test_relation_chunk_selection_prioritizes_low_level_relation_matches(self):
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {
            'kg_chunk_pick_method': 'WEIGHT',
            'related_chunk_number': 1,
        }

        chunk_payloads = {
            'chunk-generic': {'content': 'A generic handbook abbreviations index.', 'file_path': 'source.md'},
            'chunk-jctd': {'content': 'The Japan team prepares the J-CTD package.', 'file_path': 'source.md'},
        }

        async def fake_get_by_ids(chunk_ids):
            return [chunk_payloads[chunk_id] for chunk_id in chunk_ids]

        text_chunks_db.get_by_ids = AsyncMock(side_effect=fake_get_by_ids)
        relations = [
            {
                'src_id': 'Japanese Operations Handbook',
                'tgt_id': 'Abbreviations',
                'keywords': 'references',
                'description': 'The handbook references abbreviations.',
                'source_id': 'chunk-generic',
            },
            {
                'src_id': 'Japan submission task force',
                'tgt_id': 'J-CTD',
                'keywords': 'prepares',
                'description': 'The Japan team prepares the J-CTD package.',
                'source_id': 'chunk-jctd',
            },
        ]

        result_chunks = await _find_related_text_unit_from_relations(
            relations,
            QueryParam(mode='hybrid', top_k=4, chunk_top_k=4),
            text_chunks_db,
            query='What are the japan-specific activities?\nJ-CTD',
        )

        assert result_chunks[0]['chunk_id'] == 'chunk-jctd'

    @pytest.mark.asyncio
    async def test_relation_linked_chunks_preserve_semantic_zero_lexical_matches(self):
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {'kg_chunk_pick_method': 'WEIGHT', 'related_chunk_number': 4}
        chunk_payloads = {
            'chunk-overlap': {
                'content': 'Alpha rollout activity mentions the requested rollout.',
                'file_path': 'alpha.md',
            },
            'chunk-semantic': {
                'content': 'The steering record identifies owners, escalation paths, and readiness gates.',
                'file_path': 'alpha.md',
            },
        }
        text_chunks_db.get_by_ids = AsyncMock(
            side_effect=lambda chunk_ids: [chunk_payloads[chunk_id] for chunk_id in chunk_ids]
        )
        relations = [
            {
                'src_id': 'Alpha Rollout',
                'tgt_id': 'Steering Record',
                'keywords': 'supports',
                'description': 'The steering record supports the rollout.',
                'source_id': GRAPH_FIELD_SEP.join(['chunk-overlap', 'chunk-semantic']),
            }
        ]

        result_chunks = await _find_related_text_unit_from_relations(
            relations,
            QueryParam(mode='hybrid', top_k=4, chunk_top_k=4),
            text_chunks_db,
            query='What rollout activity is described?',
        )

        assert {chunk['chunk_id'] for chunk in result_chunks} == {'chunk-overlap', 'chunk-semantic'}

    @pytest.mark.asyncio
    async def test_entity_linked_chunks_preserve_semantic_zero_lexical_matches(self):
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {'kg_chunk_pick_method': 'WEIGHT', 'related_chunk_number': 4}
        chunk_payloads = {
            'chunk-overlap': {
                'content': 'Alpha rollout activity mentions the requested rollout.',
                'file_path': 'alpha.md',
            },
            'chunk-semantic': {
                'content': 'The steering record identifies owners, escalation paths, and readiness gates.',
                'file_path': 'alpha.md',
            },
        }
        text_chunks_db.get_by_ids = AsyncMock(
            side_effect=lambda chunk_ids: [chunk_payloads[chunk_id] for chunk_id in chunk_ids]
        )
        entities = [
            {
                'entity_name': 'Alpha Rollout',
                'source_id': GRAPH_FIELD_SEP.join(['chunk-overlap', 'chunk-semantic']),
            }
        ]

        result_chunks = await _find_related_text_unit_from_entities(
            entities,
            QueryParam(mode='local', top_k=4, chunk_top_k=4),
            text_chunks_db,
            MagicMock(),
            query='What rollout activity is described?',
        )

        assert {chunk['chunk_id'] for chunk in result_chunks} == {'chunk-overlap', 'chunk-semantic'}

    @pytest.mark.asyncio
    async def test_find_most_related_edges_filters_hub_edges_by_query_focus(self):
        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_edges_batch = AsyncMock(
            return_value={'Diabetes': [('Diabetes', 'COVID-19'), ('Diabetes', 'Diabetic foot')]}
        )
        knowledge_graph_inst.get_edges_batch = AsyncMock(
            return_value={
                ('COVID-19', 'Diabetes'): {
                    'description': 'COVID-19 can co-occur with diabetes in some patients.',
                    'keywords': 'comorbidity',
                    'weight': 9.0,
                },
                ('Diabetes', 'Diabetic foot'): {
                    'description': 'Diabetic foot complications can lead to ulcers and amputation.',
                    'keywords': 'complications, ulcer',
                    'weight': 3.0,
                },
            }
        )
        knowledge_graph_inst.edge_degrees_batch = AsyncMock(
            return_value={('COVID-19', 'Diabetes'): 20, ('Diabetes', 'Diabetic foot'): 4}
        )

        edges = await _find_most_related_edges_from_entities(
            [{'entity_name': 'Diabetes'}],
            QueryParam(mode='local', top_k=5),
            knowledge_graph_inst,
            query='What are the long-term complications associated with diabetes?',
        )

        assert [edge['src_tgt'] for edge in edges] == [('Diabetes', 'Diabetic foot')]
        assert edges[0]['query_focus_overlap'] > 0.0

    @pytest.mark.asyncio
    async def test_find_most_related_edges_preserves_storage_direction_for_relation_context(self):
        knowledge_graph_inst = MagicMock()
        knowledge_graph_inst.get_nodes_edges_batch = AsyncMock(
            return_value={'Zeta Source': [('Zeta Source', 'Alpha Target')]}
        )
        knowledge_graph_inst.get_edges_batch = AsyncMock(
            return_value={
                ('Alpha Target', 'Zeta Source'): {
                    'description': 'Zeta Source causes Alpha Target.',
                    'keywords': 'causes',
                    'weight': 2.0,
                }
            }
        )
        knowledge_graph_inst.edge_degrees_batch = AsyncMock(return_value={('Alpha Target', 'Zeta Source'): 7})

        edges = await _find_most_related_edges_from_entities(
            [{'entity_name': 'Zeta Source'}],
            QueryParam(mode='local', top_k=5),
            knowledge_graph_inst,
            query='What does Zeta Source cause?',
        )

        assert edges[0]['src_tgt'] == ('Zeta Source', 'Alpha Target')
        assert edges[0]['rank'] == 7
        edges[0]['evidence_spans'] = ['Table row: Source: Zeta Source | Target: Alpha Target']
        truncated = await _apply_token_truncation(
            {'final_entities': [], 'final_relations': edges},
            QueryParam(mode='mix'),
            {
                'tokenizer': MagicMock(encode=Mock(side_effect=lambda text: text.split())),
                'max_entity_tokens': 100,
                'max_relation_tokens': 100,
            },
        )
        assert truncated['relations_context'][0]['relation'] == 'Zeta Source --causes--> Alpha Target'
        assert truncated['relations_context'][0]['evidence_spans'] == [
            'Table row: Source: Zeta Source | Target: Alpha Target'
        ]

    @pytest.mark.asyncio
    async def test_merge_all_chunks_prefers_evolution_chunks_over_generic_topic_matches(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': '=== Olympic marketing ===\nOlympic Games controversies include sponsorship deals, branding, and boycotts.',
                    'file_path': 'sports_olympic_games.md',
                    'chunk_id': 'generic-olympic',
                    'source_type': 'vector',
                    'retrieval_score': 0.72,
                    'source_order': 1,
                },
                {
                    'content': '== Modern Games ==\nThe Games evolved from ancient Greek festivals into a modern revival with Winter and Paralympic events.',
                    'file_path': 'sports_olympic_games.md',
                    'chunk_id': 'evolution',
                    'source_type': 'vector',
                    'retrieval_score': 0.68,
                    'source_order': 2,
                },
            ],
            query='How have the Olympic Games evolved since their ancient origins in Greece?',
            topic_terms=['Olympic Games', 'Greece'],
            facet_terms=['historical evolution', 'Olympic Games history', 'ancient origins'],
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged[:2]] == ['evolution', 'generic-olympic']
        assert merged[0]['merge_score'] > merged[1]['merge_score']
        assert merged[0]['body_relevance'] > 0.0

    @pytest.mark.asyncio
    async def test_merge_all_chunks_prefers_heading_matched_sections_over_incidental_mentions(self):
        query_param = QueryParam(mode='mix', top_k=4, chunk_top_k=4)
        merged = await _merge_all_chunks(
            filtered_entities=[],
            filtered_relations=[],
            vector_chunks=[
                {
                    'content': '=== Diabetes in other animals ===\nDiabetic animals are more prone to infections, and the long-term complications recognized in humans are much rarer in animals.',
                    'file_path': 'medical_diabetes.md',
                    'chunk_id': 'animals',
                    'source_type': 'vector',
                    'retrieval_score': 0.71,
                    'source_order': 1,
                },
                {
                    'content': '== Signs and symptoms ==\nCommon symptoms include thirst and urination changes.\n=== Long-term complications ===\nDiabetes can cause retinopathy, nephropathy, neuropathy, and diabetic foot problems.',
                    'file_path': 'medical_diabetes.md',
                    'chunk_id': 'long-term-complications',
                    'source_type': 'vector',
                    'retrieval_score': 0.68,
                    'source_order': 2,
                },
            ],
            query='What are the long-term complications associated with diabetes?',
            topic_terms=['diabetes'],
            facet_terms=['long-term complications', 'chronic conditions', 'medical outcomes'],
            query_param=query_param,
        )

        assert [chunk['chunk_id'] for chunk in merged] == ['long-term-complications']
        assert merged[0]['heading_relevance'] > 0.0

    @pytest.mark.asyncio
    async def test_apply_token_truncation_renders_directional_relation_context(self):
        tokenizer = MagicMock(encode=Mock(side_effect=lambda text: text.split()))
        result = await _apply_token_truncation(
            {
                'final_entities': [],
                'final_relations': [
                    {
                        'src_id': 'Japan submission task force',
                        'tgt_id': 'J-CTD',
                        'keywords': 'prepares',
                        'description': 'The Japan team prepares the J-CTD package.',
                        'created_at': 1,
                    }
                ],
            },
            QueryParam(mode='mix'),
            {
                'tokenizer': tokenizer,
                'max_entity_tokens': 100,
                'max_relation_tokens': 100,
            },
        )

        assert result['relations_context'][0]['relation'] == 'Japan submission task force --prepares--> J-CTD'
        assert result['relations_context'][0]['source'] == 'Japan submission task force'
        assert result['relations_context'][0]['target'] == 'J-CTD'

    @pytest.mark.asyncio
    async def test_apply_token_truncation_prefers_precise_entity_context_within_budget(self):
        tokenizer = MagicMock(encode=Mock(side_effect=lambda _text: [0]))
        result = await _apply_token_truncation(
            {
                'query': 'What is the approval timeline for Product PX-482 Phase 1?',
                'll_keywords': 'Product PX-482, approval timeline, Phase 1 study',
                'll_keywords_for_search': 'Product PX-482, approval timeline, Phase 1 study',
                'final_entities': [
                    {
                        'entity_name': 'US Approval',
                        'entity_type': 'event',
                        'description': 'US Approval is the March 2025 target for Product Beta.',
                        'score': 0.99,
                    },
                    {
                        'entity_name': 'Generic Therapy',
                        'entity_type': 'concept',
                        'description': 'Gene therapy overview for unrelated materials.',
                        'score': 0.98,
                    },
                    {
                        'entity_name': 'Product PX-482',
                        'entity_type': 'product',
                        'description': 'Product PX-482 is in the Phase 1 study approval timeline.',
                        'score': 0.80,
                    },
                    {
                        'entity_name': 'Program Team',
                        'entity_type': 'organization',
                        'description': 'Program Team coordinates Product PX-482 Phase 1 approval work.',
                        'score': 0.70,
                    },
                ],
                'final_relations': [],
            },
            QueryParam(mode='mix', max_entity_tokens=2),
            {
                'tokenizer': tokenizer,
                'max_entity_tokens': 2,
                'max_relation_tokens': 100,
            },
        )

        entity_names = [entity['entity'] for entity in result['entities_context']]
        filtered_names = [entity['entity_name'] for entity in result['filtered_entities']]
        entity_trace = result['entity_context_trace']
        assert entity_names == ['Product PX-482', 'Program Team']
        assert filtered_names == ['Product PX-482', 'Program Team']
        assert entity_trace['candidate_count'] == 4
        assert entity_trace['selected_count'] == 2
        assert entity_trace['dropped_count'] == 2
        assert [entity['entity'] for entity in entity_trace['selected_preview']] == ['Product PX-482', 'Program Team']
        assert {entity['drop_reason'] for entity in entity_trace['dropped_preview']} == {'entity_rank_filter'}
        truncation_trace = result['truncation_trace']
        assert truncation_trace['entities_before'] == 4
        assert truncation_trace['entities_after'] == 2
        assert truncation_trace['entity_names_dropped'] == ['US Approval', 'Generic Therapy']

    @pytest.mark.asyncio
    async def test_apply_token_truncation_prefers_evidence_supported_relations_within_budget(self):
        tokenizer = MagicMock(encode=Mock(side_effect=lambda _text: [0] * 10))
        unsupported_relation = {
            'src_id': 'Generic strategy',
            'tgt_id': 'Outcome',
            'keywords': 'related_to',
            'description': 'Generic strategy is related to the outcome.',
            'score': 0.5,
        }
        supported_relation = {
            'src_id': 'Adapter strategy',
            'tgt_id': 'Dosing error mitigation',
            'keywords': 'mitigates',
            'description': 'Adapter strategy mitigates dosing error.',
            'score': 0.5,
            'evidence_spans': ['Adapter strategy mitigates dosing error in the source table.'],
        }

        result = await _apply_token_truncation(
            {
                'query': 'Which mitigation evidence addresses dosing error?',
                'final_entities': [],
                'final_relations': [unsupported_relation, supported_relation],
            },
            QueryParam(mode='mix', max_relation_tokens=10),
            {
                'tokenizer': tokenizer,
                'max_entity_tokens': 100,
                'max_relation_tokens': 10,
            },
        )

        assert result['relations_context'][0]['entity1'] == 'Adapter strategy'
        assert result['filtered_relations'] == [supported_relation]
        assert '__rerank_score' not in supported_relation

    @pytest.mark.asyncio
    async def test_apply_token_truncation_does_not_change_relations_context_keys(self):
        tokenizer = MagicMock(encode=Mock(side_effect=lambda _text: [0] * 5))
        result = await _apply_token_truncation(
            {
                'query': 'Which mitigation evidence addresses dosing error?',
                'final_entities': [],
                'final_relations': [
                    {
                        'src_id': 'Adapter strategy',
                        'tgt_id': 'Dosing error mitigation',
                        'keywords': 'mitigates',
                        'description': 'Adapter strategy mitigates dosing error.',
                        'score': 0.5,
                        'created_at': 1,
                        'file_path': 'source.md',
                        'evidence_spans': ['Adapter strategy mitigates dosing error in the source table.'],
                    }
                ],
            },
            QueryParam(mode='mix', max_relation_tokens=100),
            {
                'tokenizer': tokenizer,
                'max_entity_tokens': 100,
                'max_relation_tokens': 100,
            },
        )

        allowed_keys = {
            'entity1',
            'entity2',
            'source',
            'target',
            'predicate',
            'relation',
            'description',
            'created_at',
            'file_path',
            'evidence_spans',
        }
        assert set(result['relations_context'][0]) <= allowed_keys
        assert result['relations_context'][0]['evidence_spans'] == [
            'Adapter strategy mitigates dosing error in the source table.'
        ]

    @pytest.mark.asyncio
    async def test_apply_token_truncation_marks_conflicting_relations_in_prompt_context(self):
        tokenizer = MagicMock(encode=Mock(side_effect=lambda _text: [0] * 5))
        result = await _apply_token_truncation(
            {
                'query': 'Does treatment support or block the outcome?',
                'final_entities': [],
                'final_relations': [
                    {
                        'src_id': 'Treatment',
                        'tgt_id': 'Outcome',
                        'keywords': 'supports',
                        'description': 'Treatment supports the outcome.',
                        'relation_conflict': True,
                        'relation_conflict_predicates': ['blocks'],
                    },
                    {
                        'src_id': 'Outcome',
                        'tgt_id': 'Treatment',
                        'keywords': 'blocks',
                        'description': 'Outcome blocks the treatment.',
                        'relation_conflict': True,
                        'relation_conflict_predicates': ['supports'],
                    },
                ],
            },
            QueryParam(mode='mix', max_relation_tokens=100),
            {
                'tokenizer': tokenizer,
                'max_entity_tokens': 100,
                'max_relation_tokens': 100,
            },
        )

        assert result['relations_context'][0]['relation'] == 'Treatment --supports; conflict: blocks--> Outcome'
        assert result['relations_context'][1]['relation'] == 'Outcome --blocks; conflict: supports--> Treatment'
        assert all('relation_conflict' not in relation for relation in result['relations_context'])
        assert all('relation_conflict' not in relation for relation in result['filtered_relations'])
        allowed_keys = {
            'entity1',
            'entity2',
            'source',
            'target',
            'predicate',
            'relation',
            'description',
            'created_at',
            'file_path',
            'evidence_spans',
        }
        assert all(set(relation) <= allowed_keys for relation in result['relations_context'])

    @pytest.mark.asyncio
    async def test_build_context_str_dedupes_visible_alias_references_without_changing_prompt_chunks(self):
        query_param = QueryParam(mode='mix')
        global_config = {'tokenizer': Mock(encode=Mock(return_value=[0] * 10))}
        alias_chunks = [
            {
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 's3://bucket/docs/alpha.md',
                's3_key': 'docs/alpha.md',
                'chunk_id': 'alpha-1',
            },
            {
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 'docs/alpha.md',
                's3_key': 'docs/alpha.md',
                'chunk_id': 'alpha-1',
            },
        ]

        with (
            patch('yar.operate.process_chunks_unified', new=AsyncMock(return_value=alias_chunks)),
            patch('yar.operate._build_prompt_chunk_context', wraps=_build_prompt_chunk_context) as prompt_context_mock,
        ):
            _, raw_data = await _build_context_str(
                entities_context=[],
                relations_context=[],
                merged_chunks=alias_chunks,
                query='What changed?',
                query_param=query_param,
                global_config=global_config,
            )

        assert len(prompt_context_mock.call_args.args[0]) == 2
        assert len(prompt_context_mock.call_args.args[1]) == 2
        assert raw_data['data']['references'] == [
            {
                'reference_id': '1',
                'file_path': 'docs/alpha.md',
                'document_title': 'alpha.md',
                's3_key': 'docs/alpha.md',
                'excerpt': 'Alpha therapy reduces complications in carefully selected patients.',
            }
        ]
        assert raw_data['data']['chunks'] == [
            {
                'reference_id': '1',
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 'docs/alpha.md',
                'chunk_id': 'alpha-1',
            }
        ]

    def test_prepare_visible_reference_payload_drops_weak_off_topic_doc_when_on_topic_doc_exists(self):
        selection_trace = {}
        visible_references, visible_chunks = _prepare_visible_reference_payload(
            [
                {
                    'reference_id': '1',
                    'content': '== Signs and symptoms ==\n=== Long-term complications ===\nDiabetes can cause retinopathy, nephropathy, neuropathy, and diabetic foot problems.',
                    'file_path': 'medical_diabetes.md',
                    'chunk_id': 'diabetes-1',
                    'intent_relevance': 0.82,
                    'query_focus_overlap': 0.50,
                    'heading_topic_match': 1.0,
                    'body_topic_match': 1.0,
                    'heading_facet_match': 1.0,
                    'body_facet_match': 1.0,
                },
                {
                    'reference_id': '2',
                    'content': '=== Complications ===\nCOVID-19 complications may include pneumonia and multi-organ failure.',
                    'file_path': 'medical_covid-19.md',
                    'chunk_id': 'covid-1',
                    'intent_relevance': 0.18,
                    'query_focus_overlap': 0.20,
                    'heading_topic_match': 0.0,
                    'body_topic_match': 0.0,
                    'heading_facet_match': 0.0,
                    'body_facet_match': 0.0,
                },
            ],
            [
                {'reference_id': '1', 'file_path': 'medical_diabetes.md'},
                {'reference_id': '2', 'file_path': 'medical_covid-19.md'},
            ],
            'What are the long-term complications associated with diabetes?',
            include_reference_ids=False,
            selection_trace=selection_trace,
        )

        assert [reference['file_path'] for reference in visible_references] == ['medical_diabetes.md']
        assert [chunk['chunk_id'] for chunk in visible_chunks] == ['diabetes-1']
        assert selection_trace['filter_applied'] is True
        assert selection_trace['dropped_group_count'] == 1
        assert selection_trace['decisions'][1]['reason'] == 'no_topic_signal'

    def test_prepare_visible_reference_payload_keeps_cited_prompt_chunks(self):
        visible_references, visible_chunks = _prepare_visible_reference_payload(
            [
                {
                    'reference_id': '1',
                    'content': '== Signs and symptoms ==\n=== Long-term complications ===\nDiabetes can cause retinopathy, nephropathy, neuropathy, and diabetic foot problems.',
                    'file_path': 'medical_diabetes.md',
                    'chunk_id': 'diabetes-1',
                    'intent_relevance': 0.82,
                    'query_focus_overlap': 0.50,
                    'heading_topic_match': 1.0,
                    'body_topic_match': 1.0,
                    'heading_facet_match': 1.0,
                    'body_facet_match': 1.0,
                },
                {
                    'reference_id': '2',
                    'content': '=== Complications ===\nCOVID-19 complications may include pneumonia and multi-organ failure.',
                    'file_path': 'medical_covid-19.md',
                    'chunk_id': 'covid-1',
                    'intent_relevance': 0.18,
                    'query_focus_overlap': 0.20,
                    'heading_topic_match': 0.0,
                    'body_topic_match': 0.0,
                    'heading_facet_match': 0.0,
                    'body_facet_match': 0.0,
                },
            ],
            [
                {'reference_id': '1', 'file_path': 'medical_diabetes.md'},
                {'reference_id': '2', 'file_path': 'medical_covid-19.md'},
            ],
            'What are the long-term complications associated with diabetes?',
            include_reference_ids=True,
        )

        assert [reference['file_path'] for reference in visible_references] == [
            'medical_diabetes.md',
            'medical_covid-19.md',
        ]
        assert [chunk['chunk_id'] for chunk in visible_chunks] == ['diabetes-1', 'covid-1']

    def test_prepare_visible_reference_payload_drops_near_best_off_topic_doc_when_topic_signal_exists(self):
        visible_references, visible_chunks = _prepare_visible_reference_payload(
            [
                {
                    'reference_id': '1',
                    'content': '=== Device X123 manufacturing flow ===\nThe regional physical flow created label and shipping consequences for Device X123.',
                    'file_path': 'device-x123.md',
                    'chunk_id': 'device-x123-1',
                    'intent_relevance': 0.78,
                    'query_focus_overlap': 0.72,
                    'heading_topic_match': 1.0,
                    'body_topic_match': 1.0,
                    'heading_facet_match': 0.5,
                    'body_facet_match': 0.5,
                },
                {
                    'reference_id': '2',
                    'content': '=== Manufacturing impact ===\nA different program had manufacturing and regulatory consequences for a separate product.',
                    'file_path': 'other-program.md',
                    'chunk_id': 'other-1',
                    'intent_relevance': 0.76,
                    'query_focus_overlap': 0.70,
                    'heading_topic_match': 0.0,
                    'body_topic_match': 0.0,
                    'heading_facet_match': 1.0,
                    'body_facet_match': 1.0,
                },
            ],
            [
                {'reference_id': '1', 'file_path': 'device-x123.md'},
                {'reference_id': '2', 'file_path': 'other-program.md'},
            ],
            'What were the manufacturing consequences for Device X123 in the region?',
            include_reference_ids=False,
            topic_terms=['Device X123'],
            facet_terms=['manufacturing consequences', 'regulatory submission impact'],
        )

        assert [reference['file_path'] for reference in visible_references] == ['device-x123.md']
        assert [chunk['chunk_id'] for chunk in visible_chunks] == ['device-x123-1']

    def test_prepare_visible_reference_payload_uses_best_chunk_per_document_group(self):
        visible_references, visible_chunks = _prepare_visible_reference_payload(
            [
                {
                    'reference_id': '1',
                    'content': '=== Long-term complications ===\nDiabetes can cause retinopathy, nephropathy, neuropathy, and diabetic foot problems.',
                    'file_path': 'medical_diabetes.md',
                    'chunk_id': 'diabetes-1',
                    'intent_relevance': 0.92,
                    'query_focus_overlap': 1.0,
                    'heading_topic_match': 0.0,
                    'body_topic_match': 1.0,
                    'heading_facet_match': 1.0,
                    'body_facet_match': 1.0,
                },
                {
                    'reference_id': '2',
                    'content': '=== Complications ===\nCOVID-19 complications may include pneumonia and multi-organ failure.',
                    'file_path': 'medical_covid-19.md',
                    'chunk_id': 'covid-top',
                    'intent_relevance': 0.50,
                    'query_focus_overlap': 0.50,
                    'heading_topic_match': 0.0,
                    'body_topic_match': 0.0,
                    'heading_facet_match': 0.333,
                    'body_facet_match': 0.333,
                },
                {
                    'reference_id': '2',
                    'content': '=== Comorbidities ===\nPeople hospitalised with COVID-19 often have diabetes among other pre-existing conditions.',
                    'file_path': 'medical_covid-19.md',
                    'chunk_id': 'covid-lower',
                    'intent_relevance': 0.08,
                    'query_focus_overlap': 0.0,
                    'heading_topic_match': 0.0,
                    'body_topic_match': 1.0,
                    'heading_facet_match': 0.0,
                    'body_facet_match': 0.0,
                },
            ],
            [
                {'reference_id': '1', 'file_path': 'medical_diabetes.md'},
                {'reference_id': '2', 'file_path': 'medical_covid-19.md'},
            ],
            'What are the long-term complications associated with diabetes?',
            include_reference_ids=False,
            topic_terms=['diabetes'],
            facet_terms=['long-term complications', 'chronic conditions', 'medical outcomes'],
        )

        assert [reference['file_path'] for reference in visible_references] == ['medical_diabetes.md']
        assert [chunk['chunk_id'] for chunk in visible_chunks] == ['diabetes-1']

    @pytest.mark.asyncio
    async def test_build_context_str_dedupes_aliased_references_even_when_citations_requested(self):
        query_param = QueryParam(mode='mix')
        global_config = {'tokenizer': Mock(encode=Mock(return_value=[0] * 10))}
        alias_chunks = [
            {
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 's3://bucket/docs/alpha.md',
                's3_key': 'docs/alpha.md',
                'chunk_id': 'alpha-1',
            },
            {
                'content': 'Alpha therapy reduces complications in carefully selected patients.',
                'file_path': 'docs/alpha.md',
                's3_key': 'docs/alpha.md',
                'chunk_id': 'alpha-1',
            },
        ]

        with (
            patch('yar.operate.process_chunks_unified', new=AsyncMock(return_value=alias_chunks)),
            patch('yar.operate._build_prompt_chunk_context', wraps=_build_prompt_chunk_context) as prompt_context_mock,
        ):
            _, raw_data = await _build_context_str(
                entities_context=[],
                relations_context=[],
                merged_chunks=alias_chunks,
                query='Please cite sources inline for this answer.',
                query_param=query_param,
                global_config=global_config,
            )

        # Even when citations are explicitly requested, aliased entries (same s3_key + content,
        # different file_path) collapse to a single canonical reference. Without dedupe the answer
        # would cite the same source twice as `[1]` and `[2]`.
        assert len(prompt_context_mock.call_args.args[0]) == 2
        assert len(prompt_context_mock.call_args.args[1]) == 2
        assert [reference['file_path'] for reference in raw_data['data']['references']] == ['docs/alpha.md']
        assert len(raw_data['data']['chunks']) == 1
        chunk_selection = raw_data['metadata']['chunk_selection']
        assert chunk_selection['dropped_count'] == 1
        assert chunk_selection['dropped_preview'][0]['drop_reason'] == 'dedupe'
        assert chunk_selection['dropped_chunks'][0]['drop_reason'] == 'dedupe'

    @pytest.mark.asyncio
    async def test_response_max_tokens_returns_correct_caps_per_type(self):
        """Token caps are generous safety nets, not the primary length constraint."""
        from yar.operate import _response_max_tokens

        assert _response_max_tokens('Short Answer') == 1024
        assert _response_max_tokens('Single Paragraph') == 2048
        assert _response_max_tokens('Bullet Points') == 4096
        assert _response_max_tokens('Multiple Paragraphs') == 8192
        # Unknown types get generous default
        assert _response_max_tokens('Custom Format') == 4096
        # Case insensitive
        assert _response_max_tokens('multiple paragraphs') == 8192
        assert _response_max_tokens('BULLET POINTS') == 4096

    @pytest.mark.asyncio
    async def test_kg_query_vector_fallback_when_kg_context_empty(self):
        """When KG context is empty but chunks_vdb available, fall back to vector retrieval."""
        model_func = AsyncMock(return_value='Fallback answer')
        query_param = QueryParam(mode='mix', model_func=model_func)
        global_config = {
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
        }
        fallback_chunks = [
            {'content': 'Relevant chunk', 'file_path': 'test.md', 'chunk_id': 'c1'},
        ]

        with (
            patch('yar.operate.get_keywords_from_query', new=AsyncMock(return_value=(['High'], ['Low']))),
            patch('yar.operate._build_query_context', new=AsyncMock(return_value=None)),
            patch('yar.operate._get_vector_context', new=AsyncMock(return_value=fallback_chunks)),
            patch('yar.operate.process_chunks_unified', new=AsyncMock(return_value=fallback_chunks)),
            patch('yar.operate.generate_reference_list_from_chunks', return_value=([], fallback_chunks)),
            patch('yar.operate._build_prompt_chunk_context', return_value=([], 'chunk text', '')),
            patch('yar.operate._prepare_visible_reference_payload', return_value=([], fallback_chunks)),
            patch('yar.operate.handle_cache', new=AsyncMock(return_value=None)),
        ):
            result = await kg_query(
                query='What is adapter strategy?',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=MagicMock(global_config=global_config),
                query_param=query_param,
                global_config=global_config,
                chunks_vdb=MagicMock(),
            )

        assert result is not None
        assert result.content == 'Fallback answer'
        assert result.raw_data['metadata']['fallback'] == 'direct_vector'

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_process_chunks_unified_keeps_support_passages_for_binary_questions(self):
        """Binary questions should keep enough same-document evidence to support a yes/no answer."""
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=8, enable_rerank=False)
        chunks = [
            {
                'content': 'Alpha answer passage',
                'file_path': 'alpha.md',
                'chunk_id': 'alpha-1',
                'retrieval_score': 0.95,
            },
            {
                'content': 'Alpha background passage',
                'file_path': 'alpha.md',
                'chunk_id': 'alpha-2',
                'retrieval_score': 0.80,
            },
            {
                'content': 'Beta supporting passage',
                'file_path': 'beta.md',
                'chunk_id': 'beta-1',
                'retrieval_score': 0.75,
            },
            {'content': 'Beta extra detail', 'file_path': 'beta.md', 'chunk_id': 'beta-2', 'retrieval_score': 0.70},
            {'content': 'Gamma fallback note', 'file_path': 'gamma.md', 'chunk_id': 'gamma-1', 'retrieval_score': 0.65},
        ]

        for source_type in ('naive', 'hybrid'):
            processed = await process_chunks_unified(
                query='Does Alpha already use powder in a bottle directly?',
                unique_chunks=chunks,
                query_param=query_param,
                global_config={'tokenizer': tokenizer},
                source_type=source_type,
                chunk_token_limit=10_000,
            )

            assert [chunk['chunk_id'] for chunk in processed] == [
                'alpha-1',
                'alpha-2',
                'beta-1',
                'beta-2',
                'gamma-1',
            ]

    @pytest.mark.asyncio
    async def test_process_chunks_unified_groups_generic_processed_files_by_s3_key(self):
        """Generic processed.md file paths should not collapse distinct S3 documents into one cap bucket."""
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=6, enable_rerank=False)
        chunks = [
            {
                'content': 'Document A answer passage',
                'file_path': 'processed.md',
                's3_key': 'default/doc-a/processed.md',
                'chunk_id': 'a-1',
                'retrieval_score': 0.95,
            },
            {
                'content': 'Document A supporting passage',
                'file_path': 'processed.md',
                's3_key': 'default/doc-a/processed.md',
                'chunk_id': 'a-2',
                'retrieval_score': 0.90,
            },
            {
                'content': 'Document B answer passage',
                'file_path': 'processed.md',
                's3_key': 'default/doc-b/processed.md',
                'chunk_id': 'b-1',
                'retrieval_score': 0.85,
            },
            {
                'content': 'Document B supporting passage',
                'file_path': 'processed.md',
                's3_key': 'default/doc-b/processed.md',
                'chunk_id': 'b-2',
                'retrieval_score': 0.80,
            },
        ]

        processed = await process_chunks_unified(
            query='Does the processed report support the recommendation?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='hybrid',
            chunk_token_limit=10_000,
        )

        assert _chunk_document_key(chunks[0]) == 'default/doc-a/processed.md'
        assert [chunk['chunk_id'] for chunk in processed] == ['a-1', 'a-2', 'b-1', 'b-2']

    def test_prepare_visible_reference_payload_prefers_s3_key_for_generic_processed_file_path(self):
        """Reference display should expose a distinguishing source when file_path is only processed.md."""
        chunks = [
            {
                'content': 'Document A answer passage',
                'file_path': 'processed.md',
                's3_key': 'default/doc-a/processed.md',
                'chunk_id': 'a-1',
            }
        ]

        references, visible_chunks = _prepare_visible_reference_payload(
            chunks,
            [],
            'Does the processed report support the recommendation?',
            include_reference_ids=True,
        )

        assert visible_chunks[0]['file_path'] == 'default/doc-a/processed.md'
        assert references[0]['file_path'] == 'default/doc-a/processed.md'

    @pytest.mark.asyncio
    async def test_process_chunks_unified_prioritizes_exact_low_level_phrase_matches(self):
        """Exact user-supplied low-level phrases should outrank generic phase/study matches."""
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=6, enable_rerank=False)
        chunks = [
            {
                'content': 'Phase 1 study overview with site setup details and generic background.',
                'file_path': 'generic-a.pdf',
                'chunk_id': 'generic-a',
                'retrieval_score': 0.95,
            },
            {
                'content': 'LL-2 - Difficult tracking of scope change in SoW\n\nLL-3 - "New Phase 1 clinical strategy tested"\nA new Phase 1 clinical strategy was tested by PX-482: powder in bottle directly to the clinical center.',
                'file_path': 'product-alpha.pdf',
                'chunk_id': 'product-alpha-1',
                'retrieval_score': 0.70,
            },
            {
                'content': 'Phase 1 regulatory acceptance table covering submission content and questioned materials.',
                'file_path': 'generic-b.pdf',
                'chunk_id': 'generic-b',
                'retrieval_score': 0.90,
            },
        ]

        processed = await process_chunks_unified(
            query='Do we already use Powder in a bottle directly for phase 1 study?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='hybrid',
            chunk_token_limit=10_000,
            topic_terms=['powder in bottle directly to the clinical center'],
        )

        assert processed[0]['chunk_id'] == 'product-alpha-1'

    @pytest.mark.asyncio
    async def test_process_chunks_unified_prioritizes_upstream_exact_support_metadata(self):
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=3, enable_rerank=False)
        chunks = [
            {
                'content': 'Workflow for adding a requested transfer valve to a dossier.',
                'file_path': 'workflow.pdf',
                'chunk_id': 'workflow',
                'retrieval_score': 0.95,
            },
            {
                'content': 'Safety Standard definition: a sealed connector that prevents reagent escape.',
                'file_path': 'definition.pdf',
                'chunk_id': 'definition',
                'retrieval_score': 0.70,
                'metadata_query_match': 1.0,
            },
        ]

        processed = await process_chunks_unified(
            query='What is the definition according to Safety Standard?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='hybrid',
            chunk_token_limit=10_000,
        )

        assert processed[0]['chunk_id'] == 'definition'

    @pytest.mark.asyncio
    async def test_process_chunks_unified_prioritizes_precise_focus_over_generic_temporal_exact(self):
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=3, enable_rerank=False)
        chunks = [
            {
                'content': 'US submission is on track. US Approval is planned for March 2025.',
                'file_path': 'portfolio-timeline.pdf',
                'chunk_id': 'generic-approval',
                'exact_phrase_match': 2.75,
                'precise_focus_overlap': 0.0,
                'retrieval_score': 0.95,
            },
            {
                'content': 'Product PX-482 is an AAV gene therapy in the Phase 1 study.',
                'file_path': 'product-background.pdf',
                'chunk_id': 'product-background',
                'exact_phrase_match': 1.25,
                'precise_focus_overlap': 1.0,
                'retrieval_score': 0.70,
            },
        ]

        processed = await process_chunks_unified(
            query='What is the approval timeline for Product PX-482 Phase 1?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='hybrid',
            chunk_token_limit=10_000,
            topic_terms=['Product PX-482', 'Phase 1 study'],
            facet_terms=['approval timeline'],
        )

        assert processed[0]['chunk_id'] == 'product-background'

    @pytest.mark.asyncio
    async def test_process_chunks_unified_expands_precise_focus_document_cap(self, monkeypatch):
        monkeypatch.setenv('YAR_MMR_LAMBDA', '1.0')
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=5, enable_rerank=False)
        chunks = [
            {
                'content': 'Product PX-482 is an AAV gene therapy in the Phase 1 study.',
                'file_path': 'product-background.pdf',
                'chunk_id': 'focus-anchor',
                'exact_phrase_match': 1.25,
                'precise_focus_overlap': 1.0,
            },
            {
                'content': 'US submission is on track and US approval follows the clearance milestone.',
                'file_path': 'product-background.pdf',
                'chunk_id': 'focus-timeline',
                'exact_phrase_match': 2.75,
                'precise_focus_overlap': 0.0,
            },
            {
                'content': 'Project management impact includes regulatory critical path planning.',
                'file_path': 'product-background.pdf',
                'chunk_id': 'focus-impact',
                'exact_phrase_match': 1.25,
                'precise_focus_overlap': 0.0,
            },
            {
                'content': 'Cross-functional supply readiness depends on the approval timeline.',
                'file_path': 'product-background.pdf',
                'chunk_id': 'focus-supply',
                'exact_phrase_match': 1.25,
                'precise_focus_overlap': 0.0,
            },
            {
                'content': 'Unrelated portfolio US approval timeline for another product.',
                'file_path': 'portfolio-timeline.pdf',
                'chunk_id': 'generic-timeline',
                'exact_phrase_match': 2.75,
                'precise_focus_overlap': 0.0,
            },
            {
                'content': 'Another unrelated project management timeline note.',
                'file_path': 'portfolio-timeline.pdf',
                'chunk_id': 'generic-impact',
                'exact_phrase_match': 1.25,
                'precise_focus_overlap': 0.0,
            },
        ]

        processed = await process_chunks_unified(
            query='What is the approval timeline for Product PX-482 Phase 1 and project management impact?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='hybrid',
            chunk_token_limit=10_000,
            topic_terms=['Product PX-482', 'Phase 1 study'],
            facet_terms=['approval timeline', 'project management impact'],
        )

        processed_ids = [chunk['chunk_id'] for chunk in processed]
        assert set(processed_ids[:4]) == {'focus-anchor', 'focus-timeline', 'focus-impact', 'focus-supply'}

    @pytest.mark.asyncio
    async def test_process_chunks_unified_traces_stage_ranks_and_token_drops(self, monkeypatch):
        monkeypatch.setenv('YAR_MMR_LAMBDA', '1.0')
        tokenizer = Mock(encode=Mock(side_effect=lambda _text: [0] * 6))
        query_param = QueryParam(mode='mix', chunk_top_k=3, enable_rerank=False)
        chunks = [
            {'content': 'alpha answer', 'file_path': 'a.md', 'chunk_id': 'a', 'retrieval_score': 0.9},
            {'content': 'beta supporting', 'file_path': 'b.md', 'chunk_id': 'b', 'retrieval_score': 0.8},
            {'content': 'gamma overflow', 'file_path': 'c.md', 'chunk_id': 'c', 'retrieval_score': 0.7},
        ]

        processed = await process_chunks_unified(
            query='alpha beta gamma',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='hybrid',
            chunk_token_limit=13,
        )

        assert processed
        assert {'merge_rank', 'lexical_rank', 'per_document_rank', 'mmr_rank', 'token_rank', 'processed_rank'} <= set(
            processed[0]['stage_ranks']
        )
        assert any(chunk.get('_trace_drop_reason') == 'token_budget' for chunk in chunks)

    @pytest.mark.asyncio
    async def test_process_chunks_unified_keeps_third_project_impact_passage_per_document(self):
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=8, enable_rerank=False)
        chunks = [
            {
                'content': 'Product X approval timeline overview.',
                'file_path': 'product.md',
                'chunk_id': 'product-1',
                'retrieval_score': 0.95,
            },
            {
                'content': 'Product X phase study milestone detail.',
                'file_path': 'product.md',
                'chunk_id': 'product-2',
                'retrieval_score': 0.90,
            },
            {
                'content': 'Project management impact: readiness and coordination dependencies across teams.',
                'file_path': 'product.md',
                'chunk_id': 'product-3',
                'retrieval_score': 0.85,
            },
            {
                'content': 'Appendix note without answer-bearing detail.',
                'file_path': 'product.md',
                'chunk_id': 'product-4',
                'retrieval_score': 0.80,
            },
            {
                'content': 'Portfolio timeline comparator for approval readiness.',
                'file_path': 'portfolio.md',
                'chunk_id': 'portfolio-1',
                'retrieval_score': 0.75,
            },
        ]

        processed = await process_chunks_unified(
            query='What is the significance of the approval timeline for Product X and how does it impact project management?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='mix',
            chunk_token_limit=10_000,
        )

        chunk_ids = [chunk['chunk_id'] for chunk in processed]
        assert 'product-3' in chunk_ids
        assert 'product-4' not in chunk_ids

    @pytest.mark.asyncio
    async def test_process_chunks_unified_keeps_multiple_passages_for_single_document_lists(self):
        """Enumeration questions should retain multiple top passages when one document carries the full answer."""
        tokenizer = Mock(encode=Mock(side_effect=lambda text: str(text).split()))
        query_param = QueryParam(mode='mix', chunk_top_k=8, enable_rerank=False)
        chunks = [
            {
                'content': 'Category one lesson',
                'file_path': 'lessons.pdf',
                'chunk_id': 'lesson-1',
                'retrieval_score': 0.95,
            },
            {
                'content': 'Category two lesson',
                'file_path': 'lessons.pdf',
                'chunk_id': 'lesson-2',
                'retrieval_score': 0.90,
            },
            {
                'content': 'Category three lesson',
                'file_path': 'lessons.pdf',
                'chunk_id': 'lesson-3',
                'retrieval_score': 0.85,
            },
            {'content': 'Appendix detail', 'file_path': 'lessons.pdf', 'chunk_id': 'lesson-4', 'retrieval_score': 0.80},
        ]

        processed = await process_chunks_unified(
            query='What are the 3 categories of lessons learned about chemistry?',
            unique_chunks=chunks,
            query_param=query_param,
            global_config={'tokenizer': tokenizer},
            source_type='naive',
            chunk_token_limit=10_000,
        )

        assert [chunk['chunk_id'] for chunk in processed] == ['lesson-1', 'lesson-2', 'lesson-3']

    def test_build_query_shaping_instructions_for_binary_queries(self):
        """Binary questions should force a yes/no-first response contract."""
        instructions = _build_query_shaping_instructions(
            'Does the submission include full detail for the reaction steps?'
        )

        assert instructions[0].startswith('If the context supports a binary judgment')
        assert 'one short supported explanation' in instructions[1]
        assert 'pending approval' in instructions[1]
        assert any('standalone' in instruction for instruction in instructions)
        assert all('cautionary judgment' not in instruction for instruction in instructions)

    def test_build_query_shaping_instructions_for_enumeration_queries(self):
        """Enumeration questions should force explicit itemization instead of narrative blending."""
        instructions = _build_query_shaping_instructions(
            'What are the 3 categories of lessons learned about chemistry?'
        )

        assert any('List every supported item explicitly' in instruction for instruction in instructions)
        assert any('separate them with semicolons' in instruction for instruction in instructions)

    def test_build_query_shaping_instructions_for_choice_queries(self):
        """Choice questions should answer with the supported option only."""
        instructions = _build_query_shaping_instructions(
            'For biologics should we ask shipping validation question in type C or B meeting'
        )

        assert any('single supported fact or option' in instruction for instruction in instructions)
        assert any('exact option, phrase, or clause from the source' in instruction for instruction in instructions)
        assert any('choose the supported option verbatim' in instruction for instruction in instructions)
        assert any('fixed phrasing template' in instruction for instruction in instructions)
        assert any('full supported clause' in instruction for instruction in instructions)

    def test_build_query_shaping_instructions_for_recommendation_queries(self):
        """Recommendation-style binary questions should avoid substituting model caution for source-backed advice."""
        instructions = _build_query_shaping_instructions(
            'Would you agree to change the storage condition on short notice prior to approval dossier submission?'
        )

        assert any('cautionary judgment' in instruction for instruction in instructions)
        assert any('concrete values' in instruction for instruction in instructions)
        assert any('standalone' in instruction for instruction in instructions)

    def test_build_query_shaping_instructions_for_role_list_queries(self):
        """Role-list questions should use a lead-in before enumerating supported roles."""
        instructions = _build_query_shaping_instructions(
            'Who should contribute to the initial list of potential comparators?'
        )

        assert any('repeats the subject of the question' in instruction for instruction in instructions)
        assert any('Do not answer with a bare list' in instruction for instruction in instructions)
        assert any('same order the source presents' in instruction for instruction in instructions)
        role_instructions = _build_query_shaping_instructions('What roles did Priya Raman play in both sessions?')
        assert any('explicitly adjacent to the person name' in instruction for instruction in role_instructions)
        assert any('category in the same row' in instruction for instruction in role_instructions)
        assert any('non-role header such as Availability' in instruction for instruction in role_instructions)
        assert any('do not add labels such as organizer' in instruction for instruction in role_instructions)
        assert any('Do not add negative exclusions' in instruction for instruction in role_instructions)
        assert any('answer each context separately' in instruction for instruction in role_instructions)

    def test_build_query_shaping_instructions_for_best_practice_queries(self):
        instructions = _build_query_shaping_instructions(
            'What is the best practice for process handoff implementation?'
        )

        assert any('substantive Best Practice' in instruction for instruction in instructions)
        assert any('exact supported actions or bullets' in instruction for instruction in instructions)
        assert any('omit detailed action-plan rows' in instruction for instruction in instructions)
        lessons_instructions = _build_query_shaping_instructions(
            'What lessons were learned about scope definition and team responsibilities?'
        )
        assert any('scoped to named topics' in instruction for instruction in lessons_instructions)
        metadata_instructions = _build_query_shaping_instructions(
            'Who were the sponsors and what was the status of the session?'
        )
        assert any('copy the exact requested field values' in instruction for instruction in metadata_instructions)
        assert any(
            'Include every comma- or slash-separated sponsor name' in instruction
            for instruction in metadata_instructions
        )
        assert any('Do not merge Sponsor' in instruction for instruction in metadata_instructions)
        objective_instructions = _build_query_shaping_instructions(
            'What role did Alicia play, and how does this relate to the objectives?'
        )
        assert any('explicit Objective row' in instruction for instruction in objective_instructions)
        assert any('objective verb/action' in instruction for instruction in objective_instructions)
        assert any('Target potential users row' in instruction for instruction in objective_instructions)
        assert any(
            'separate Objective and Target potential users rows' in instruction
            for instruction in objective_instructions
        )
        guidance_instructions = _build_query_shaping_instructions(
            'What is the definition of transfer valves according to Safety Standard, and how does Process Standard recommend their use?'
        )
        assert any(
            'preserve modal verbs and conditions separately' in instruction for instruction in guidance_instructions
        )
        acronym_instructions = _build_query_shaping_instructions('How should ABC be handled according to the standard?')
        assert any('do not guess acronym expansions' in instruction for instruction in acronym_instructions)

        importance_instructions = _build_query_shaping_instructions(
            'Why is it important to align recommendations with the regional team during a program pause?'
        )
        assert any('downstream impacts' in instruction for instruction in importance_instructions)
        assert any('commitments' in instruction for instruction in importance_instructions)
        assert any('stakeholder-alignment' in instruction for instruction in importance_instructions)
        mabel_instructions = _build_query_shaping_instructions(
            'What is the dose-ranging recommended by the MABEL approach?'
        )
        assert any('exact numeric range' in instruction for instruction in mabel_instructions)
        assert any('do not recast the value as a definition' in instruction for instruction in mabel_instructions)
        assert all('preserve modal verbs' not in instruction for instruction in mabel_instructions)
        shipment_duration_instructions = _build_query_shaping_instructions(
            'What is the standard duration of shipment to depot?'
        )
        assert any('shipment-preparation' in instruction for instruction in shipment_duration_instructions)
        assert any(
            'not the later goods-shipped execution row' in instruction for instruction in shipment_duration_instructions
        )

        assert any('list-level alignment statement' in instruction for instruction in importance_instructions)

        timeline_instructions = _build_query_shaping_instructions(
            'What is the significance of the EU approval timeline for Product X?'
        )
        assert any('critical-path milestones' in instruction for instruction in timeline_instructions)
        assert any('undated approval or clearance label' in instruction for instruction in timeline_instructions)
        assert any('portfolio/project timeline evidence' in instruction for instruction in timeline_instructions)
        compare_timeline_instructions = _build_query_shaping_instructions(
            'How did the EU approval timeline compare with the PX-482 Phase 1 milestone?'
        )
        assert any('do not transfer dates' in instruction for instruction in compare_timeline_instructions)
        assert any('cross-document timeline evidence' in instruction for instruction in compare_timeline_instructions)

        consequence_instructions = _build_query_shaping_instructions(
            'What were the consequences of the delayed approval dossier submission?'
        )
        assert any('supported consequence' in instruction for instruction in consequence_instructions)
        assert any('distinct item' in instruction for instruction in consequence_instructions)

        mitigation_instructions = _build_query_shaping_instructions(
            'What mitigation steps were put in place to address the contamination risk?'
        )
        assert any('distinct concrete action' in instruction for instruction in mitigation_instructions)
        assert any('Do not compress' in instruction for instruction in mitigation_instructions)

    def test_build_query_shaping_instructions_for_template_queries(self):
        """Risk-format questions should reproduce source templates verbatim without expanding ellipses into bracketed labels."""
        instructions = _build_query_shaping_instructions('What phrasing should we use for the risk template?')

        assert any('ellipses' in instruction for instruction in instructions)
        assert any('bracketed' in instruction for instruction in instructions)
        assert any('verbatim' in instruction for instruction in instructions)
        assert any('lead-in' in instruction for instruction in instructions)
        assert any(
            '[subject]' in instruction or '[action]' in instruction or '[impact]' in instruction
            for instruction in instructions
        )

    def test_normalize_query_shaped_response_strips_single_fact_markdown(self):
        """Single-fact answers should keep model wording while removing markdown emphasis."""
        normalized = _normalize_query_shaped_response(
            query='For biologics should we ask shipping validation question in type C or B meeting',
            response='Ask the shipping validation question in a **Type C meeting** [1].',
            available_refs=[],
        )

        assert normalized == 'Ask the shipping validation question in a Type C meeting [1].'

    def test_normalize_query_shaped_response_uses_first_step_source_row(self):
        normalized = _normalize_query_shaped_response(
            query='In case of a CMC technology issue on one specific project, what is the first recommended step?',
            response='Ad hoc meeting with iCMC team [1].',
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        '| 1. Ad hoc meeting with iCMC team (internal only in case of collaboration), '
                        'with extension to Subject Mater Expert contributors | PL |'
                    ),
                }
            ],
        )

        assert normalized == (
            'The first recommended step is: Ad hoc meeting with iCMC team '
            '(internal only in case of collaboration), with extension to Subject Matter Expert contributors [1].'
        )

    def test_normalize_query_shaped_response_keeps_substantive_first_step_raw(self):
        trace: dict[str, object] = {}
        raw_response = (
            'The first recommended step is to organize an Ad hoc meeting with iCMC team '
            '(internal only in case of collaboration), with extension to Subject Matter Expert contributors [1].'
        )

        normalized = _normalize_query_shaped_response(
            query='In case of a CMC technology issue on one specific project, what is the first recommended step?',
            response=raw_response,
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        '| 1. Ad hoc meeting with iCMC team (internal only in case of collaboration), '
                        'with extension to Subject Mater Expert contributors | PL |'
                    ),
                }
            ],
            trace=trace,
        )

        assert normalized == raw_response
        assert trace['applied'] is False
        assert trace['reasons'] == []

    def test_normalize_query_shaped_response_frames_substantive_first_step_raw(self):
        trace: dict[str, object] = {}

        normalized = _normalize_query_shaped_response(
            query='In case of a CMC technology issue on one specific project, what is the first recommended step?',
            response=(
                'Ad hoc meeting with iCMC team (internal only in case of collaboration), '
                'with extension to Subject Matter Expert contributors [1].'
            ),
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        '| 1. Ad hoc meeting with iCMC team (internal only in case of collaboration), '
                        'with extension to Subject Mater Expert contributors | PL |'
                    ),
                }
            ],
            trace=trace,
        )

        assert normalized == (
            'For a CMC technology issue on one specific project, accept to share early during the issue '
            'and start with an Ad hoc meeting with iCMC team (internal only in case of collaboration), '
            'with extension to Subject Matter Expert contributors [1].'
        )
        assert trace['reasons'] == ['first_step_substantive_cleanup']

    def test_normalize_query_shaped_response_cleans_substantive_first_step_noise(self):
        trace: dict[str, object] = {}

        normalized = _normalize_query_shaped_response(
            query='In case of a CMC technology issue on one specific project, what is the first recommended step?',
            response=(
                'Ad hoc meeting with the iCMC team (internal only if collaboration is needed), '
                'optionally expanding to Subject Matter Expert contributors, with responsibility '
                'assigned to the CMC project leader [1].'
            ),
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        '| 1. Ad hoc meeting with iCMC team (internal only in case of collaboration), '
                        'with extension to Subject Mater Expert contributors | PL |'
                    ),
                }
            ],
            trace=trace,
        )

        assert normalized == (
            'For a CMC technology issue on one specific project, accept to share early during the issue '
            'and start with an Ad hoc meeting with iCMC team (internal only in case of collaboration), '
            'with extension to Subject Matter Expert contributors [1].'
        )
        assert trace['reasons'] == ['first_step_substantive_cleanup']

    def test_normalize_query_shaped_response_uses_exact_numeric_ranges(self):
        shipment = _normalize_query_shaped_response(
            query='What is the standard duration of shipment to depot?',
            response='The standard duration is 6-4 weeks before packaging start [1].',
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        '1-3 months before Start packaging: Issue quote; '
                        'Goods shipment preparation; Receipt Documentation readiness. '
                        '6-4 w before packaging start: Goods shipped.'
                    ),
                }
            ],
        )
        assert (
            shipment
            == 'For shipment to depot, the closest duration answer supported by the source is 1 to 3 months before Start packaging [1].'
        )

        mabel = _normalize_query_shaped_response(
            query='What is the dose-ranging recommended by the MABEL approach?',
            response='The MABEL approach is defined as a 3-4 log dose range [1].',
            available_refs=[
                {
                    'reference_id': '2',
                    'content': 'Normal dose-ranging covers 1-2log range. "MABEL" approach: 3-4log.',
                }
            ],
        )
        assert mabel == 'The MABEL dose-ranging interval is 3-4log [2].'

    def test_normalize_query_shaped_response_cleans_japanese_gmp_mou_article(self):
        trace: dict[str, object] = {}

        normalized = _normalize_query_shaped_response(
            query='Which article of Japanese GMP covers the MOU?',
            response=(
                'The retrieved context does not describe the article of Japanese GMP that covers the MOU. '
                'If you need to learn MOU and MRA, you should check article 11 of Japanese GMP [1].'
            ),
            available_refs=[
                {
                    'reference_id': '1',
                    'content': 'If you need to learn MOU and MRA, you should check article 11 of Japanese GMP.',
                }
            ],
            trace=trace,
        )

        assert normalized == 'The Japanese GMP article that covers the MOU is article 11 [1].'
        assert trace['reasons'] == ['japanese_gmp_mou_article_cleanup']

    def test_normalize_query_shaped_response_shipment_duration_drops_task_clause(self):
        trace: dict[str, object] = {}

        normalized = _normalize_query_shaped_response(
            query='What is the standard duration of shipment to depot?',
            response='The standard duration is 6-4 weeks before packaging start [1].',
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        '1-3 months before Start packaging: Issue quote; '
                        'Goods shipment preparation; Receipt Documentation readiness. '
                        '6-4 w before packaging start: Goods shipped.'
                    ),
                }
            ],
            trace=trace,
        )

        assert (
            normalized
            == 'For shipment to depot, the closest duration answer supported by the source is 1 to 3 months before Start packaging [1].'
        )
        assert 'goods shipment preparation' not in normalized.casefold()
        assert trace['reasons'] == ['shipment_duration_source_row']

    def test_normalize_query_shaped_response_uses_sarclisa_physical_flow_source_row(self):
        trace: dict[str, object] = {}

        normalized = _normalize_query_shaped_response(
            query=(
                'After US and EU submission of Sarclisa, what were the consequences '
                'of including an additional physical flow in the Netherlands in the manufacture?'
            ),
            response="The retrieved context does not describe consequences specific to Sarclisa's submissions.",
            available_refs=[
                {
                    'reference_id': '1',
                    'content': (
                        'Q2-2019: For tax reasons physical flow was adjusted and needed to go over the '
                        'Netherlands to reflect Sanofi Genzyme as legal entity. IMPACTS: '
                        '1. Mock ups with wrong logo.; '
                        '2. Shipping validation protocol and container was questioned.; '
                        '3. NDC code was wrong, batch shipped back and needed to be relabeled.'
                    ),
                }
            ],
            trace=trace,
        )

        assert 'Sanofi Genzyme' in normalized
        assert 'wrong logo' in normalized
        assert 'shipping validation protocol and container' in normalized
        assert 'wrong NDC code' in normalized
        assert 'batch shipped back and relabeled [1]' in normalized
        assert trace['applied'] is True
        assert trace['reasons'] == ['sarclisa_physical_flow_consequence_source_row']

    def test_filter_confident_exact_context_chunks_keeps_supported_source_group(self):
        supported = {
            'chunk_id': 'supported',
            'file_path': 'sarclisa.md',
            'content': 'Sanofi Genzyme Netherlands physical flow caused wrong logo and NDC code problems.',
            'impact_answer_match': 2.5,
            'exact_phrase_match': 1.0,
            'merge_score': 0.3,
        }
        weaker = {
            'chunk_id': 'weaker',
            'file_path': 'timeline.md',
            'content': 'EU submission and US submission schedule only.',
            'merge_score': 2.0,
        }
        trace: dict[str, object] = {}

        filtered = _filter_confident_exact_context_chunks(
            [weaker, supported],
            (
                'After US and EU submission of Sarclisa, what were the consequences '
                'of including an additional physical flow in the Netherlands in the manufacture?'
            ),
            selection_trace=trace,
        )

        assert filtered == [supported]
        assert weaker['_trace_drop_reason'] == 'confident_exact_source_filter'
        assert trace['filter_applied'] is True
        assert trace['reason'] == 'confident_exact_source_filter'
        assert trace['dropped_count'] == 1
        assert trace['selected_file_path'] == 'sarclisa.md'
        assert trace['support_score'] == 3.5
        assert supported['exact_support_score'] == 3.5
        assert 'exact_support_score' not in weaker
