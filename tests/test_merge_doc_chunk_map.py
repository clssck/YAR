"""Tests for Phase 3 document attribution in merge_nodes_and_edges."""

from __future__ import annotations

from collections import defaultdict

import pytest

from yar.operate import GRAPH_FIELD_SEP

EntityName = str
RelationPair = tuple[str, str]


def compute_doc_entity_sets(
    all_nodes: dict[EntityName, list[dict[str, str]]],
    all_entity_names: set[EntityName],
    doc_chunk_map: dict[str, set[str]] | None,
    *,
    doc_id: str | None = None,
) -> dict[str, set[EntityName]]:
    """Mirror Phase 3 entity attribution for multi-doc and single-doc indexing."""
    if doc_chunk_map:
        entity_source_chunks: dict[EntityName, set[str]] = defaultdict(set)
        for entity_name, entity_data_list in all_nodes.items():
            for entry in entity_data_list:
                source_id = entry.get('source_id', '')
                if source_id:
                    entity_source_chunks[entity_name].update(
                        chunk_id for chunk_id in source_id.split(GRAPH_FIELD_SEP) if chunk_id
                    )

        result: dict[str, set[EntityName]] = {}
        for mapped_doc_id, chunk_keys in doc_chunk_map.items():
            result[mapped_doc_id] = {
                name
                for name in all_entity_names
                if entity_source_chunks.get(name, set()) & chunk_keys
            }
        return result

    if doc_id:
        return {doc_id: set(all_entity_names)}

    return {}


def compute_doc_relation_sets(
    all_edges: dict[RelationPair, list[dict[str, str]]],
    all_relation_pairs: set[RelationPair],
    doc_chunk_map: dict[str, set[str]] | None,
    *,
    doc_id: str | None = None,
) -> dict[str, set[RelationPair]]:
    """Mirror Phase 3 relation attribution for multi-doc and single-doc indexing."""
    if doc_chunk_map:
        relation_source_chunks: dict[RelationPair, set[str]] = defaultdict(set)
        for edge_key, edge_data_list in all_edges.items():
            for entry in edge_data_list:
                source_id = entry.get('source_id', '')
                if source_id:
                    relation_source_chunks[edge_key].update(
                        chunk_id for chunk_id in source_id.split(GRAPH_FIELD_SEP) if chunk_id
                    )

        result: dict[str, set[RelationPair]] = {}
        for mapped_doc_id, chunk_keys in doc_chunk_map.items():
            result[mapped_doc_id] = {
                edge_key
                for edge_key in all_relation_pairs
                if relation_source_chunks.get(edge_key, set()) & chunk_keys
            }
        return result

    if doc_id:
        return {doc_id: set(all_relation_pairs)}

    return {}


@pytest.mark.offline
class TestDocChunkMapAttribution:
    """Tests for document attribution from chunk memberships."""

    def test_entities_attributed_to_correct_docs(self) -> None:
        all_nodes = {
            'EntityA': [{'source_id': 'chunk-1', 'entity_type': 'TYPE'}],
            'EntityB': [{'source_id': 'chunk-2', 'entity_type': 'TYPE'}],
            'EntityC': [
                {
                    'source_id': f'chunk-1{GRAPH_FIELD_SEP}chunk-2',
                    'entity_type': 'TYPE',
                }
            ],
        }
        all_entity_names = {'EntityA', 'EntityB', 'EntityC'}
        doc_chunk_map = {'doc1': {'chunk-1'}, 'doc2': {'chunk-2'}}

        result = compute_doc_entity_sets(all_nodes, all_entity_names, doc_chunk_map)

        assert result['doc1'] == {'EntityA', 'EntityC'}
        assert result['doc2'] == {'EntityB', 'EntityC'}
        assert 'EntityB' not in result['doc1']
        assert 'EntityA' not in result['doc2']

    def test_relations_attributed_to_correct_docs(self) -> None:
        relation_ab = tuple(sorted(('EntityA', 'EntityB')))
        relation_bc = tuple(sorted(('EntityB', 'EntityC')))
        relation_ac = tuple(sorted(('EntityA', 'EntityC')))
        all_edges = {
            relation_ab: [{'source_id': 'chunk-1', 'weight': '1'}],
            relation_bc: [{'source_id': 'chunk-2', 'weight': '1'}],
            relation_ac: [{'source_id': f'chunk-1{GRAPH_FIELD_SEP}chunk-2', 'weight': '1'}],
        }
        all_relation_pairs = {relation_ab, relation_bc, relation_ac}
        doc_chunk_map = {'doc1': {'chunk-1'}, 'doc2': {'chunk-2'}}

        result = compute_doc_relation_sets(all_edges, all_relation_pairs, doc_chunk_map)

        assert result['doc1'] == {relation_ab, relation_ac}
        assert result['doc2'] == {relation_bc, relation_ac}
        assert relation_bc not in result['doc1']
        assert relation_ab not in result['doc2']

    def test_single_doc_id_is_backward_compatible(self) -> None:
        relation_ab = tuple(sorted(('EntityA', 'EntityB')))
        relation_bc = tuple(sorted(('EntityB', 'EntityC')))
        all_nodes = {
            'EntityA': [{'source_id': 'chunk-1', 'entity_type': 'TYPE'}],
            'EntityB': [{'source_id': 'chunk-2', 'entity_type': 'TYPE'}],
            'EntityC': [{'source_id': '', 'entity_type': 'TYPE'}],
        }
        all_edges = {
            relation_ab: [{'source_id': 'chunk-1', 'weight': '1'}],
            relation_bc: [{'source_id': '', 'weight': '1'}],
        }
        all_entity_names = {'EntityA', 'EntityB', 'EntityC'}
        all_relation_pairs = {relation_ab, relation_bc}
        doc_id = 'legacy-doc'

        entity_result = compute_doc_entity_sets(
            all_nodes,
            all_entity_names,
            None,
            doc_id=doc_id,
        )
        relation_result = compute_doc_relation_sets(
            all_edges,
            all_relation_pairs,
            None,
            doc_id=doc_id,
        )

        assert entity_result == {doc_id: all_entity_names}
        assert relation_result == {doc_id: all_relation_pairs}
