"""
Tests for yar/utils_graph.py - Graph utility functions.

This module tests:
- _merge_attributes helper function
- DeletionResult usage patterns
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from yar.base import DeletionResult
from yar.constants import GRAPH_FIELD_SEP
from yar.utils_graph import _merge_attributes, acreate_relation


class _AsyncNoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_acreate_relation_uses_canonical_projection_shape():
    class FakeGraphStorage:
        def __init__(self):
            self.nodes = {'Z Source', 'A Target'}
            self.edges = {}

        async def has_node(self, node_id):
            return node_id in self.nodes

        async def has_edge(self, source_node_id, target_node_id):
            return frozenset((source_node_id, target_node_id)) in self.edges

        async def upsert_edge(self, source_node_id, target_node_id, edge_data):
            self.edges[frozenset((source_node_id, target_node_id))] = edge_data

        async def get_edge(self, source_node_id, target_node_id):
            return self.edges.get(frozenset((source_node_id, target_node_id)))

        async def index_done_callback(self):
            return None

    class FakeVectorStorage:
        def __init__(self):
            self.global_config = {}
            self.records = {}

        async def upsert(self, data):
            self.records.update(data)

        async def get_by_id(self, record_id):
            return self.records.get(record_id)

        async def index_done_callback(self):
            return None

    class FakeRelationChunksStorage:
        def __init__(self):
            self.records = {}

        async def upsert(self, data):
            self.records.update(data)

        async def index_done_callback(self):
            return None

    graph = FakeGraphStorage()
    relationships_vdb = FakeVectorStorage()
    relation_chunks = FakeRelationChunksStorage()

    with patch('yar.utils_graph.get_storage_keyed_lock', return_value=_AsyncNoopLock()):
        result = await acreate_relation(
            graph,
            FakeVectorStorage(),
            relationships_vdb,
            'Z Source',
            'A Target',
            {
                'description': 'Z Source tracks A Target owners.',
                'keywords': 'tracks, organizes',
                'source_id': 'chunk-manual',
                'file_path': 'manual.md',
            },
            relation_chunks,
        )

    edge_data = result['graph_data']
    assert set(edge_data) == {'weight', 'description', 'keywords', 'source_id', 'file_path', 'created_at', 'truncate'}
    assert edge_data['keywords'] == 'tracks, organizes'
    assert 'search_hints: tracks organizes source target owners' in result['vector_data']['content']
    assert set(result['vector_data']) == {
        'src_id',
        'tgt_id',
        'source_id',
        'content',
        'keywords',
        'description',
        'weight',
        'file_path',
    }
    assert relation_chunks.records['A Target<SEP>Z Source']['chunk_ids'] == ['chunk-manual']


class TestMergeAttributes:
    """Tests for _merge_attributes helper function."""

    def test_empty_data_list(self):
        """Test with empty data list."""
        result = _merge_attributes([], {})
        assert result == {}

    def test_single_entity(self):
        """Test merging single entity (no merge needed)."""
        data = [{'description': 'A person', 'entity_type': 'PERSON'}]
        strategy = {'description': 'concatenate', 'entity_type': 'keep_first'}

        result = _merge_attributes(data, strategy)

        assert result['description'] == 'A person'
        assert result['entity_type'] == 'PERSON'

    def test_concatenate_strategy(self):
        """Test concatenate merge strategy."""
        data = [
            {'description': 'First description'},
            {'description': 'Second description'},
            {'description': 'Third description'},
        ]
        strategy = {'description': 'concatenate'}

        result = _merge_attributes(data, strategy)

        assert GRAPH_FIELD_SEP in result['description']
        assert 'First description' in result['description']
        assert 'Second description' in result['description']
        assert 'Third description' in result['description']

    def test_keep_first_strategy(self):
        """Test keep_first merge strategy."""
        data = [
            {'entity_type': 'PERSON'},
            {'entity_type': 'ORGANIZATION'},
            {'entity_type': 'LOCATION'},
        ]
        strategy = {'entity_type': 'keep_first'}

        result = _merge_attributes(data, strategy)

        assert result['entity_type'] == 'PERSON'

    def test_keep_last_strategy(self):
        """Test keep_last merge strategy."""
        data = [
            {'entity_type': 'PERSON'},
            {'entity_type': 'ORGANIZATION'},
            {'entity_type': 'LOCATION'},
        ]
        strategy = {'entity_type': 'keep_last'}

        result = _merge_attributes(data, strategy)

        assert result['entity_type'] == 'LOCATION'

    def test_join_unique_strategy(self):
        """Test join_unique merge strategy with GRAPH_FIELD_SEP separator."""
        # Values already separated by GRAPH_FIELD_SEP
        data = [
            {'source_id': f'chunk1{GRAPH_FIELD_SEP}chunk2'},
            {'source_id': f'chunk2{GRAPH_FIELD_SEP}chunk3'},
            {'source_id': 'chunk4'},
        ]
        strategy = {'source_id': 'join_unique'}

        result = _merge_attributes(data, strategy)

        # Should contain unique items
        items = result['source_id'].split(GRAPH_FIELD_SEP)
        assert len(items) == len(set(items))  # All unique
        assert 'chunk1' in items
        assert 'chunk2' in items
        assert 'chunk3' in items
        assert 'chunk4' in items

    def test_join_unique_comma_strategy(self):
        """Test join_unique_comma merge strategy."""
        data = [
            {'keywords': 'python, coding'},
            {'keywords': 'coding, programming'},
            {'keywords': 'development'},
        ]
        strategy = {'keywords': 'join_unique_comma'}

        result = _merge_attributes(data, strategy)

        # Should contain unique items separated by comma
        items = [item.strip() for item in result['keywords'].split(',')]
        assert len(items) == len(set(items))  # All unique
        assert 'python' in items
        assert 'coding' in items
        assert 'programming' in items
        assert 'development' in items

    def test_max_strategy_numeric(self):
        """Test max merge strategy with numeric values."""
        data = [
            {'weight': 0.5},
            {'weight': 0.9},
            {'weight': 0.3},
        ]
        strategy = {'weight': 'max'}

        result = _merge_attributes(data, strategy)

        assert result['weight'] == 0.9

    def test_max_strategy_string_numbers(self):
        """Test max merge strategy with string numeric values."""
        data = [
            {'weight': '0.5'},
            {'weight': '0.9'},
            {'weight': '0.3'},
        ]
        strategy = {'weight': 'max'}

        result = _merge_attributes(data, strategy)

        assert result['weight'] == 0.9

    def test_max_strategy_mixed_types(self):
        """Test max strategy falls back gracefully for non-numeric."""
        data = [
            {'weight': 'high'},
            {'weight': 0.5},
        ]
        strategy = {'weight': 'max'}

        result = _merge_attributes(data, strategy)
        # Should get the max of valid numeric values
        assert result['weight'] == 0.5

    def test_default_strategy_is_keep_first(self):
        """Test that unknown strategies default to keep_first."""
        data = [
            {'custom_field': 'first'},
            {'custom_field': 'second'},
        ]
        strategy = {}  # No strategy defined for custom_field

        result = _merge_attributes(data, strategy)

        assert result['custom_field'] == 'first'

    def test_filter_none_only_false(self):
        """Test filter_none_only=False filters all falsy values."""
        data = [
            {'description': ''},  # Empty string - falsy
            {'description': 'Valid description'},
            {'description': None},  # None - falsy
        ]
        strategy = {'description': 'keep_first'}

        result = _merge_attributes(data, strategy, filter_none_only=False)

        assert result['description'] == 'Valid description'

    def test_filter_none_only_true(self):
        """Test filter_none_only=True only filters None values."""
        data = [
            {'description': ''},  # Empty string - kept
            {'description': 'Valid description'},
            {'description': None},  # None - filtered
        ]
        strategy = {'description': 'keep_first'}

        result = _merge_attributes(data, strategy, filter_none_only=True)

        # Empty string should be kept
        assert result['description'] == ''

    def test_multiple_fields_different_strategies(self):
        """Test merging with different strategies for different fields."""
        data = [
            {
                'description': 'First desc',
                'entity_type': 'PERSON',
                'source_id': 'chunk1',
                'weight': 0.5,
            },
            {
                'description': 'Second desc',
                'entity_type': 'ORGANIZATION',
                'source_id': 'chunk2',
                'weight': 0.8,
            },
        ]
        strategy = {
            'description': 'concatenate',
            'entity_type': 'keep_first',
            'source_id': 'join_unique',
            'weight': 'max',
        }

        result = _merge_attributes(data, strategy)

        assert 'First desc' in result['description']
        assert 'Second desc' in result['description']
        assert result['entity_type'] == 'PERSON'
        assert 'chunk1' in result['source_id']
        assert 'chunk2' in result['source_id']
        assert result['weight'] == 0.8

    def test_missing_keys_in_some_data(self):
        """Test merging when some entities lack certain keys."""
        data = [
            {'description': 'Has both', 'entity_type': 'PERSON'},
            {'description': 'Only desc'},  # No entity_type
            {'entity_type': 'LOCATION'},  # No description
        ]
        strategy = {
            'description': 'concatenate',
            'entity_type': 'keep_first',
        }

        result = _merge_attributes(data, strategy)

        assert 'Has both' in result['description']
        assert 'Only desc' in result['description']
        assert result['entity_type'] == 'PERSON'

    def test_all_none_values(self):
        """Test merging when all values for a key are None."""
        data = [
            {'description': None},
            {'description': None},
        ]
        strategy = {'description': 'concatenate'}

        result = _merge_attributes(data, strategy)

        # Key should not appear in result since all values are filtered
        assert 'description' not in result

    def test_entity_merge_typical_scenario(self):
        """Test typical entity merge scenario."""
        # Simulating merging duplicate entities
        data = [
            {
                'entity_id': 'John Smith',
                'description': 'CEO of Company A',
                'entity_type': 'PERSON',
                'source_id': f'doc1-chunk1{GRAPH_FIELD_SEP}doc1-chunk2',
            },
            {
                'entity_id': 'J. Smith',
                'description': 'Business leader and entrepreneur',
                'entity_type': 'PERSON',
                'source_id': 'doc2-chunk1',
            },
        ]
        strategy = {
            'description': 'concatenate',
            'entity_type': 'keep_first',
            'source_id': 'join_unique',
        }

        result = _merge_attributes(data, strategy)

        assert 'CEO' in result['description']
        assert 'entrepreneur' in result['description']
        assert result['entity_type'] == 'PERSON'

    def test_relation_merge_typical_scenario(self):
        """Test typical relation merge scenario."""
        # Simulating merging duplicate relationships
        data = [
            {
                'description': 'works at',
                'keywords': 'employment, job',
                'weight': 0.7,
                'source_id': 'chunk1',
            },
            {
                'description': 'is employed by',
                'keywords': 'work, career',
                'weight': 0.9,
                'source_id': 'chunk2',
            },
        ]
        strategy = {
            'description': 'concatenate',
            'keywords': 'join_unique_comma',
            'source_id': 'join_unique',
            'weight': 'max',
        }

        result = _merge_attributes(data, strategy, filter_none_only=True)

        assert 'works at' in result['description']
        assert 'is employed by' in result['description']
        assert result['weight'] == 0.9


class TestDeletionResultUsage:
    """Tests for DeletionResult dataclass usage patterns."""

    def test_success_result(self):
        """Test creating a success deletion result."""
        result = DeletionResult(
            status='success',
            doc_id='entity_123',
            message='Entity deleted successfully',
            status_code=200,
        )

        assert result.status == 'success'
        assert result.doc_id == 'entity_123'
        assert result.status_code == 200

    def test_not_found_result(self):
        """Test creating a not_found deletion result."""
        result = DeletionResult(
            status='not_found',
            doc_id='missing_entity',
            message='Entity not found',
            status_code=404,
        )

        assert result.status == 'not_found'
        assert result.status_code == 404

    def test_fail_result(self):
        """Test creating a failure deletion result."""
        result = DeletionResult(
            status='fail',
            doc_id='entity_456',
            message='Database error during deletion',
            status_code=500,
        )

        assert result.status == 'fail'
        assert result.status_code == 500

    def test_result_equality(self):
        """Test deletion result equality."""
        result1 = DeletionResult(
            status='success',
            doc_id='entity_1',
            message='Deleted',
            status_code=200,
        )
        result2 = DeletionResult(
            status='success',
            doc_id='entity_1',
            message='Deleted',
            status_code=200,
        )

        assert result1 == result2


class TestMergeAttributesEdgeCases:
    """Edge case tests for _merge_attributes."""

    def test_empty_strings_in_concatenate(self):
        """Test concatenate with empty strings."""
        data = [
            {'description': 'First'},
            {'description': ''},
            {'description': 'Third'},
        ]
        strategy = {'description': 'concatenate'}

        # With filter_none_only=False (default), empty strings are filtered
        result = _merge_attributes(data, strategy, filter_none_only=False)
        parts = result['description'].split(GRAPH_FIELD_SEP)
        assert '' not in parts

    def test_whitespace_handling_in_join_unique_comma(self):
        """Test join_unique_comma handles whitespace properly."""
        data = [
            {'keywords': '  python  ,  coding  '},
            {'keywords': 'coding,   programming   '},
        ]
        strategy = {'keywords': 'join_unique_comma'}

        result = _merge_attributes(data, strategy)

        items = result['keywords'].split(',')
        for item in items:
            # Items should be trimmed
            assert item == item.strip()

    def test_single_item_join_unique(self):
        """Test join_unique with single item."""
        data = [{'source_id': 'only_one'}]
        strategy = {'source_id': 'join_unique'}

        result = _merge_attributes(data, strategy)

        assert result['source_id'] == 'only_one'

    def test_numeric_zero_with_max(self):
        """Test max strategy handles zero correctly."""
        data = [
            {'weight': 0},
            {'weight': -5},
            {'weight': -1},
        ]
        strategy = {'weight': 'max'}

        result = _merge_attributes(data, strategy, filter_none_only=True)

        assert result['weight'] == 0

    def test_large_data_list(self):
        """Test merging many items."""
        data = [{'value': f'item_{i}'} for i in range(100)]
        strategy = {'value': 'concatenate'}

        result = _merge_attributes(data, strategy)

        parts = result['value'].split(GRAPH_FIELD_SEP)
        assert len(parts) == 100

    def test_special_characters_in_values(self):
        """Test handling special characters."""
        data = [
            {'description': 'Contains "quotes" and \'apostrophes\''},
            {'description': 'Has\nnewlines\tand\ttabs'},
        ]
        strategy = {'description': 'concatenate'}

        result = _merge_attributes(data, strategy)

        assert 'quotes' in result['description']
        assert 'newlines' in result['description']

    def test_unicode_characters(self):
        """Test handling unicode characters."""
        data = [
            {'description': '日本語テキスト'},
            {'description': 'English text'},
            {'description': 'Émojis 🎉'},
        ]
        strategy = {'description': 'concatenate'}

        result = _merge_attributes(data, strategy)

        assert '日本語' in result['description']
        assert 'Émojis' in result['description']
