"""
Tests for yar/types.py - Type definitions and Pydantic models.

This module tests:
- GPTKeywordExtractionFormat model with required and optional fields
- KnowledgeGraphNode model with id, labels, and properties
- KnowledgeGraphEdge model with source, target, and optional type
- KnowledgeGraph model with nodes, edges, and is_truncated flag
- Type alias LLMFunc for async callable signatures
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yar.types import (
    GPTKeywordExtractionFormat,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    LLMFunc,
)


class TestGPTKeywordExtractionFormat:
    """Tests for GPTKeywordExtractionFormat Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid GPTKeywordExtractionFormat instance."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=['AI', 'Machine Learning'],
            low_level_keywords=['neural networks', 'deep learning'],
        )
        assert fmt.high_level_keywords == ['AI', 'Machine Learning']
        assert fmt.low_level_keywords == ['neural networks', 'deep learning']

    def test_empty_keyword_lists(self):
        """Test creating instance with empty keyword lists."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=[],
            low_level_keywords=[],
        )
        assert fmt.high_level_keywords == []
        assert fmt.low_level_keywords == []

    def test_single_keyword_each(self):
        """Test with single keyword in each list."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=['Technology'],
            low_level_keywords=['GPU'],
        )
        assert len(fmt.high_level_keywords) == 1
        assert len(fmt.low_level_keywords) == 1
        assert fmt.high_level_keywords[0] == 'Technology'
        assert fmt.low_level_keywords[0] == 'GPU'

    def test_many_keywords(self):
        """Test with many keywords in lists."""
        high = [f'high_{i}' for i in range(100)]
        low = [f'low_{i}' for i in range(100)]
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=high,
            low_level_keywords=low,
        )
        assert len(fmt.high_level_keywords) == 100
        assert len(fmt.low_level_keywords) == 100

    def test_missing_high_level_keywords_raises(self):
        """Test missing high_level_keywords raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GPTKeywordExtractionFormat(low_level_keywords=['test'])
        assert 'high_level_keywords' in str(exc_info.value)

    def test_missing_low_level_keywords_raises(self):
        """Test missing low_level_keywords raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GPTKeywordExtractionFormat(high_level_keywords=['test'])
        assert 'low_level_keywords' in str(exc_info.value)

    def test_non_list_high_level_keywords_raises(self):
        """Test non-list high_level_keywords raises ValidationError."""
        with pytest.raises(ValidationError):
            GPTKeywordExtractionFormat(
                high_level_keywords='not_a_list',
                low_level_keywords=['test'],
            )

    def test_non_list_low_level_keywords_raises(self):
        """Test non-list low_level_keywords raises ValidationError."""
        with pytest.raises(ValidationError):
            GPTKeywordExtractionFormat(
                high_level_keywords=['test'],
                low_level_keywords='not_a_list',
            )

    def test_non_string_keywords_raises(self):
        """Test non-string keywords in lists raise ValidationError."""
        with pytest.raises(ValidationError):
            GPTKeywordExtractionFormat(
                high_level_keywords=[1, 2, 3],
                low_level_keywords=['test'],
            )

    def test_mixed_types_in_keywords_raises(self):
        """Test mixed types in keyword lists raise ValidationError."""
        with pytest.raises(ValidationError):
            GPTKeywordExtractionFormat(
                high_level_keywords=['valid', 123],
                low_level_keywords=['test'],
            )

    def test_special_characters_in_keywords(self):
        """Test keywords with special characters are accepted."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=['AI/ML', 'Deep-Learning', 'Neural_Networks'],
            low_level_keywords=['GPU-acceleration', 'NLP@scale'],
        )
        assert 'AI/ML' in fmt.high_level_keywords
        assert 'Deep-Learning' in fmt.high_level_keywords

    def test_whitespace_in_keywords(self):
        """Test keywords with whitespace are preserved."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=['keyword with spaces', '  leading spaces'],
            low_level_keywords=['multiple   internal spaces'],
        )
        assert fmt.high_level_keywords[0] == 'keyword with spaces'
        assert fmt.high_level_keywords[1] == '  leading spaces'

    def test_unicode_keywords(self):
        """Test Unicode keywords are accepted."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=['机器学习', 'IA', '학습'],
            low_level_keywords=['深层网络', 'ニューラルネットワーク'],
        )
        assert '机器学习' in fmt.high_level_keywords
        assert 'ニューラルネットワーク' in fmt.low_level_keywords

    def test_model_serialization(self):
        """Test model can be serialized to dict."""
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=['AI'],
            low_level_keywords=['ML'],
        )
        data = fmt.model_dump()
        assert data['high_level_keywords'] == ['AI']
        assert data['low_level_keywords'] == ['ML']

    def test_model_from_dict(self):
        """Test creating model from dict."""
        data = {
            'high_level_keywords': ['Category1', 'Category2'],
            'low_level_keywords': ['Detail1', 'Detail2'],
        }
        fmt = GPTKeywordExtractionFormat(**data)
        assert fmt.high_level_keywords == ['Category1', 'Category2']
        assert fmt.low_level_keywords == ['Detail1', 'Detail2']


class TestKnowledgeGraphNode:
    """Tests for KnowledgeGraphNode Pydantic model."""

    def test_valid_creation(self):
        """Test creating a valid KnowledgeGraphNode."""
        node = KnowledgeGraphNode(
            id='node_1',
            labels=['Person', 'Entity'],
            properties={'name': 'John Doe', 'age': 30},
        )
        assert node.id == 'node_1'
        assert node.labels == ['Person', 'Entity']
        assert node.properties == {'name': 'John Doe', 'age': 30}

    def test_empty_labels_list(self):
        """Test node with empty labels list."""
        node = KnowledgeGraphNode(
            id='node_1',
            labels=[],
            properties={},
        )
        assert node.labels == []
        assert node.properties == {}

    def test_single_label(self):
        """Test node with single label."""
        node = KnowledgeGraphNode(
            id='node_1',
            labels=['Person'],
            properties={},
        )
        assert len(node.labels) == 1
        assert node.labels[0] == 'Person'

    def test_multiple_labels(self):
        """Test node with multiple labels."""
        labels = ['Person', 'Employee', 'Manager', 'Developer']
        node = KnowledgeGraphNode(
            id='node_1',
            labels=labels,
            properties={},
        )
        assert node.labels == labels

    def test_complex_properties(self):
        """Test node with complex properties."""
        props = {
            'name': 'John',
            'age': 30,
            'salary': 100000.50,
            'active': True,
            'tags': ['developer', 'python'],
            'metadata': {'department': 'Engineering', 'level': 5},
            'score': None,
        }
        node = KnowledgeGraphNode(
            id='node_1',
            labels=['Person'],
            properties=props,
        )
        assert node.properties == props
        assert node.properties['tags'] == ['developer', 'python']
        assert node.properties['metadata']['department'] == 'Engineering'

    def test_empty_properties(self):
        """Test node with empty properties dict."""
        node = KnowledgeGraphNode(
            id='node_1',
            labels=['Entity'],
            properties={},
        )
        assert node.properties == {}

    def test_missing_id_raises(self):
        """Test missing id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphNode(labels=['Person'], properties={})
        assert 'id' in str(exc_info.value)

    def test_missing_labels_raises(self):
        """Test missing labels raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphNode(id='node_1', properties={})
        assert 'labels' in str(exc_info.value)

    def test_missing_properties_raises(self):
        """Test missing properties raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphNode(id='node_1', labels=['Person'])
        assert 'properties' in str(exc_info.value)

    def test_non_string_id_raises(self):
        """Test non-string id raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphNode(
                id=123,
                labels=['Person'],
                properties={},
            )

    def test_non_list_labels_raises(self):
        """Test non-list labels raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphNode(
                id='node_1',
                labels='Person',
                properties={},
            )

    def test_non_dict_properties_raises(self):
        """Test non-dict properties raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphNode(
                id='node_1',
                labels=['Person'],
                properties='not_a_dict',
            )

    def test_uuid_style_id(self):
        """Test UUID-style node IDs."""
        node = KnowledgeGraphNode(
            id='550e8400-e29b-41d4-a716-446655440000',
            labels=['Entity'],
            properties={},
        )
        assert node.id == '550e8400-e29b-41d4-a716-446655440000'

    def test_numeric_string_id(self):
        """Test numeric string as ID."""
        node = KnowledgeGraphNode(
            id='12345',
            labels=['Entity'],
            properties={},
        )
        assert node.id == '12345'

    def test_special_characters_in_id(self):
        """Test special characters in ID."""
        node = KnowledgeGraphNode(
            id='node_1-2:3.4@5',
            labels=['Entity'],
            properties={},
        )
        assert node.id == 'node_1-2:3.4@5'

    def test_model_serialization(self):
        """Test model can be serialized to dict."""
        node = KnowledgeGraphNode(
            id='node_1',
            labels=['Person'],
            properties={'name': 'Alice'},
        )
        data = node.model_dump()
        assert data['id'] == 'node_1'
        assert data['labels'] == ['Person']
        assert data['properties'] == {'name': 'Alice'}

    def test_model_from_dict(self):
        """Test creating model from dict."""
        data = {
            'id': 'node_xyz',
            'labels': ['Place', 'City'],
            'properties': {'name': 'New York', 'population': 8000000},
        }
        node = KnowledgeGraphNode(**data)
        assert node.id == 'node_xyz'
        assert node.labels == ['Place', 'City']


class TestKnowledgeGraphEdge:
    """Tests for KnowledgeGraphEdge Pydantic model."""

    def test_valid_creation_with_type(self):
        """Test creating a valid KnowledgeGraphEdge with type."""
        edge = KnowledgeGraphEdge(
            id='edge_1',
            type='KNOWS',
            source='node_1',
            target='node_2',
            properties={'since': 2020, 'weight': 0.8},
        )
        assert edge.id == 'edge_1'
        assert edge.type == 'KNOWS'
        assert edge.source == 'node_1'
        assert edge.target == 'node_2'
        assert edge.properties == {'since': 2020, 'weight': 0.8}

    def test_creation_with_none_type(self):
        """Test creating edge with type=None (untyped edge)."""
        edge = KnowledgeGraphEdge(
            id='edge_2',
            type=None,
            source='node_1',
            target='node_2',
            properties={},
        )
        assert edge.type is None
        assert edge.source == 'node_1'
        assert edge.target == 'node_2'

    def test_type_field_accepts_none(self):
        """Test type field accepts None value."""
        edge = KnowledgeGraphEdge(
            id='edge_3',
            type=None,
            source='node_1',
            target='node_2',
            properties={},
        )
        assert edge.type is None

    def test_empty_properties(self):
        """Test edge with empty properties."""
        edge = KnowledgeGraphEdge(
            id='edge_1',
            type='RELATED_TO',
            source='node_1',
            target='node_2',
            properties={},
        )
        assert edge.properties == {}

    def test_complex_properties(self):
        """Test edge with complex properties."""
        props = {
            'weight': 0.95,
            'confidence': 0.87,
            'created_at': '2024-01-01',
            'metadata': {'source': 'extracted', 'verified': True},
            'tags': ['important', 'verified'],
        }
        edge = KnowledgeGraphEdge(
            id='edge_1',
            type='KNOWS',
            source='node_1',
            target='node_2',
            properties=props,
        )
        assert edge.properties == props
        assert edge.properties['metadata']['verified'] is True

    def test_missing_id_raises(self):
        """Test missing id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphEdge(
                type='KNOWS',
                source='node_1',
                target='node_2',
                properties={},
            )
        assert 'id' in str(exc_info.value)

    def test_missing_source_raises(self):
        """Test missing source raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphEdge(
                id='edge_1',
                type='KNOWS',
                target='node_2',
                properties={},
            )
        assert 'source' in str(exc_info.value)

    def test_missing_target_raises(self):
        """Test missing target raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphEdge(
                id='edge_1',
                type='KNOWS',
                source='node_1',
                properties={},
            )
        assert 'target' in str(exc_info.value)

    def test_missing_properties_raises(self):
        """Test missing properties raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeGraphEdge(
                id='edge_1',
                type='KNOWS',
                source='node_1',
                target='node_2',
            )
        assert 'properties' in str(exc_info.value)

    def test_non_string_id_raises(self):
        """Test non-string id raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphEdge(
                id=123,
                type='KNOWS',
                source='node_1',
                target='node_2',
                properties={},
            )

    def test_non_string_source_raises(self):
        """Test non-string source raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphEdge(
                id='edge_1',
                type='KNOWS',
                source=123,
                target='node_2',
                properties={},
            )

    def test_non_string_target_raises(self):
        """Test non-string target raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphEdge(
                id='edge_1',
                type='KNOWS',
                source='node_1',
                target=456,
                properties={},
            )

    def test_non_dict_properties_raises(self):
        """Test non-dict properties raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraphEdge(
                id='edge_1',
                type='KNOWS',
                source='node_1',
                target='node_2',
                properties='not_a_dict',
            )

    def test_self_loop_edge(self):
        """Test edge where source and target are the same (self-loop)."""
        edge = KnowledgeGraphEdge(
            id='edge_self',
            type='SELF_REFERENCE',
            source='node_1',
            target='node_1',
            properties={'loop': True},
        )
        assert edge.source == edge.target

    def test_various_edge_types(self):
        """Test various edge type strings."""
        types = ['KNOWS', 'RELATED_TO', 'PARENT_OF', 'HAS_PROPERTY', 'BELONGS_TO']
        for edge_type in types:
            edge = KnowledgeGraphEdge(
                id=f'edge_{edge_type}',
                type=edge_type,
                source='node_1',
                target='node_2',
                properties={},
            )
            assert edge.type == edge_type

    def test_model_serialization(self):
        """Test model can be serialized to dict."""
        edge = KnowledgeGraphEdge(
            id='edge_1',
            type='KNOWS',
            source='node_1',
            target='node_2',
            properties={'strength': 'strong'},
        )
        data = edge.model_dump()
        assert data['id'] == 'edge_1'
        assert data['type'] == 'KNOWS'
        assert data['source'] == 'node_1'
        assert data['target'] == 'node_2'
        assert data['properties'] == {'strength': 'strong'}

    def test_model_from_dict(self):
        """Test creating model from dict."""
        data = {
            'id': 'edge_xy',
            'type': 'MENTIONS',
            'source': 'node_a',
            'target': 'node_b',
            'properties': {'count': 5},
        }
        edge = KnowledgeGraphEdge(**data)
        assert edge.id == 'edge_xy'
        assert edge.type == 'MENTIONS'


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph Pydantic model."""

    def test_empty_graph(self):
        """Test creating an empty knowledge graph."""
        graph = KnowledgeGraph()
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.is_truncated is False

    def test_default_values(self):
        """Test default values are set correctly."""
        graph = KnowledgeGraph()
        assert isinstance(graph.nodes, list)
        assert isinstance(graph.edges, list)
        assert isinstance(graph.is_truncated, bool)
        assert graph.is_truncated is False

    def test_single_node_graph(self):
        """Test graph with single node."""
        node = KnowledgeGraphNode(
            id='node_1',
            labels=['Entity'],
            properties={'name': 'Test'},
        )
        graph = KnowledgeGraph(nodes=[node], edges=[])
        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == 'node_1'
        assert graph.edges == []

    def test_graph_with_nodes_and_edges(self):
        """Test graph with both nodes and edges."""
        nodes = [
            KnowledgeGraphNode(id='n1', labels=['A'], properties={}),
            KnowledgeGraphNode(id='n2', labels=['B'], properties={}),
            KnowledgeGraphNode(id='n3', labels=['C'], properties={}),
        ]
        edges = [
            KnowledgeGraphEdge(id='e1', type='LINK', source='n1', target='n2', properties={}),
            KnowledgeGraphEdge(id='e2', type='LINK', source='n2', target='n3', properties={}),
        ]
        graph = KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=False)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.is_truncated is False

    def test_is_truncated_true(self):
        """Test graph with is_truncated=True."""
        node = KnowledgeGraphNode(id='n1', labels=['X'], properties={})
        graph = KnowledgeGraph(nodes=[node], edges=[], is_truncated=True)
        assert graph.is_truncated is True

    def test_is_truncated_false(self):
        """Test graph with explicit is_truncated=False."""
        graph = KnowledgeGraph(nodes=[], edges=[], is_truncated=False)
        assert graph.is_truncated is False

    def test_large_graph(self):
        """Test creating a large graph."""
        nodes = [
            KnowledgeGraphNode(id=f'node_{i}', labels=['Entity'], properties={'index': i})
            for i in range(100)
        ]
        edges = [
            KnowledgeGraphEdge(
                id=f'edge_{i}',
                type='CONNECTS',
                source=f'node_{i}',
                target=f'node_{(i+1) % 100}',
                properties={},
            )
            for i in range(100)
        ]
        graph = KnowledgeGraph(nodes=nodes, edges=edges)
        assert len(graph.nodes) == 100
        assert len(graph.edges) == 100

    def test_graph_with_none_type_edges(self):
        """Test graph with untyped edges (type=None)."""
        nodes = [
            KnowledgeGraphNode(id='n1', labels=['A'], properties={}),
            KnowledgeGraphNode(id='n2', labels=['B'], properties={}),
        ]
        edges = [
            KnowledgeGraphEdge(id='e1', type=None, source='n1', target='n2', properties={}),
        ]
        graph = KnowledgeGraph(nodes=nodes, edges=edges)
        assert graph.edges[0].type is None

    def test_invalid_nodes_type_raises(self):
        """Test non-list nodes raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(nodes='not_a_list', edges=[])

    def test_invalid_edges_type_raises(self):
        """Test non-list edges raises ValidationError."""
        with pytest.raises(ValidationError):
            KnowledgeGraph(nodes=[], edges='not_a_list')

    def test_invalid_is_truncated_type_raises(self):
        """Test non-coercible is_truncated raises ValidationError."""
        # Note: Pydantic v2 coerces truthy strings to bool, so 'yes' becomes True
        # Use a list instead which cannot be coerced to bool
        with pytest.raises(ValidationError):
            KnowledgeGraph(nodes=[], edges=[], is_truncated=['not', 'a', 'bool'])

    def test_node_list_independence(self):
        """Test that node lists are independent between instances."""
        node1 = KnowledgeGraphNode(id='n1', labels=['A'], properties={})
        graph1 = KnowledgeGraph(nodes=[node1])
        graph2 = KnowledgeGraph(nodes=[])

        assert len(graph1.nodes) == 1
        assert len(graph2.nodes) == 0
        assert graph1.nodes is not graph2.nodes

    def test_edge_list_independence(self):
        """Test that edge lists are independent between instances."""
        edge1 = KnowledgeGraphEdge(id='e1', type='X', source='n1', target='n2', properties={})
        graph1 = KnowledgeGraph(edges=[edge1])
        graph2 = KnowledgeGraph(edges=[])

        assert len(graph1.edges) == 1
        assert len(graph2.edges) == 0
        assert graph1.edges is not graph2.edges

    def test_model_serialization(self):
        """Test model can be serialized to dict."""
        nodes = [KnowledgeGraphNode(id='n1', labels=['A'], properties={})]
        edges = [KnowledgeGraphEdge(id='e1', type='X', source='n1', target='n2', properties={})]
        graph = KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=True)

        data = graph.model_dump()
        assert len(data['nodes']) == 1
        assert len(data['edges']) == 1
        assert data['is_truncated'] is True

    def test_model_from_dict(self):
        """Test creating model from dict with nested structures."""
        data = {
            'nodes': [
                {'id': 'n1', 'labels': ['Type1'], 'properties': {'key': 'value'}},
                {'id': 'n2', 'labels': ['Type2'], 'properties': {}},
            ],
            'edges': [
                {'id': 'e1', 'type': 'LINKS', 'source': 'n1', 'target': 'n2', 'properties': {}},
            ],
            'is_truncated': False,
        }
        graph = KnowledgeGraph(**data)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.nodes[0].id == 'n1'
        assert graph.edges[0].type == 'LINKS'

    def test_complex_nested_structure(self):
        """Test graph with complex nested properties."""
        node = KnowledgeGraphNode(
            id='complex_node',
            labels=['Entity'],
            properties={
                'metadata': {
                    'source': 'extraction',
                    'confidence': 0.95,
                    'attributes': {
                        'nested': {
                            'deep': {
                                'value': 'found',
                            }
                        }
                    },
                },
                'tags': ['important', 'verified'],
            },
        )
        edge = KnowledgeGraphEdge(
            id='complex_edge',
            type='COMPLEX_TYPE',
            source='complex_node',
            target='other_node',
            properties={
                'metadata': {
                    'weight': 0.9,
                    'relations': ['direct', 'indirect'],
                },
            },
        )
        graph = KnowledgeGraph(nodes=[node], edges=[edge], is_truncated=False)
        assert graph.nodes[0].properties['metadata']['attributes']['nested']['deep']['value'] == 'found'
        assert graph.edges[0].properties['metadata']['weight'] == 0.9


class TestLLMFunc:
    """Tests for LLMFunc type alias."""

    def test_llm_func_type_exists(self):
        """Test that LLMFunc type alias is defined and accessible."""
        assert LLMFunc is not None

    def test_llm_func_is_callable_type(self):
        """Test LLMFunc represents a callable type alias."""
        import typing
        from collections.abc import Callable

        # LLMFunc should be a generic alias for Callable
        # Check it's a parameterized Callable type
        origin = typing.get_origin(LLMFunc)
        assert origin is Callable

    async def test_llm_func_signature(self):
        """Test a function matching LLMFunc signature."""

        async def mock_llm_func(prompt: str) -> str:
            """Mock LLM function that matches LLMFunc signature."""
            return 'response'

        # Function can be called and returns awaitable
        result = await mock_llm_func('test prompt')
        assert result == 'response'
        assert isinstance(result, str)

    async def test_llm_func_with_kwargs(self):
        """Test LLMFunc with additional keyword arguments."""

        async def llm_with_kwargs(prompt: str, **kwargs) -> str:
            """LLM function with kwargs."""
            return f"response with {len(kwargs)} kwargs"

        result = await llm_with_kwargs('test', model='gpt4', temperature=0.7)
        assert isinstance(result, str)

    async def test_llm_func_with_multiple_args(self):
        """Test LLMFunc accepting multiple arguments."""

        async def complex_llm(prompt: str, context: str = '', **kwargs) -> str:
            """LLM function with multiple arguments."""
            return f"{prompt} - {context}"

        result = await complex_llm('question', context='info', top_k=5)
        assert 'question' in result
        assert 'info' in result


class TestTypeIntegration:
    """Integration tests for types working together."""

    def test_keyword_extraction_in_graph_properties(self):
        """Test using keyword extraction format in graph properties."""
        keywords = GPTKeywordExtractionFormat(
            high_level_keywords=['AI', 'ML'],
            low_level_keywords=['neural', 'tensor'],
        )
        # Store as dict in properties
        node = KnowledgeGraphNode(
            id='keyword_node',
            labels=['KeywordSet'],
            properties={'keywords': keywords.model_dump()},
        )
        assert node.properties['keywords']['high_level_keywords'] == ['AI', 'ML']

    def test_build_knowledge_graph_programmatically(self):
        """Test building a complete knowledge graph programmatically."""
        nodes = [
            KnowledgeGraphNode(
                id='ai',
                labels=['Technology', 'Concept'],
                properties={'name': 'Artificial Intelligence'},
            ),
            KnowledgeGraphNode(
                id='ml',
                labels=['Technology', 'Concept'],
                properties={'name': 'Machine Learning'},
            ),
            KnowledgeGraphNode(
                id='dl',
                labels=['Technology', 'Concept'],
                properties={'name': 'Deep Learning'},
            ),
        ]

        edges = [
            KnowledgeGraphEdge(
                id='ai_ml',
                type='ENCOMPASSES',
                source='ai',
                target='ml',
                properties={'relationship': 'subset'},
            ),
            KnowledgeGraphEdge(
                id='ml_dl',
                type='ENCOMPASSES',
                source='ml',
                target='dl',
                properties={'relationship': 'subset'},
            ),
        ]

        graph = KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=False)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.nodes[0].properties['name'] == 'Artificial Intelligence'

    def test_graph_serialization_roundtrip(self):
        """Test serializing and deserializing graph maintains data integrity."""
        original_graph = KnowledgeGraph(
            nodes=[
                KnowledgeGraphNode(
                    id='n1',
                    labels=['Entity'],
                    properties={'data': {'nested': 'value'}},
                )
            ],
            edges=[
                KnowledgeGraphEdge(
                    id='e1',
                    type='REL',
                    source='n1',
                    target='n2',
                    properties={'weight': 1.0},
                )
            ],
            is_truncated=True,
        )

        # Serialize
        serialized = original_graph.model_dump()

        # Deserialize
        restored_graph = KnowledgeGraph(**serialized)

        # Verify
        assert len(restored_graph.nodes) == len(original_graph.nodes)
        assert len(restored_graph.edges) == len(original_graph.edges)
        assert restored_graph.is_truncated == original_graph.is_truncated
        assert restored_graph.nodes[0].id == original_graph.nodes[0].id
        assert restored_graph.edges[0].type == original_graph.edges[0].type

    def test_edge_cases_with_all_models(self):
        """Test edge cases across all model types."""
        # Empty strings
        fmt = GPTKeywordExtractionFormat(
            high_level_keywords=[''],
            low_level_keywords=[''],
        )
        assert fmt.high_level_keywords == ['']

        # Unicode
        node = KnowledgeGraphNode(
            id='🔗',
            labels=['😀'],
            properties={'emoji': '🎯'},
        )
        assert '🔗' in node.id

        # Very long strings
        long_str = 'a' * 10000
        edge = KnowledgeGraphEdge(
            id='long_edge',
            type='LONG_TYPE',
            source='n1',
            target='n2',
            properties={'data': long_str},
        )
        assert len(edge.properties['data']) == 10000
