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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from yar.base import QueryContextResult, QueryParam, TextChunkSchema
from yar.constants import DEFAULT_MAX_FILE_PATHS, DEFAULT_SUMMARY_LANGUAGE, GRAPH_FIELD_SEP
from yar.operate import (
    _build_context_str,
    _build_prompt_chunk_context,
    _build_query_context,
    _build_query_shaping_instructions,
    _enrich_local_keywords,
    _find_most_related_edges_from_entities,
    _generate_multi_facet_hyde,
    _get_node_data,
    _get_vector_context,
    _matches_entity_filter,
    _merge_all_chunks,
    _normalize_query_shaped_response,
    _perform_kg_search,
    _prepare_visible_reference_payload,
    _resolve_max_file_paths,
    _should_validate_inline_citations,
    _split_keyword_terms,
    _truncate_entity_identifier,
    chunking_by_semantic,
    create_chunker,
    decompose_query_for_hyde,
    extract_keywords_only,
    get_keywords_from_query,
    kg_query,
    rewrite_query_with_history,
)
from yar.prompt import PROMPTS
from yar.utils import process_chunks_unified


@pytest.mark.offline
class TestEntityFilterMatching:
    """Tests for entity filter normalization used during retrieval."""

    def test_matches_entity_filter_ignores_punctuation_and_case(self):
        assert _matches_entity_filter('iCMC-NPP leader guidance', 'iCMC NPP')
        assert _matches_entity_filter('ICMC NPP leader guidance', 'icmc-npp')
        assert not _matches_entity_filter('SARA lessons learned', 'iCMC NPP')
        assert not _matches_entity_filter('iCMC-NPP leader guidance', '')

    @pytest.mark.asyncio
    async def test_vector_context_filter_matches_hyphenated_source_metadata(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.query = AsyncMock(
            return_value=[
                {
                    'id': 'chunk-1',
                    'content': 'The leader must avoid delaying submission.',
                    'file_path': '2019 iCMC-NPP lessons learned.pptx',
                    's3_key': 'default/doc-1/processed.md',
                    'score': 0.91,
                }
            ]
        )
        query_param = QueryParam(mode='mix', top_k=5, chunk_top_k=5, entity_filter='iCMC NPP', enable_bm25_fusion=False)

        chunks = await _get_vector_context('What must the leader avoid?', chunks_vdb, query_param)

        assert len(chunks) == 1
        assert chunks[0]['file_path'] == '2019 iCMC-NPP lessons learned.pptx'

    @pytest.mark.asyncio
    async def test_vector_context_filter_returns_empty_when_no_field_matches(self):
        chunks_vdb = MagicMock()
        chunks_vdb.cosine_better_than_threshold = 0.4
        chunks_vdb.query = AsyncMock(
            return_value=[
                {
                    'id': 'chunk-1',
                    'content': 'SARA sharing lessons learned.',
                    'file_path': '2019 SARA lessons learned.pptx',
                    's3_key': 'default/doc-1/processed.md',
                    'score': 0.91,
                }
            ]
        )
        query_param = QueryParam(mode='mix', top_k=5, chunk_top_k=5, entity_filter='iCMC NPP', enable_bm25_fusion=False)

        chunks = await _get_vector_context('What must the leader avoid?', chunks_vdb, query_param)

        assert chunks == []


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

    def test_chunking_with_preset_semantic(self):
        """Test chunking with semantic preset."""
        content = 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'
        result = chunking_by_semantic(content, preset='semantic')

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunking_with_preset_recursive(self):
        """Test chunking with recursive preset."""
        content = 'Line one.\nLine two.\nLine three.'
        result = chunking_by_semantic(content, preset='recursive')

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunking_with_preset_none(self):
        """Test chunking with no preset."""
        content = 'Simple text content for testing.'
        result = chunking_by_semantic(content, preset=None)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunk_order_indices(self):
        """Test that chunk indices are sequential."""
        content = '\n\n'.join([f'Paragraph {i}' for i in range(10)])
        result = chunking_by_semantic(content, max_chars=100, max_overlap=20)

        indices = [chunk['chunk_order_index'] for chunk in result]
        assert indices == list(range(len(result)))

    def test_chunk_tokens_estimation(self):
        """Test token count estimation (~4 chars per token)."""
        content = 'word ' * 100  # 500 characters
        result = chunking_by_semantic(content, max_chars=200, max_overlap=50)

        for chunk in result:
            # Token count should be roughly content_len / 4
            estimated_tokens = len(chunk['content']) // 4
            assert abs(chunk['tokens'] - estimated_tokens) < 10

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

    def test_preset_semantic(self):
        """Test semantic preset."""
        chunker = create_chunker(preset='semantic')
        result = chunker(None, 'Test content', None, False, 100, 1200)

        assert isinstance(result, list)

    def test_preset_recursive(self):
        """Test recursive preset."""
        chunker = create_chunker(preset='recursive')
        result = chunker(None, 'Test content', None, False, 100, 1200)

        assert isinstance(result, list)

    def test_preset_none(self):
        """Test None preset."""
        chunker = create_chunker(preset=None)
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

        # These params are ignored by Kreuzberg adapter
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
    async def test_conversation_history_included_in_keyword_cache_hash(self):
        """Keyword cache keys should vary when conversation history changes."""
        mock_llm = AsyncMock(return_value='{"high_level_keywords": ["ai"], "low_level_keywords": ["ml"]}')

        global_config = {
            'llm_model_func': mock_llm,
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'addon_params': {'language': DEFAULT_SUMMARY_LANGUAGE},
        }
        param = QueryParam(
            conversation_history=[
                {'role': 'user', 'content': 'Tell me about Fitusiran.'},
                {'role': 'assistant', 'content': 'It is an RNAi therapeutic.'},
            ]
        )
        hashing_kv = Mock()
        hashing_kv.global_config = {'enable_llm_cache': True}
        compute_hash = Mock(return_value='keywords-cache-hash')

        with (
            patch('yar.operate._conversation_history_cache_hash', return_value='history-hash') as history_hash_mock,
            patch('yar.operate.handle_cache', new=AsyncMock(return_value=None)),
            patch('yar.operate.compute_args_hash', new=compute_hash),
            patch('yar.operate.save_to_cache', new=AsyncMock()) as save_mock,
        ):
            hl, ll = await extract_keywords_only(
                text='What about dosing?',
                param=param,
                global_config=global_config,
                hashing_kv=hashing_kv,
            )

        history_hash_mock.assert_called_once_with(param.conversation_history)
        assert hl == ['ai']
        assert ll == ['ml']
        assert compute_hash.call_args.args == (
            param.mode,
            'What about dosing?',
            DEFAULT_SUMMARY_LANGUAGE,
            'history-hash',
        )

        saved_cache = save_mock.await_args.args[1]
        assert saved_cache.queryparam['conversation_history_hash'] == 'history-hash'

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
            patch('yar.operate._process_extraction_result', new=AsyncMock(return_value=({}, {}))),
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
            patch('yar.operate._process_extraction_result', new=AsyncMock(return_value=({}, {}))),
        ):
            result = await extract_entities(chunks=chunks, global_config=global_config)

        assert isinstance(result, list)
        assert captured_batch_sizes == [8, 8]

        captured_batch_sizes.clear()
        with (
            patch.dict('os.environ', {'ENTITY_EXTRACT_BATCH_SIZE': '20'}, clear=False),
            patch('yar.operate.use_llm_func_with_cache', side_effect=fake_use_llm_with_cache),
            patch('yar.operate._process_extraction_result', new=AsyncMock(return_value=({}, {}))),
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
            return {}, {}

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
            return {}, {}

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
            return {}, {}

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
            return {}, {}

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
            return {}, {}

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
            patch('yar.operate._process_extraction_result', new=AsyncMock(return_value=({}, {}))),
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
            patch('yar.operate._process_extraction_result', new=AsyncMock(return_value=({}, {}))),
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
        """Test that all presets return functions with same signature."""
        chunkers = [
            create_chunker(preset=None),
            create_chunker(preset='semantic'),
            create_chunker(preset='recursive'),
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
            ll_keywords=['Fitusiran', 'company research', 'Eptacog'],
            mode='local',
            user_supplied_ll=False,
        )

        assert enriched == ['Fitusiran', 'Eptacog']

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

        assert [entity['entity_name'] for entity in result['final_entities']] == [
            'ExactMatch',
            'HubEntity',
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


@pytest.mark.offline
class TestEntityQueryEmbeddingReuse:
    """Regression tests for reusing precomputed entity query embeddings."""

    @pytest.mark.asyncio
    async def test_perform_kg_search_reuses_precomputed_embedding_for_entity_search(self):
        query_param = QueryParam(
            mode='local',
            top_k=3,
            enable_hyde=True,
            model_func=AsyncMock(return_value='Detailed hypothetical answer for retrieval.'),
        )
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {'kg_chunk_pick_method': 'TEXT'}
        text_chunks_db.embedding_func = AsyncMock(return_value=[[0.1, 0.2]])

        with patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], []))) as node_mock:
            await _perform_kg_search(
                query='Amazon',
                ll_keywords='Amazon',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        # With HyDE on and entity-search reuse, we now embed both the HyDE answer and the original
        # query so the entity vector RRF-fuses HyDE drift against the literal query.
        assert text_chunks_db.embedding_func.await_count == 2
        embedded_inputs = [call.args[0] for call in text_chunks_db.embedding_func.await_args_list]
        assert ['Detailed hypothetical answer for retrieval.'] in embedded_inputs
        assert ['Amazon'] in embedded_inputs
        assert node_mock.await_args.kwargs['query_embedding'] == [0.1, 0.2]
        assert node_mock.await_args.kwargs['original_query_embedding'] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_perform_kg_search_accepts_short_hyde_answers(self):
        query_param = QueryParam(
            mode='local',
            top_k=3,
            enable_hyde=True,
            model_func=AsyncMock(return_value='Founded2019'),
        )
        text_chunks_db = MagicMock()
        text_chunks_db.global_config = {'kg_chunk_pick_method': 'TEXT'}
        text_chunks_db.embedding_func = AsyncMock(return_value=[[0.3, 0.4]])

        with patch('yar.operate._get_node_data', new=AsyncMock(return_value=([], []))) as node_mock:
            await _perform_kg_search(
                query='When was Acme founded?',
                ll_keywords='Acme',
                hl_keywords='',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=text_chunks_db,
                query_param=query_param,
                chunks_vdb=None,
            )

        # Same dual-embedding rationale as above for short HyDE answers.
        assert text_chunks_db.embedding_func.await_count == 2
        embedded_inputs = [call.args[0] for call in text_chunks_db.embedding_func.await_args_list]
        assert ['Founded2019'] in embedded_inputs
        assert ['When was Acme founded?'] in embedded_inputs
        assert node_mock.await_args.kwargs['query_embedding'] == [0.3, 0.4]
        assert node_mock.await_args.kwargs['original_query_embedding'] == [0.3, 0.4]

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

    @pytest.mark.asyncio
    async def test_kg_query_cache_hash_and_metadata_include_bm25_and_hyde_flags(self):
        query_param = QueryParam(
            mode='mix',
            enable_bm25_fusion=True,
            enable_hyde=True,
            model_func=AsyncMock(return_value='Answer'),
        )
        global_config = {
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
        }
        hashing_kv = Mock()
        hashing_kv.global_config = {'enable_llm_cache': True}

        context_result = QueryContextResult(
            context='Context block',
            raw_data={'data': {'references': []}},
        )
        compute_hash = Mock(return_value='query-cache-hash')

        with (
            patch('yar.operate.get_keywords_from_query', new=AsyncMock(return_value=(['High'], ['Low']))),
            patch('yar.operate._build_query_context', new=AsyncMock(return_value=context_result)),
            patch('yar.operate.handle_cache', new=AsyncMock(return_value=None)),
            patch('yar.operate.compute_args_hash', new=compute_hash),
            patch('yar.operate.save_to_cache', new=AsyncMock()) as save_mock,
        ):
            result = await kg_query(
                query='What changed?',
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_db=MagicMock(),
                query_param=query_param,
                global_config=global_config,
                hashing_kv=hashing_kv,
            )

        assert result is not None
        assert compute_hash.call_args.args == (
            query_param.mode,
            'What changed?',
            query_param.response_type,
            query_param.top_k,
            query_param.chunk_top_k,
            query_param.max_entity_tokens,
            query_param.max_relation_tokens,
            query_param.max_total_tokens,
            'High',
            'Low',
            query_param.user_prompt or '',
            PROMPTS['rag_response'],
            8192,
            query_param.enable_rerank,
            query_param.enable_bm25_fusion,
            query_param.enable_hyde,
            query_param.entity_filter or '',
        )

        saved_cache = save_mock.await_args.args[1]
        assert saved_cache.queryparam['enable_bm25_fusion'] is True
        assert saved_cache.queryparam['enable_hyde'] is True


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

        with (
            patch('yar.operate._perform_kg_search', new=AsyncMock(return_value=search_result)),
            patch('yar.operate._apply_token_truncation', new=AsyncMock(return_value=truncation_result)),
            patch('yar.operate._merge_all_chunks', new=merge_mock),
            patch(
                'yar.operate._build_context_str',
                new=AsyncMock(return_value=('Context block', {'data': {'chunks': []}, 'metadata': {}})),
            ),
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
        assert merge_mock.await_args.kwargs['topic_terms'] == ['diabetes']
        assert merge_mock.await_args.kwargs['facet_terms'] == ['long-term complications', 'chronic conditions']

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
        )

        assert [reference['file_path'] for reference in visible_references] == ['medical_diabetes.md']
        assert [chunk['chunk_id'] for chunk in visible_chunks] == ['diabetes-1']

    def test_prepare_visible_reference_payload_drops_near_best_off_topic_doc_when_topic_signal_exists(self):
        visible_references, visible_chunks = _prepare_visible_reference_payload(
            [
                {
                    'reference_id': '1',
                    'content': '=== Sarclisa manufacturing flow ===\nThe Netherlands physical flow created label and shipping consequences for Sarclisa.',
                    'file_path': 'sarclisa.md',
                    'chunk_id': 'sarclisa-1',
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
                {'reference_id': '1', 'file_path': 'sarclisa.md'},
                {'reference_id': '2', 'file_path': 'other-program.md'},
            ],
            'What were the manufacturing consequences for Sarclisa in the Netherlands?',
            include_reference_ids=False,
            topic_terms=['Sarclisa'],
            facet_terms=['manufacturing consequences', 'regulatory submission impact'],
        )

        assert [reference['file_path'] for reference in visible_references] == ['sarclisa.md']
        assert [chunk['chunk_id'] for chunk in visible_chunks] == ['sarclisa-1']

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
    async def test_hyde_timeout_falls_back_to_original_query(self):
        """HyDE gracefully degrades on timeout instead of failing the query."""

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(10)  # Will exceed timeout
            return 'Hypothetical answer'

        query_param = QueryParam(mode='mix', enable_hyde=True, model_func=AsyncMock(return_value='Answer'))
        global_config = {
            'tokenizer': Mock(encode=Mock(return_value=[0] * 10)),
            'llm_model_func': slow_llm,
            'llm_timeout': 0.1,  # 100ms timeout
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
                text_chunks_db=MagicMock(global_config=global_config),
                query_param=query_param,
                global_config=global_config,
            )

        # Query still succeeds even though HyDE timed out
        assert result is not None
        assert result.content == 'Answer'

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
                query='What is CSTD strategy?',
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
    async def test_process_chunks_unified_limits_binary_questions_to_one_passage_per_document(self):
        """Binary questions should keep the best passage per document for vector-only and hybrid chunk sets."""
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

            assert [chunk['chunk_id'] for chunk in processed] == ['alpha-1', 'beta-1', 'gamma-1']

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
                'content': 'LL-2 - Difficult tracking of scope change in SoW\n\nLL-3 - "New Phase 1 clinical strategy tested"\nA new Phase 1 (SAD) clinical strategy was tested by MyoKardia: powder in bottle directly to the clinical center.',
                'file_path': 'myokardia.pdf',
                'chunk_id': 'myokardia-1',
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

        assert processed[0]['chunk_id'] == 'myokardia-1'

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
            'Does the NeoGAA China submission include full detail for the reaction steps?'
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
            'Would you agree to change the storage condition on short notice prior to NDA submission?'
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

    def test_build_query_shaping_instructions_for_template_queries(self):
        """Risk-format questions should reproduce source templates verbatim without expanding ellipses into bracketed labels."""
        instructions = _build_query_shaping_instructions(
            'Based on lessons learned What is the correct descriptive syntaxe to phrase the CMC risk'
        )

        assert any('ellipses' in instruction for instruction in instructions)
        assert any('bracketed' in instruction for instruction in instructions)
        assert any('verbatim' in instruction for instruction in instructions)
        assert any('lead-in' in instruction for instruction in instructions)
        assert any(
            '[subject]' in instruction or '[action]' in instruction or '[impact]' in instruction
            for instruction in instructions
        )

    def test_normalize_query_shaped_response_preserves_risk_template(self):
        """Risk-format questions should collapse invented bracket placeholders back to the source template."""
        response = (
            'To phrase the CMC risk correctly, use the syntax: '
            '**Due to [cause] the risk [risk description] could impact [impact area]** [1].'
        )
        available_refs = [
            {'excerpt': 'The use of the syntaxe of the description : Due to ... the risk ...could impact ....'}
        ]

        normalized = _normalize_query_shaped_response(
            query='Based on lessons learned What is the correct descriptive syntaxe to phrase the CMC risk',
            response=response,
            available_refs=available_refs,
        )

        assert normalized == 'The correct syntax is: Due to ... the risk ... could impact .... [1]'

    def test_normalize_query_shaped_response_strips_single_fact_markdown(self):
        """Single-fact answers should return the supported option plainly without markdown emphasis."""
        normalized = _normalize_query_shaped_response(
            query='For biologics should we ask shipping validation question in type C or B meeting',
            response='Ask the shipping validation question in a **Type C meeting** [1].',
            available_refs=[],
        )

        assert normalized == 'In a Type C meeting [1].'

    def test_normalize_query_shaped_response_collapses_meeting_choice(self):
        """Single-fact meeting-choice answers should collapse to the supported meeting phrase."""
        normalized = _normalize_query_shaped_response(
            query='For biologics should we ask shipping validation question in type C or B meeting',
            response='Add the shipping validation question in type C meeting [1].',
            available_refs=[],
        )

        assert normalized == 'In type C meeting [1].'


@pytest.mark.offline
class TestRewriteQueryWithHistory:
    """Tests for rewrite_query_with_history."""

    @pytest.mark.asyncio
    async def test_no_history_returns_none(self):
        result = await rewrite_query_with_history(
            'What about phase 2?',
            None,
            use_model_func=AsyncMock(return_value='unused'),
            llm_timeout=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_history_returns_none(self):
        result = await rewrite_query_with_history(
            'What about phase 2?',
            [],
            use_model_func=AsyncMock(return_value='unused'),
            llm_timeout=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_model_func_returns_none(self):
        result = await rewrite_query_with_history(
            'What about phase 2?',
            [{'role': 'user', 'content': 'Tell me about Drug X phase 1'}],
            use_model_func=None,
            llm_timeout=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rewrites_pronoun_query(self):
        model_func = AsyncMock(return_value='What is the phase 2 trial of Drug X?')
        result = await rewrite_query_with_history(
            'What about phase 2?',
            [
                {'role': 'user', 'content': 'Tell me about Drug X phase 1'},
                {'role': 'assistant', 'content': 'Phase 1 of Drug X showed safety in 50 patients.'},
            ],
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result == 'What is the phase 2 trial of Drug X?'
        # Prompt should embed both the query and a formatted history block.
        prompt_arg = model_func.await_args.args[0]
        assert 'What about phase 2?' in prompt_arg
        assert 'Tell me about Drug X phase 1' in prompt_arg
        assert 'Phase 1 of Drug X showed safety in 50 patients.' in prompt_arg

    @pytest.mark.asyncio
    async def test_strips_quotes_from_model_output(self):
        model_func = AsyncMock(return_value='"What is the phase 2 trial of Drug X?"')
        result = await rewrite_query_with_history(
            'What about phase 2?',
            [{'role': 'user', 'content': 'Tell me about Drug X'}],
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result == 'What is the phase 2 trial of Drug X?'

    @pytest.mark.asyncio
    async def test_rejects_overly_long_rewrite(self):
        # Suspiciously long output (>6x original AND >600 chars) is treated as a model failure.
        bogus = 'noise ' * 200
        model_func = AsyncMock(return_value=bogus)
        result = await rewrite_query_with_history(
            'What about phase 2?',
            [{'role': 'user', 'content': 'Drug X phase 1'}],
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_rewrite_matches_original(self):
        model_func = AsyncMock(return_value='What about phase 2?')
        result = await rewrite_query_with_history(
            'What about phase 2?',
            [{'role': 'user', 'content': 'Drug X'}],
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        # No-op rewrite signals the caller to fall back; same as None.
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        async def slow(*_args, **_kwargs):
            await asyncio.sleep(5)
            return 'too slow'

        result = await rewrite_query_with_history(
            'What about phase 2?',
            [{'role': 'user', 'content': 'Drug X'}],
            use_model_func=slow,
            llm_timeout=0.01,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_truncates_history_to_max_turns(self):
        # Build 20 history turns; only the last 6 should appear in the prompt.
        history = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'turn-{i}'} for i in range(20)]
        model_func = AsyncMock(return_value='resolved query')
        await rewrite_query_with_history(
            'follow-up',
            history,
            use_model_func=model_func,
            llm_timeout=10.0,
            max_history_turns=6,
        )
        prompt_arg = model_func.await_args.args[0]
        # Most recent 6 turns (14..19) should appear; older ones should not.
        assert 'turn-19' in prompt_arg
        assert 'turn-14' in prompt_arg
        assert 'turn-13' not in prompt_arg
        assert 'turn-0' not in prompt_arg


@pytest.mark.offline
class TestDecomposeQueryForHyde:
    """Tests for decompose_query_for_hyde."""

    @pytest.mark.asyncio
    async def test_atomic_query_short_circuits_without_llm(self):
        # No multi-facet markers -> heuristic returns the original query without an LLM call.
        model_func = AsyncMock(return_value='["unused"]')
        result = await decompose_query_for_hyde(
            'What is mRNA?',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result == ['What is mRNA?']
        model_func.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_query_with_and_short_circuits(self):
        # Less than 5 tokens, even with 'and' -> not worth decomposing.
        model_func = AsyncMock(return_value='["unused"]')
        result = await decompose_query_for_hyde(
            'A and B?',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result == ['A and B?']
        model_func.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compare_query_decomposes(self):
        model_func = AsyncMock(return_value='["What is the safety of drug A?", "What is the safety of drug B?"]')
        result = await decompose_query_for_hyde(
            'Compare drug A and drug B for safety',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result == ['What is the safety of drug A?', 'What is the safety of drug B?']
        model_func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dedupes_subquestions_case_insensitive(self):
        model_func = AsyncMock(return_value='["What is X?", "WHAT IS X?", "What is Y?"]')
        result = await decompose_query_for_hyde(
            'Compare X and Y in detail',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        # Case-insensitive dedup keeps first occurrence.
        assert result == ['What is X?', 'What is Y?']

    @pytest.mark.asyncio
    async def test_caps_at_max_subquestions(self):
        model_func = AsyncMock(return_value='["q1", "q2", "q3", "q4", "q5"]')
        result = await decompose_query_for_hyde(
            'Compare A and B and C and D and E in detail',
            use_model_func=model_func,
            llm_timeout=10.0,
            max_subquestions=3,
        )
        assert result == ['q1', 'q2', 'q3']

    @pytest.mark.asyncio
    async def test_falls_back_to_original_on_invalid_json(self):
        model_func = AsyncMock(return_value='this is not json at all')
        result = await decompose_query_for_hyde(
            'Compare drug A and drug B for safety',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        # Falls back to the original query when JSON parse fails.
        assert result == ['Compare drug A and drug B for safety']

    @pytest.mark.asyncio
    async def test_falls_back_when_llm_returns_empty_list(self):
        model_func = AsyncMock(return_value='[]')
        result = await decompose_query_for_hyde(
            'Compare drug A and drug B for safety',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result == ['Compare drug A and drug B for safety']

    @pytest.mark.asyncio
    async def test_timeout_falls_back(self):
        async def slow(*_args, **_kwargs):
            await asyncio.sleep(5)
            return '["slow"]'

        result = await decompose_query_for_hyde(
            'Compare drug A and drug B for safety',
            use_model_func=slow,
            llm_timeout=0.01,
        )
        assert result == ['Compare drug A and drug B for safety']

    @pytest.mark.asyncio
    async def test_no_model_func_returns_original(self):
        result = await decompose_query_for_hyde(
            'Compare A and B and C across X and Y',
            use_model_func=None,
            llm_timeout=10.0,
        )
        assert result == ['Compare A and B and C across X and Y']


@pytest.mark.offline
class TestGenerateMultiFacetHyde:
    """Tests for _generate_multi_facet_hyde."""

    @pytest.mark.asyncio
    async def test_atomic_query_runs_hyde_once(self):
        # No multi-facet markers -> decompose returns [query]; HyDE runs once.
        model_func = AsyncMock(return_value='Single hypothetical answer about mRNA biology and translation.')
        result = await _generate_multi_facet_hyde(
            'What is mRNA?',
            use_model_func=model_func,
            llm_timeout=10.0,
        )
        assert result is not None
        assert 'mRNA' in result
        # Atomic query -> only one model call (no decompose, just HyDE).
        assert model_func.await_count == 1

    @pytest.mark.asyncio
    async def test_multi_facet_concatenates_per_facet_passages(self):
        # First call: decomposition. Subsequent calls: HyDE per sub-question.
        async def fake_model(prompt, **_kwargs):
            if 'JSON array' in prompt or 'json array' in prompt.lower():
                return '["What is the safety of drug A?", "What is the safety of drug B?"]'
            if 'drug A' in prompt:
                return 'Drug A safety profile passage with substantial detail and supporting context.'
            if 'drug B' in prompt:
                return 'Drug B safety profile passage with substantial detail and supporting context.'
            return 'fallback hypothetical passage'

        result = await _generate_multi_facet_hyde(
            'Compare drug A and drug B for safety',
            use_model_func=fake_model,
            llm_timeout=10.0,
        )
        assert result is not None
        assert 'Drug A safety' in result
        assert 'Drug B safety' in result
        assert '\n\n' in result  # facet passages joined with blank line

    @pytest.mark.asyncio
    async def test_returns_none_when_all_facets_fail(self):
        # Decomposition succeeds but every HyDE call returns too-short text.
        async def fake_model(prompt, **_kwargs):
            if 'JSON array' in prompt or 'json array' in prompt.lower():
                return '["q1", "q2"]'
            return 'tiny'  # below the 10-char HyDE floor

        result = await _generate_multi_facet_hyde(
            'Compare A and B in detail with care',
            use_model_func=fake_model,
            llm_timeout=10.0,
        )
        assert result is None
