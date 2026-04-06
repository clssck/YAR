"""
Tests for yar/base.py - Core data structures and abstract base classes.

This module tests:
- QueryParam dataclass with defaults and validation
- DocStatus enum values
- DocProcessingStatus dataclass and __post_init__ logic
- DeletionResult dataclass
- QueryResult dataclass with properties
- QueryContextResult dataclass
- StoragesStatus enum
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from yar.base import (
    DeletionResult,
    DocProcessingStatus,
    DocStatus,
    QueryContextResult,
    QueryParam,
    QueryResult,
    StoragesStatus,
    TextChunkSchema,
 )
from yar.evaluation.e2e_test_harness import resolve_dataset_path
from yar.evaluation.eval_rag_quality import (
    RAGEvaluator,
    _calculate_ragas_score,
    _has_complete_metrics,
 )
from yar.yar import YAR

if TYPE_CHECKING:
    pass


class TestDocStatus:
    """Tests for DocStatus enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert DocStatus.PENDING == 'pending'
        assert DocStatus.PROCESSING == 'processing'
        assert DocStatus.PREPROCESSED == 'preprocessed'
        assert DocStatus.PROCESSED == 'processed'
        assert DocStatus.FAILED == 'failed'

    def test_enum_is_string(self):
        """Test DocStatus is a string enum."""
        assert isinstance(DocStatus.PENDING, str)
        assert DocStatus.PENDING == 'pending'

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert 'pending' in [s.value for s in DocStatus]
        assert 'invalid' not in [s.value for s in DocStatus]


class TestStoragesStatus:
    """Tests for StoragesStatus enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert StoragesStatus.NOT_CREATED == 'not_created'
        assert StoragesStatus.CREATED == 'created'
        assert StoragesStatus.INITIALIZED == 'initialized'
        assert StoragesStatus.FINALIZED == 'finalized'

    def test_enum_is_string(self):
        """Test StoragesStatus is a string enum."""
        assert isinstance(StoragesStatus.CREATED, str)


class TestQueryParam:
    """Tests for QueryParam dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        param = QueryParam()
        assert param.mode == 'mix'
        assert param.only_need_context is False
        assert param.only_need_prompt is False
        assert param.response_type == 'Multiple Paragraphs'
        assert param.stream is False
        assert param.hl_keywords == []
        assert param.ll_keywords == []
        assert param.enable_rerank is False
    def test_custom_values(self):
        """Test custom values override defaults."""
        param = QueryParam(
            mode='local',
            only_need_context=True,
            stream=True,
            top_k=100,
            hl_keywords=['test', 'keyword'],
        )
        assert param.mode == 'local'
        assert param.only_need_context is True
        assert param.stream is True
        assert param.top_k == 100
        assert param.hl_keywords == ['test', 'keyword']

    def test_mode_literal_values(self):
        """Test all valid mode values."""
        valid_modes = ['local', 'global', 'hybrid', 'naive', 'mix', 'bypass']
        for mode in valid_modes:
            param = QueryParam(mode=mode)  # type: ignore[arg-type]
            assert param.mode == mode

    def test_model_func_default_none(self):
        """Test model_func defaults to None."""
        param = QueryParam()
        assert param.model_func is None

    def test_token_limits(self):
        """Test token limit fields exist and have reasonable defaults."""
        param = QueryParam()
        assert param.max_entity_tokens > 0
        assert param.max_relation_tokens > 0
        assert param.max_total_tokens > 0


class TestDocProcessingStatus:
    """Tests for DocProcessingStatus dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        status = DocProcessingStatus(
            content_summary='Test content...',
            content_length=1000,
            file_path='/path/to/file.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        assert status.content_summary == 'Test content...'
        assert status.content_length == 1000
        assert status.file_path == '/path/to/file.txt'
        assert status.status == DocStatus.PENDING

    def test_optional_fields_default(self):
        """Test optional fields have correct defaults."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        assert status.track_id is None
        assert status.chunks_count is None
        assert status.error_msg is None
        assert status.s3_key is None
        assert status.metadata == {}

    def test_post_init_multimodal_false_processed(self):
        """Test __post_init__ converts PROCESSED to PREPROCESSED when multimodal_processed=False."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            multimodal_processed=False,
        )
        # Status should be converted to PREPROCESSED
        assert status.status == DocStatus.PREPROCESSED

    def test_post_init_multimodal_true_processed(self):
        """Test __post_init__ keeps PROCESSED when multimodal_processed=True."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            multimodal_processed=True,
        )
        # Status should remain PROCESSED
        assert status.status == DocStatus.PROCESSED

    def test_post_init_multimodal_none(self):
        """Test __post_init__ does nothing when multimodal_processed=None."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            multimodal_processed=None,
        )
        # Status should remain PROCESSED
        assert status.status == DocStatus.PROCESSED

    def test_chunks_list_default_is_none(self):
        """Test chunks_list defaults to None (not an empty list)."""
        status = DocProcessingStatus(
            content_summary='Test1',
            content_length=100,
            file_path='/test1.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        assert status.chunks_list is None


class TestDeletionResult:
    """Tests for DeletionResult dataclass."""

    def test_success_result(self):
        """Test successful deletion result."""
        result = DeletionResult(
            status='success',
            doc_id='doc_123',
            message='Document deleted successfully',
        )
        assert result.status == 'success'
        assert result.doc_id == 'doc_123'
        assert result.status_code == 200

    def test_not_found_result(self):
        """Test not found deletion result."""
        result = DeletionResult(
            status='not_found',
            doc_id='doc_456',
            message='Document not found',
            status_code=404,
        )
        assert result.status == 'not_found'
        assert result.status_code == 404

    def test_fail_result(self):
        """Test failed deletion result."""
        result = DeletionResult(
            status='fail',
            doc_id='doc_789',
            message='Deletion failed',
            status_code=500,
        )
        assert result.status == 'fail'
        assert result.status_code == 500

    def test_file_path_optional(self):
        """Test file_path is optional."""
        result = DeletionResult(
            status='success',
            doc_id='doc_123',
            message='Deleted',
        )
        assert result.file_path is None

        result_with_path = DeletionResult(
            status='success',
            doc_id='doc_123',
            message='Deleted',
            file_path='/path/to/file.txt',
        )
        assert result_with_path.file_path == '/path/to/file.txt'


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = QueryResult()
        assert result.content is None
        assert result.response_iterator is None
        assert result.raw_data is None
        assert result.is_streaming is False

    def test_non_streaming_result(self):
        """Test non-streaming result with content."""
        result = QueryResult(
            content='This is the answer',
            is_streaming=False,
        )
        assert result.content == 'This is the answer'
        assert result.is_streaming is False

    def test_reference_list_property_with_data(self):
        """Test reference_list property extracts references from raw_data."""
        result = QueryResult(
            content='Answer',
            raw_data={
                'data': {
                    'references': [
                        {'reference_id': '1', 'file_path': '/doc1.pdf'},
                        {'reference_id': '2', 'file_path': '/doc2.pdf'},
                    ]
                }
            },
        )
        refs = result.reference_list
        assert len(refs) == 2
        assert refs[0]['reference_id'] == '1'
        assert refs[1]['file_path'] == '/doc2.pdf'

    def test_reference_list_property_empty(self):
        """Test reference_list property returns empty list when no data."""
        result = QueryResult()
        assert result.reference_list == []

        result_with_empty_data = QueryResult(raw_data={})
        assert result_with_empty_data.reference_list == []

    def test_metadata_property_with_data(self):
        """Test metadata property extracts metadata from raw_data."""
        result = QueryResult(
            raw_data={
                'metadata': {
                    'query_mode': 'mix',
                    'keywords': ['test', 'query'],
                }
            }
        )
        metadata = result.metadata
        assert metadata['query_mode'] == 'mix'
        assert metadata['keywords'] == ['test', 'query']

    def test_metadata_property_empty(self):
        """Test metadata property returns empty dict when no data."""
        result = QueryResult()
        assert result.metadata == {}


class TestQueryContextResult:
    """Tests for QueryContextResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation with required fields."""
        result = QueryContextResult(
            context='Relevant context for the query...',
            raw_data={'data': {}, 'metadata': {}},
        )
        assert result.context == 'Relevant context for the query...'
        assert result.coverage_level == 'good'  # default

    def test_limited_coverage(self):
        """Test limited coverage level."""
        result = QueryContextResult(
            context='Sparse context...',
            raw_data={},
            coverage_level='limited',
        )
        assert result.coverage_level == 'limited'

    def test_reference_list_property(self):
        """Test reference_list property."""
        result = QueryContextResult(
            context='Context',
            raw_data={
                'data': {
                    'references': [{'reference_id': '1', 'file_path': '/test.pdf'}]
                }
            },
        )
        assert len(result.reference_list) == 1


class TestTextChunkSchema:
    """Tests for TextChunkSchema TypedDict."""

    def test_required_fields(self):
        """Test creating a chunk with required fields."""
        chunk: TextChunkSchema = {
            'tokens': 100,
            'content': 'This is the chunk content',
            'full_doc_id': 'doc_123',
            'chunk_order_index': 0,
        }
        assert chunk['tokens'] == 100
        assert chunk['content'] == 'This is the chunk content'

    def test_optional_fields(self):
        """Test creating a chunk with optional fields."""
        chunk: TextChunkSchema = {
            'tokens': 100,
            'content': 'Content',
            'full_doc_id': 'doc_123',
            'chunk_order_index': 0,
            'file_path': '/path/to/file.txt',
            's3_key': 'bucket/key',
            'char_start': 0,
            'char_end': 100,
        }
        assert chunk['file_path'] == '/path/to/file.txt'
        assert chunk['s3_key'] == 'bucket/key'
        assert chunk['char_start'] == 0
        assert chunk['char_end'] == 100


# Integration tests for edge cases


class TestEdgeCases:
    """Edge case tests for base module."""

    def test_doc_processing_status_with_error(self):
        """Test DocProcessingStatus with error message."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.FAILED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            error_msg='Processing failed: Out of memory',
        )
        assert status.status == DocStatus.FAILED
        assert 'Out of memory' in status.error_msg

    def test_query_result_streaming_mode(self):
        """Test QueryResult in streaming mode."""

        async def mock_iterator():
            yield 'chunk1'
            yield 'chunk2'

        result = QueryResult(
            response_iterator=mock_iterator(),
            is_streaming=True,
        )
        assert result.is_streaming is True
        assert result.content is None
        assert result.response_iterator is not None

    def test_query_param_with_all_options(self):
        """Test QueryParam with many options set."""
        param = QueryParam(
            mode='hybrid',
            only_need_context=True,
            only_need_prompt=False,
            response_type='Bullet Points',
            stream=True,
            top_k=50,
            chunk_top_k=20,
            max_entity_tokens=8000,
            max_relation_tokens=8000,
            max_total_tokens=32000,
            hl_keywords=['high', 'level'],
            ll_keywords=['low', 'level'],
            enable_hyde=True,
            enable_bm25_fusion=True,
            bm25_weight=0.4,
            entity_filter='Fitusiran',
        )
        assert param.mode == 'hybrid'
        assert param.response_type == 'Bullet Points'
        assert param.top_k == 50
        assert param.max_total_tokens == 32000
        assert param.enable_hyde is True
        assert param.enable_bm25_fusion is True
        assert param.bm25_weight == 0.4
        assert param.entity_filter == 'Fitusiran'


@pytest.mark.offline
class TestYARQueryMethods:
    """Tests for YAR query wrapper semantics."""

    @pytest.mark.asyncio
    async def test_aquery_data_clones_all_query_fields(self):
        """Data-only path should preserve newer QueryParam fields."""
        rag = YAR.__new__(YAR)
        rag.chunk_entity_relation_graph = object()
        rag.entities_vdb = object()
        rag.relationships_vdb = object()
        rag.text_chunks = object()
        rag.chunks_vdb = object()
        rag.llm_response_cache = object()
        rag._query_done = AsyncMock()

        original_param = QueryParam(
            mode='mix',
            stream=True,
            only_need_context=False,
            only_need_prompt=True,
            response_type='Bullet Points',
            top_k=7,
            chunk_top_k=4,
            max_entity_tokens=123,
            max_relation_tokens=456,
            max_total_tokens=789,
            hl_keywords=['high'],
            ll_keywords=['low'],
            conversation_history=[{'role': 'user', 'content': 'Earlier context'}],
            model_func=AsyncMock(return_value='custom model'),
            user_prompt='Be precise',
            enable_rerank=True,
            enable_hyde=True,
            enable_bm25_fusion=True,
            bm25_weight=0.55,
            entity_filter='Fitusiran',
        )

        returned_result = QueryResult(
            content='',
            raw_data={'status': 'success', 'message': 'ok', 'data': {}, 'metadata': {}},
        )

        with (
            patch('yar.yar.asdict', return_value={}),
            patch('yar.yar.kg_query', new=AsyncMock(return_value=returned_result)) as kg_query_mock,
        ):
            result = await YAR.aquery_data(rag, 'test query', original_param)

        cloned_param = kg_query_mock.await_args.args[5]
        assert cloned_param is not original_param
        assert cloned_param.only_need_context is True
        assert cloned_param.only_need_prompt is False
        assert cloned_param.stream is False
        assert cloned_param.enable_hyde is True
        assert cloned_param.enable_bm25_fusion is True
        assert cloned_param.bm25_weight == 0.55
        assert cloned_param.entity_filter == 'Fitusiran'
        assert cloned_param.conversation_history == [{'role': 'user', 'content': 'Earlier context'}]
        assert cloned_param.user_prompt == 'Be precise'
        assert cloned_param.model_func is original_param.model_func
        assert original_param.only_need_context is False
        assert original_param.only_need_prompt is True
        assert original_param.stream is True
        assert result == {'status': 'success', 'message': 'ok', 'data': {}, 'metadata': {}}

    @pytest.mark.asyncio
    async def test_aquery_raises_for_backend_failure_payload(self):
        """Backward-compatible wrapper should raise on real backend failure."""
        rag = YAR.__new__(YAR)
        rag.aquery_llm = AsyncMock(
            return_value={
                'status': 'failure',
                'message': 'Query failed: upstream timeout',
                'metadata': {},
                'data': {},
                'llm_response': {'content': None, 'response_iterator': None, 'is_streaming': False},
            }
        )

        with pytest.raises(RuntimeError, match='Query failed: upstream timeout'):
            await YAR.aquery(rag, 'test query')

    @pytest.mark.asyncio
    async def test_aquery_no_results_returns_user_facing_content(self):
        """No-results path should remain truthful and non-exceptional."""
        rag = YAR.__new__(YAR)
        rag.aquery_llm = AsyncMock(
            return_value={
                'status': 'failure',
                'message': 'Query returned no results',
                'metadata': {'failure_reason': 'no_results'},
                'data': {},
                'llm_response': {
                    'content': 'No relevant context found for the query.',
                    'response_iterator': None,
                    'is_streaming': False,
                },
            }
        )

        result = await YAR.aquery(rag, 'test query')

        assert result == 'No relevant context found for the query.'


class TestEvaluationHarnessHelpers:
    """Focused tests for evaluation script helpers."""

    def test_resolve_dataset_path_prefers_explicit_path(self, tmp_path):
        """Explicit dataset path should win over bundled fallbacks."""
        explicit_dataset = tmp_path / 'explicit.json'
        explicit_dataset.write_text('{"test_cases": []}', encoding='utf-8')

        bundled_dataset = tmp_path / 'bundled.json'
        bundled_dataset.write_text('{"test_cases": [{"question": "q", "ground_truth": "a"}]}', encoding='utf-8')

        resolved = resolve_dataset_path(explicit_dataset, [bundled_dataset])

        assert resolved == explicit_dataset

    def test_resolve_dataset_path_raises_actionable_error_when_missing(self, tmp_path):
        """Missing bundled datasets should raise a clear actionable error."""
        missing_candidates = [tmp_path / 'missing-a.json', tmp_path / 'missing-b.json']

        with pytest.raises(FileNotFoundError, match='Provide --dataset') as exc_info:
            resolve_dataset_path(None, missing_candidates)

        message = str(exc_info.value)
        assert 'missing-a.json' in message
        assert 'missing-b.json' in message

    def test_incomplete_metrics_do_not_produce_successful_ragas_score(self):
        """Any missing or NaN metric should invalidate aggregate success scoring."""
        metrics = {
            'faithfulness': 0.9,
            'answer_relevance': float('nan'),
            'context_recall': 0.8,
            'context_precision': 0.7,
        }

        assert _has_complete_metrics(metrics) is False
        assert _calculate_ragas_score(metrics) != _calculate_ragas_score(metrics)

    def test_benchmark_stats_exclude_incomplete_metric_rows(self):
        """Benchmark aggregates must not count incomplete metric rows as successes."""
        evaluator = object.__new__(RAGEvaluator)

        stats = RAGEvaluator._calculate_benchmark_stats(
            evaluator,
            [
                {
                    'status': 'success',
                    'metrics': {
                        'faithfulness': 0.8,
                        'answer_relevance': 0.7,
                        'context_recall': 0.9,
                        'context_precision': 0.6,
                    },
                    'ragas_score': 0.75,
                },
                {
                    'status': 'incomplete',
                    'metrics': {
                        'faithfulness': 1.0,
                        'answer_relevance': 1.0,
                        'context_recall': float('nan'),
                        'context_precision': 1.0,
                    },
                    'ragas_score': float('nan'),
                },
                {
                    'status': 'error',
                    'metrics': {},
                    'ragas_score': 0,
                },
            ],
        )

        assert stats['total_tests'] == 3
        assert stats['successful_tests'] == 1
        assert stats['incomplete_tests'] == 1
        assert stats['failed_tests'] == 2
        assert stats['average_metrics']['ragas_score'] == 0.75
        assert stats['max_ragas_score'] == 0.75
        assert stats['min_ragas_score'] == 0.75