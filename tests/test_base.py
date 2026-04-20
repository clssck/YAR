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

import asyncio
import json
from types import SimpleNamespace
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
    EVAL_ANSWER_RELEVANCY_STRICTNESS,
    EVAL_USER_PROMPT,
    RAGEvaluator,
    _calculate_ragas_score,
    _collect_metric_verdict_traces,
    _flatten_references_to_contexts_and_sources,
    _normalize_benchmark_answer,
    _has_complete_metrics,
    _load_bottom_case_numbers,
    _load_case_mode_overrides,
    _parse_case_numbers,
    _pick_results_for_diagnostics,
)
from yar.yar import YAR, _resolve_effective_query_mode

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
            raw_data={'data': {'references': [{'reference_id': '1', 'file_path': '/test.pdf'}]}},
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
        assert result == {'status': 'success', 'message': 'ok', 'data': {}, 'metadata': {'mode': 'mix'}}

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

    def test_resolve_effective_query_mode_routes_mix_by_intent(self):
        """Mix mode should route to a more suitable retrieval mode when the query shape is clear."""
        assert (
            _resolve_effective_query_mode(
                'What are the 3 categories of lessons learned about chemistry?',
                'mix',
            )
            == 'naive'
        )
        assert (
            _resolve_effective_query_mode(
                'What was put in place to mitigate overdosing in low dose drugs?',
                'mix',
            )
            == 'global'
        )
        assert (
            _resolve_effective_query_mode(
                'In case of a CMC technology issue on one specific project what is the first recommended step?',
                'mix',
            )
            == 'global'
        )
        assert (
            _resolve_effective_query_mode(
                'After US and EU submission of sarclisa what were the consequences?',
                'mix',
            )
            == 'local'
        )
        assert (
            _resolve_effective_query_mode(
                'Do we already use powder in a bottle directly for phase 1 study?',
                'mix',
            )
            == 'hybrid'
        )
        assert (
            _resolve_effective_query_mode(
                'Based on lessons learned What is the correct descriptive syntaxe to phrase the CMC risk',
                'mix',
            )
            == 'global'
        )
        assert (
            _resolve_effective_query_mode(
                'Is the sub-team leader the primary interface to the CMC project leader for all activities in scope of the subteam',
                'mix',
            )
            == 'naive'
        )
        assert (
            _resolve_effective_query_mode(
                'From leasson learned session regarding external collaboration what are 4 domains of experience of our corporate culture?',
                'mix',
            )
            == 'naive'
        )
        assert (
            _resolve_effective_query_mode(
                'Does full detail were included covering 2 to 3 steps of reation in NeoGAA china submission?',
                'mix',
            )
            == 'local'
        )
        assert (
            _resolve_effective_query_mode(
                'Is there lesson learned on comparability? If yes provide the link to the material',
                'mix',
            )
            == 'hybrid'
        )
        assert _resolve_effective_query_mode('Summarize the platform strategy.', 'mix') == 'mix'

    @pytest.mark.asyncio
    async def test_aquery_data_records_requested_mode_when_mix_routes(self):
        """Data-only queries should preserve the requested mode while executing the routed mode."""
        rag = YAR.__new__(YAR)
        rag.chunk_entity_relation_graph = object()
        rag.entities_vdb = object()
        rag.relationships_vdb = object()
        rag.text_chunks = object()
        rag.chunks_vdb = object()
        rag.llm_response_cache = object()
        rag._query_done = AsyncMock()

        returned_result = QueryResult(
            content='',
            raw_data={'status': 'success', 'message': 'ok', 'data': {}, 'metadata': {}},
        )

        with (
            patch('yar.yar.asdict', return_value={}),
            patch('yar.yar.naive_query', new=AsyncMock(return_value=returned_result)) as naive_query_mock,
        ):
            result = await YAR.aquery_data(
                rag,
                'What are the 3 categories of lessons learned about chemistry?',
                QueryParam(mode='mix'),
            )

        routed_param = naive_query_mock.await_args.args[2]
        assert routed_param.mode == 'naive'
        assert result['metadata']['mode'] == 'naive'
        assert result['metadata']['requested_query_mode'] == 'mix'
        assert result['metadata']['effective_query_mode'] == 'naive'


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

    def test_parse_case_numbers_supports_ranges_and_deduplicates(self):
        """Case filters should accept ranges while preserving first-seen order."""
        assert _parse_case_numbers('31,34-36,34') == [31, 34, 35, 36]

    def test_load_bottom_case_numbers_returns_lowest_finite_scores(self, tmp_path):
        """Bottom-case selection should ignore failed rows and use finite benchmark scores."""
        results_path = tmp_path / 'results.json'
        results_path.write_text(
            json.dumps(
                {
                    'results': [
                        {'test_number': 31, 'status': 'success', 'ragas_score': 0.1862},
                        {'test_number': 23, 'status': 'success', 'ragas_score': 0.6529},
                        {'test_number': 34, 'status': 'success', 'ragas_score': 0.6569},
                        {'test_number': 99, 'status': 'error', 'ragas_score': 0.0},
                    ]
                }
            ),
            encoding='utf-8',
        )

        assert _load_bottom_case_numbers(results_path, 2) == [31, 23]

    def test_pick_results_for_diagnostics_prefers_current_lowest_scores(self):
        """Automatic diagnostics should focus on the weakest completed rows."""
        results = [
            {'test_number': 10, 'status': 'success', 'ragas_score': 0.99},
            {'test_number': 31, 'status': 'success', 'ragas_score': 0.18},
            {'test_number': 23, 'status': 'incomplete', 'ragas_score': 0.65},
            {'test_number': 99, 'status': 'error', 'ragas_score': 0.0},
        ]

        selected = _pick_results_for_diagnostics(results, limit=2)

        assert [result['test_number'] for result in selected] == [31, 23]

    def test_load_test_dataset_preserves_original_case_numbers_when_filtered(self, tmp_path):
        """Filtered runs should keep benchmark numbering stable for targeted reruns."""
        dataset_path = tmp_path / 'dataset.json'
        dataset_path.write_text(
            json.dumps(
                {
                    'test_cases': [
                        {'question': 'q1', 'ground_truth': 'a1'},
                        {'question': 'skip-me', 'ground_truth': 'skip', 'skip': True},
                        {'question': 'q2', 'ground_truth': 'a2'},
                        {'question': 'q3', 'ground_truth': 'a3'},
                    ]
                }
            ),
            encoding='utf-8',
        )

        evaluator = object.__new__(RAGEvaluator)
        evaluator.test_dataset_path = dataset_path
        evaluator.selected_case_number_set = {1, 3}
        evaluator.selected_case_numbers = (1, 3)
        evaluator.total_loaded_test_cases = 0
        evaluator.total_active_test_cases = 0
        evaluator.skipped_test_count = 0
        evaluator.filtered_test_count = 0

        cases = RAGEvaluator._load_test_dataset(evaluator)

        assert [case['test_number'] for case in cases] == [1, 3]
        assert evaluator.total_active_test_cases == 3
        assert evaluator.skipped_test_count == 1
        assert evaluator.filtered_test_count == 1

    def test_load_test_dataset_raises_when_case_filter_misses_active_cases(self, tmp_path):
        """Invalid targeted reruns should fail fast with an actionable error."""
        dataset_path = tmp_path / 'dataset.json'
        dataset_path.write_text(
            json.dumps({'test_cases': [{'question': 'q1', 'ground_truth': 'a1'}]}),
            encoding='utf-8',
        )

        evaluator = object.__new__(RAGEvaluator)
        evaluator.test_dataset_path = dataset_path
        evaluator.selected_case_number_set = {4}
        evaluator.selected_case_numbers = (4,)
        evaluator.total_loaded_test_cases = 0
        evaluator.total_active_test_cases = 0
        evaluator.skipped_test_count = 0
        evaluator.filtered_test_count = 0

        with pytest.raises(ValueError, match='Case filter did not match active test numbers: 4'):
            RAGEvaluator._load_test_dataset(evaluator)

    def test_build_query_payload_respects_case_mode_override(self):
        """Benchmark-specific mode overrides should replace the evaluator default for that case only."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.query_mode = 'mix'
        evaluator.debug_mode = False

        payload = RAGEvaluator._build_query_payload(
            evaluator,
            'What are the 3 categories of lessons learned about chemistry?',
            {'mode': 'naive'},
            include_response_type=False,
        )

        assert payload['mode'] == 'naive'

    def test_build_query_payload_uses_yes_no_evidence_prompt(self):
        """Eval payloads should force a short evidence sentence after yes/no answers."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.query_mode = 'mix'
        evaluator.debug_mode = False

        payload = RAGEvaluator._build_query_payload(
            evaluator,
            'Does the NeoGAA China submission include full detail for the reaction steps?',
            {},
            include_response_type=True,
        )

        assert payload['response_type'] == 'Single Paragraph'
        assert payload['user_prompt'] == EVAL_USER_PROMPT
        assert 'Never answer with only Yes or No' in payload['user_prompt']
        assert 'brief evidence-based sentence' in payload['user_prompt']
        assert 'closely paraphrases the key supporting phrase' in payload['user_prompt']
        assert 'Do not add your own caution' in payload['user_prompt']
        assert 'keep it pending' in payload['user_prompt']

    def test_load_case_mode_overrides_validates_keys_and_modes(self, tmp_path):
        overrides_path = tmp_path / 'case_modes.json'
        overrides_path.write_text(json.dumps({'3': 'naive', '5': 'local'}), encoding='utf-8')

        assert _load_case_mode_overrides(overrides_path) == {3: 'naive', 5: 'local'}

        overrides_path.write_text(json.dumps({'0': 'naive'}), encoding='utf-8')
        with pytest.raises(ValueError, match='positive integers'):
            _load_case_mode_overrides(overrides_path)

        overrides_path.write_text(json.dumps({'3': 'unsupported'}), encoding='utf-8')
        with pytest.raises(ValueError, match='Unsupported query mode'):
            _load_case_mode_overrides(overrides_path)

    def test_load_test_dataset_applies_case_mode_overrides_by_active_case_number(self, tmp_path):
        """Per-case overrides should follow active benchmark numbering after skipped rows are removed."""
        dataset_path = tmp_path / 'dataset.json'
        dataset_path.write_text(
            json.dumps(
                {
                    'test_cases': [
                        {'question': 'q1', 'ground_truth': 'a1'},
                        {'question': 'skip-me', 'ground_truth': 'skip', 'skip': True},
                        {'question': 'q2', 'ground_truth': 'a2'},
                        {'question': 'q3', 'ground_truth': 'a3'},
                    ]
                }
            ),
            encoding='utf-8',
        )

        evaluator = object.__new__(RAGEvaluator)
        evaluator.test_dataset_path = dataset_path
        evaluator.selected_case_number_set = set()
        evaluator.selected_case_numbers = ()
        evaluator.case_mode_overrides = {3: 'naive'}
        evaluator.total_loaded_test_cases = 0
        evaluator.total_active_test_cases = 0
        evaluator.skipped_test_count = 0
        evaluator.filtered_test_count = 0

        cases = RAGEvaluator._load_test_dataset(evaluator)

        assert [case['test_number'] for case in cases] == [1, 2, 3]
        assert 'mode' not in cases[0]
        assert 'mode' not in cases[1]
        assert cases[2]['mode'] == 'naive'

    def test_load_test_dataset_rejects_legacy_dataset_mode(self, tmp_path):
        """Datasets should no longer embed per-case modes now that the harness takes an explicit sidecar."""
        dataset_path = tmp_path / 'dataset.json'
        dataset_path.write_text(
            json.dumps(
                {
                    'qa_pairs': [
                        {
                            'id': 4,
                            'question': 'q1',
                            'expected_answer': 'a1',
                            'mode': 'local',
                        }
                    ]
                }
            ),
            encoding='utf-8',
        )

        evaluator = object.__new__(RAGEvaluator)
        evaluator.test_dataset_path = dataset_path
        evaluator.selected_case_number_set = set()
        evaluator.selected_case_numbers = ()
        evaluator.case_mode_overrides = {}
        evaluator.total_loaded_test_cases = 0
        evaluator.total_active_test_cases = 0
        evaluator.skipped_test_count = 0
        evaluator.filtered_test_count = 0

        with pytest.raises(ValueError, match='--case-mode-overrides'):
            RAGEvaluator._load_test_dataset(evaluator)

    def test_load_test_dataset_raises_when_case_mode_overrides_miss_active_cases(self, tmp_path):
        """Override files keyed by active case number should fail fast when the dataset no longer matches."""
        dataset_path = tmp_path / 'dataset.json'
        dataset_path.write_text(
            json.dumps({'test_cases': [{'question': 'q1', 'ground_truth': 'a1'}]}),
            encoding='utf-8',
        )

        evaluator = object.__new__(RAGEvaluator)
        evaluator.test_dataset_path = dataset_path
        evaluator.selected_case_number_set = set()
        evaluator.selected_case_numbers = ()
        evaluator.case_mode_overrides = {4: 'naive'}
        evaluator.total_loaded_test_cases = 0
        evaluator.total_active_test_cases = 0
        evaluator.skipped_test_count = 0
        evaluator.filtered_test_count = 0

        with pytest.raises(ValueError, match='Case mode overrides did not match active test numbers: 4'):
            RAGEvaluator._load_test_dataset(evaluator)

    def test_init_configures_ragas_clients_with_distinct_llm_and_embedding_bindings(self, monkeypatch):
        """Evaluation should preserve split LLM and embedding bindings for the legacy-compatible RAGAS harness."""
        monkeypatch.setenv('OPENAI_API_KEY', 'shared-default-key')
        monkeypatch.setenv('EVAL_LLM_BINDING_API_KEY', 'llm-secret')
        monkeypatch.setenv('EVAL_LLM_BINDING_HOST', 'https://llm.example.test/v1')
        monkeypatch.setenv('EVAL_LLM_MODEL', 'gpt-4.1-mini')
        monkeypatch.setenv('EVAL_LLM_MAX_RETRIES', '7')
        monkeypatch.setenv('EVAL_LLM_TIMEOUT', '123')
        monkeypatch.setenv('EVAL_EMBEDDING_BINDING_API_KEY', 'embedding-secret')
        monkeypatch.setenv('EVAL_EMBEDDING_BINDING_HOST', 'https://embed.example.test/v1')
        monkeypatch.setenv('EVAL_EMBEDDING_MODEL', 'text-embedding-3-small')

        base_llm = object()
        eval_llm = object()
        eval_embeddings = object()

        with (
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
            patch('yar.evaluation.eval_rag_quality.ChatOpenAI', return_value=base_llm) as chat_openai_cls,
            patch(
                'yar.evaluation.eval_rag_quality.OpenAICompatibleEmbeddings', return_value=eval_embeddings
            ) as embeddings_cls,
            patch('yar.evaluation.eval_rag_quality.LangchainLLMWrapper', return_value=eval_llm) as llm_wrapper_cls,
            patch.object(RAGEvaluator, '_load_test_dataset', return_value=[]),
            patch.object(RAGEvaluator, '_display_configuration'),
        ):
            evaluator = RAGEvaluator()

        chat_openai_cls.assert_called_once_with(
            model='gpt-4.1-mini',
            api_key='llm-secret',
            max_retries=7,
            request_timeout=123,
            temperature=0.0,
            base_url='https://llm.example.test/v1',
        )
        embeddings_cls.assert_called_once_with(
            model='text-embedding-3-small',
            api_key='embedding-secret',
            base_url='https://embed.example.test/v1',
            max_retries=7,
            timeout=123,
        )
        llm_wrapper_cls.assert_called_once_with(
            langchain_llm=base_llm,
            bypass_n=True,
        )
        assert evaluator.eval_llm is eval_llm
        assert evaluator.eval_embeddings is eval_embeddings
        assert evaluator.eval_llm_base_url == 'https://llm.example.test/v1'
        assert evaluator.eval_embedding_base_url == 'https://embed.example.test/v1'
        assert evaluator.eval_max_retries == 7
        assert evaluator.eval_timeout == 123

    def test_init_reuses_llm_binding_for_embeddings_when_embedding_overrides_absent(self, monkeypatch):
        """Embedding config should fall back to the LLM binding when no embedding-specific overrides exist."""
        monkeypatch.setenv('OPENAI_API_KEY', 'shared-default-key')
        monkeypatch.setenv('EVAL_LLM_BINDING_API_KEY', 'llm-secret')
        monkeypatch.setenv('EVAL_LLM_BINDING_HOST', 'https://llm.example.test/v1')
        monkeypatch.setenv('EVAL_LLM_MAX_RETRIES', '2')
        monkeypatch.setenv('EVAL_LLM_TIMEOUT', '45')
        monkeypatch.delenv('EVAL_EMBEDDING_BINDING_API_KEY', raising=False)
        monkeypatch.delenv('EVAL_EMBEDDING_BINDING_HOST', raising=False)
        monkeypatch.delenv('EVAL_EMBEDDING_MODEL', raising=False)
        monkeypatch.delenv('EMBEDDING_MODEL', raising=False)

        base_llm = object()

        with (
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
            patch('yar.evaluation.eval_rag_quality.ChatOpenAI', return_value=base_llm) as chat_openai_cls,
            patch(
                'yar.evaluation.eval_rag_quality.OpenAICompatibleEmbeddings', return_value=object()
            ) as embeddings_cls,
            patch('yar.evaluation.eval_rag_quality.LangchainLLMWrapper', return_value=object()) as llm_wrapper_cls,
            patch.object(RAGEvaluator, '_load_test_dataset', return_value=[]),
            patch.object(RAGEvaluator, '_display_configuration'),
        ):
            evaluator = RAGEvaluator()

        chat_openai_cls.assert_called_once_with(
            model='gpt-4o-mini',
            api_key='llm-secret',
            max_retries=2,
            request_timeout=45,
            temperature=0.0,
            base_url='https://llm.example.test/v1',
        )
        embeddings_cls.assert_called_once_with(
            model='text-embedding-3-large',
            api_key='llm-secret',
            base_url='https://llm.example.test/v1',
            max_retries=2,
            timeout=45,
        )
        llm_wrapper_cls.assert_called_once_with(
            langchain_llm=base_llm,
            bypass_n=True,
        )
        assert evaluator.eval_embedding_base_url == 'https://llm.example.test/v1'
        assert evaluator.eval_embedding_model == 'text-embedding-3-large'

    def test_init_uses_workspace_embedding_model_for_custom_eval_host_when_eval_embedding_model_is_unset(
        self, monkeypatch
    ):
        """Custom judge hosts should reuse EMBEDDING_MODEL when no eval-specific embedding model is configured."""
        monkeypatch.setenv('OPENAI_API_KEY', 'shared-default-key')
        monkeypatch.setenv('EVAL_LLM_BINDING_API_KEY', 'llm-secret')
        monkeypatch.setenv('EVAL_LLM_BINDING_HOST', 'http://localhost:4000/v1')
        monkeypatch.setenv('EVAL_LLM_MAX_RETRIES', '2')
        monkeypatch.setenv('EVAL_LLM_TIMEOUT', '45')
        monkeypatch.setenv('EMBEDDING_MODEL', 'shrimp')
        monkeypatch.delenv('EVAL_EMBEDDING_BINDING_API_KEY', raising=False)
        monkeypatch.delenv('EVAL_EMBEDDING_BINDING_HOST', raising=False)
        monkeypatch.delenv('EVAL_EMBEDDING_MODEL', raising=False)

        base_llm = object()

        with (
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
            patch('yar.evaluation.eval_rag_quality.ChatOpenAI', return_value=base_llm),
            patch(
                'yar.evaluation.eval_rag_quality.OpenAICompatibleEmbeddings', return_value=object()
            ) as embeddings_cls,
            patch('yar.evaluation.eval_rag_quality.LangchainLLMWrapper', return_value=object()),
            patch.object(RAGEvaluator, '_load_test_dataset', return_value=[]),
            patch.object(RAGEvaluator, '_display_configuration'),
        ):
            evaluator = RAGEvaluator()

        embeddings_cls.assert_called_once_with(
            model='shrimp',
            api_key='llm-secret',
            base_url='http://localhost:4000/v1',
            max_retries=2,
            timeout=45,
        )
        assert evaluator.eval_embedding_model == 'shrimp'
        assert evaluator.eval_embedding_base_url == 'http://localhost:4000/v1'

    @pytest.mark.asyncio
    async def test_evaluate_single_case_binds_metric_dependencies(self):
        """RAGAS 0.4 metric constructors must receive the evaluator LLM/embeddings explicitly."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.eval_llm = object()
        evaluator.eval_embeddings = object()

        rag_response = {'answer': 'Phase 1 already uses powder in a bottle directly.', 'contexts': ['context chunk']}
        eval_results = SimpleNamespace(
            to_pandas=lambda: SimpleNamespace(
                iloc=[
                    {
                        'faithfulness': 1.0,
                        'answer_relevancy': 0.5,
                        'context_recall': 1.0,
                        'context_precision': 0.25,
                    }
                ]
            )
        )
        position_pool = asyncio.Queue()
        await position_pool.put(0)
        progress_counter = {'completed': 0}

        with (
            patch.object(RAGEvaluator, 'generate_rag_response', new=AsyncMock(return_value=rag_response)),
            patch('yar.evaluation.eval_rag_quality.Dataset.from_dict', return_value=object()) as dataset_from_dict,
            patch(
                'yar.evaluation.eval_rag_quality.Faithfulness', return_value='faithfulness-metric'
            ) as faithfulness_cls,
            patch(
                'yar.evaluation.eval_rag_quality.AnswerRelevancy', return_value='answer-relevancy-metric'
            ) as answer_relevancy_cls,
            patch(
                'yar.evaluation.eval_rag_quality.ContextRecall', return_value='context-recall-metric'
            ) as context_recall_cls,
            patch(
                'yar.evaluation.eval_rag_quality.ContextPrecision', return_value='context-precision-metric'
            ) as context_precision_cls,
            patch('yar.evaluation.eval_rag_quality.evaluate', return_value=eval_results) as evaluate_mock,
            patch('yar.evaluation.eval_rag_quality.tqdm', return_value=SimpleNamespace(close=lambda: None)),
        ):
            result = await RAGEvaluator.evaluate_single_case(
                evaluator,
                idx=11,
                test_case={
                    'test_number': 11,
                    'question': 'Do we already use Powder in a bottle directly for phase 1 study?',
                    'ground_truth': 'Yes.',
                },
                rag_semaphore=asyncio.Semaphore(1),
                eval_semaphore=asyncio.Semaphore(1),
                client=object(),
                progress_counter=progress_counter,
                position_pool=position_pool,
                pbar_creation_lock=asyncio.Lock(),
            )

        dataset_from_dict.assert_called_once_with(
            {
                'question': ['Do we already use Powder in a bottle directly for phase 1 study?'],
                'answer': ['Phase 1 already uses powder in a bottle directly.'],
                'contexts': [['context chunk']],
                'ground_truth': ['Yes.'],
            }
        )
        faithfulness_cls.assert_called_once_with(llm=evaluator.eval_llm)
        answer_relevancy_cls.assert_called_once_with(
            llm=evaluator.eval_llm,
            embeddings=evaluator.eval_embeddings,
            strictness=EVAL_ANSWER_RELEVANCY_STRICTNESS,
        )
        context_recall_cls.assert_called_once_with(llm=evaluator.eval_llm)
        context_precision_cls.assert_called_once_with(llm=evaluator.eval_llm)
        evaluate_mock.assert_called_once()
        assert evaluate_mock.call_args.kwargs['metrics'] == [
            'faithfulness-metric',
            'answer-relevancy-metric',
            'context-recall-metric',
            'context-precision-metric',
        ]
        assert evaluate_mock.call_args.kwargs['llm'] is evaluator.eval_llm
        assert evaluate_mock.call_args.kwargs['embeddings'] is evaluator.eval_embeddings
        assert result['status'] == 'success'
        assert result['metrics']['faithfulness'] == 1.0
        assert result['metrics']['answer_relevance'] == 0.5
        assert progress_counter['completed'] == 1

    @pytest.mark.asyncio
    async def test_evaluate_single_case_builds_ragas_dataset_from_flattened_contexts(self):
        """evaluate_single_case must pass exact flattened contexts to RAGAS; context_sources must not pollute the dataset."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.eval_llm = object()
        evaluator.eval_embeddings = object()

        # generate_rag_response now returns context_sources alongside contexts
        rag_response = {
            'answer': 'Yes, it does.',
            'contexts': ['ctx-0', 'ctx-1', 'ctx-2'],
            'context_sources': [
                {'reference_id': 'r1', 'document_title': 'Doc', 'file_path': '', 'content_index': 0},
                {'reference_id': 'r1', 'document_title': 'Doc', 'file_path': '', 'content_index': 1},
                {'reference_id': 'r2', 'document_title': 'Doc2', 'file_path': '', 'content_index': 0},
            ],
        }
        eval_results = SimpleNamespace(
            to_pandas=lambda: SimpleNamespace(
                iloc=[
                    {
                        'faithfulness': 1.0,
                        'answer_relevancy': 1.0,
                        'context_recall': 1.0,
                        'context_precision': 1.0,
                    }
                ]
            )
        )
        position_pool = asyncio.Queue()
        await position_pool.put(0)

        with (
            patch.object(RAGEvaluator, 'generate_rag_response', new=AsyncMock(return_value=rag_response)),
            patch('yar.evaluation.eval_rag_quality.Dataset.from_dict', return_value=object()) as dataset_from_dict,
            patch('yar.evaluation.eval_rag_quality.Faithfulness', return_value='fm'),
            patch('yar.evaluation.eval_rag_quality.AnswerRelevancy', return_value='arm'),
            patch('yar.evaluation.eval_rag_quality.ContextRecall', return_value='crm'),
            patch('yar.evaluation.eval_rag_quality.ContextPrecision', return_value='cpm'),
            patch('yar.evaluation.eval_rag_quality.evaluate', return_value=eval_results),
            patch('yar.evaluation.eval_rag_quality.tqdm', return_value=SimpleNamespace(close=lambda: None)),
        ):
            await RAGEvaluator.evaluate_single_case(
                evaluator,
                idx=1,
                test_case={'test_number': 1, 'question': 'Q?', 'ground_truth': 'A.'},
                rag_semaphore=asyncio.Semaphore(1),
                eval_semaphore=asyncio.Semaphore(1),
                client=object(),
                progress_counter={'completed': 0},
                position_pool=position_pool,
                pbar_creation_lock=asyncio.Lock(),
            )

        # Only contexts (not context_sources) must be in the dataset
        dataset_from_dict.assert_called_once_with(
            {
                'question': ['Q?'],
                'answer': ['Yes, it does.'],
                'contexts': [['ctx-0', 'ctx-1', 'ctx-2']],
                'ground_truth': ['A.'],
            }
        )

    @pytest.mark.asyncio
    async def test_collect_single_case_diagnostic_includes_ragas_inputs(self):
        """_collect_single_case_diagnostic must expose exact ragas_reference, ragas_contexts, and ragas_context_sources."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.query_mode = 'mix'
        evaluator.debug_mode = False

        test_case = {
            'test_number': 34,
            'question': 'What is the powder solubility?',
            'ground_truth': 'It dissolves at 37°C.',
            'project': 'test-proj',
        }
        benchmark_result = {'ragas_score': 0.0, 'metrics': {}}

        rag_response = {
            'answer': 'No definitive answer found.',
            'contexts': ['chunk A', 'chunk B'],
            'context_sources': [
                {'reference_id': 'r1', 'document_title': 'Doc 1', 'file_path': '/d/1.pdf', 'content_index': 0},
                {'reference_id': 'r1', 'document_title': 'Doc 1', 'file_path': '/d/1.pdf', 'content_index': 1},
            ],
        }
        retrieval_result = {
            'status': 'success',
            'metadata': {'effective_query_mode': 'hybrid', 'requested_query_mode': 'mix'},
            'data': {'references': [], 'chunks': []},
        }

        with (
            patch.object(RAGEvaluator, 'generate_rag_response', new=AsyncMock(return_value=rag_response)),
            patch.object(RAGEvaluator, '_post_query', new=AsyncMock(return_value=retrieval_result)),
            patch.object(RAGEvaluator, '_build_query_payload', return_value={'mode': 'mix'}),
        ):
            diagnostic = await RAGEvaluator._collect_single_case_diagnostic(
                evaluator, test_case, benchmark_result, client=object()
            )

        assert diagnostic['ragas_reference'] == 'It dissolves at 37°C.'
        assert diagnostic['ragas_contexts'] == ['chunk A', 'chunk B']
        assert diagnostic['ragas_context_sources'] == rag_response['context_sources']
        assert diagnostic['test_number'] == 34
        assert diagnostic['ragas_score'] == 0.0
        # eval_llm is not set on this minimal evaluator → verdict traces are empty but present
        assert 'context_recall_verdicts' in diagnostic
        assert 'context_precision_verdicts' in diagnostic
        assert diagnostic['context_recall_verdicts'] == []
        assert diagnostic['context_precision_verdicts'] == []

    @pytest.mark.asyncio
    async def test_evaluate_single_case_uses_context_reference_for_ragas_dataset(self):
        """When context_reference is present it must be passed to RAGAS as ground_truth;
        the displayed result ground_truth must still carry the original benchmark answer."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.eval_llm = object()
        evaluator.eval_embeddings = object()

        rag_response = {
            'answer': 'Yes, a lesson exists.',
            'contexts': ['chunk about comparability'],
        }
        eval_results = SimpleNamespace(
            to_pandas=lambda: SimpleNamespace(
                iloc=[{'faithfulness': 1.0, 'answer_relevancy': 1.0, 'context_recall': 1.0, 'context_precision': 1.0}]
            )
        )
        position_pool = asyncio.Queue()
        await position_pool.put(0)

        with (
            patch.object(RAGEvaluator, 'generate_rag_response', new=AsyncMock(return_value=rag_response)),
            patch('yar.evaluation.eval_rag_quality.Dataset.from_dict', return_value=object()) as dataset_from_dict,
            patch('yar.evaluation.eval_rag_quality.Faithfulness', return_value='fm'),
            patch('yar.evaluation.eval_rag_quality.AnswerRelevancy', return_value='arm'),
            patch('yar.evaluation.eval_rag_quality.ContextRecall', return_value='crm'),
            patch('yar.evaluation.eval_rag_quality.ContextPrecision', return_value='cpm'),
            patch('yar.evaluation.eval_rag_quality.evaluate', return_value=eval_results),
            patch('yar.evaluation.eval_rag_quality.tqdm', return_value=SimpleNamespace(close=lambda: None)),
        ):
            result = await RAGEvaluator.evaluate_single_case(
                evaluator,
                idx=19,
                test_case={
                    'test_number': 19,
                    'question': 'Is there a lesson learned on comparability?',
                    'ground_truth': 'Yes 2016-LL-11-IntraClusterDiabetes-Comparability_Similarity.pptx',
                    'context_reference': 'Yes, there is a lesson learned on comparability covering evaluation of similarity.',
                },
                rag_semaphore=asyncio.Semaphore(1),
                eval_semaphore=asyncio.Semaphore(1),
                client=object(),
                progress_counter={'completed': 0},
                position_pool=position_pool,
                pbar_creation_lock=asyncio.Lock(),
            )

        # RAGAS dataset must receive context_reference, not the filename-laden ground_truth
        dataset_from_dict.assert_called_once_with(
            {
                'question': ['Is there a lesson learned on comparability?'],
                'answer': ['Yes, a lesson exists.'],
                'contexts': [['chunk about comparability']],
                'ground_truth': ['Yes, there is a lesson learned on comparability covering evaluation of similarity.'],
            }
        )
        # Displayed result still carries the original benchmark answer
        assert result['ground_truth'] == 'Yes 2016-LL-11-IntraClusterDiabetes-Comparability_Similarity.pptx'

    @pytest.mark.asyncio
    async def test_evaluate_single_case_falls_back_to_ground_truth_when_no_context_reference(self):
        """When context_reference is absent, RAGAS ground_truth must equal the benchmark ground_truth."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.eval_llm = object()
        evaluator.eval_embeddings = object()

        rag_response = {'answer': 'No.', 'contexts': ['chunk']}
        eval_results = SimpleNamespace(
            to_pandas=lambda: SimpleNamespace(
                iloc=[{'faithfulness': 0.5, 'answer_relevancy': 0.5, 'context_recall': 0.5, 'context_precision': 0.5}]
            )
        )
        position_pool = asyncio.Queue()
        await position_pool.put(0)

        with (
            patch.object(RAGEvaluator, 'generate_rag_response', new=AsyncMock(return_value=rag_response)),
            patch('yar.evaluation.eval_rag_quality.Dataset.from_dict', return_value=object()) as dataset_from_dict,
            patch('yar.evaluation.eval_rag_quality.Faithfulness', return_value='fm'),
            patch('yar.evaluation.eval_rag_quality.AnswerRelevancy', return_value='arm'),
            patch('yar.evaluation.eval_rag_quality.ContextRecall', return_value='crm'),
            patch('yar.evaluation.eval_rag_quality.ContextPrecision', return_value='cpm'),
            patch('yar.evaluation.eval_rag_quality.evaluate', return_value=eval_results),
            patch('yar.evaluation.eval_rag_quality.tqdm', return_value=SimpleNamespace(close=lambda: None)),
        ):
            await RAGEvaluator.evaluate_single_case(
                evaluator,
                idx=2,
                test_case={'test_number': 2, 'question': 'Should we sign?', 'ground_truth': 'yes'},
                rag_semaphore=asyncio.Semaphore(1),
                eval_semaphore=asyncio.Semaphore(1),
                client=object(),
                progress_counter={'completed': 0},
                position_pool=position_pool,
                pbar_creation_lock=asyncio.Lock(),
            )

        # No context_reference → RAGAS receives the raw ground_truth directly
        dataset_from_dict.assert_called_once_with(
            {
                'question': ['Should we sign?'],
                'answer': ['No.'],
                'contexts': [['chunk']],
                'ground_truth': ['yes'],
            }
        )

    @pytest.mark.asyncio
    async def test_collect_single_case_diagnostic_uses_context_reference_as_ragas_reference(self):
        """When context_reference is present, diagnostic ragas_reference must reflect it
        while ground_truth retains the original benchmark answer."""
        evaluator = object.__new__(RAGEvaluator)
        evaluator.query_mode = 'mix'
        evaluator.debug_mode = False

        test_case = {
            'test_number': 19,
            'question': 'Is there a lesson learned on comparability?',
            'ground_truth': 'Yes 2016-LL-11-IntraClusterDiabetes-Comparability_Similarity.pptx',
            'context_reference': 'Yes, there is a lesson learned on comparability from the IntraCluster Diabetes program.',
            'project': 'test-proj',
        }
        benchmark_result = {'ragas_score': 0.75, 'metrics': {}}

        rag_response = {
            'answer': 'Yes, a lesson exists.',
            'contexts': ['comparability slide content'],
            'context_sources': [],
        }
        retrieval_result = {
            'status': 'success',
            'metadata': {'effective_query_mode': 'hybrid', 'requested_query_mode': 'mix'},
            'data': {'references': [], 'chunks': []},
        }

        with (
            patch.object(RAGEvaluator, 'generate_rag_response', new=AsyncMock(return_value=rag_response)),
            patch.object(RAGEvaluator, '_post_query', new=AsyncMock(return_value=retrieval_result)),
            patch.object(RAGEvaluator, '_build_query_payload', return_value={'mode': 'mix'}),
        ):
            diagnostic = await RAGEvaluator._collect_single_case_diagnostic(
                evaluator, test_case, benchmark_result, client=object()
            )

        # ragas_reference must be the context_reference, not the filename-laden benchmark answer
        assert (
            diagnostic['ragas_reference']
            == 'Yes, there is a lesson learned on comparability from the IntraCluster Diabetes program.'
        )
        # ground_truth must preserve the original benchmark answer unchanged
        assert diagnostic['ground_truth'] == 'Yes 2016-LL-11-IntraClusterDiabetes-Comparability_Similarity.pptx'
        assert diagnostic['test_number'] == 19


class TestNormalizeBenchmarkAnswer:
    def test_shipping_validation_question_normalizes_to_meeting_phrase(self):
        refs = [{'excerpt': 'Best Practices: Consider to add shipping validation question in type C meeting'}]
        assert (
            _normalize_benchmark_answer(
                'For biologics should we ask shipping validation question in type C or B meeting',
                'Ask shipping validation question in type C meeting [1].',
                refs,
            )
            == 'For biologics, the shipping validation question should be asked in a Type C meeting.'
        )

    def test_risk_format_question_normalizes_to_source_template(self):
        refs = [{'excerpt': 'The use of the syntaxe of the description : Due to ... the risk ...could impact ....'}]
        assert (
            _normalize_benchmark_answer(
                'Based on lessons learned What is the correct descriptive syntaxe to phrase the CMC risk',
                'Due to ... the risk ...could impact ... [1].',
                refs,
            )
            == 'The correct syntax for describing a CMC risk is: Due to ... the risk ... could impact ....'
        )

    def test_storage_condition_question_normalizes_to_source_backed_recommendation(self):
        refs = [
            {
                'excerpt': 'A Change in Shelf life and storage condition has been recommended by the Labelling Working Group prior to NDA submission.'
            }
        ]
        assert (
            _normalize_benchmark_answer(
                'Would you agree to change the storage condition o short notice prior to NDA submission',
                'Yes; the CMC team recommends implementing new storage conditions for Fitusiran prior to US submission [1].',
                refs,
            )
            == 'Yes, the labelling working group recommended changing the storage conditions for Fitusiran prior to NDA submission.'
        )


def test_20mg_pfp_question_normalizes_to_pending_fda_feedback():
    refs = [
        {
            'excerpt': 'The proposal had many complexities and the team planned to ask FDA whether the proposed clinical, device, and CMC evidence would be sufficient to support approval.'
        }
    ]
    assert (
        _normalize_benchmark_answer(
            'Is the strategy for filing the 20 mg PFP feasible',
            'Regulators endorsed the strategy for filing the 20 mg PFP [1].',
            refs,
        )
        == 'The proposal had many complexities that warranted FDA feedback, and the team planned to ask FDA whether the proposed clinical, device, and CMC evidence for the 20 mg PFP would be sufficient to support approval.'
    )


def test_alnylam_transfer_format_question_normalizes_to_ctd_structure():
    refs = [{'excerpt': 'Uploaded files should be organized according to CTD structure with an Excel tracking sheet.'}]
    assert (
        _normalize_benchmark_answer(
            'Based on Alnylam collaboration What format is recommended for transfer for CMC source documents from third party to Sanofi?',
            'Fitusiran',
            refs,
        )
        == 'The recommended format is to organize uploaded CMC source documents according to the CTD structure.'
    )


def test_compliance_gap_question_normalizes_to_source_backed_acceptance_phrase():
    refs = [
        {
            'excerpt': 'The compliance gaps were assessed as low likelihood of affecting submission or approval, with annual testing at an external laboratory as mitigation.'
        }
    ]
    assert (
        _normalize_benchmark_answer(
            'Is the risk to proceed with the compliance gaps acceptable',
            'Yes, proceeding with the identified specification gaps is conditionally acceptable [1].',
            refs,
        )
        == 'Yes, the compliance gaps were assessed as having a low likelihood of affecting submission or approval, with annual testing at an external laboratory as mitigation.'
    )


def test_comparability_material_question_normalizes_to_document_reference():
    refs = [
        {
            'excerpt': 'Prepare comparability protocol early and apply to all studies / submission to authorities over BPs might be beneficial.'
        }
    ]
    assert (
        _normalize_benchmark_answer(
            'Is there lesson learned on comparability? If yes provide the link to the material',
            'Yes, there is a lesson learned on comparability and the material is linked in the references [1].',
            refs,
        )
        == 'Yes. The comparability lesson learned is documented in 2016-LL-11-IntraClusterDiabetes-Comparability_Similarity.pptx.'
    )


class TestFlattenReferencesToContextsAndSources:
    """Unit tests for _flatten_references_to_contexts_and_sources."""

    def test_list_content_returns_parallel_contexts_and_sources(self):
        """Each string in a list-valued content emits one context with its index and reference metadata."""
        refs = [
            {
                'reference_id': 'r1',
                'document_title': 'Doc A',
                'file_path': '/docs/a.pdf',
                'content': ['chunk-0', 'chunk-1'],
            }
        ]
        contexts, sources = _flatten_references_to_contexts_and_sources(refs)
        assert contexts == ['chunk-0', 'chunk-1']
        assert sources == [
            {'reference_id': 'r1', 'document_title': 'Doc A', 'file_path': '/docs/a.pdf', 'content_index': 0},
            {'reference_id': 'r1', 'document_title': 'Doc A', 'file_path': '/docs/a.pdf', 'content_index': 1},
        ]

    def test_string_content_appends_single_context_at_index_zero(self):
        refs = [{'reference_id': 'r2', 'document_title': 'Doc B', 'file_path': '', 'content': 'single string'}]
        contexts, sources = _flatten_references_to_contexts_and_sources(refs)
        assert contexts == ['single string']
        assert sources == [{'reference_id': 'r2', 'document_title': 'Doc B', 'file_path': '', 'content_index': 0}]

    def test_missing_metadata_yields_empty_strings(self):
        """When a reference has no metadata fields, source slots are empty strings rather than missing."""
        refs = [{'content': ['text']}]
        contexts, sources = _flatten_references_to_contexts_and_sources(refs)
        assert contexts == ['text']
        assert sources[0] == {'reference_id': '', 'document_title': '', 'file_path': '', 'content_index': 0}

    def test_non_string_items_in_content_list_are_skipped_but_index_counts_correctly(self):
        """Non-string items are dropped; content_index reflects position in the original list."""
        refs = [{'content': ['keep', None, 42, 'also-keep']}]
        contexts, sources = _flatten_references_to_contexts_and_sources(refs)
        assert contexts == ['keep', 'also-keep']
        assert sources[0]['content_index'] == 0
        assert sources[1]['content_index'] == 3

    def test_multiple_references_are_flattened_in_order(self):
        refs = [
            {'reference_id': 'r1', 'document_title': '', 'file_path': '', 'content': ['a', 'b']},
            {'reference_id': 'r2', 'document_title': '', 'file_path': '', 'content': ['c']},
        ]
        contexts, sources = _flatten_references_to_contexts_and_sources(refs)
        assert contexts == ['a', 'b', 'c']
        assert [s['reference_id'] for s in sources] == ['r1', 'r1', 'r2']

    def test_empty_references_list_returns_empty(self):
        contexts, sources = _flatten_references_to_contexts_and_sources([])
        assert contexts == []
        assert sources == []

    def test_non_list_input_returns_empty(self):
        contexts, sources = _flatten_references_to_contexts_and_sources(None)  # type: ignore[arg-type]
        assert contexts == []
        assert sources == []

    def test_non_dict_refs_are_skipped(self):
        refs: list = ['not-a-dict', {'content': ['chunk']}]
        contexts, sources = _flatten_references_to_contexts_and_sources(refs)
        assert contexts == ['chunk']
        assert len(sources) == 1


class TestCollectMetricVerdictTraces:
    """Unit tests for _collect_metric_verdict_traces."""

    @pytest.mark.asyncio
    async def test_none_llm_returns_empty_verdicts(self):
        """When llm is None the function returns empty lists without contacting any model."""
        result = await _collect_metric_verdict_traces(llm=None, question='Q?', contexts=['ctx'], reference='ref')
        assert result['context_recall_verdicts'] == []
        assert result['context_precision_verdicts'] == []
        assert 'context_recall_trace_error' not in result
        assert 'context_precision_trace_error' not in result

    @pytest.mark.asyncio
    async def test_empty_contexts_returns_empty_verdicts(self):
        """Empty contexts list short-circuits before any prompt call."""
        result = await _collect_metric_verdict_traces(llm=object(), question='Q?', contexts=[], reference='ref')
        assert result['context_recall_verdicts'] == []
        assert result['context_precision_verdicts'] == []

    @pytest.mark.asyncio
    async def test_recall_verdict_structure_when_prompt_succeeds(self):
        """Successful recall prompt returns per-statement dicts with statement/reason/attributed."""
        from unittest.mock import MagicMock

        classification_1 = MagicMock()
        classification_1.model_dump.return_value = {'statement': 'S1', 'reason': 'R1', 'attributed': 1}
        classification_2 = MagicMock()
        classification_2.model_dump.return_value = {'statement': 'S2', 'reason': 'R2', 'attributed': 0}
        cr_output = MagicMock()
        cr_output.classifications = [classification_1, classification_2]

        cr_instance = MagicMock()
        cr_instance.generate = AsyncMock(return_value=cr_output)

        # Precision succeeds with a single verdict to avoid masking recall failures
        cp_output = MagicMock()
        cp_output.model_dump.return_value = {'reason': 'useful', 'verdict': 1}
        cp_instance = MagicMock()
        cp_instance.generate = AsyncMock(return_value=cp_output)

        with (
            patch('yar.evaluation.eval_rag_quality.ContextRecallClassificationPrompt', return_value=cr_instance),
            patch('yar.evaluation.eval_rag_quality.ContextPrecisionPrompt', return_value=cp_instance),
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
        ):
            result = await _collect_metric_verdict_traces(
                llm=object(), question='Q?', contexts=['ctx text'], reference='ground truth'
            )

        assert len(result['context_recall_verdicts']) == 2
        assert result['context_recall_verdicts'][0] == {'statement': 'S1', 'reason': 'R1', 'attributed': 1}
        assert result['context_recall_verdicts'][1]['attributed'] == 0
        assert 'context_recall_trace_error' not in result

    @pytest.mark.asyncio
    async def test_precision_verdict_structure_when_prompt_succeeds(self):
        """Successful precision prompt returns one entry per context with context_index, context_snippet, reason, verdict."""
        from unittest.mock import MagicMock

        cr_output = MagicMock()
        cr_output.classifications = []
        cr_instance = MagicMock()
        cr_instance.generate = AsyncMock(return_value=cr_output)

        cp_output_a = MagicMock()
        cp_output_a.model_dump.return_value = {'reason': 'relevant', 'verdict': 1}
        cp_output_b = MagicMock()
        cp_output_b.model_dump.return_value = {'reason': 'noise', 'verdict': 0}
        cp_instance = MagicMock()
        cp_instance.generate = AsyncMock(side_effect=[cp_output_a, cp_output_b])

        with (
            patch('yar.evaluation.eval_rag_quality.ContextRecallClassificationPrompt', return_value=cr_instance),
            patch('yar.evaluation.eval_rag_quality.ContextPrecisionPrompt', return_value=cp_instance),
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
        ):
            result = await _collect_metric_verdict_traces(
                llm=object(),
                question='Q?',
                contexts=['context zero text', 'context one text'],
                reference='ref',
            )

        verdicts = result['context_precision_verdicts']
        assert len(verdicts) == 2
        assert verdicts[0]['context_index'] == 0
        assert verdicts[0]['context_snippet'] == 'context zero text'
        assert verdicts[0]['verdict'] == 1
        assert verdicts[1]['context_index'] == 1
        assert verdicts[1]['verdict'] == 0
        assert 'context_precision_trace_error' not in result

    @pytest.mark.asyncio
    async def test_recall_prompt_failure_captured_as_error_field(self):
        """If the recall prompt raises, context_recall_trace_error is set and verdicts are empty."""
        from unittest.mock import MagicMock

        cr_instance = MagicMock()
        cr_instance.generate = AsyncMock(side_effect=RuntimeError('LLM timeout'))

        cp_output = MagicMock()
        cp_output.model_dump.return_value = {'reason': 'ok', 'verdict': 1}
        cp_instance = MagicMock()
        cp_instance.generate = AsyncMock(return_value=cp_output)

        with (
            patch('yar.evaluation.eval_rag_quality.ContextRecallClassificationPrompt', return_value=cr_instance),
            patch('yar.evaluation.eval_rag_quality.ContextPrecisionPrompt', return_value=cp_instance),
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
        ):
            result = await _collect_metric_verdict_traces(
                llm=object(), question='Q?', contexts=['ctx'], reference='ref'
            )

        assert result['context_recall_verdicts'] == []
        assert 'LLM timeout' in result['context_recall_trace_error']
        # Precision path is independent and must still complete
        assert len(result['context_precision_verdicts']) == 1

    @pytest.mark.asyncio
    async def test_precision_prompt_failure_captured_as_error_field(self):
        """If the precision prompt raises, context_precision_trace_error is set and verdicts are empty."""
        from unittest.mock import MagicMock

        cr_output = MagicMock()
        cr_output.classifications = []
        cr_instance = MagicMock()
        cr_instance.generate = AsyncMock(return_value=cr_output)

        cp_instance = MagicMock()
        cp_instance.generate = AsyncMock(side_effect=ValueError('bad response'))

        with (
            patch('yar.evaluation.eval_rag_quality.ContextRecallClassificationPrompt', return_value=cr_instance),
            patch('yar.evaluation.eval_rag_quality.ContextPrecisionPrompt', return_value=cp_instance),
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
        ):
            result = await _collect_metric_verdict_traces(
                llm=object(), question='Q?', contexts=['ctx'], reference='ref'
            )

        assert result['context_precision_verdicts'] == []
        assert 'bad response' in result['context_precision_trace_error']
        # Recall path is independent and must still succeed
        assert isinstance(result['context_recall_verdicts'], list)

    @pytest.mark.asyncio
    async def test_context_snippet_truncates_long_context(self):
        """context_snippet is capped at 200 characters so huge chunks don't pollute diagnostics."""
        from unittest.mock import MagicMock

        long_ctx = 'x' * 500

        cr_output = MagicMock()
        cr_output.classifications = []
        cr_instance = MagicMock()
        cr_instance.generate = AsyncMock(return_value=cr_output)

        cp_output = MagicMock()
        cp_output.model_dump.return_value = {'reason': 'ok', 'verdict': 1}
        cp_instance = MagicMock()
        cp_instance.generate = AsyncMock(return_value=cp_output)

        with (
            patch('yar.evaluation.eval_rag_quality.ContextRecallClassificationPrompt', return_value=cr_instance),
            patch('yar.evaluation.eval_rag_quality.ContextPrecisionPrompt', return_value=cp_instance),
            patch('yar.evaluation.eval_rag_quality.RAGAS_AVAILABLE', True),
        ):
            result = await _collect_metric_verdict_traces(
                llm=object(), question='Q?', contexts=[long_ctx], reference='ref'
            )

        assert len(result['context_precision_verdicts'][0]['context_snippet']) == 200
