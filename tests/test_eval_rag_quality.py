from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yar.evaluation.eval_rag_quality as eval_rag_quality
from yar.evaluation.eval_rag_quality import (
    RAGEvaluator,
    _calculate_retrieval_metrics,
    _collect_expected_documents,
    _extract_retrieved_documents,
    _references_from_chunks,
    _relationship_evidence_to_contexts_and_sources,
    _render_case_diagnostics_markdown,
)
from yar.tracing import TraceConfig, TraceManager


def test_eval_query_payload_disables_cache_by_default(tmp_path: Path, monkeypatch):
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(
        json.dumps({'test_cases': [{'question': 'Q?', 'ground_truth': 'A.'}]}),
        encoding='utf-8',
    )
    monkeypatch.delenv('EVAL_DISABLE_CACHE', raising=False)

    evaluator = RAGEvaluator(test_dataset_path=dataset, retrieval_only=True)

    payload = evaluator._build_query_payload('Q?', include_response_type=False)

    assert payload['disable_cache'] is True


def test_eval_query_payload_allows_cache_when_requested(tmp_path: Path, monkeypatch):
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(
        json.dumps({'test_cases': [{'question': 'Q?', 'ground_truth': 'A.'}]}),
        encoding='utf-8',
    )
    monkeypatch.setenv('EVAL_DISABLE_CACHE', 'false')

    evaluator = RAGEvaluator(test_dataset_path=dataset, retrieval_only=True)

    payload = evaluator._build_query_payload('Q?', include_response_type=False)

    assert payload['disable_cache'] is False


def test_evaluator_tracing_is_off_by_default(tmp_path: Path, monkeypatch):
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(
        json.dumps({'test_cases': [{'question': 'Q?', 'ground_truth': 'A.'}]}),
        encoding='utf-8',
    )
    monkeypatch.delenv('YAR_EVAL_TRACE_ENABLED', raising=False)
    monkeypatch.setenv('YAR_TRACE_ENABLED', 'true')

    evaluator = RAGEvaluator(test_dataset_path=dataset, retrieval_only=True)

    assert evaluator.tracing.active is False
    assert evaluator.tracing.config.project_name == 'yar-eval'


def test_evaluator_tracing_activates_when_env_set(tmp_path: Path, monkeypatch):
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(
        json.dumps({'test_cases': [{'question': 'Q?', 'ground_truth': 'A.'}]}),
        encoding='utf-8',
    )
    tracing = TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-eval',
            endpoint=None,
        ),
        tracer=object(),
    )
    monkeypatch.setattr(eval_rag_quality.TraceManager, 'from_env', lambda **_kwargs: tracing)
    monkeypatch.setenv('YAR_EVAL_TRACE_ENABLED', 'true')
    monkeypatch.delenv('YAR_TRACE_PROJECT', raising=False)
    monkeypatch.delenv('PHOENIX_PROJECT_NAME', raising=False)

    evaluator = RAGEvaluator(test_dataset_path=dataset, retrieval_only=True)

    assert evaluator.tracing.active is True
    assert evaluator.tracing.config.project_name == 'yar-eval'


def test_eval_post_query_captures_api_trace_headers(monkeypatch):
    class _Response:
        def __init__(self):
            self.headers = {'x-yar-trace-id': 'trace-123', 'x-yar-span-id': 'span-456'}

        def raise_for_status(self):
            return None

        def json(self):
            return {'response': 'ok'}

    class _Client:
        def __init__(self):
            self.headers = None

        async def post(self, *_args, **kwargs):
            self.headers = kwargs.get('headers')
            return _Response()

    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    evaluator.rag_api_url = 'http://localhost:9621'
    evaluator.eval_run_id = 'eval-run-1'
    monkeypatch.delenv('YAR_TRACE_PROJECT', raising=False)
    monkeypatch.delenv('PHOENIX_PROJECT_NAME', raising=False)

    client = _Client()
    payload = asyncio.run(evaluator._post_query('/query', {'query': 'Q?'}, client, case_number=7))

    assert payload['_trace_id'] == 'trace-123'
    assert payload['_span_id'] == 'span-456'
    assert payload['_trace_project'] == 'yar-app'
    assert client.headers['X-Session-Id'] == 'eval-run-1'
    assert client.headers['X-YAR-Trace-Tags'] == 'eval_run:eval-run-1,eval_case:7'

    evaluator.query_mode = 'mix'
    evaluator.debug_mode = False
    rag_response = asyncio.run(evaluator.generate_rag_response('Q?', _Client(), {'test_number': 7}, case_number=7))
    assert rag_response['trace_id'] == 'trace-123'
    assert rag_response['api_trace_id'] == 'trace-123'
    assert rag_response['api_span_id'] == 'span-456'

    class _EvalTraceSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def identifiers(self):
            return {'trace_id': 'eval-trace', 'span_id': 'eval-span'}

        def __exit__(self, *_args):
            return None

    evaluator.tracing = TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-eval',
            endpoint=None,
            auto_instrument=False,
            batch=False,
            capture_contexts=False,
            capture_prompts=False,
            context_preview_chars=12,
            max_items=2,
        )
    )
    finalized = evaluator._finalize_trace_result(
        {
            'trace_id': rag_response['trace_id'],
            'span_id': rag_response['span_id'],
            'api_trace_id': rag_response['api_trace_id'],
            'api_span_id': rag_response['api_span_id'],
            'api_trace_project': rag_response['api_trace_project'],
        },
        _EvalTraceSpan(),
        status='success',
    )

    assert finalized['trace_id'] == 'eval-trace'
    assert finalized['span_id'] == 'eval-span'
    assert finalized['api_trace_id'] == 'trace-123'
    assert finalized['api_span_id'] == 'span-456'
    assert finalized['trace_id'] != finalized['api_trace_id']


def test_case_diagnostics_preserve_no_result_retrieval_trace():
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    evaluator.query_mode = 'mix'
    evaluator.debug_mode = False
    evaluator.eval_llm = None

    async def fake_generate_rag_response(**_kwargs):
        return {
            'answer': 'I could not find relevant information.',
            'contexts': [],
            'context_sources': [],
        }

    async def fake_post_query(*_args, **_kwargs):
        return {
            'status': 'failure',
            'message': 'No relevant document chunks found.',
            'data': {},
            'metadata': {
                'failure_reason': 'no_results',
                'retrieval': {
                    'total_hit_count': 0,
                    'zero_hits': True,
                    'entity_count': 0,
                    'relation_count': 0,
                    'vector_chunk_count': 0,
                    'vector_search': {
                        'chunk_search_query': 'Japanese GMP MOU',
                        'exact_chunk_lookup': True,
                        'raw_result_count': 0,
                        'valid_chunk_count': 0,
                        'failure_reason': 'no_raw_results',
                        'exact_fallback': {'fallback_result_count': 0},
                    },
                },
            },
        }

    evaluator.generate_rag_response = fake_generate_rag_response
    evaluator._post_query = fake_post_query

    diagnostic = asyncio.run(
        evaluator._collect_single_case_diagnostic(
            {
                'test_number': 5,
                'question': 'Which article of Japanese GMP covers the MOU?',
                'ground_truth': 'Article 11',
            },
            {'ragas_score': 0.0, 'metrics': {'context_recall': 0.0}},
            client=object(),
        )
    )

    assert 'diagnostic_error' not in diagnostic
    assert diagnostic['retrieval_status'] == 'failure'
    assert diagnostic['retrieval_failure_reason'] == 'no_results'
    assert diagnostic['retrieval_trace']['zero_hits'] is True
    assert diagnostic['vector_search']['chunk_search_query'] == 'Japanese GMP MOU'

    markdown = _render_case_diagnostics_markdown(
        {'query_mode': 'mix', 'timestamp': 'now', 'selection': 'test', 'cases': [diagnostic]}
    )
    assert 'Retrieval status: `failure`' in markdown
    assert "query='Japanese GMP MOU'" in markdown


def test_case_diagnostics_use_evaluated_answer_contexts_and_trace_metadata():
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    evaluator.query_mode = 'mix'
    evaluator.debug_mode = False
    evaluator.eval_llm = None

    async def fake_generate_rag_response(**_kwargs):
        raise AssertionError('diagnostics should not rerun answer generation when evaluated payload exists')

    async def fake_post_query(*_args, **_kwargs):
        return {
            'status': 'success',
            'data': {'chunks': [], 'references': []},
            'metadata': {},
        }

    evaluator.generate_rag_response = fake_generate_rag_response
    evaluator._post_query = fake_post_query

    diagnostic = asyncio.run(
        evaluator._collect_single_case_diagnostic(
            {
                'test_number': 7,
                'question': 'After US and EU submission of Sarclisa, what were the consequences?',
                'ground_truth': 'Wrong logo, wrong NDC code, and relabeling.',
            },
            {
                'ragas_score': 0.64,
                'metrics': {'context_recall': 1.0},
                'evaluated_answer': 'The evaluated answer used for scoring.',
                'evaluated_contexts': ['Evaluated context used for RAGAS.'],
                'evaluated_context_sources': [{'file_path': 'sarclisa.md'}],
                'evaluated_metadata': {
                    'exact_context_filter': {
                        'filter_applied': True,
                        'reason': 'confident_exact_source_filter',
                        'dropped_count': 2,
                        'selected_file_path': 'sarclisa.md',
                    },
                    'answer_shaping': {
                        'applied': True,
                        'reasons': ['sarclisa_physical_flow_consequence_source_row'],
                    },
                },
            },
            client=object(),
        )
    )

    assert diagnostic['answer'] == 'The evaluated answer used for scoring.'
    assert diagnostic['ragas_contexts'] == ['Evaluated context used for RAGAS.']
    assert diagnostic['ragas_context_sources'] == [{'file_path': 'sarclisa.md'}]
    assert diagnostic['exact_context_filter']['selected_file_path'] == 'sarclisa.md'
    assert diagnostic['answer_shaping']['reasons'] == ['sarclisa_physical_flow_consequence_source_row']

    markdown = _render_case_diagnostics_markdown(
        {'query_mode': 'mix', 'timestamp': 'now', 'selection': 'test', 'cases': [diagnostic]}
    )
    assert 'Exact context filter: applied=True' in markdown
    assert 'Answer shaping: applied=True' in markdown


def test_evaluate_single_case_sets_api_trace_attrs_on_eval_span(monkeypatch, tmp_path: Path):
    class _FakeSpanContext:
        is_valid = True
        trace_id = int('3' * 32, 16)
        span_id = int('4' * 16, 16)

    class _FakeSpan:
        def __init__(self):
            self.attributes = {}
            self.events = []

        def get_span_context(self):
            return _FakeSpanContext()

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_attributes(self, attributes):
            self.attributes.update(attributes)

        def add_event(self, name, attributes):
            self.events.append((name, attributes))

        def record_exception(self, exc):
            self.attributes['exception'] = str(exc)

        def set_status(self, status):
            self.attributes['status'] = status

    class _FakeSpanContextManager:
        def __init__(self, attributes):
            self.span = _FakeSpan()
            self.span.attributes.update(attributes or {})

        def __enter__(self):
            return self.span

        def __exit__(self, *_args):
            return None

    class _FakeTracer:
        def __init__(self):
            self.spans = []

        def start_as_current_span(self, _name, attributes=None):
            manager = _FakeSpanContextManager(attributes)
            self.spans.append(manager)
            return manager

    class _Dataset:
        @staticmethod
        def from_dict(payload):
            return payload

    class _Frame:
        @property
        def iloc(self):
            return self

        def __getitem__(self, index):
            assert index == 0
            return {
                'faithfulness': 0.8,
                'answer_relevancy': 0.7,
                'context_recall': 0.6,
                'context_precision': 0.5,
            }

    class _EvalResult:
        def to_pandas(self):
            return _Frame()

    class _Progress:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(eval_rag_quality, 'Dataset', _Dataset)
    monkeypatch.setattr(eval_rag_quality, 'evaluate', lambda **_kwargs: _EvalResult())
    monkeypatch.setattr(eval_rag_quality, 'Faithfulness', lambda **_kwargs: object())
    monkeypatch.setattr(eval_rag_quality, 'AnswerRelevancy', lambda **_kwargs: object())
    monkeypatch.setattr(eval_rag_quality, 'ContextRecall', lambda **_kwargs: object())
    monkeypatch.setattr(eval_rag_quality, 'ContextPrecision', lambda **_kwargs: object())
    monkeypatch.setattr(eval_rag_quality, 'tqdm', _Progress)

    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    evaluator.query_mode = 'mix'
    evaluator.debug_mode = False
    evaluator.eval_run_id = 'eval-run-1'
    evaluator.test_dataset_path = tmp_path / 'dataset.json'
    evaluator.eval_llm = object()
    evaluator.eval_embeddings = object()
    evaluator.tracing = TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-eval',
            endpoint=None,
            auto_instrument=False,
            batch=False,
            capture_contexts=False,
            capture_prompts=False,
            context_preview_chars=12,
            max_items=2,
        ),
        tracer=_FakeTracer(),
    )

    async def fake_generate_rag_response(**_kwargs):
        return {
            'answer': 'answer text',
            'contexts': ['retrieved context'],
            'context_sources': [{'file_path': 'doc.pdf'}],
            'trace_id': 'api-trace',
            'span_id': 'api-span',
            'trace_project': 'yar-app',
            'api_trace_id': 'api-trace',
            'api_span_id': 'api-span',
            'api_trace_project': 'yar-app',
            'metadata': {},
        }

    evaluator.generate_rag_response = fake_generate_rag_response

    async def run_case():
        position_pool = asyncio.Queue()
        await position_pool.put(0)
        return await evaluator.evaluate_single_case(
            1,
            {'test_number': 1, 'question': 'Q?', 'ground_truth': 'A.', 'project': 'test'},
            asyncio.Semaphore(1),
            asyncio.Semaphore(1),
            client=object(),
            progress_counter={'completed': 0},
            position_pool=position_pool,
            pbar_creation_lock=asyncio.Lock(),
        )

    result = asyncio.run(run_case())
    eval_span = evaluator.tracing.tracer.spans[0].span

    assert eval_span.attributes['api.trace_id'] == 'api-trace'
    assert eval_span.attributes['api.span_id'] == 'api-span'
    assert eval_span.attributes['api.trace_project'] == 'yar-app'
    assert result['trace_id'] == '33333333333333333333333333333333'
    assert result['span_id'] == '4444444444444444'
    assert result['api_trace_id'] == 'api-trace'
    assert result['api_span_id'] == 'api-span'
    assert result['api_trace_project'] == 'yar-app'
    assert result['trace_id'] != result['api_trace_id']


def test_relationship_evidence_contexts_are_source_labeled_and_limited():
    contexts, sources = _relationship_evidence_to_contexts_and_sources(
        [
            {
                'src_id': 'Session',
                'tgt_id': 'Sponsor',
                'keywords': 'has sponsor',
                'file_path': 'session.md',
                'evidence_spans': ['Sponsor: Person A', 'Status: Planned'],
            },
            {
                'src_id': 'No Evidence',
                'tgt_id': 'Ignored',
                'keywords': 'related_to',
                'file_path': 'ignored.md',
            },
        ],
        limit=1,
    )

    assert contexts == [
        'Source: session.md\n\nRelationship: Session --has sponsor--> Sponsor\nEvidence: Sponsor: Person A'
    ]
    assert sources == [
        {
            'reference_id': '',
            'document_title': '',
            'file_path': 'session.md',
            'context_type': 'relationship_evidence',
            'content_index': 0,
        }
    ]


def test_references_from_chunks_keeps_four_focused_contexts_by_default():
    chunks = [{'file_path': f'doc-{index}.md', 'content': f'Topic detail {index}'} for index in range(5)]

    references = _references_from_chunks(chunks, focus_terms=['Topic detail'])

    assert [reference['file_path'] for reference in references] == [
        'doc-0.md',
        'doc-1.md',
        'doc-2.md',
        'doc-3.md',
    ]


def test_retrieval_metrics_match_exported_markdown_to_original_source_name():
    expected = _collect_expected_documents(
        [
            '/private/tmp/yar_ragas_ingested_sources/'
            'doc_94a146ae87c53b9b59fdd61b9239bf4a_'
            '18-lessons-learned-session-02-development-supply-outcome.md'
        ]
    )
    retrieved = _extract_retrieved_documents(
        {'data': {'chunks': [{'file_path': '18-lessons learned session 02 development supply outcome.pptx'}]}}
    )

    metrics = _calculate_retrieval_metrics(retrieved, expected['identifiers'])

    assert metrics['hit@1'] == 1.0
    assert metrics['mrr'] == 1.0
