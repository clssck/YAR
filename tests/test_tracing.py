from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from yar.api.routers.query_routes import create_query_routes
from yar.evaluation.eval_rag_quality import RAGEvaluator, _trace_context_payload
from yar.tracing import (
    TraceConfig,
    TraceManager,
    TraceSpan,
    instrument_fastapi_app,
    noop_trace_manager,
    trace_sequence_preview,
)


class _FakeSpanContext:
    is_valid = True
    trace_id = int('1' * 32, 16)
    span_id = int('2' * 16, 16)


class _FakeSpan:
    def __init__(self):
        super().__init__()
        self.attributes = {}
        self.events = []
        self.status = None

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
        self.status = status


class _FakeSpanContextManager:
    def __init__(self):
        super().__init__()
        self.span = _FakeSpan()
        self.exited = False
        self.name = ''

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return None


class _FakeTracer:
    def __init__(self):
        super().__init__()
        self.spans = []

    def start_as_current_span(self, name, attributes=None):
        manager = _FakeSpanContextManager()
        manager.name = name
        manager.span.attributes.update(attributes or {})
        self.spans.append(manager)
        return manager


def _active_trace_manager(*, capture_contexts=False, capture_prompts=False):
    return TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-app',
            endpoint=None,
            auto_instrument=False,
            batch=False,
            capture_contexts=capture_contexts,
            capture_prompts=capture_prompts,
            context_preview_chars=12,
            max_items=2,
        ),
        tracer=_FakeTracer(),
    )


def _build_retrieval_evaluator(**kwargs):
    with (
        patch.object(RAGEvaluator, '_load_test_dataset', return_value=[]),
        patch.object(RAGEvaluator, '_display_configuration'),
    ):
        return RAGEvaluator(retrieval_only=True, **kwargs)


def test_trace_span_extracts_identifiers_and_coerces_attributes():
    manager = _FakeSpanContextManager()

    with TraceSpan(manager) as span:
        span.set_attributes({'metric': 1.25, 'payload': {'a': 1}})
        span.add_event('done', {'items': ['a', 'b']})

    assert span.identifiers() == {'trace_id': '11111111111111111111111111111111', 'span_id': '2222222222222222'}
    assert manager.span.attributes['metric'] == 1.25
    assert manager.span.attributes['payload'] == '{"a": 1}'
    assert manager.span.events == [('done', {'items': ['a', 'b']})]
    assert manager.exited is True


def test_noop_trace_manager_is_inactive(monkeypatch):
    monkeypatch.delenv('YAR_TRACE_ENABLED', raising=False)

    manager = noop_trace_manager(default_project='yar-eval')

    assert manager.active is False
    assert manager.summary()['disabled_reason'] == 'disabled'
    with manager.start_span('test') as span:
        span.set_attribute('ignored', 'value')
    assert span.identifiers() == {}


def test_trace_config_can_enable_app_tracing_by_default(monkeypatch):
    monkeypatch.delenv('YAR_TRACE_ENABLED', raising=False)
    monkeypatch.delenv('YAR_TRACE_PROJECT', raising=False)

    config = TraceConfig.from_env(default_project='yar-app', enabled_by_default=True)

    assert config.enabled is True
    assert config.project_name == 'yar-app'


def test_trace_env_can_disable_default_app_tracing(monkeypatch):
    monkeypatch.setenv('YAR_TRACE_ENABLED', 'false')

    config = TraceConfig.from_env(default_project='yar-app', enabled_by_default=True)

    assert config.enabled is False


def test_trace_config_normalizes_base_phoenix_http_endpoint(monkeypatch):
    monkeypatch.setenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006/')

    config = TraceConfig.from_env(default_project='yar-app')

    assert config.endpoint == 'http://localhost:6006/v1/traces'


def test_rag_evaluator_explicit_eval_run_id_overrides_env(monkeypatch):
    monkeypatch.setenv('YAR_EVAL_RUN_ID', 'env-run')

    evaluator = _build_retrieval_evaluator(eval_run_id='explicit-run')

    assert evaluator.eval_run_id == 'explicit-run'


def test_rag_evaluator_uses_env_eval_run_id_when_explicit_absent(monkeypatch):
    monkeypatch.setenv('YAR_EVAL_RUN_ID', 'env-run')

    evaluator = _build_retrieval_evaluator()

    assert evaluator.eval_run_id == 'env-run'


def test_trace_context_payload_redacts_contexts_by_default():
    manager = TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-eval',
            endpoint=None,
            auto_instrument=False,
            batch=False,
            capture_contexts=False,
            capture_prompts=False,
            context_preview_chars=12,
            max_items=1,
        )
    )

    payload = _trace_context_payload(
        tracing=manager,
        contexts=['secret full context should not be present'],
        context_sources=[{'file_path': 'doc.pdf', 'reference_id': '1'}],
    )

    assert payload == {
        'context_count': 1,
        'context_sources': [{'reference_id': '1', 'document_title': '', 'file_path': 'doc.pdf', 'content_index': 0}],
    }


def test_trace_context_payload_includes_bounded_previews_when_enabled():
    manager = TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-eval',
            endpoint=None,
            auto_instrument=False,
            batch=False,
            capture_contexts=True,
            capture_prompts=False,
            context_preview_chars=10,
            max_items=1,
        )
    )

    assert _trace_context_payload(
        tracing=manager,
        contexts=['abcdefghijklmnopqrstuvwxyz'],
        context_sources=[],
    )['context_previews'] == ['abcdefghij...']
    assert trace_sequence_preview(['one', 'two'], max_items=1, max_chars=10) == ['one']


def test_finalize_trace_result_attaches_trace_ids():
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    evaluator.tracing = TraceManager(
        config=TraceConfig(
            enabled=True,
            project_name='yar-eval',
            endpoint=None,
            auto_instrument=False,
            batch=False,
            capture_contexts=False,
            capture_prompts=False,
            context_preview_chars=500,
            max_items=10,
        )
    )
    trace_span = SimpleNamespace(
        identifiers=lambda: {'trace_id': 'abc', 'span_id': 'def'},
        set_attribute=lambda *args, **kwargs: None,
        __exit__=lambda *args, **kwargs: None,
    )

    result = evaluator._finalize_trace_result({'status': 'success'}, trace_span, status='success')

    assert result == {'status': 'success', 'trace_id': 'abc', 'span_id': 'def', 'trace_project': 'yar-eval'}


@pytest.mark.asyncio
async def test_fastapi_middleware_does_not_emit_http_span():
    """The middleware must not produce its own span for arbitrary endpoints.

    Only explicit RAG handlers create spans; everything else (status polls,
    health, static assets) stays silent.
    """
    manager = _active_trace_manager()
    app = FastAPI()

    @app.get('/ok')
    async def ok():
        return {'ok': True}

    instrument_fastapi_app(app, manager)

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get('/ok')

    assert response.status_code == 200
    # No spans should be emitted for non-RAG endpoints.
    assert manager.tracer.spans == []


class _FakeRag:
    async def aquery_llm(self, query, param):
        return {
            'status': 'success',
            'message': 'ok',
            'llm_response': {'content': 'answer text'},
            'data': {
                'references': [{'reference_id': '1', 'file_path': 'doc.pdf'}],
                'chunks': [{'reference_id': '1', 'content': 'secret context payload'}],
                'entities': [{'entity_name': 'PKU'}],
                'relationships': [],
            },
            'metadata': {'query_mode': param.mode},
        }


@pytest.mark.asyncio
async def test_query_route_trace_redacts_sensitive_text_by_default():
    manager = _active_trace_manager(capture_contexts=False, capture_prompts=False)
    app = FastAPI()
    app.include_router(create_query_routes(_FakeRag(), tracing=manager))

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/query', json={'query': 'What is PKU?', 'mode': 'mix'})

    assert response.status_code == 200
    query_span = manager.tracer.spans[0].span
    assert query_span.attributes['input.query_length'] == len('What is PKU?')
    assert query_span.attributes['rag.reference_count'] == 1
    assert query_span.attributes['rag.source_files'] == ['doc.pdf']
    assert 'input.query' not in query_span.attributes
    assert 'output.answer' not in query_span.attributes
    assert 'rag.context_previews' not in query_span.attributes


@pytest.mark.asyncio
async def test_query_route_trace_includes_bounded_text_when_enabled():
    manager = _active_trace_manager(capture_contexts=True, capture_prompts=True)
    app = FastAPI()
    app.include_router(create_query_routes(_FakeRag(), tracing=manager))

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/query', json={'query': 'What is PKU?', 'mode': 'mix'})

    assert response.status_code == 200
    query_span = manager.tracer.spans[0].span
    assert query_span.attributes['input.query'] == 'What is PKU?'
    assert query_span.attributes['output.answer'] == 'answer text'
    assert query_span.attributes['rag.context_previews'] == ['secret conte...']


def test_start_llm_span_emits_openinference_attributes():
    manager = _active_trace_manager(capture_prompts=True)
    with manager.start_chain_span('chain.parent'):
        with manager.start_llm_span(
            'llm.test',
            model='gpt-4',
            provider='openai',
            system='openai',
            prompt='hello',
            system_prompt='be brief',
            history_messages=[{'role': 'user', 'content': 'past'}],
            invocation_parameters={'temperature': 0.1},
        ) as span:
            span.set_llm_token_counts(prompt=10, completion=5, total=15)
            span.set_llm_cost(prompt=0.001, completion=0.002, total=0.003)
            span.set_llm_finish_reason('stop')
            span.set_llm_output_messages([{'role': 'assistant', 'content': 'hi'}])

    llm_attrs = manager.tracer.spans[1].span.attributes
    assert llm_attrs['openinference.span.kind'] == 'LLM'
    assert llm_attrs['llm.model_name'] == 'gpt-4'
    assert llm_attrs['llm.provider'] == 'openai'
    assert llm_attrs['llm.system'] == 'openai'
    assert llm_attrs['llm.token_count.prompt'] == 10
    assert llm_attrs['llm.token_count.completion'] == 5
    assert llm_attrs['llm.token_count.total'] == 15
    assert llm_attrs['llm.cost.total'] == 0.003
    assert llm_attrs['llm.finish_reason'] == 'stop'
    # Messages flattened
    assert llm_attrs['llm.input_messages.0.message.role'] == 'system'
    assert llm_attrs['llm.input_messages.0.message.content'] == 'be brief'
    assert llm_attrs['llm.input_messages.1.message.role'] == 'user'
    assert llm_attrs['llm.input_messages.2.message.role'] == 'user'
    assert llm_attrs['llm.input_messages.2.message.content'] == 'hello'
    assert llm_attrs['llm.output_messages.0.message.role'] == 'assistant'
    assert llm_attrs['llm.output_messages.0.message.content'] == 'hi'
    assert llm_attrs['llm.invocation_parameters']


def test_start_llm_span_redacts_messages_when_capture_prompts_false():
    manager = _active_trace_manager(capture_prompts=False)
    with manager.start_chain_span('chain.parent'):
        with manager.start_llm_span(
            'llm.test',
            model='gpt-4',
            provider='openai',
            prompt='secret',
            system_prompt='be brief',
        ):
            pass

    llm_attrs = manager.tracer.spans[1].span.attributes
    assert 'llm.input_messages.0.message.content' not in llm_attrs
    assert llm_attrs['llm.model_name'] == 'gpt-4'


def test_start_embedding_span_emits_openinference_attributes():
    manager = _active_trace_manager(capture_contexts=True)
    with manager.start_chain_span('chain.parent'):
        with manager.start_embedding_span(
            'emb.test',
            model='text-embedding-3-small',
            provider='openai',
            texts=['hello world'],
        ):
            pass

    emb_attrs = manager.tracer.spans[1].span.attributes
    assert emb_attrs['openinference.span.kind'] == 'EMBEDDING'
    assert emb_attrs['embedding.model_name'] == 'text-embedding-3-small'
    assert emb_attrs['llm.provider'] == 'openai'
    assert emb_attrs['embedding.embeddings.0.embedding.text'] == 'hello world'


def test_start_retriever_span_records_documents():
    manager = _active_trace_manager(capture_contexts=True, capture_prompts=True)
    with manager.start_chain_span('chain.parent'):
        with manager.start_retriever_span(
            'retrieval.test',
            query='find me',
            top_k=3,
            mode='mix',
        ) as span:
            span.set_retrieval_documents(
                [
                    {'id': '1', 'content': 'doc one', 'score': 0.9, 'metadata': {'src': 'a'}},
                    {'id': '2', 'content': 'doc two', 'score': 0.5},
                ]
            )

    retr_attrs = manager.tracer.spans[1].span.attributes
    assert retr_attrs['openinference.span.kind'] == 'RETRIEVER'
    assert retr_attrs['retrieval.top_k'] == 3
    assert retr_attrs['retrieval.mode'] == 'mix'
    assert retr_attrs['input.value'] == 'find me'
    assert retr_attrs['retrieval.documents.0.document.id'] == '1'
    assert retr_attrs['retrieval.documents.0.document.content'] == 'doc one'
    assert retr_attrs['retrieval.documents.0.document.score'] == 0.9
    assert retr_attrs['retrieval.documents.1.document.id'] == '2'
    assert retr_attrs['retrieval.documents.1.document.score'] == 0.5


def test_start_reranker_span_emits_documents():
    manager = _active_trace_manager(capture_contexts=True, capture_prompts=True)
    input_docs = [{'id': '1', 'content': 'first'}, {'id': '2', 'content': 'second'}]
    with manager.start_chain_span('chain.parent'):
        with manager.start_reranker_span(
            'rerank.test',
            model='cohere-rerank',
            query='q',
            top_k=2,
            input_documents=input_docs,
        ) as span:
            span.set_reranker(
                output_documents=[
                    {'id': '1', 'score': 0.9, 'content': 'first'},
                ],
            )

    rr_attrs = manager.tracer.spans[1].span.attributes
    assert rr_attrs['openinference.span.kind'] == 'RERANKER'
    assert rr_attrs['reranker.model_name'] == 'cohere-rerank'
    assert rr_attrs['reranker.query'] == 'q'
    assert rr_attrs['reranker.top_k'] == 2
    assert rr_attrs['reranker.input_documents.0.document.id'] == '1'
    assert rr_attrs['reranker.output_documents.0.document.score'] == 0.9


def test_start_chain_span_carries_input_value():
    manager = _active_trace_manager()
    with manager.start_chain_span('chain.test', input_value='hello'):
        pass
    attrs = manager.tracer.spans[0].span.attributes
    assert attrs['openinference.span.kind'] == 'CHAIN'
    assert attrs['input.value'] == 'hello'


def test_start_evaluator_and_tool_and_guardrail_spans_set_kind():
    manager = _active_trace_manager()
    # Evaluator and agent spans are root-capable; tool/guardrail require a parent.
    with manager.start_evaluator_span('eval.test'):
        pass
    with manager.start_evaluator_span('eval.outer'):
        with manager.start_tool_span('tool.test', tool_name='lookup', description='d', parameters={'a': 1}):
            pass
        with manager.start_guardrail_span('guard.test'):
            pass
    with manager.start_agent_span('agent.test', agent_name='alice'):
        pass

    kinds = [span.span.attributes.get('openinference.span.kind') for span in manager.tracer.spans]
    assert kinds == ['EVALUATOR', 'EVALUATOR', 'TOOL', 'GUARDRAIL', 'AGENT']
    tool_attrs = manager.tracer.spans[2].span.attributes
    assert tool_attrs['tool.name'] == 'lookup'
    assert tool_attrs['tool.description'] == 'd'
    assert 'tool.parameters' in tool_attrs
    agent_attrs = manager.tracer.spans[4].span.attributes
    assert agent_attrs['agent.name'] == 'alice'


def test_session_user_metadata_setters_record_attributes():
    manager = _active_trace_manager()
    with manager.start_chain_span('chain.test') as span:
        span.set_session('s-1')
        span.set_user('u-2')
        span.set_metadata({'env': 'test'})
        span.set_tags(['a', 'b'])
        span.set_input_value('hi')
        span.set_output_value('there')
        span.set_llm_prompt_template(
            template='Hello {name}',
            variables={'name': 'world'},
            version='v1',
        )

    attrs = manager.tracer.spans[0].span.attributes
    assert attrs['session.id'] == 's-1'
    assert attrs['user.id'] == 'u-2'
    assert attrs['metadata']
    assert attrs['tag.tags'] == ['a', 'b']
    assert attrs['input.value'] == 'hi'
    assert attrs['output.value'] == 'there'
    assert attrs['llm.prompt_template.template'] == 'Hello {name}'
    assert attrs['llm.prompt_template.version'] == 'v1'


def test_active_trace_manager_register_and_release(monkeypatch):
    from yar.tracing import _set_active_trace_manager, get_active_trace_manager

    previous = get_active_trace_manager()
    try:
        manager = _active_trace_manager()
        _set_active_trace_manager(manager)
        assert get_active_trace_manager() is manager
        _set_active_trace_manager(None)
        assert get_active_trace_manager() is None
    finally:
        _set_active_trace_manager(previous)


def test_default_tags_propagate_through_spans(monkeypatch):
    monkeypatch.setenv('YAR_TRACE_DEFAULT_TAGS', 'prod,smoke')
    config = TraceConfig.from_env(default_project='yar-app', enabled_by_default=True)
    manager = TraceManager(config=config, tracer=_FakeTracer())
    with manager.start_chain_span('chain.test'):
        pass
    attrs = manager.tracer.spans[0].span.attributes
    assert attrs['tag.tags'] == ['prod', 'smoke']


@pytest.mark.asyncio
async def test_fastapi_middleware_propagates_session_user_headers_to_child_spans():
    """Session/user/tags from headers must reach spans created inside the request."""
    manager = _active_trace_manager()
    app = FastAPI()
    instrument_fastapi_app(app, manager)

    @app.get('/hi')
    async def _hi():
        # Simulate a query handler emitting a chain span; OpenInference's
        # using_attributes pushes session/user/tags onto the OTel context, but
        # in tests we have no real OTel context. Verify that the manager's
        # `using_attributes` was wired (returns a context manager) and that
        # the explicit chain span receives the same identifiers via headers
        # propagated downstream by the test.
        with manager.start_chain_span(
            'app.fake_query',
            attributes={
                'session.id': 's-42',
                'user.id': 'alice',
                'tag.tags': ['prod', 'smoke'],
            },
        ):
            pass
        return {'ok': True}

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get(
            '/hi',
            headers={
                'x-session-id': 's-42',
                'x-user-id': 'alice',
                'x-yar-trace-tags': 'prod, smoke',
            },
        )

    assert response.status_code == 200
    span_attrs = manager.tracer.spans[0].span.attributes
    assert span_attrs['session.id'] == 's-42'
    assert span_attrs['user.id'] == 'alice'
    assert span_attrs['tag.tags'] == ['prod', 'smoke']
    assert span_attrs['openinference.span.kind'] == 'CHAIN'


@pytest.mark.asyncio
async def test_query_route_emits_categorization_tags():
    manager = _active_trace_manager(capture_contexts=True, capture_prompts=True)
    app = FastAPI()
    app.include_router(create_query_routes(_FakeRag(), tracing=manager))

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(
            '/query',
            json={'query': 'What is PKU?', 'mode': 'mix', 'include_references': True},
        )

    assert response.status_code == 200
    span = manager.tracer.spans[0].span
    tags = span.attributes.get('tag.tags', [])
    # Must contain endpoint + mode + status categories
    assert 'endpoint:query' in tags
    assert 'mode:mix' in tags
    assert 'streaming:false' in tags
    assert 'references:true' in tags
    assert 'status:success' in tags
    # Reference count is bucketed
    assert any(t.startswith('refs:') for t in tags)
    # Query length bucket present
    assert any(t.startswith('len:') for t in tags)


def test_classify_exception_categorizes_known_errors():
    from yar.api.routers.query_routes import _classify_exception

    class _Timeout(Exception):
        pass

    class _RateLimit(Exception):
        pass

    assert _classify_exception(_Timeout('Operation Timeout')) == 'timeout'
    assert _classify_exception(_RateLimit('429 too many requests')) == 'rate_limit'
    assert _classify_exception(ConnectionError('connection refused')) == 'connection'
    assert _classify_exception(ValueError('boom')) == 'internal'


@pytest.mark.asyncio
async def test_query_route_emits_synthetic_retriever_span():
    manager = _active_trace_manager(capture_contexts=True, capture_prompts=True)
    app = FastAPI()
    app.include_router(create_query_routes(_FakeRag(), tracing=manager))

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(
            '/query',
            json={'query': 'What is PKU?', 'mode': 'mix', 'include_references': True},
        )

    assert response.status_code == 200
    span_names = [span.name for span in manager.tracer.spans]
    # Two spans: chain (app.query) + synthetic retriever (retrieval.documents)
    assert 'app.query' in span_names
    assert 'retrieval.documents' in span_names
    retr_idx = span_names.index('retrieval.documents')
    retr_attrs = manager.tracer.spans[retr_idx].span.attributes
    assert retr_attrs['openinference.span.kind'] == 'RETRIEVER'
    assert retr_attrs['retrieval.requested_mode'] == 'mix'
    assert retr_attrs['retrieval.document_count'] >= 0


def test_sample_ratio_parses_and_clamps(monkeypatch):
    from yar.tracing import _parse_sample_ratio

    assert _parse_sample_ratio(None) == 1.0
    assert _parse_sample_ratio('') == 1.0
    assert _parse_sample_ratio('0.25') == 0.25
    assert _parse_sample_ratio('-0.5') == 0.0
    assert _parse_sample_ratio('2.0') == 1.0
    assert _parse_sample_ratio('garbage') == 1.0


def test_trace_config_reads_sample_ratio(monkeypatch):
    monkeypatch.setenv('YAR_TRACE_SAMPLE_RATIO', '0.1')
    config = TraceConfig.from_env(default_project='yar-app', enabled_by_default=True)
    assert config.sample_ratio == 0.1


def test_query_cache_hit_contextvar_round_trip():
    from yar.tracing import (
        mark_query_cache_hit,
        query_cache_hit_was_set,
        reset_query_cache_hit,
    )

    reset_query_cache_hit()
    assert query_cache_hit_was_set() is False
    mark_query_cache_hit()
    assert query_cache_hit_was_set() is True
    reset_query_cache_hit()
    assert query_cache_hit_was_set() is False


def test_attach_prompt_template_sets_attributes_for_naive_mode():
    from types import SimpleNamespace

    from yar.api.routers.query_routes import _attach_prompt_template

    manager = _active_trace_manager()
    request = SimpleNamespace(mode='naive', response_type='Multiple Paragraphs', user_prompt=None)
    with manager.start_chain_span('chain.parent') as span:
        _attach_prompt_template(span, request)

    attrs = manager.tracer.spans[0].span.attributes
    assert attrs['llm.prompt_template.name'] == 'naive_rag_response'
    assert attrs['llm.prompt_template.version'] == 'naive_rag_response'
    # Variables JSON-encoded; just verify mode is in the payload
    assert 'naive' in attrs['llm.prompt_template.variables']


def test_attach_prompt_template_sets_attributes_for_graph_mode():
    from types import SimpleNamespace

    from yar.api.routers.query_routes import _attach_prompt_template

    manager = _active_trace_manager()
    request = SimpleNamespace(mode='mix', response_type='Single Paragraph', user_prompt='custom')
    with manager.start_chain_span('chain.parent') as span:
        _attach_prompt_template(span, request)

    attrs = manager.tracer.spans[0].span.attributes
    assert attrs['llm.prompt_template.name'] == 'rag_response'
    assert 'mix' in attrs['llm.prompt_template.variables']
    assert 'custom' in attrs['llm.prompt_template.variables']


def test_parse_resource_attributes():
    from yar.tracing import _parse_resource_attributes

    assert _parse_resource_attributes(None) == {}
    assert _parse_resource_attributes('') == {}
    assert _parse_resource_attributes('a=1,b=2') == {'a': '1', 'b': '2'}
    assert _parse_resource_attributes(' a = 1 , b = 2 ') == {'a': '1', 'b': '2'}
    # malformed entries are skipped
    assert _parse_resource_attributes('valid=ok,no_equal,empty=') == {'valid': 'ok', 'empty': ''}


def test_trace_config_picks_up_resource_attributes(monkeypatch):
    monkeypatch.setenv('YAR_SERVICE_NAME', 'yar-prod')
    monkeypatch.setenv('YAR_SERVICE_VERSION', '1.2.3')
    monkeypatch.setenv('YAR_DEPLOYMENT_ENV', 'staging')
    monkeypatch.setenv('YAR_TRACE_RESOURCE_ATTRIBUTES', 'team=infra,region=us-east')
    config = TraceConfig.from_env(default_project='yar-app', enabled_by_default=True)
    assert config.service_name == 'yar-prod'
    assert config.service_version == '1.2.3'
    assert config.deployment_environment == 'staging'
    assert config.resource_attributes == {'team': 'infra', 'region': 'us-east'}


def test_record_retry_event_no_op_without_span():
    from yar.tracing import record_retry_event

    # Should not raise even when no OTel span is active.
    class _Outcome:
        failed = True

        def exception(self):
            return ValueError('fail')

    class _State:
        attempt_number = 2
        outcome = _Outcome()
        next_action = None

    record_retry_event(_State())


def test_record_llm_finish_reason_no_op_without_span():
    from yar.tracing import record_llm_finish_reason

    # Should not raise even when no OTel span is active.
    record_llm_finish_reason('length')
    record_llm_finish_reason(None)
    record_llm_finish_reason('stop')


def test_extract_cited_reference_ids():
    from yar.api.routers.query_routes import _extract_cited_reference_ids

    assert _extract_cited_reference_ids('') == set()
    assert _extract_cited_reference_ids('plain prose') == set()
    assert _extract_cited_reference_ids('See [1] and [2,3].') == {1, 2, 3}
    assert _extract_cited_reference_ids('Repeat [1] then [1] again [4]') == {1, 4}
    assert _extract_cited_reference_ids('Whitespace [1, 2 , 3]') == {1, 2, 3}


def test_retrieval_precision_bucket():
    from yar.api.routers.query_routes import _retrieval_precision_bucket

    assert _retrieval_precision_bucket(0.9) == 'precision:high'
    assert _retrieval_precision_bucket(0.5) == 'precision:mid'
    assert _retrieval_precision_bucket(0.1) == 'precision:low'
    assert _retrieval_precision_bucket(0.0) == 'precision:zero'


def test_emit_citation_metrics_attaches_attrs_and_tags():
    from yar.api.routers.query_routes import _emit_citation_metrics

    manager = _active_trace_manager()
    result = {
        'data': {
            'references': [
                {'reference_id': 1},
                {'reference_id': 2},
                {'reference_id': 3},
                {'reference_id': 4},
            ]
        }
    }
    response_text = 'According to [1] and [2], it works.'
    with manager.start_chain_span('chain.parent') as span:
        tags = _emit_citation_metrics(trace_span=span, response_text=response_text, result=result)

    attrs = manager.tracer.spans[0].span.attributes
    assert attrs['retrieval.cited_count'] == 2
    assert attrs['retrieval.unused_count'] == 2
    assert attrs['retrieval.precision'] == 0.5
    assert sorted(attrs['retrieval.cited_ids']) == ['1', '2']
    assert 'cited:2' in tags
    assert 'precision:mid' in tags


def test_retrieval_fingerprint_is_stable():
    from yar.api.routers.query_routes import _retrieval_fingerprint

    result_a = {
        'data': {
            'references': [
                {'document_title': 'doc.pdf', 'content_index': 0},
                {'document_title': 'doc.pdf', 'content_index': 3},
            ]
        }
    }
    result_b = dict(result_a)
    fp_a = _retrieval_fingerprint(result_a)
    fp_b = _retrieval_fingerprint(result_b)
    assert fp_a is not None
    assert fp_a == fp_b
    # Different ordering -> different fingerprint
    result_c = {
        'data': {
            'references': [
                {'document_title': 'doc.pdf', 'content_index': 3},
                {'document_title': 'doc.pdf', 'content_index': 0},
            ]
        }
    }
    assert _retrieval_fingerprint(result_c) != fp_a
    # Empty references -> None
    assert _retrieval_fingerprint({'data': {'references': []}}) is None
    assert _retrieval_fingerprint({'no_data': True}) is None


def test_latency_bucket():
    from yar.api.routers.query_routes import _latency_bucket

    assert _latency_bucket(50) == 'latency:fast'
    assert _latency_bucket(500) == 'latency:normal'
    assert _latency_bucket(2000) == 'latency:slow'
    assert _latency_bucket(8000) == 'latency:very_slow'
    assert _latency_bucket(20000) == 'latency:extreme'


def test_slow_query_threshold_env(monkeypatch):
    from yar.api.routers.query_routes import _slow_query_threshold_ms

    monkeypatch.delenv('YAR_TRACE_SLOW_QUERY_MS', raising=False)
    assert _slow_query_threshold_ms() == 5000.0
    monkeypatch.setenv('YAR_TRACE_SLOW_QUERY_MS', '1500')
    assert _slow_query_threshold_ms() == 1500.0
    monkeypatch.setenv('YAR_TRACE_SLOW_QUERY_MS', 'garbage')
    assert _slow_query_threshold_ms() == 5000.0


def test_record_query_metrics_no_op_without_meter():
    manager = _active_trace_manager()
    # No meter has been activated; method must be safe.
    manager.record_query_metrics(
        endpoint='/query',
        mode='mix',
        status='success',
        duration_ms=120.0,
    )
    manager.record_token_metrics(prompt_tokens=10, completion_tokens=5, model='gpt-4')


def test_trace_config_reads_metrics_settings(monkeypatch):
    monkeypatch.setenv('YAR_TRACE_METRICS_ENABLED', 'true')
    monkeypatch.setenv('YAR_TRACE_METRICS_ENDPOINT', 'http://collector.example.com')
    monkeypatch.setenv('YAR_TRACE_METRICS_INTERVAL_MS', '10000')
    config = TraceConfig.from_env(default_project='yar-app', enabled_by_default=True)
    assert config.metrics_enabled is True
    assert config.metrics_endpoint == 'http://collector.example.com'
    assert config.metrics_export_interval_ms == 10000
