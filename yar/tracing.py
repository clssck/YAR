"""Arize Phoenix / OpenInference tracing integration for YAR.

This module owns the full tracing surface for the project. Goals:

1.  Emit OpenInference-compliant spans for LLM, embedding, retriever,
    reranker, chain, tool, guardrail and evaluator workloads so Arize
    Phoenix classifies and indexes them correctly.
2.  Surface project-wide context (session, user, tags, metadata, prompt
    templates) through OpenInference helpers when available.
3.  Expose a thin facade over ``phoenix.client.Client`` for datasets,
    experiments, span/trace/session annotations, and prompt management.
4.  Degrade safely when Phoenix/OpenInference are missing or disabled:
    every call returns a no-op object and never raises.

The legacy entry points (``start_span``, ``configure_tracing``,
``noop_trace_manager``, ``instrument_fastapi_app``,
``trace_sequence_preview``) remain stable so existing call sites keep
working.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from yar.utils import logger

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'off'}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.casefold().strip()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning('Invalid integer for %s=%r; using %s', name, value, default)
        return default


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(',') if item.strip()]


def _parse_sample_ratio(raw: str | None) -> float:
    if not raw:
        return 1.0
    try:
        ratio = float(raw)
    except ValueError:
        logger.warning('Invalid YAR_TRACE_SAMPLE_RATIO=%r; using 1.0', raw)
        return 1.0
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


def _parse_resource_attributes(raw: str | None) -> dict[str, str]:
    """Parse comma-separated ``key=value`` pairs into a string dict."""
    if not raw:
        return {}
    out: dict[str, str] = {}
    for chunk in raw.split(','):
        if '=' not in chunk:
            continue
        key, value = chunk.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return out


def _normalize_phoenix_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    normalized = endpoint.rstrip('/')
    if normalized.startswith(('http://', 'https://')) and '/v1/traces' not in normalized:
        return f'{normalized}/v1/traces'
    return normalized


def _strip_traces_suffix(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    if endpoint.endswith('/v1/traces'):
        return endpoint[: -len('/v1/traces')]
    return endpoint


# ---------------------------------------------------------------------------
# OpenInference semconv (lazy / soft imports so the module degrades safely)
# ---------------------------------------------------------------------------


def _load_semconv() -> dict[str, Any]:
    try:
        from openinference.semconv.trace import (  # type: ignore[import-not-found]
            DocumentAttributes,
            EmbeddingAttributes,
            MessageAttributes,
            OpenInferenceMimeTypeValues,
            OpenInferenceSpanKindValues,
            RerankerAttributes,
            SpanAttributes,
            ToolCallAttributes,
        )
    except Exception:  # pragma: no cover - dependency missing
        return {}
    return {
        'SpanAttributes': SpanAttributes,
        'OpenInferenceSpanKindValues': OpenInferenceSpanKindValues,
        'OpenInferenceMimeTypeValues': OpenInferenceMimeTypeValues,
        'DocumentAttributes': DocumentAttributes,
        'EmbeddingAttributes': EmbeddingAttributes,
        'MessageAttributes': MessageAttributes,
        'RerankerAttributes': RerankerAttributes,
        'ToolCallAttributes': ToolCallAttributes,
    }


_SEMCONV: dict[str, Any] = _load_semconv()


def _semconv_value(group: str, attr: str, fallback: str) -> str:
    cls = _SEMCONV.get(group)
    if cls is None:
        return fallback
    return getattr(cls, attr, fallback)


def _span_attr(name: str, fallback: str) -> str:
    return _semconv_value('SpanAttributes', name, fallback)


def _kind_value(name: str, fallback: str) -> str:
    cls = _SEMCONV.get('OpenInferenceSpanKindValues')
    if cls is None:
        return fallback
    member = getattr(cls, name, None)
    if member is None:
        return fallback
    return getattr(member, 'value', fallback)


def _mime_value(name: str, fallback: str) -> str:
    cls = _SEMCONV.get('OpenInferenceMimeTypeValues')
    if cls is None:
        return fallback
    member = getattr(cls, name, None)
    if member is None:
        return fallback
    return getattr(member, 'value', fallback)


# ---------------------------------------------------------------------------
# Attribute coercion
# ---------------------------------------------------------------------------


AttributeValue = str | int | float | bool | list[str]


def _json_default(value: Any) -> str:
    return str(value)


def _coerce_attribute(value: Any) -> AttributeValue | None:
    """Convert arbitrary values to OpenTelemetry-safe scalar/list attributes."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if all(isinstance(item, str) for item in items):
            return items
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _coerce_attributes(attributes: Mapping[str, Any] | None) -> dict[str, AttributeValue]:
    coerced: dict[str, AttributeValue] = {}
    for key, value in (attributes or {}).items():
        coerced_value = _coerce_attribute(value)
        if coerced_value is None:
            continue
        coerced[str(key)] = coerced_value
    return coerced


def _truncate(value: Any, max_chars: int) -> str:
    text = ' '.join(str(value or '').split())
    if max_chars and len(text) > max_chars:
        text = f'{text[:max_chars]}...'
    return text


# ---------------------------------------------------------------------------
# TraceConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceConfig:
    """Environment-driven tracing configuration.

    Tracing is intentionally opt-in. When disabled, or when Phoenix /
    OpenInference dependencies are unavailable, callers receive no-op spans
    and continue normally.
    """

    enabled: bool
    project_name: str
    endpoint: str | None
    auto_instrument: bool = False
    batch: bool = False
    capture_contexts: bool = True
    capture_prompts: bool = True
    capture_embeddings: bool = True
    context_preview_chars: int = 500
    max_items: int = 50
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    instrument_openai: bool = False
    instrument_litellm: bool = False
    instrument_httpx: bool = False
    session_header: str = 'x-session-id'
    user_header: str = 'x-user-id'
    tag_header: str = 'x-yar-trace-tags'
    default_tags: list[str] = field(default_factory=list)
    sample_ratio: float = 1.0
    service_name: str = 'yar'
    service_version: str | None = None
    deployment_environment: str | None = None
    resource_attributes: dict[str, str] = field(default_factory=dict)
    metrics_enabled: bool = False
    metrics_endpoint: str | None = None
    metrics_export_interval_ms: int = 60000

    @classmethod
    def from_env(cls, *, default_project: str = 'yar', enabled_by_default: bool = False) -> TraceConfig:
        headers_env = os.getenv('PHOENIX_CLIENT_HEADERS') or os.getenv('YAR_TRACE_HEADERS')
        headers: dict[str, str] = {}
        if headers_env:
            for chunk in headers_env.split(','):
                if '=' in chunk:
                    k, v = chunk.split('=', 1)
                    if k.strip():
                        headers[k.strip()] = v.strip()

        project_name = (
            os.getenv('YAR_TRACE_PROJECT')
            or os.getenv('PHOENIX_PROJECT_NAME')
            or default_project
        )
        if _env_bool('YAR_EVAL_TRACE_ENABLED', False):
            project_name = 'yar-eval'

        return cls(
            enabled=_env_bool('YAR_TRACE_ENABLED', enabled_by_default),
            project_name=project_name,
            endpoint=_normalize_phoenix_endpoint(os.getenv('PHOENIX_COLLECTOR_ENDPOINT')),
            api_key=os.getenv('PHOENIX_API_KEY') or None,
            headers=headers,
            auto_instrument=_env_bool('YAR_TRACE_AUTO_INSTRUMENT', False),
            batch=_env_bool('YAR_TRACE_BATCH', False),
            capture_contexts=_env_bool('YAR_TRACE_CAPTURE_CONTEXTS', True),
            capture_prompts=_env_bool('YAR_TRACE_CAPTURE_PROMPTS', True),
            capture_embeddings=_env_bool('YAR_TRACE_CAPTURE_EMBEDDINGS', True),
            context_preview_chars=max(0, _env_int('YAR_TRACE_CONTEXT_PREVIEW_CHARS', 500)),
            max_items=max(1, _env_int('YAR_TRACE_MAX_ITEMS', 50)),
            instrument_openai=_env_bool('YAR_TRACE_INSTRUMENT_OPENAI', False),
            instrument_litellm=_env_bool('YAR_TRACE_INSTRUMENT_LITELLM', False),
            instrument_httpx=_env_bool('YAR_TRACE_INSTRUMENT_HTTPX', False),
            session_header=os.getenv('YAR_TRACE_SESSION_HEADER', 'x-session-id'),
            user_header=os.getenv('YAR_TRACE_USER_HEADER', 'x-user-id'),
            tag_header=os.getenv('YAR_TRACE_TAG_HEADER', 'x-yar-trace-tags'),
            default_tags=_env_list('YAR_TRACE_DEFAULT_TAGS'),
            sample_ratio=_parse_sample_ratio(os.getenv('YAR_TRACE_SAMPLE_RATIO')),
            service_name=os.getenv('YAR_SERVICE_NAME', 'yar'),
            service_version=os.getenv('YAR_SERVICE_VERSION') or None,
            deployment_environment=os.getenv('YAR_DEPLOYMENT_ENV') or os.getenv('DEPLOYMENT_ENVIRONMENT') or None,
            resource_attributes=_parse_resource_attributes(os.getenv('YAR_TRACE_RESOURCE_ATTRIBUTES')),
            metrics_enabled=_env_bool('YAR_TRACE_METRICS_ENABLED', False),
            metrics_endpoint=os.getenv('YAR_TRACE_METRICS_ENDPOINT')
            or os.getenv('OTEL_EXPORTER_OTLP_METRICS_ENDPOINT')
            or os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'),
            metrics_export_interval_ms=max(1000, _env_int('YAR_TRACE_METRICS_INTERVAL_MS', 60000)),
        )


# ---------------------------------------------------------------------------
# OpenInference attribute payload builders
# ---------------------------------------------------------------------------


def _input_value_payload(value: Any, *, mime: str = 'text/plain') -> dict[str, Any]:
    if value is None:
        return {}
    return {
        _span_attr('INPUT_VALUE', 'input.value'): value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=_json_default),
        _span_attr('INPUT_MIME_TYPE', 'input.mime_type'): mime,
    }


def _output_value_payload(value: Any, *, mime: str = 'text/plain') -> dict[str, Any]:
    if value is None:
        return {}
    return {
        _span_attr('OUTPUT_VALUE', 'output.value'): value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=_json_default),
        _span_attr('OUTPUT_MIME_TYPE', 'output.mime_type'): mime,
    }


def _llm_messages_payload(prefix: str, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Encode messages as flattened OpenInference attributes.

    Phoenix accepts ``llm.input_messages.<i>.message.role`` /
    ``llm.input_messages.<i>.message.content`` style attributes.
    """
    role_key = _semconv_value('MessageAttributes', 'MESSAGE_ROLE', 'message.role')
    content_key = _semconv_value('MessageAttributes', 'MESSAGE_CONTENT', 'message.content')
    name_key = _semconv_value('MessageAttributes', 'MESSAGE_NAME', 'message.name')
    payload: dict[str, Any] = {}
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            continue
        role = msg.get('role')
        content = msg.get('content')
        name = msg.get('name')
        if role is not None:
            payload[f'{prefix}.{idx}.{role_key}'] = str(role)
        if content is not None:
            payload[f'{prefix}.{idx}.{content_key}'] = (
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=_json_default)
            )
        if name is not None:
            payload[f'{prefix}.{idx}.{name_key}'] = str(name)
    return payload


def _retrieval_documents_payload(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = _span_attr('RETRIEVAL_DOCUMENTS', 'retrieval.documents')
    id_key = _semconv_value('DocumentAttributes', 'DOCUMENT_ID', 'document.id')
    content_key = _semconv_value('DocumentAttributes', 'DOCUMENT_CONTENT', 'document.content')
    score_key = _semconv_value('DocumentAttributes', 'DOCUMENT_SCORE', 'document.score')
    metadata_key = _semconv_value('DocumentAttributes', 'DOCUMENT_METADATA', 'document.metadata')
    payload: dict[str, Any] = {}
    for idx, doc in enumerate(documents):
        if not isinstance(doc, Mapping):
            continue
        if (doc_id := doc.get('id')) is not None:
            payload[f'{base}.{idx}.{id_key}'] = str(doc_id)
        if (content := doc.get('content')) is not None:
            payload[f'{base}.{idx}.{content_key}'] = str(content)
        if (score := doc.get('score')) is not None:
            try:
                payload[f'{base}.{idx}.{score_key}'] = float(score)
            except (TypeError, ValueError):
                payload[f'{base}.{idx}.{score_key}'] = str(score)
        metadata = doc.get('metadata')
        if metadata is not None:
            payload[f'{base}.{idx}.{metadata_key}'] = (
                metadata
                if isinstance(metadata, str)
                else json.dumps(metadata, ensure_ascii=False, default=_json_default)
            )
    return payload


def _reranker_documents_payload(prefix: str, documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    id_key = _semconv_value('DocumentAttributes', 'DOCUMENT_ID', 'document.id')
    content_key = _semconv_value('DocumentAttributes', 'DOCUMENT_CONTENT', 'document.content')
    score_key = _semconv_value('DocumentAttributes', 'DOCUMENT_SCORE', 'document.score')
    payload: dict[str, Any] = {}
    for idx, doc in enumerate(documents):
        if not isinstance(doc, Mapping):
            continue
        if (doc_id := doc.get('id')) is not None:
            payload[f'{prefix}.{idx}.{id_key}'] = str(doc_id)
        if (content := doc.get('content')) is not None:
            payload[f'{prefix}.{idx}.{content_key}'] = str(content)
        if (score := doc.get('score')) is not None:
            try:
                payload[f'{prefix}.{idx}.{score_key}'] = float(score)
            except (TypeError, ValueError):
                payload[f'{prefix}.{idx}.{score_key}'] = str(score)
    return payload


def _embedding_payload(texts: Sequence[str], *, max_chars: int, max_items: int) -> dict[str, Any]:
    base = _span_attr('EMBEDDING_EMBEDDINGS', 'embedding.embeddings')
    text_key = _semconv_value('EmbeddingAttributes', 'EMBEDDING_TEXT', 'embedding.text')
    payload: dict[str, Any] = {}
    for idx, text in enumerate(list(texts)[:max_items]):
        payload[f'{base}.{idx}.{text_key}'] = _truncate(text, max_chars)
    return payload


# ---------------------------------------------------------------------------
# TraceSpan
# ---------------------------------------------------------------------------


class TraceSpan:
    """Span wrapper that degrades to no-op when tracing is unavailable."""

    def __init__(
        self,
        span_cm: Any = None,
        *,
        accumulate_tokens: bool = False,
        rollup_tokens: bool = False,
    ):
        super().__init__()
        self._span_cm = span_cm
        self._span: Any = None
        self._depth_token: contextvars.Token[int] | None = None
        self._token_accumulator_token: contextvars.Token[dict[str, int] | None] | None = None
        self._accumulate_tokens = accumulate_tokens
        self._rollup_tokens = rollup_tokens
        self.trace_id: str | None = None
        self.span_id: str | None = None

    # Context-manager plumbing -------------------------------------------------

    def __enter__(self) -> TraceSpan:
        # Track YAR-managed nesting depth even when the underlying tracer is a
        # fake/test double that doesn't propagate OpenTelemetry context.
        if self._rollup_tokens:
            # Fresh accumulator for this chain; descendants add their token
            # counts here, we drain on __exit__.
            self._token_accumulator_token = _push_chain_token_accumulator()
        if self._span_cm is None:
            return self
        self._depth_token = _push_active_span()
        try:
            self._span = self._span_cm.__enter__()
            context = self._span.get_span_context() if self._span else None
            if context and getattr(context, 'is_valid', False):
                self.trace_id = f'{context.trace_id:032x}'
                self.span_id = f'{context.span_id:016x}'
                # Stash on a contextvar so the FastAPI middleware can surface
                # them as response headers — the chain ``with`` block exits
                # inside the route handler, so the middleware can no longer
                # read ``get_current_span()`` after ``call_next``.
                _last_request_trace_id.set(self.trace_id)
                _last_request_span_id.set(self.span_id)
        except Exception as exc:
            logger.warning('Failed to enter YAR trace span: %s', exc)
            self._span = None
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool | None:
        if exc_value is not None:
            self.record_exception(exc_value)
            self.set_status_error(str(exc_value))
        if self._rollup_tokens:
            acc = _get_chain_token_accumulator() or {}
            if acc:
                self._write_token_count_attrs(
                    prompt=acc.get('prompt'),
                    completion=acc.get('completion'),
                    total=acc.get('total'),
                )
        if self._span_cm is None:
            self._reset_token_accumulator()
            return None
        try:
            return self._span_cm.__exit__(exc_type, exc_value, traceback)
        except Exception as exc:
            logger.warning('Failed to exit YAR trace span: %s', exc)
            return None
        finally:
            if self._depth_token is not None:
                _pop_active_span(self._depth_token)
                self._depth_token = None
            self._reset_token_accumulator()

    def _reset_token_accumulator(self) -> None:
        if self._token_accumulator_token is not None:
            with contextlib.suppress(ValueError, LookupError):
                _pop_chain_token_accumulator(self._token_accumulator_token)
            self._token_accumulator_token = None

    # Generic attribute helpers -----------------------------------------------

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is None:
            return
        coerced = _coerce_attribute(value)
        if coerced is None:
            return
        try:
            self._span.set_attribute(key, coerced)
        except Exception as exc:
            logger.debug('Failed to set YAR trace attribute %s: %s', key, exc)

    def set_attributes(self, attributes: Mapping[str, Any] | None) -> None:
        if self._span is None or not attributes:
            return
        try:
            self._span.set_attributes(_coerce_attributes(attributes))
        except Exception as exc:
            logger.debug('Failed to set YAR trace attributes: %s', exc)

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        if self._span is None:
            return
        try:
            self._span.add_event(name, _coerce_attributes(attributes))
        except Exception as exc:
            logger.debug('Failed to add YAR trace event %s: %s', name, exc)

    def record_exception(self, exc: BaseException) -> None:
        if self._span is None:
            return
        try:
            self._span.record_exception(exc)
        except Exception as trace_exc:
            logger.debug('Failed to record YAR trace exception: %s', trace_exc)

    def set_status_error(self, description: str) -> None:
        if self._span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            self._span.set_status(Status(StatusCode.ERROR, description))
        except Exception as exc:
            logger.debug('Failed to set YAR trace error status: %s', exc)

    def identifiers(self) -> dict[str, str]:
        identifiers: dict[str, str] = {}
        if self.trace_id:
            identifiers['trace_id'] = self.trace_id
        if self.span_id:
            identifiers['span_id'] = self.span_id
        return identifiers

    # OpenInference convenience setters ---------------------------------------

    def set_input_value(self, value: Any, *, mime: str = 'text/plain') -> None:
        self.set_attributes(_input_value_payload(value, mime=mime))

    def set_output_value(self, value: Any, *, mime: str = 'text/plain') -> None:
        self.set_attributes(_output_value_payload(value, mime=mime))

    def set_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        if not metadata:
            return
        try:
            payload = json.dumps(dict(metadata), ensure_ascii=False, default=_json_default)
        except Exception as exc:
            logger.debug('Failed to serialize trace metadata: %s', exc)
            return
        self.set_attribute(_span_attr('METADATA', 'metadata'), payload)

    def set_tags(self, tags: Sequence[str] | None) -> None:
        if not tags:
            return
        self.set_attribute(_span_attr('TAG_TAGS', 'tag.tags'), [str(t) for t in tags])

    def set_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        self.set_attribute(_span_attr('SESSION_ID', 'session.id'), str(session_id))

    def set_user(self, user_id: str | None) -> None:
        if not user_id:
            return
        self.set_attribute(_span_attr('USER_ID', 'user.id'), str(user_id))

    # LLM-specific setters -----------------------------------------------------

    def set_llm_model(
        self, *, model: str | None = None, provider: str | None = None, system: str | None = None
    ) -> None:
        attrs: dict[str, Any] = {}
        if model is not None:
            attrs[_span_attr('LLM_MODEL_NAME', 'llm.model_name')] = str(model)
        if provider is not None:
            attrs[_span_attr('LLM_PROVIDER', 'llm.provider')] = str(provider)
        if system is not None:
            attrs[_span_attr('LLM_SYSTEM', 'llm.system')] = str(system)
        self.set_attributes(attrs)

    def set_llm_invocation_parameters(self, parameters: Mapping[str, Any] | None) -> None:
        if not parameters:
            return
        try:
            payload = json.dumps(dict(parameters), ensure_ascii=False, default=_json_default)
        except Exception as exc:
            logger.debug('Failed to serialize llm invocation parameters: %s', exc)
            return
        self.set_attribute(
            _span_attr('LLM_INVOCATION_PARAMETERS', 'llm.invocation_parameters'),
            payload,
        )

    def set_llm_input_messages(self, messages: Sequence[Mapping[str, Any]] | None) -> None:
        if not messages:
            return
        prefix = _span_attr('LLM_INPUT_MESSAGES', 'llm.input_messages')
        self.set_attributes(_llm_messages_payload(prefix, messages))

    def set_llm_output_messages(self, messages: Sequence[Mapping[str, Any]] | None) -> None:
        if not messages:
            return
        prefix = _span_attr('LLM_OUTPUT_MESSAGES', 'llm.output_messages')
        self.set_attributes(_llm_messages_payload(prefix, messages))

    def set_llm_token_counts(
        self,
        *,
        prompt: int | None = None,
        completion: int | None = None,
        total: int | None = None,
        cache_read: int | None = None,
        cache_write: int | None = None,
        reasoning: int | None = None,
    ) -> None:
        self._write_token_count_attrs(
            prompt=prompt,
            completion=completion,
            total=total,
            cache_read=cache_read,
            cache_write=cache_write,
            reasoning=reasoning,
        )
        if self._accumulate_tokens:
            _add_chain_tokens(prompt=prompt, completion=completion, total=total)

    def _write_token_count_attrs(
        self,
        *,
        prompt: int | None = None,
        completion: int | None = None,
        total: int | None = None,
        cache_read: int | None = None,
        cache_write: int | None = None,
        reasoning: int | None = None,
    ) -> None:
        attrs: dict[str, Any] = {}
        if prompt is not None:
            attrs[_span_attr('LLM_TOKEN_COUNT_PROMPT', 'llm.token_count.prompt')] = int(prompt)
        if completion is not None:
            attrs[_span_attr('LLM_TOKEN_COUNT_COMPLETION', 'llm.token_count.completion')] = int(completion)
        if total is not None:
            attrs[_span_attr('LLM_TOKEN_COUNT_TOTAL', 'llm.token_count.total')] = int(total)
        if cache_read is not None:
            attrs[
                _span_attr('LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ', 'llm.token_count.prompt_details.cache_read')
            ] = int(cache_read)
        if cache_write is not None:
            attrs[
                _span_attr('LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE', 'llm.token_count.prompt_details.cache_write')
            ] = int(cache_write)
        if reasoning is not None:
            attrs[
                _span_attr(
                    'LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING', 'llm.token_count.completion_details.reasoning'
                )
            ] = int(reasoning)
        self.set_attributes(attrs)

    def set_llm_cost(
        self,
        *,
        prompt: float | None = None,
        completion: float | None = None,
        total: float | None = None,
    ) -> None:
        attrs: dict[str, Any] = {}
        if prompt is not None:
            attrs[_span_attr('LLM_COST_PROMPT', 'llm.cost.prompt')] = float(prompt)
        if completion is not None:
            attrs[_span_attr('LLM_COST_COMPLETION', 'llm.cost.completion')] = float(completion)
        if total is not None:
            attrs[_span_attr('LLM_COST_TOTAL', 'llm.cost.total')] = float(total)
        self.set_attributes(attrs)

    def set_llm_finish_reason(self, reason: str | None) -> None:
        if not reason:
            return
        self.set_attribute(_span_attr('LLM_FINISH_REASON', 'llm.finish_reason'), str(reason))

    def set_llm_prompt_template(
        self,
        *,
        template: str | None = None,
        variables: Mapping[str, Any] | None = None,
        version: str | None = None,
    ) -> None:
        attrs: dict[str, Any] = {}
        if template is not None:
            attrs[_span_attr('LLM_PROMPT_TEMPLATE', 'llm.prompt_template.template')] = str(template)
        if variables is not None:
            try:
                attrs[_span_attr('LLM_PROMPT_TEMPLATE_VARIABLES', 'llm.prompt_template.variables')] = json.dumps(
                    dict(variables), ensure_ascii=False, default=_json_default
                )
            except Exception as exc:
                logger.debug('Failed to serialize prompt template variables: %s', exc)
        if version is not None:
            attrs[_span_attr('LLM_PROMPT_TEMPLATE_VERSION', 'llm.prompt_template.version')] = str(version)
        self.set_attributes(attrs)

    # Embedding ----------------------------------------------------------------

    def set_embedding_model(self, *, model: str | None = None, provider: str | None = None) -> None:
        attrs: dict[str, Any] = {}
        if model is not None:
            attrs[_span_attr('EMBEDDING_MODEL_NAME', 'embedding.model_name')] = str(model)
        if provider is not None:
            attrs[_span_attr('LLM_PROVIDER', 'llm.provider')] = str(provider)
        self.set_attributes(attrs)

    def set_embedding_inputs(self, texts: Sequence[str], *, max_chars: int, max_items: int) -> None:
        if not texts:
            return
        self.set_attributes(_embedding_payload(texts, max_chars=max_chars, max_items=max_items))

    # Retrieval ---------------------------------------------------------------

    def set_retrieval_documents(self, documents: Sequence[Mapping[str, Any]] | None) -> None:
        if not documents:
            return
        self.set_attributes(_retrieval_documents_payload(documents))

    # Reranker ----------------------------------------------------------------

    def set_reranker(
        self,
        *,
        model: str | None = None,
        query: str | None = None,
        top_k: int | None = None,
        input_documents: Sequence[Mapping[str, Any]] | None = None,
        output_documents: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        attrs: dict[str, Any] = {}
        if model is not None:
            attrs[_semconv_value('RerankerAttributes', 'RERANKER_MODEL_NAME', 'reranker.model_name')] = str(model)
        if query is not None:
            attrs[_semconv_value('RerankerAttributes', 'RERANKER_QUERY', 'reranker.query')] = str(query)
        if top_k is not None:
            attrs[_semconv_value('RerankerAttributes', 'RERANKER_TOP_K', 'reranker.top_k')] = int(top_k)
        if input_documents:
            attrs.update(
                _reranker_documents_payload(
                    _semconv_value('RerankerAttributes', 'RERANKER_INPUT_DOCUMENTS', 'reranker.input_documents'),
                    input_documents,
                )
            )
        if output_documents:
            attrs.update(
                _reranker_documents_payload(
                    _semconv_value('RerankerAttributes', 'RERANKER_OUTPUT_DOCUMENTS', 'reranker.output_documents'),
                    output_documents,
                )
            )
        self.set_attributes(attrs)

    # Tool / Prompt -----------------------------------------------------------

    def set_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        attrs: dict[str, Any] = {}
        if name is not None:
            attrs[_span_attr('TOOL_NAME', 'tool.name')] = str(name)
        if description is not None:
            attrs[_span_attr('TOOL_DESCRIPTION', 'tool.description')] = str(description)
        if parameters is not None:
            try:
                attrs[_span_attr('TOOL_PARAMETERS', 'tool.parameters')] = json.dumps(
                    dict(parameters), ensure_ascii=False, default=_json_default
                )
            except Exception as exc:
                logger.debug('Failed to serialize tool parameters: %s', exc)
        self.set_attributes(attrs)

    def set_prompt(
        self,
        *,
        prompt_id: str | None = None,
        url: str | None = None,
        vendor: str | None = None,
    ) -> None:
        attrs: dict[str, Any] = {}
        if prompt_id is not None:
            attrs[_span_attr('PROMPT_ID', 'prompt.id')] = str(prompt_id)
        if url is not None:
            attrs[_span_attr('PROMPT_URL', 'prompt.url')] = str(url)
        if vendor is not None:
            attrs[_span_attr('PROMPT_VENDOR', 'prompt.vendor')] = str(vendor)
        self.set_attributes(attrs)


# ---------------------------------------------------------------------------
# TraceManager
# ---------------------------------------------------------------------------


_active_state: dict[str, TraceManager | None] = {'manager': None}


def get_active_trace_manager() -> TraceManager | None:
    """Return the process-wide trace manager, if installed."""
    return _active_state['manager']


def _set_active_trace_manager(manager: TraceManager | None) -> None:
    _active_state['manager'] = manager


_active_span_depth: contextvars.ContextVar[int] = contextvars.ContextVar('yar_trace_active_span_depth', default=0)
_last_request_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'yar_trace_last_request_trace_id', default=None
)
_last_request_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'yar_trace_last_request_span_id', default=None
)


def _push_active_span() -> contextvars.Token[int]:
    return _active_span_depth.set(_active_span_depth.get() + 1)


def _pop_active_span(token: contextvars.Token[int]) -> None:
    with contextlib.suppress(ValueError, LookupError):
        _active_span_depth.reset(token)


_chain_token_accumulator: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    'yar_trace_chain_token_accumulator', default=None
)


def _push_chain_token_accumulator() -> contextvars.Token[dict[str, int] | None]:
    """Begin a fresh chain-level token accumulator for any descendant LLM spans."""
    return _chain_token_accumulator.set({})


def _pop_chain_token_accumulator(token: contextvars.Token[dict[str, int] | None]) -> None:
    with contextlib.suppress(ValueError, LookupError):
        _chain_token_accumulator.reset(token)


def _get_chain_token_accumulator() -> dict[str, int] | None:
    return _chain_token_accumulator.get()


def _add_chain_tokens(
    *,
    prompt: int | None = None,
    completion: int | None = None,
    total: int | None = None,
) -> None:
    """Add LLM token counts to the active chain accumulator, if any.

    Becomes a no-op when no chain-rollup span is active in the current
    contextvars scope (e.g. an LLM span fired outside of a chain).
    """
    acc = _chain_token_accumulator.get()
    if acc is None:
        return
    if prompt:
        acc['prompt'] = acc.get('prompt', 0) + int(prompt)
    if completion:
        acc['completion'] = acc.get('completion', 0) + int(completion)
    if total:
        acc['total'] = acc.get('total', 0) + int(total)


def _has_recording_parent_span() -> bool:
    """Return True when a YAR-managed span (or any OTel span) is currently active."""
    if _active_span_depth.get() > 0:
        return True
    try:
        from opentelemetry import trace as _otel_trace
    except Exception:  # pragma: no cover - opentelemetry missing
        return False
    span = _otel_trace.get_current_span()
    if span is None:
        return False
    is_recording = getattr(span, 'is_recording', None)
    if callable(is_recording) and not is_recording():
        return False
    context = span.get_span_context()
    return bool(context and getattr(context, 'is_valid', False))


def record_retry_event(retry_state: Any) -> None:
    """Emit a span event on the current OTel span describing a retry attempt.

    Designed to be called from a tenacity ``before_sleep`` hook. It is safe
    to call when no span is active; the call simply no-ops.
    """
    try:
        from opentelemetry import trace as _otel_trace
    except Exception:  # pragma: no cover - dependency missing
        return
    span = _otel_trace.get_current_span()
    if span is None:
        return
    is_recording = getattr(span, 'is_recording', None)
    if callable(is_recording) and not is_recording():
        return
    attempt = getattr(retry_state, 'attempt_number', 0)
    outcome = getattr(retry_state, 'outcome', None)
    seconds = getattr(retry_state, 'next_action', None)
    sleep_for = getattr(seconds, 'sleep', None) if seconds is not None else None
    exc = outcome.exception() if outcome is not None and getattr(outcome, 'failed', False) else None
    attrs: dict[str, Any] = {'retry.attempt': int(attempt)}
    if sleep_for is not None:
        attrs['retry.sleep_seconds'] = float(sleep_for)
    if exc is not None:
        attrs['retry.exception_type'] = type(exc).__name__
        attrs['retry.exception_message'] = str(exc)[:500]
    try:
        span.add_event('retry.attempt', _coerce_attributes(attrs))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug('Failed to emit retry span event: %s', e)


def record_llm_finish_reason(reason: str | None) -> None:
    """Set ``llm.finish_reason`` and ``llm.output_truncated`` on the current span.

    Called by the OpenAI client wrapper after a non-streaming completion so the
    LLM span exposes whether output was cut off (``finish_reason == 'length'``).
    """
    if not reason:
        return
    try:
        from opentelemetry import trace as _otel_trace
    except Exception:  # pragma: no cover
        return
    span = _otel_trace.get_current_span()
    if span is None:
        return
    is_recording = getattr(span, 'is_recording', None)
    if callable(is_recording) and not is_recording():
        return
    try:
        span.set_attribute('llm.finish_reason', str(reason))
        span.set_attribute('llm.output_truncated', reason == 'length')
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug('Failed to set llm.finish_reason: %s', exc)


_query_cache_hit: contextvars.ContextVar[bool] = contextvars.ContextVar('yar_query_cache_hit', default=False)


def mark_query_cache_hit() -> None:
    """Mark the current query as having served its LLM response from cache.

    Called from inside ``yar.operate`` whenever the LLM response cache produced
    a hit instead of running the model. The route handler reads this flag
    after ``aquery_llm`` returns and tags the chain span accordingly.
    """
    _query_cache_hit.set(True)


def reset_query_cache_hit() -> contextvars.Token[bool]:
    """Reset the cache-hit flag. Returns a token to restore prior state."""
    return _query_cache_hit.set(False)


def query_cache_hit_was_set() -> bool:
    """Return True when ``mark_query_cache_hit`` was called in this context."""
    return _query_cache_hit.get()


class TraceManager:
    """Phoenix / OpenInference manager used by eval and API code."""

    def __init__(
        self,
        *,
        config: TraceConfig,
        tracer: Any = None,
        tracer_provider: Any = None,
        disabled_reason: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.tracer = tracer
        self.tracer_provider = tracer_provider
        self.disabled_reason = disabled_reason
        self._client: Any = None
        self._client_attempted = False
        self._instrumentors: list[Any] = []
        self._meter_provider: Any = None
        self._meter: Any = None
        self._instruments: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self.config.enabled and self.tracer is not None

    @classmethod
    def from_env(cls, *, default_project: str = 'yar', enabled_by_default: bool = False) -> TraceManager:
        config = TraceConfig.from_env(default_project=default_project, enabled_by_default=enabled_by_default)
        if not config.enabled:
            return cls(config=config, disabled_reason='disabled')

        try:
            from opentelemetry import trace
            from phoenix.otel import register
        except ImportError as exc:
            logger.warning(
                'YAR tracing requested but Phoenix/OpenTelemetry dependencies are unavailable: %s. '
                'Install with `pip install -e .[observability]`.',
                exc,
            )
            return cls(config=config, disabled_reason='missing_dependencies')

        register_kwargs: dict[str, Any] = {
            'project_name': config.project_name,
            'auto_instrument': config.auto_instrument,
            'batch': config.batch,
        }
        if config.endpoint:
            register_kwargs['endpoint'] = config.endpoint
        if config.api_key:
            register_kwargs['api_key'] = config.api_key
        if config.headers:
            register_kwargs['headers'] = dict(config.headers)

        if config.sample_ratio < 1.0:
            # phoenix.otel.register honours OTel's standard sampler env vars.
            os.environ.setdefault('OTEL_TRACES_SAMPLER', 'parentbased_traceidratio')
            os.environ['OTEL_TRACES_SAMPLER_ARG'] = f'{config.sample_ratio:.6f}'

        resource_pairs: list[str] = [f'service.name={config.service_name}']
        if config.service_version:
            resource_pairs.append(f'service.version={config.service_version}')
        if config.deployment_environment:
            resource_pairs.append(f'deployment.environment={config.deployment_environment}')
        for key, value in config.resource_attributes.items():
            resource_pairs.append(f'{key}={value}')
        existing_resource = os.environ.get('OTEL_RESOURCE_ATTRIBUTES', '').strip()
        if existing_resource:
            resource_pairs.append(existing_resource)
        os.environ['OTEL_RESOURCE_ATTRIBUTES'] = ','.join(resource_pairs)

        try:
            tracer_provider = register(**register_kwargs)
            tracer = trace.get_tracer('yar')
        except Exception as exc:
            logger.warning('YAR tracing requested but Phoenix registration failed: %s', exc)
            return cls(config=config, disabled_reason='registration_failed')

        logger.info(
            'YAR tracing enabled: project=%s endpoint=%s api_key=%s',
            config.project_name,
            config.endpoint or 'default',
            'set' if config.api_key else 'unset',
        )
        manager = cls(config=config, tracer=tracer, tracer_provider=tracer_provider)
        manager._activate_instrumentors()
        if config.metrics_enabled:
            manager._activate_metrics()
        return manager

    # Instrumentation ----------------------------------------------------------

    def _activate_instrumentors(self) -> None:
        if self.config.instrument_openai:
            self._activate_named_instrumentor(
                'openinference.instrumentation.openai',
                'OpenAIInstrumentor',
            )
        if self.config.instrument_litellm:
            self._activate_named_instrumentor(
                'openinference.instrumentation.litellm',
                'LiteLLMInstrumentor',
            )
        if self.config.instrument_httpx:
            self._activate_named_instrumentor(
                'opentelemetry.instrumentation.httpx',
                'HTTPXClientInstrumentor',
            )

    def _activate_named_instrumentor(self, module_name: str, class_name: str) -> None:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception as exc:
            logger.warning(
                'YAR tracing requested instrumentor %s.%s but it is unavailable: %s',
                module_name,
                class_name,
                exc,
            )
            return
        try:
            instrumentor = cls()
            instrumentor.instrument(tracer_provider=self.tracer_provider)
            self._instrumentors.append(instrumentor)
            logger.info('Activated tracing instrumentor: %s.%s', module_name, class_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('Failed to activate %s.%s: %s', module_name, class_name, exc)

    def _activate_metrics(self) -> None:
        """Set up an OTel MeterProvider that exports to OTLP HTTP."""
        try:
            from opentelemetry import metrics
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        except Exception as exc:
            logger.warning('OTel metrics SDK unavailable: %s', exc)
            return

        endpoint = self.config.metrics_endpoint
        exporter_kwargs: dict[str, Any] = {}
        if endpoint:
            # OTLP metric exporter expects a fully-qualified /v1/metrics endpoint.
            normalized = endpoint.rstrip('/')
            if not normalized.endswith('/v1/metrics'):
                normalized = f'{normalized}/v1/metrics'
            exporter_kwargs['endpoint'] = normalized
        if self.config.headers:
            exporter_kwargs['headers'] = dict(self.config.headers)

        try:
            exporter = OTLPMetricExporter(**exporter_kwargs)
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metrics_export_interval_ms,
            )
            provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(provider)
            self._meter_provider = provider
            self._meter = metrics.get_meter('yar')
            logger.info('YAR metrics enabled: endpoint=%s', endpoint or 'default')
        except Exception as exc:
            logger.warning('Failed to set up OTel MeterProvider: %s', exc)

    def _instrument(self, name: str, factory: Any) -> Any:
        if self._meter is None:
            return None
        existing = self._instruments.get(name)
        if existing is not None:
            return existing
        try:
            inst = factory(self._meter)
        except Exception as exc:
            logger.debug('Failed to create instrument %s: %s', name, exc)
            return None
        self._instruments[name] = inst
        return inst

    def record_query_metrics(
        self,
        *,
        endpoint: str,
        mode: str,
        status: str,
        duration_ms: float,
        cached: bool = False,
        error_type: str | None = None,
    ) -> None:
        """Emit OTel metrics for a completed query.

        Increments ``yar.queries.total`` (and ``yar.queries.errors`` on
        failure) and observes ``yar.query.duration_ms``. All instruments
        carry endpoint/mode/status labels for slicing in dashboards.
        """
        if self._meter is None:
            return
        labels = {'endpoint': endpoint, 'mode': mode, 'status': status, 'cached': str(cached).lower()}
        counter = self._instrument(
            'queries.total',
            lambda m: m.create_counter(
                'yar.queries.total',
                description='Total YAR query requests by endpoint/mode/status.',
                unit='1',
            ),
        )
        if counter is not None:
            counter.add(1, attributes=labels)
        if status != 'success' or error_type:
            err_counter = self._instrument(
                'queries.errors',
                lambda m: m.create_counter(
                    'yar.queries.errors',
                    description='YAR query errors by endpoint/mode/error_type.',
                    unit='1',
                ),
            )
            if err_counter is not None:
                err_counter.add(
                    1,
                    attributes={
                        'endpoint': endpoint,
                        'mode': mode,
                        'error_type': error_type or 'unknown',
                    },
                )
        histogram = self._instrument(
            'query.duration',
            lambda m: m.create_histogram(
                'yar.query.duration_ms',
                description='YAR query end-to-end latency in milliseconds.',
                unit='ms',
            ),
        )
        if histogram is not None:
            histogram.record(float(duration_ms), attributes=labels)

    def record_token_metrics(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        model: str,
    ) -> None:
        if self._meter is None:
            return
        prompt_counter = self._instrument(
            'llm.tokens.prompt',
            lambda m: m.create_counter('yar.llm.tokens.prompt', unit='token'),
        )
        completion_counter = self._instrument(
            'llm.tokens.completion',
            lambda m: m.create_counter('yar.llm.tokens.completion', unit='token'),
        )
        labels = {'model': model}
        if prompt_counter is not None and prompt_tokens:
            prompt_counter.add(int(prompt_tokens), attributes=labels)
        if completion_counter is not None and completion_tokens:
            completion_counter.add(int(completion_tokens), attributes=labels)

    # Span construction --------------------------------------------------------

    def start_span(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        return self._build_span(name=name, attributes=attributes, kind=None)

    def start_chain_span(
        self,
        name: str,
        *,
        input_value: Any = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        return self._build_span(
            name=name,
            attributes=attributes,
            kind=_kind_value('CHAIN', 'CHAIN'),
            input_value=input_value,
            rollup_tokens=True,
        )

    def start_llm_span(
        self,
        name: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        system: str | None = None,
        prompt: Any = None,
        system_prompt: str | None = None,
        history_messages: Sequence[Mapping[str, Any]] | None = None,
        invocation_parameters: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        kind = _kind_value('LLM', 'LLM')
        extra: dict[str, Any] = {}
        if model is not None:
            extra[_span_attr('LLM_MODEL_NAME', 'llm.model_name')] = model
        if provider is not None:
            extra[_span_attr('LLM_PROVIDER', 'llm.provider')] = provider
        if system is not None:
            extra[_span_attr('LLM_SYSTEM', 'llm.system')] = system
        if invocation_parameters is not None:
            try:
                extra[_span_attr('LLM_INVOCATION_PARAMETERS', 'llm.invocation_parameters')] = json.dumps(
                    dict(invocation_parameters), ensure_ascii=False, default=_json_default
                )
            except Exception as exc:
                logger.debug('Failed to serialize llm invocation parameters: %s', exc)

        if self.config.capture_prompts:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            for msg in history_messages or []:
                if isinstance(msg, Mapping):
                    messages.append({'role': msg.get('role', 'user'), 'content': msg.get('content', '')})
            if prompt is not None:
                messages.append({'role': 'user', 'content': prompt})
            if messages:
                prefix = _span_attr('LLM_INPUT_MESSAGES', 'llm.input_messages')
                extra.update(_llm_messages_payload(prefix, messages))
                extra.update(
                    _input_value_payload(
                        json.dumps(messages, ensure_ascii=False, default=_json_default),
                        mime=_mime_value('JSON', 'application/json'),
                    )
                )

        if attributes:
            extra.update(dict(attributes))
        return self._build_span(name=name, attributes=extra, kind=kind, require_parent=True, accumulate_tokens=True)

    def start_embedding_span(
        self,
        name: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        texts: Sequence[str] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        kind = _kind_value('EMBEDDING', 'EMBEDDING')
        extra: dict[str, Any] = {}
        if model is not None:
            extra[_span_attr('EMBEDDING_MODEL_NAME', 'embedding.model_name')] = model
        if provider is not None:
            extra[_span_attr('LLM_PROVIDER', 'llm.provider')] = provider
        if texts and (self.config.capture_contexts or self.config.capture_embeddings):
            extra.update(
                _embedding_payload(
                    list(texts),
                    max_chars=self.config.context_preview_chars,
                    max_items=self.config.max_items,
                )
            )
        if attributes:
            extra.update(dict(attributes))
        return self._build_span(name=name, attributes=extra, kind=kind, require_parent=True)

    def start_retriever_span(
        self,
        name: str,
        *,
        query: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        kind = _kind_value('RETRIEVER', 'RETRIEVER')
        extra: dict[str, Any] = {}
        if query is not None and self.config.capture_prompts:
            extra.update(_input_value_payload(query, mime='text/plain'))
        if top_k is not None:
            extra['retrieval.top_k'] = int(top_k)
        if mode is not None:
            extra['retrieval.mode'] = str(mode)
        if attributes:
            extra.update(dict(attributes))
        return self._build_span(name=name, attributes=extra, kind=kind, require_parent=True)

    def start_reranker_span(
        self,
        name: str,
        *,
        model: str | None = None,
        query: str | None = None,
        top_k: int | None = None,
        input_documents: Sequence[Mapping[str, Any]] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        kind = _kind_value('RERANKER', 'RERANKER')
        extra: dict[str, Any] = {}
        if model is not None:
            extra[_semconv_value('RerankerAttributes', 'RERANKER_MODEL_NAME', 'reranker.model_name')] = model
        if query is not None and self.config.capture_prompts:
            extra[_semconv_value('RerankerAttributes', 'RERANKER_QUERY', 'reranker.query')] = query
        if top_k is not None:
            extra[_semconv_value('RerankerAttributes', 'RERANKER_TOP_K', 'reranker.top_k')] = int(top_k)
        if input_documents and self.config.capture_contexts:
            extra.update(
                _reranker_documents_payload(
                    _semconv_value('RerankerAttributes', 'RERANKER_INPUT_DOCUMENTS', 'reranker.input_documents'),
                    input_documents,
                )
            )
        if attributes:
            extra.update(dict(attributes))
        return self._build_span(name=name, attributes=extra, kind=kind, require_parent=True)

    def start_tool_span(
        self,
        name: str,
        *,
        tool_name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        kind = _kind_value('TOOL', 'TOOL')
        extra: dict[str, Any] = {}
        if tool_name is not None:
            extra[_span_attr('TOOL_NAME', 'tool.name')] = tool_name
        if description is not None:
            extra[_span_attr('TOOL_DESCRIPTION', 'tool.description')] = description
        if parameters is not None:
            with contextlib.suppress(Exception):
                extra[_span_attr('TOOL_PARAMETERS', 'tool.parameters')] = json.dumps(
                    dict(parameters), ensure_ascii=False, default=_json_default
                )
        if attributes:
            extra.update(dict(attributes))
        return self._build_span(name=name, attributes=extra, kind=kind, require_parent=True)

    def start_guardrail_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        return self._build_span(
            name=name,
            attributes=attributes,
            kind=_kind_value('GUARDRAIL', 'GUARDRAIL'),
            require_parent=True,
        )

    def start_evaluator_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        return self._build_span(name=name, attributes=attributes, kind=_kind_value('EVALUATOR', 'EVALUATOR'))

    def start_agent_span(
        self,
        name: str,
        *,
        agent_name: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceSpan:
        kind = _kind_value('AGENT', 'AGENT')
        extra: dict[str, Any] = {}
        if agent_name is not None:
            extra[_span_attr('AGENT_NAME', 'agent.name')] = agent_name
        if attributes:
            extra.update(dict(attributes))
        return self._build_span(name=name, attributes=extra, kind=kind)

    def _build_span(
        self,
        *,
        name: str,
        attributes: Mapping[str, Any] | None,
        kind: str | None,
        input_value: Any = None,
        require_parent: bool = False,
        accumulate_tokens: bool = False,
        rollup_tokens: bool = False,
    ) -> TraceSpan:
        if not self.active:
            return TraceSpan(accumulate_tokens=accumulate_tokens, rollup_tokens=rollup_tokens)
        if require_parent and not _has_recording_parent_span():
            return TraceSpan(accumulate_tokens=accumulate_tokens, rollup_tokens=rollup_tokens)
        merged: dict[str, Any] = {}
        if kind is not None:
            merged[_span_attr('OPENINFERENCE_SPAN_KIND', 'openinference.span.kind')] = kind
        if input_value is not None:
            merged.update(
                _input_value_payload(
                    input_value,
                    mime='text/plain' if isinstance(input_value, str) else _mime_value('JSON', 'application/json'),
                )
            )
        if self.config.default_tags:
            merged[_span_attr('TAG_TAGS', 'tag.tags')] = list(self.config.default_tags)
        if attributes:
            merged.update(dict(attributes))
        try:
            return TraceSpan(
                self.tracer.start_as_current_span(name, attributes=_coerce_attributes(merged)),
                accumulate_tokens=accumulate_tokens,
                rollup_tokens=rollup_tokens,
            )
        except Exception as exc:
            logger.warning('Failed to start YAR trace span %s: %s', name, exc)
            return TraceSpan(accumulate_tokens=accumulate_tokens, rollup_tokens=rollup_tokens)

    # Context-manager helpers --------------------------------------------------

    def using_session(self, session_id: str | None) -> Any:
        if not session_id:
            return nullcontext()
        try:
            from openinference.instrumentation import using_session
        except Exception:  # pragma: no cover - dependency missing
            return nullcontext()
        return using_session(str(session_id))

    def using_user(self, user_id: str | None) -> Any:
        if not user_id:
            return nullcontext()
        try:
            from openinference.instrumentation import using_user
        except Exception:  # pragma: no cover - dependency missing
            return nullcontext()
        return using_user(str(user_id))

    def using_metadata(self, metadata: Mapping[str, Any] | None) -> Any:
        if not metadata:
            return nullcontext()
        try:
            from openinference.instrumentation import using_metadata
        except Exception:  # pragma: no cover - dependency missing
            return nullcontext()
        return using_metadata(dict(metadata))

    def using_tags(self, tags: Sequence[str] | None) -> Any:
        if not tags:
            return nullcontext()
        try:
            from openinference.instrumentation import using_tags
        except Exception:  # pragma: no cover - dependency missing
            return nullcontext()
        return using_tags(list(tags))

    def using_prompt_template(
        self,
        *,
        template: str | None = None,
        variables: Mapping[str, Any] | None = None,
        version: str | None = None,
    ) -> Any:
        if template is None and not variables:
            return nullcontext()
        try:
            from openinference.instrumentation import using_prompt_template
        except Exception:  # pragma: no cover - dependency missing
            return nullcontext()
        kwargs: dict[str, Any] = {}
        if template is not None:
            kwargs['template'] = str(template)
        if variables is not None:
            kwargs['variables'] = dict(variables)
        if version is not None:
            kwargs['version'] = str(version)
        try:
            return using_prompt_template(**kwargs)
        except Exception:  # pragma: no cover - dependency missing
            return nullcontext()

    def using_attributes(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        prompt_template: str | None = None,
        prompt_template_variables: Mapping[str, Any] | None = None,
        prompt_template_version: str | None = None,
    ) -> contextlib.AbstractContextManager[Any]:
        """Stack OpenInference context managers; safe when dependencies are missing."""
        try:
            from openinference.instrumentation import using_attributes
        except Exception:  # pragma: no cover
            return nullcontext()
        kwargs: dict[str, Any] = {}
        if session_id is not None:
            kwargs['session_id'] = str(session_id)
        if user_id is not None:
            kwargs['user_id'] = str(user_id)
        if metadata is not None:
            kwargs['metadata'] = dict(metadata)
        if tags is not None:
            kwargs['tags'] = list(tags)
        if prompt_template is not None:
            kwargs['prompt_template'] = str(prompt_template)
        if prompt_template_variables is not None:
            kwargs['prompt_template_variables'] = dict(prompt_template_variables)
        if prompt_template_version is not None:
            kwargs['prompt_template_version'] = str(prompt_template_version)
        if not kwargs:
            return nullcontext()
        try:
            return using_attributes(**kwargs)
        except Exception:  # pragma: no cover
            return nullcontext()

    # Phoenix client -----------------------------------------------------------

    def client(self) -> Any | None:
        """Return a cached ``phoenix.client.Client`` (lazy)."""
        if self._client is not None or self._client_attempted:
            return self._client
        self._client_attempted = True
        try:
            from phoenix.client import Client
        except Exception as exc:
            logger.debug('phoenix.client not importable: %s', exc)
            return None
        endpoint = _strip_traces_suffix(self.config.endpoint)
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs['base_url'] = endpoint
        if self.config.api_key:
            kwargs['api_key'] = self.config.api_key
        if self.config.headers:
            kwargs['headers'] = dict(self.config.headers)
        try:
            self._client = Client(**kwargs)
        except Exception as exc:
            logger.warning('Failed to construct phoenix.client.Client: %s', exc)
            self._client = None
        return self._client

    def has_client(self) -> bool:
        return self.client() is not None

    # Datasets -----------------------------------------------------------------

    def upsert_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        examples: Sequence[Mapping[str, Any]] | None = None,
        inputs: Sequence[Mapping[str, Any]] | None = None,
        outputs: Sequence[Mapping[str, Any]] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any | None:
        client = self.client()
        if client is None:
            logger.debug('upsert_dataset skipped: phoenix client unavailable')
            return None
        try:
            return client.datasets.create_dataset(
                name=name,
                description=description,
                examples=list(examples) if examples is not None else None,
                inputs=list(inputs) if inputs is not None else None,
                outputs=list(outputs) if outputs is not None else None,
                metadata=list(metadata) if metadata is not None else None,
            )
        except Exception as exc:
            logger.warning('Failed to upsert Phoenix dataset %s: %s', name, exc)
            return None

    def add_dataset_examples(
        self,
        dataset: Any,
        *,
        examples: Sequence[Mapping[str, Any]] | None = None,
        inputs: Sequence[Mapping[str, Any]] | None = None,
        outputs: Sequence[Mapping[str, Any]] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.datasets.add_examples_to_dataset(
                dataset=dataset,
                examples=list(examples) if examples is not None else None,
                inputs=list(inputs) if inputs is not None else None,
                outputs=list(outputs) if outputs is not None else None,
                metadata=list(metadata) if metadata is not None else None,
            )
        except Exception as exc:
            logger.warning('Failed to extend Phoenix dataset: %s', exc)
            return None

    # Experiments --------------------------------------------------------------

    def run_experiment(
        self,
        *,
        dataset: Any,
        task: Callable[..., Any],
        evaluators: Sequence[Callable[..., Any]] | None = None,
        experiment_name: str | None = None,
        experiment_description: str | None = None,
        experiment_metadata: Mapping[str, Any] | None = None,
        concurrency: int | None = None,
        dry_run: bool = False,
        print_summary: bool = False,
    ) -> Any | None:
        client = self.client()
        if client is None:
            logger.warning('run_experiment skipped: phoenix client unavailable')
            return None
        try:
            return client.experiments.run_experiment(
                dataset=dataset,
                task=task,
                evaluators=list(evaluators) if evaluators is not None else None,
                experiment_name=experiment_name,
                experiment_description=experiment_description,
                experiment_metadata=dict(experiment_metadata) if experiment_metadata else None,
                concurrency=concurrency,
                dry_run=dry_run,
                print_summary=print_summary,
            )
        except Exception as exc:
            logger.warning('Phoenix experiment %s failed: %s', experiment_name, exc)
            return None

    def evaluate_experiment(
        self,
        *,
        experiment: Any,
        evaluators: Sequence[Callable[..., Any]],
        concurrency: int | None = None,
        dry_run: bool = False,
        print_summary: bool = False,
    ) -> Any | None:
        client = self.client()
        if client is None:
            logger.warning('evaluate_experiment skipped: phoenix client unavailable')
            return None
        try:
            return client.experiments.evaluate_experiment(
                experiment=experiment,
                evaluators=list(evaluators),
                concurrency=concurrency,
                dry_run=dry_run,
                print_summary=print_summary,
            )
        except Exception as exc:
            logger.warning('Phoenix experiment evaluation failed: %s', exc)
            return None

    # Annotations --------------------------------------------------------------

    def add_span_annotation(
        self,
        *,
        span_id: str,
        annotation_name: str,
        label: str | None = None,
        score: float | None = None,
        explanation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        annotator_kind: str = 'HUMAN',
    ) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.spans.add_span_annotation(
                span_id=span_id,
                annotation_name=annotation_name,
                label=label,
                score=score,
                explanation=explanation,
                metadata=dict(metadata) if metadata else None,
                annotator_kind=annotator_kind,
            )
        except Exception as exc:
            logger.warning('Failed to add span annotation: %s', exc)
            return None

    def add_trace_annotation(
        self,
        *,
        trace_id: str,
        annotation_name: str,
        label: str | None = None,
        score: float | None = None,
        explanation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        annotator_kind: str = 'HUMAN',
    ) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.traces.add_trace_annotation(
                trace_id=trace_id,
                annotation_name=annotation_name,
                label=label,
                score=score,
                explanation=explanation,
                metadata=dict(metadata) if metadata else None,
                annotator_kind=annotator_kind,
            )
        except Exception as exc:
            logger.warning('Failed to add trace annotation: %s', exc)
            return None

    def add_session_annotation(
        self,
        *,
        session_id: str,
        annotation_name: str,
        label: str | None = None,
        score: float | None = None,
        explanation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        annotator_kind: str = 'HUMAN',
    ) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.sessions.add_session_annotation(
                session_id=session_id,
                annotation_name=annotation_name,
                label=label,
                score=score,
                explanation=explanation,
                metadata=dict(metadata) if metadata else None,
                annotator_kind=annotator_kind,
            )
        except Exception as exc:
            logger.warning('Failed to add session annotation: %s', exc)
            return None

    def log_span_annotations_dataframe(self, dataframe: Any) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.spans.log_span_annotations_dataframe(dataframe=dataframe)
        except Exception as exc:
            logger.warning('Failed to log span annotations dataframe: %s', exc)
            return None

    def add_span_note(self, *, span_id: str, note: str) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.spans.add_span_note(span_id=span_id, note=note)
        except Exception as exc:
            logger.warning('Failed to add span note: %s', exc)
            return None

    # Prompts ------------------------------------------------------------------

    def get_prompt(self, *, name: str, version: str | None = None, tag: str | None = None) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            kwargs: dict[str, Any] = {'name': name}
            if version is not None:
                kwargs['version'] = version
            if tag is not None:
                kwargs['tag'] = tag
            return client.prompts.get(**kwargs)
        except Exception as exc:
            logger.warning('Failed to fetch Phoenix prompt %s: %s', name, exc)
            return None

    def upsert_prompt(
        self,
        *,
        name: str,
        version: Any,
        prompt_description: str | None = None,
        tag: str | None = None,
    ) -> Any | None:
        client = self.client()
        if client is None:
            return None
        try:
            return client.prompts.create(
                name=name,
                version=version,
                prompt_description=prompt_description,
                tag=tag,
            )
        except Exception as exc:
            logger.warning('Failed to upsert Phoenix prompt %s: %s', name, exc)
            return None

    # Lifecycle ----------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            'enabled': self.config.enabled,
            'active': self.active,
            'project_name': self.config.project_name,
            'endpoint': self.config.endpoint,
            'disabled_reason': self.disabled_reason,
            'capture_contexts': self.config.capture_contexts,
            'capture_prompts': self.config.capture_prompts,
            'capture_embeddings': self.config.capture_embeddings,
            'has_client': self.has_client(),
            'instrumentors': [type(i).__name__ for i in self._instrumentors],
        }

    def shutdown(self) -> None:
        for instrumentor in self._instrumentors:
            try:
                instrumentor.uninstrument()
            except Exception as exc:
                logger.debug('Failed to uninstrument %s: %s', type(instrumentor).__name__, exc)
        self._instrumentors.clear()
        if self.tracer_provider is None:
            if get_active_trace_manager() is self:
                _set_active_trace_manager(None)
            return
        shutdown = getattr(self.tracer_provider, 'shutdown', None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                logger.warning('Failed to shut down YAR tracing provider: %s', exc)
        if get_active_trace_manager() is self:
            _set_active_trace_manager(None)


# ---------------------------------------------------------------------------
# Module-level conveniences
# ---------------------------------------------------------------------------


def noop_trace_manager(*, default_project: str = 'yar') -> TraceManager:
    return TraceManager(
        config=TraceConfig.from_env(default_project=default_project),
        disabled_reason='disabled',
    )


def configure_tracing(*, default_project: str = 'yar', enabled_by_default: bool = False) -> TraceManager:
    manager = TraceManager.from_env(default_project=default_project, enabled_by_default=enabled_by_default)
    _set_active_trace_manager(manager)
    return manager


def trace_sequence_preview(values: Iterable[Any], *, max_items: int, max_chars: int) -> list[str]:
    """Return bounded string previews safe for default trace attributes."""
    previews: list[str] = []
    for value in list(values)[:max_items]:
        previews.append(_truncate(value, max_chars))
    return previews


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------


def instrument_fastapi_app(app: Any, tracing: TraceManager) -> None:
    """Attach a context-propagation middleware to a FastAPI app.

    This middleware does **not** create ``http.request`` spans. Only the
    explicit RAG handlers (``app.query``, ``app.query_stream``,
    ``app.query_data``) emit traces, so polling, health, static and webui
    forwarding endpoints stay silent.

    What the middleware still does:

    * reads optional ``X-Session-Id`` / ``X-User-Id`` / ``X-YAR-Trace-Tags``
      headers (configurable via ``YAR_TRACE_*_HEADER`` env vars) and pushes
      them onto the OpenInference context so any downstream span (e.g. the
      ``app.query`` chain span) inherits them.
    * surfaces the resulting ``trace_id`` / ``span_id`` as
      ``x-yar-trace-id`` / ``x-yar-span-id`` response headers when a span
      was actually recorded inside the request.

    Disabled or unavailable tracing remains a true no-op.
    """
    app.state.tracing = tracing
    if not tracing.active or getattr(app.state, 'yar_tracing_middleware_installed', False):
        return

    app.state.yar_tracing_middleware_installed = True

    session_header = tracing.config.session_header.lower()
    user_header = tracing.config.user_header.lower()
    tag_header = tracing.config.tag_header.lower()

    @app.middleware('http')
    async def yar_trace_middleware(request: Any, call_next: Any) -> Any:
        headers = getattr(request, 'headers', None)
        session_id = headers.get(session_header) if headers else None
        user_id = headers.get(user_header) if headers else None
        tags_raw = headers.get(tag_header) if headers else None
        tag_values: list[str] = []
        if tags_raw:
            tag_values = [item.strip() for item in str(tags_raw).split(',') if item.strip()]

        ctx = tracing.using_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tag_values or None,
        )

        with ctx:
            response = await call_next(request)

        # Surface the chain trace_id / span_id as response headers so eval
        # tooling can correlate API responses to Phoenix traces without
        # parsing the body. The chain span has already exited by the time we
        # land here, so we read from contextvars that ``TraceSpan.__enter__``
        # populated rather than ``get_current_span``.
        try:
            trace_id_hex = _last_request_trace_id.get()
            span_id_hex = _last_request_span_id.get()
            if trace_id_hex:
                response_headers = getattr(response, 'headers', None)
                if response_headers is not None:
                    response_headers['x-yar-trace-id'] = trace_id_hex
                    if span_id_hex:
                        response_headers['x-yar-span-id'] = span_id_hex
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug('Failed to set x-yar-trace-id response header: %s', exc)
        return response


__all__ = [
    'TraceConfig',
    'TraceManager',
    'TraceSpan',
    'configure_tracing',
    'get_active_trace_manager',
    'instrument_fastapi_app',
    'noop_trace_manager',
    'trace_sequence_preview',
]
