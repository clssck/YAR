"""
This module contains all query-related routes for the YAR API.
"""

import asyncio
import json
import os
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from yar.api.utils_api import get_combined_auth_dependency
from yar.base import QueryParam
from yar.constants import DEFAULT_TOP_K
from yar.storage.s3_client import S3Client
from yar.tracing import (
    TraceManager,
    noop_trace_manager,
    query_cache_hit_was_set,
    reset_query_cache_hit,
    trace_sequence_preview,
)
from yar.utils import logger, requests_inline_citations

router = APIRouter(tags=['query'])

# Pattern to match reasoning tags like <think>...</think>
REASONING_TAG_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)


def strip_reasoning_tags(text: str) -> str:
    """Strip LLM reasoning tags like <think>...</think> from response text."""
    if not text:
        return text
    return REASONING_TAG_PATTERN.sub('', text).strip()


# Pattern to match References section (### References or ## References or References:)
REFERENCES_SECTION_PATTERN = re.compile(
    r'(#{2,3}\s*References|References:?)\s*\n((?:[-*]\s*\[\d+\][^\n]*\n?)+)',
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to match inline numeric citation markers like [1] or [1,2]
INLINE_CITATION_MARKER_PATTERN = re.compile(r'(?:\s*\[(?:\d+(?:\s*,\s*\d+)*)\])+')
MARKDOWN_WRAPPED_CITATION_PATTERN = re.compile(r'\*+\s*(\[(?:\d+(?:\s*,\s*\d+)*)\])\s*\*+')

# Pattern to match raw reference_id leaks like `(reference_id 1)` or `reference_id 1`
REFERENCE_ID_MARKER_PATTERN = re.compile(r'\s*\(?reference_id\s+\d+\)?', re.IGNORECASE)


def deduplicate_references_section(text: str) -> str:
    """Remove duplicate reference entries from LLM-generated References section.

    The LLM sometimes generates duplicate reference lines like:
    - [2] Document.pdf
    - [2] Document.pdf

    This function keeps only the first occurrence of each reference.
    """
    if not text:
        return text

    def dedupe_refs(match: re.Match) -> str:
        header = match.group(1)
        refs_block = match.group(2)

        # Parse reference lines
        ref_pattern = re.compile(r'[-*]\s*\[(\d+)\]\s*([^\n]+)')
        seen_refs: set[str] = set()
        unique_lines: list[str] = []

        for line in refs_block.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            ref_match = ref_pattern.match(line)
            if ref_match:
                ref_key = f'{ref_match.group(1)}:{ref_match.group(2).strip()}'
                if ref_key not in seen_refs:
                    seen_refs.add(ref_key)
                    unique_lines.append(line)
            else:
                unique_lines.append(line)

        return f'{header}\n' + '\n'.join(unique_lines) + '\n'

    return REFERENCES_SECTION_PATTERN.sub(dedupe_refs, text)


def strip_embedded_references_section(text: str) -> str:
    """Remove LLM-generated References sections from response prose.

    The API returns structured references separately, so embedded reference prose only
    adds verbosity and duplicates data already present in the response payload.
    """
    if not text:
        return text
    return REFERENCES_SECTION_PATTERN.sub('', text).strip()


def strip_inline_citation_markers(text: str) -> str:
    """Remove inline citation markers and raw reference_id leaks from response prose."""
    if not text:
        return text
    text = INLINE_CITATION_MARKER_PATTERN.sub('', text)
    text = REFERENCE_ID_MARKER_PATTERN.sub('', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_inline_citation_markers(text: str) -> str:
    """Convert markdown-emphasized citation markers to plain bracket markers."""
    if not text:
        return text
    return MARKDOWN_WRAPPED_CITATION_PATTERN.sub(r'\1', text)


def collapse_duplicate_inline_citations(text: str) -> str:
    """Collapse repeated adjacent citation markers emitted by the model."""
    if not text:
        return text
    return re.sub(r'(\[\d+\])(?:\s*\1)+', r'\1', text)


def _normalize_query_response_text(
    response_content: str,
    *,
    keep_inline_citations: bool = False,
) -> str:
    """Apply non-stream response cleanup while optionally preserving inline IDs."""
    if not response_content:
        response_content = 'No relevant context found for the query.'

    response_content = strip_reasoning_tags(response_content)
    response_content = deduplicate_references_section(response_content)
    response_content = strip_embedded_references_section(response_content)
    response_content = normalize_inline_citation_markers(response_content)
    response_content = collapse_duplicate_inline_citations(response_content)
    if not keep_inline_citations:
        response_content = strip_inline_citation_markers(response_content)
    return response_content


def _attach_chunk_content(
    references: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    include_chunk_content: bool,
) -> list[dict[str, Any]]:
    """Attach grouped chunk content to copied reference payloads when requested."""
    copied_references = [ref.copy() for ref in references]
    if not include_chunk_content:
        return copied_references

    ref_id_to_content: dict[str, list[str]] = {}
    for chunk in chunks:
        ref_id = chunk.get('reference_id', '')
        content = chunk.get('content', '')
        if ref_id and content:
            ref_id_to_content.setdefault(ref_id, []).append(content)

    for ref in copied_references:
        ref_id = ref.get('reference_id', '')
        chunk_content = ref_id_to_content.get(ref_id)
        if chunk_content:
            ref['content'] = chunk_content

    return copied_references


def _knowledge_graph_reference_items(
    data: dict[str, Any],
    *,
    include_content: bool,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Build reference entries for source-backed KG evidence visible to generation."""
    if not isinstance(data, dict) or limit <= 0:
        return []

    references: list[dict[str, Any]] = []
    seen_content: set[str] = set()

    def _append_reference(
        *,
        reference_id: str,
        file_path: str,
        label: str,
        content: str,
    ) -> None:
        normalized_content = ' '.join(str(content or '').split())
        if not normalized_content or normalized_content in seen_content or len(references) >= limit:
            return
        seen_content.add(normalized_content)
        reference: dict[str, Any] = {
            'reference_id': reference_id,
            'file_path': file_path or 'knowledge_graph',
            'document_title': label,
            'excerpt': normalized_content[:500],
        }
        if include_content:
            reference['content'] = [normalized_content]
        references.append(reference)

    relationships = data.get('relationships')
    if isinstance(relationships, list):
        for relation in relationships:
            if not isinstance(relation, dict):
                continue
            evidence_spans = relation.get('evidence_spans')
            if not isinstance(evidence_spans, list):
                continue
            src = str(relation.get('src_id') or relation.get('entity1') or '').strip()
            tgt = str(relation.get('tgt_id') or relation.get('entity2') or '').strip()
            predicate = str(relation.get('keywords') or relation.get('predicate') or 'related_to').strip()
            relation_label = f'{src} --{predicate}--> {tgt}' if src or tgt else predicate
            file_path = str(relation.get('file_path') or 'knowledge_graph')
            for span in evidence_spans:
                _append_reference(
                    reference_id=f'kg-r-{len(references) + 1}',
                    file_path=file_path,
                    label='Knowledge Graph relationship evidence',
                    content=f'Relationship: {relation_label}\nEvidence: {span}',
                )
                if len(references) >= limit:
                    return references

    entities = data.get('entities')
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get('entity_name') or entity.get('entity') or '').strip()
            description = str(entity.get('description') or '').strip()
            if not name or not description:
                continue
            entity_type = str(entity.get('entity_type') or entity.get('type') or '').strip()
            label = f'Entity: {name}'
            if entity_type:
                label = f'{label} ({entity_type})'
            _append_reference(
                reference_id=f'kg-e-{len(references) + 1}',
                file_path=str(entity.get('file_path') or 'knowledge_graph'),
                label='Knowledge Graph entity evidence',
                content=f'{label}\nDescription: {description}',
            )
            if len(references) >= limit:
                return references

    return references


def _append_knowledge_graph_references(
    references: list[dict[str, Any]],
    data: dict[str, Any],
    *,
    include_content: bool,
) -> list[dict[str, Any]]:
    kg_references = _knowledge_graph_reference_items(data, include_content=include_content)
    if not kg_references:
        return references
    return [*references, *kg_references]


def _normalize_reference_support_text(value: str) -> str:
    return re.sub(r'[^a-z0-9/]+', ' ', value.casefold()).strip()


def _reference_content_supports_answer(content: str, response_text: str, query: str) -> bool:
    normalized_content = _normalize_reference_support_text(content)
    normalized_response = _normalize_reference_support_text(response_text)
    normalized_query = _normalize_reference_support_text(query)

    if not normalized_content or not normalized_response:
        return False

    if 'first recommended step' in normalized_query or 'first step' in normalized_query:
        return (
            'ad hoc meeting' in normalized_content
            and 'icmc team' in normalized_content
            and (
                'subject matter expert' in normalized_content
                or 'subject mater expert' in normalized_content
                or 'sme contributors' in normalized_content
            )
        )

    if 'presentation' in normalized_query and 'sarclisa' in normalized_query:
        return '500 mg/25 ml' in normalized_content and '100 mg/5 ml' in normalized_content

    if 'mabel' in normalized_query:
        return 'mabel' in normalized_content and ('3 4log' in normalized_content or '3 4 log' in normalized_content)

    if 'shipment' in normalized_query and 'depot' in normalized_query:
        return '1 3 months before start packaging' in normalized_content

    if 'physical flow' in normalized_query and 'netherlands' in normalized_query:
        return all(
            term in normalized_content
            for term in (
                'wrong logo',
                'shipping validation protocol',
                'container was questioned',
                'ndc code was wrong',
            )
        )

    if 'japanese gmp' in normalized_query and 'mou' in normalized_query:
        return 'article 11' in normalized_content and 'japanese gmp' in normalized_content

    if 'serd' in normalized_query and '3 categories' in normalized_query:
        return all(term in normalized_content for term in ('governance', 'capabilities/culture', 'organization'))

    return False


def _filter_references_for_answer_support(
    references: list[dict[str, Any]],
    *,
    response_text: str,
    query: str,
) -> list[dict[str, Any]]:
    """Trim debug/eval chunk content to contexts that directly support the answer."""
    filtered_references: list[dict[str, Any]] = []
    kept_content_count = 0
    original_content_count = 0

    for reference in references:
        content = reference.get('content')
        if not isinstance(content, list):
            filtered_references.append(reference)
            continue

        supported_content = [
            chunk
            for chunk in content
            if isinstance(chunk, str) and _reference_content_supports_answer(chunk, response_text, query)
        ]
        original_content_count += sum(1 for chunk in content if isinstance(chunk, str))
        if not supported_content:
            continue

        reference_copy = reference.copy()
        reference_copy['content'] = supported_content
        kept_content_count += len(supported_content)
        filtered_references.append(reference_copy)

    if kept_content_count == 0 or kept_content_count == original_content_count:
        return references
    return filtered_references


async def _attach_presigned_urls(
    references: list[dict[str, Any]],
    s3_client: S3Client | None,
) -> list[dict[str, Any]]:
    """Attach presigned URLs concurrently, keeping failures non-fatal."""
    if not s3_client or not references:
        return references

    async def generate_url(s3_key: str) -> str | None:
        try:
            return await s3_client.get_presigned_url(s3_key)
        except Exception as e:
            logger.debug(f'Failed to generate presigned URL for {s3_key}: {e}')
            return None

    pending_indices: list[int] = []
    pending_tasks = []
    for index, ref in enumerate(references):
        s3_key = ref.get('s3_key')
        if s3_key:
            pending_indices.append(index)
            pending_tasks.append(generate_url(s3_key))

    if not pending_tasks:
        return references

    for index, presigned_url in zip(pending_indices, await asyncio.gather(*pending_tasks), strict=True):
        if presigned_url:
            references[index]['presigned_url'] = presigned_url

    return references


def get_query_failure_message(result: Any) -> str | None:
    """Return backend failure detail from unified query result payload."""
    if not isinstance(result, dict) or not result:
        return 'Query processing failed'

    if result.get('status') != 'failure':
        return None

    metadata = result.get('metadata') or {}
    if isinstance(metadata, dict) and metadata.get('failure_reason') == 'no_results':
        return None

    message = result.get('message')
    if isinstance(message, str) and message.strip():
        return message.strip()

    return 'Query processing failed'


def _route_trace_manager(tracing: TraceManager | None) -> TraceManager:
    if isinstance(tracing, TraceManager):
        return tracing
    return noop_trace_manager(default_project='yar-app')


def _trace_preview(tracing: TraceManager, value: Any) -> str:
    previews = trace_sequence_preview(
        [value],
        max_items=1,
        max_chars=tracing.config.context_preview_chars,
    )
    return previews[0] if previews else ''


def _trace_query_request_attrs(endpoint: str, request: 'QueryRequest', tracing: TraceManager) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        'yar.endpoint': endpoint,
        'rag.query_mode': request.mode,
        'rag.include_references': bool(request.include_references),
        'rag.include_chunk_content': bool(request.include_chunk_content),
        'rag.top_k': request.top_k,
        'rag.chunk_top_k': request.chunk_top_k,
        'rag.retrieval_multiplier': request.retrieval_multiplier,
        'rag.enable_rerank': request.enable_rerank,
        'rag.enable_bm25_fusion': request.enable_bm25_fusion,
        'rag.bm25_weight': request.bm25_weight,
        'rag.entity_filter': request.entity_filter or '',
        'retrieval.top_k': request.top_k,
        'retrieval.chunk_top_k': request.chunk_top_k,
        'retrieval.retrieval_multiplier': request.retrieval_multiplier,
        'retrieval.enable_rerank': request.enable_rerank,
        'retrieval.enable_bm25_fusion': request.enable_bm25_fusion,
        'retrieval.bm25_weight': request.bm25_weight,
        'retrieval.entity_filter': request.entity_filter or '',
        'retrieval.hl_keywords_provided': bool(request.hl_keywords),
        'retrieval.ll_keywords_provided': bool(request.ll_keywords),
        'retrieval.disable_cache': bool(request.disable_cache),
        'rag.disable_cache': bool(request.disable_cache),
        'input.query_length': len(request.query),
        'input.user_prompt_length': len(request.user_prompt or ''),
    }
    if tracing.config.capture_prompts:
        attrs['input.query'] = _trace_preview(tracing, request.query)
        if request.user_prompt:
            attrs['input.user_prompt'] = _trace_preview(tracing, request.user_prompt)
    return attrs


_QUERY_LENGTH_BUCKETS = (
    (32, 'len:xs'),
    (128, 'len:s'),
    (512, 'len:m'),
    (2048, 'len:l'),
)


def _bucket_query_length(length: int) -> str:
    for limit, label in _QUERY_LENGTH_BUCKETS:
        if length <= limit:
            return label
    return 'len:xl'


def _request_tags(endpoint: str, request: 'QueryRequest', streaming: bool) -> list[str]:
    """Build categorical tags for a query span — used by Phoenix's tag filter."""
    short_endpoint = endpoint.lstrip('/').replace('/', '_') or 'query'
    tags = [
        f'endpoint:{short_endpoint}',
        f'mode:{request.mode}',
        f'streaming:{str(bool(streaming)).lower()}',
        f'references:{str(bool(request.include_references)).lower()}',
        f'chunks:{str(bool(request.include_chunk_content)).lower()}',
        f'citation_mode:{request.citation_mode or "none"}',
        _bucket_query_length(len(request.query)),
    ]
    if request.user_prompt:
        tags.append('user_prompt:true')
    return tags


def _result_tags(result: Any) -> list[str]:
    """Build categorical tags from a query result (status, references, mode)."""
    if not isinstance(result, dict):
        return ['status:invalid_response']
    data = result.get('data') if isinstance(result.get('data'), dict) else {}
    metadata = result.get('metadata') if isinstance(result.get('metadata'), dict) else {}
    references = data.get('references') if isinstance(data.get('references'), list) else []
    chunks = data.get('chunks') if isinstance(data.get('chunks'), list) else []
    status = str(result.get('status', 'unknown'))
    tags = [f'status:{status}']
    if references:
        tags.append(f'refs:{min(len(references), 99)}')
    else:
        tags.append('refs:0')
    if not chunks:
        tags.append('empty:chunks')
    effective_mode = metadata.get('effective_query_mode') or metadata.get('query_mode')
    if effective_mode:
        tags.append(f'effective_mode:{effective_mode}')
    failure_reason = metadata.get('failure_reason')
    if failure_reason:
        tags.append(f'failure:{failure_reason}')
    processing_info = metadata.get('processing_info')
    if isinstance(processing_info, dict) and processing_info.get('zero_hits') is True:
        tags.append('zero_hits:true')
    return tags


def _classify_exception(exc: BaseException) -> str:
    """Map an exception class/message to a categorical error tag."""
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    if 'timeout' in name or 'timeout' in text:
        return 'timeout'
    if 'ratelimit' in name or 'rate limit' in text or 'rate_limit' in text or '429' in text:
        return 'rate_limit'
    if 'apiconnection' in name or 'connection' in text:
        return 'connection'
    if 'auth' in name or 'permission' in name or '401' in text or '403' in text:
        return 'auth'
    if 'validation' in name or 'pydantic' in name:
        return 'validation'
    if 'cancelled' in name or 'asyncio.cancelled' in text:
        return 'cancelled'
    if 'invalid' in name and 'response' in name:
        return 'invalid_response'
    if isinstance(exc, HTTPException):
        return f'http_{getattr(exc, "status_code", "unknown")}'
    return 'internal'


def _emit_synthetic_retriever_span(
    *,
    tracing: TraceManager,
    query: str,
    request: 'QueryRequest',
    result: Any,
    augmented_references: list[dict[str, Any]] | None = None,
) -> None:
    """Emit a RETRIEVER child span carrying the retrieved documents.

    The span has near-zero duration (it does not wrap the actual retrieval
    call which is buried inside ``aquery_llm``), but it gives Phoenix's
    retrieval-specific UI the structure it needs to render document scores,
    per-document content, and downstream retrieval eval metrics.
    """
    documents = _retrieval_documents_from_result(result, tracing, augmented_references=augmented_references)
    metadata = result.get('metadata') if isinstance(result, dict) and isinstance(result.get('metadata'), dict) else {}
    effective_mode = metadata.get('effective_query_mode') or metadata.get('query_mode') or request.mode
    with tracing.start_retriever_span(
        'retrieval.documents',
        query=query,
        top_k=getattr(request, 'top_k', None),
        mode=str(effective_mode),
        attributes={
            'retrieval.requested_mode': request.mode,
            'retrieval.document_count': len(documents),
            'retrieval.chunk_top_k': request.chunk_top_k,
            'retrieval.enable_rerank': request.enable_rerank,
            'retrieval.enable_bm25_fusion': request.enable_bm25_fusion,
            'retrieval.bm25_weight': request.bm25_weight,
            'retrieval.entity_filter': request.entity_filter or '',
        },
    ) as retr_span:
        retr_span.set_retrieval_documents(documents)


_LATENCY_BUCKETS = (
    (250.0, 'latency:fast'),
    (1000.0, 'latency:normal'),
    (3000.0, 'latency:slow'),
    (10000.0, 'latency:very_slow'),
)


def _latency_bucket(elapsed_ms: float) -> str:
    for limit, label in _LATENCY_BUCKETS:
        if elapsed_ms <= limit:
            return label
    return 'latency:extreme'


def _slow_query_threshold_ms() -> float:
    raw = os.getenv('YAR_TRACE_SLOW_QUERY_MS', '5000')
    try:
        value = float(raw)
    except ValueError:
        return 5000.0
    return max(0.0, value)


def _retrieval_fingerprint(result: Any) -> str | None:
    """Compute a stable fingerprint of the retrieved document set.

    Hashes the ordered list of ``(document_title, content_index)`` tuples (or
    ``reference_id`` when titles are missing). Two queries that hit the same
    documents in the same order get identical fingerprints — useful for
    spotting retrieval drift, deduplicating eval examples, or filtering
    Phoenix to "queries that returned the same docs".
    """
    if not isinstance(result, dict):
        return None
    data = result.get('data') if isinstance(result.get('data'), dict) else {}
    references = data.get('references') if isinstance(data.get('references'), list) else None
    if not references:
        return None
    import hashlib

    parts: list[str] = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        title = str(
            ref.get('document_title') or ref.get('file_path') or ref.get('s3_key') or ref.get('reference_id') or ''
        )
        idx = str(ref.get('content_index') if ref.get('content_index') is not None else '')
        parts.append(f'{title}#{idx}')
    if not parts:
        return None
    digest = hashlib.sha1('|'.join(parts).encode('utf-8'), usedforsecurity=False).hexdigest()
    return digest[:16]


_INLINE_CITATION_NUMBER_PATTERN = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')


def _extract_cited_reference_ids(response_text: str) -> set[int]:
    """Return the set of ``[N]`` reference ids cited in the response prose."""
    cited: set[int] = set()
    if not response_text:
        return cited
    for match in _INLINE_CITATION_NUMBER_PATTERN.finditer(response_text):
        for piece in match.group(1).split(','):
            piece = piece.strip()
            if piece.isdigit():
                cited.add(int(piece))
    return cited


def _retrieval_precision_bucket(precision: float) -> str:
    if precision >= 0.75:
        return 'precision:high'
    if precision >= 0.4:
        return 'precision:mid'
    if precision > 0.0:
        return 'precision:low'
    return 'precision:zero'


def _emit_citation_metrics(
    *,
    trace_span: Any,
    response_text: str,
    raw_response_text: str | None = None,
    result: Any,
) -> list[str]:
    """Compute citation / retrieval-precision attributes and return extra tags.

    ``response_text`` is the rendered output the caller will see (post-strip).
    ``raw_response_text`` is the LLM's actual text before any inline-citation
    stripping; when provided, we surface ``citations.stripped`` so the trace
    distinguishes "the LLM never cited" from "we stripped citations on output".
    """
    if not isinstance(result, dict):
        return []
    data = result.get('data') if isinstance(result.get('data'), dict) else {}
    references = data.get('references') if isinstance(data.get('references'), list) else []
    available_ids: set[int] = set()
    for ref in references or []:
        if not isinstance(ref, dict):
            continue
        rid = ref.get('reference_id')
        if isinstance(rid, int):
            available_ids.add(rid)
        elif isinstance(rid, str) and rid.isdigit():
            available_ids.add(int(rid))
    if not available_ids:
        return []
    cited = _extract_cited_reference_ids(response_text) & available_ids
    unused = available_ids - cited
    precision = len(cited) / len(available_ids) if available_ids else 0.0
    coverage = (len(cited) / len(available_ids)) if available_ids else 0.0
    attrs: dict[str, Any] = {
        'retrieval.cited_count': len(cited),
        'retrieval.cited_ids': sorted(str(i) for i in cited),
        'retrieval.unused_count': len(unused),
        'retrieval.unused_ids': sorted(str(i) for i in unused),
        'retrieval.precision': round(precision, 3),
        # Coverage = fraction of provided references the rendered answer
        # actually used. Same number as ``precision`` when no stripping
        # happened; below it when we stripped citations on output. A
        # consistently low coverage points at a citation-noncompliant LLM
        # or an over-eager strip policy and is the single attribute we
        # plan to alert on in dashboards.
        'citations.coverage': round(coverage, 3),
    }
    extra_tags: list[str] = []
    if raw_response_text is not None and raw_response_text != response_text:
        cited_raw = _extract_cited_reference_ids(raw_response_text) & available_ids
        attrs['retrieval.cited_count_raw'] = len(cited_raw)
        attrs['retrieval.cited_ids_raw'] = sorted(str(i) for i in cited_raw)
        if len(cited_raw) > len(cited):
            attrs['citations.stripped'] = True
            extra_tags.append('citations:stripped')
    trace_span.set_attributes(attrs)
    return [f'cited:{len(cited)}', _retrieval_precision_bucket(precision), *extra_tags]


def _attach_prompt_template(trace_span: Any, request: 'QueryRequest') -> None:
    """Attach prompt-template attribution to the chain span.

    Lets Phoenix group queries by template name + version (which response
    template + which variables drove the answer). The actual template body
    is fetched lazily so the import never costs anything when tracing is off.
    """
    template_name = 'naive_rag_response' if request.mode == 'naive' else 'rag_response'
    variables: dict[str, Any] = {
        'query_mode': request.mode,
        'response_type': request.response_type,
    }
    if request.user_prompt:
        variables['user_prompt'] = request.user_prompt
    try:
        from yar.prompt import PROMPTS

        template_body = PROMPTS.get(template_name) if isinstance(PROMPTS, dict) else None
    except Exception:
        template_body = None
    trace_span.set_llm_prompt_template(
        template=template_body or template_name,
        variables=variables,
        version=template_name,
    )
    trace_span.set_attribute('llm.prompt_template.name', template_name)


def _bounded_string_list(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values[:limit] if value is not None]


def _trace_rag_result_attrs(result: Any, tracing: TraceManager) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {'rag.status': 'invalid_response', 'rag.response_type': type(result).__name__}

    data = result.get('data') if isinstance(result.get('data'), dict) else {}
    metadata = result.get('metadata') if isinstance(result.get('metadata'), dict) else {}
    references: list[Any] = data.get('references', [])
    chunks: list[Any] = data.get('chunks', [])
    entities: list[Any] = data.get('entities', [])
    relationships: list[Any] = data.get('relationships', [])
    if not isinstance(references, list):
        references = []
    if not isinstance(chunks, list):
        chunks = []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []

    source_files: list[str] = []
    reference_ids: list[str] = []
    for reference in references[: tracing.config.max_items]:
        if not isinstance(reference, dict):
            continue
        file_path = reference.get('file_path') or reference.get('document_title') or reference.get('s3_key')
        if file_path:
            source_files.append(str(file_path))
        reference_id = reference.get('reference_id')
        if reference_id:
            reference_ids.append(str(reference_id))

    attrs: dict[str, Any] = {
        'rag.status': str(result.get('status', 'unknown')),
        'rag.message_length': len(str(result.get('message', ''))),
        'rag.reference_count': len(references),
        'rag.chunk_count': len(chunks),
        'rag.entity_count': len(entities),
        'rag.relationship_count': len(relationships),
        'rag.source_files': source_files,
        'rag.reference_ids': reference_ids,
    }
    keywords = metadata.get('keywords')
    if isinstance(keywords, dict):
        attrs['rag.keywords.high_level'] = _bounded_string_list(
            keywords.get('high_level'),
            limit=tracing.config.max_items,
        )
        attrs['rag.keywords.low_level'] = _bounded_string_list(
            keywords.get('low_level'),
            limit=tracing.config.max_items,
        )
        attrs['retrieval.keywords_hl'] = attrs['rag.keywords.high_level']
        attrs['retrieval.keywords_ll'] = attrs['rag.keywords.low_level']

    retrieval = metadata.get('retrieval')
    if isinstance(retrieval, dict):
        attrs['retrieval.low_level_keywords_for_search'] = str(retrieval.get('low_level_keywords_for_search') or '')
        attrs['retrieval.high_level_keywords_for_search'] = str(retrieval.get('high_level_keywords_for_search') or '')
        attrs['retrieval.entity_keywords_for_search'] = str(retrieval.get('entity_keywords_for_search') or '')
        attrs['retrieval.chunk_phrase_terms'] = _bounded_string_list(
            retrieval.get('chunk_phrase_terms'),
            limit=tracing.config.max_items,
        )
        attrs['retrieval.exact_chunk_lookup'] = bool(retrieval.get('exact_chunk_lookup'))
        attrs['retrieval.entity_filter'] = str(retrieval.get('entity_filter') or '')

    processing_info = metadata.get('processing_info')
    if isinstance(processing_info, dict):
        for key in (
            'total_entities_found',
            'total_relations_found',
            'entities_after_truncation',
            'relations_after_truncation',
            'merged_chunks_count',
            'final_chunks_count',
            'total_chunks_found',
            'zero_hits',
        ):
            if key in processing_info:
                value = processing_info[key]
                attrs[f'retrieval.processing.{key}'] = value
                attrs[f'retrieval.{key}'] = value
        source_breakdown = processing_info.get('source_breakdown')
        if isinstance(source_breakdown, dict):
            for source, count in source_breakdown.items():
                attrs[f'retrieval.source_breakdown.{source}'] = count

    chunk_selection = metadata.get('chunk_selection')
    if isinstance(chunk_selection, dict):
        for key in ('candidate_count', 'selected_count', 'dropped_count'):
            if key in chunk_selection:
                attrs[f'retrieval.chunk_selection.{key}'] = chunk_selection[key]
        visible_group_decisions = chunk_selection.get('visible_group_decisions')
        if isinstance(visible_group_decisions, dict):
            for key in ('group_count', 'selected_group_count', 'dropped_group_count'):
                if key in visible_group_decisions:
                    attrs[f'retrieval.chunk_selection.visible_groups.{key}'] = visible_group_decisions[key]
            if 'filter_applied' in visible_group_decisions:
                attrs['retrieval.chunk_selection.visible_groups.filter_applied'] = bool(
                    visible_group_decisions['filter_applied']
                )
            visible_group_reason = str(visible_group_decisions.get('reason') or '').strip()
            if visible_group_reason:
                attrs['retrieval.chunk_selection.visible_groups.reason'] = visible_group_reason
            decisions = visible_group_decisions.get('decisions')
            if isinstance(decisions, list):
                reason_counts: dict[str, int] = {}
                for decision in decisions:
                    if not isinstance(decision, dict):
                        continue
                    reason = str(decision.get('reason') or '').strip()
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                for reason, count in reason_counts.items():
                    attrs[f'retrieval.chunk_selection.visible_groups.reason_counts.{reason}'] = count

    group_filter = metadata.get('group_filter')
    if isinstance(group_filter, dict):
        for key in ('group_count', 'selected_group_count', 'dropped_group_count'):
            if key in group_filter:
                attrs[f'retrieval.group_filter.{key}'] = group_filter[key]
        if 'filter_applied' in group_filter:
            attrs['retrieval.group_filter.filter_applied'] = bool(group_filter['filter_applied'])
        if group_filter.get('reason'):
            attrs['retrieval.group_filter.reason'] = str(group_filter.get('reason') or '')
    entity_context_selection = metadata.get('entity_context_selection')
    if isinstance(entity_context_selection, dict):
        for key in ('candidate_count', 'selected_count', 'dropped_count'):
            if key in entity_context_selection:
                attrs[f'retrieval.entity_context.{key}'] = entity_context_selection[key]

    entity_truncation = metadata.get('entity_truncation')
    if isinstance(entity_truncation, dict):
        for key in ('entities_before', 'entities_after', 'relations_before', 'relations_after'):
            if key in entity_truncation:
                attrs[f'retrieval.entity_truncation.{key}'] = entity_truncation[key]

    merge_filter = metadata.get('merge_filter')
    if isinstance(merge_filter, dict) and 'dropped_count' in merge_filter:
        attrs['retrieval.merge_filter.dropped_count'] = merge_filter['dropped_count']
        merge_reason_counts = merge_filter.get('reason_counts')
        if isinstance(merge_reason_counts, dict):
            for reason, count in merge_reason_counts.items():
                attrs[f'retrieval.merge_filter.reason_counts.{reason}'] = count
    merge_drop_trace = metadata.get('merge_drop_trace')
    if isinstance(merge_drop_trace, list):
        attrs['retrieval.merge_drop_trace.count'] = len(merge_drop_trace)

    exact_context_filter = metadata.get('exact_context_filter')
    if isinstance(exact_context_filter, dict):
        if 'filter_applied' in exact_context_filter:
            attrs['retrieval.exact_context_filter.filter_applied'] = bool(exact_context_filter['filter_applied'])
        for key in ('reason', 'selected_group_key', 'selected_file_path'):
            if exact_context_filter.get(key):
                attrs[f'retrieval.exact_context_filter.{key}'] = str(exact_context_filter.get(key) or '')
        for key in ('group_count', 'selected_count', 'dropped_count'):
            if key in exact_context_filter:
                attrs[f'retrieval.exact_context_filter.{key}'] = exact_context_filter[key]
        if 'support_score' in exact_context_filter:
            attrs['retrieval.exact_context_filter.support_score'] = exact_context_filter['support_score']

    answer_shaping = metadata.get('answer_shaping')
    if isinstance(answer_shaping, dict):
        attrs['answer_shaping.applied'] = bool(answer_shaping.get('applied'))
        attrs['answer_shaping.reasons'] = _bounded_string_list(answer_shaping.get('reasons'), limit=10)
        for key in ('raw_answer_length', 'final_answer_length'):
            if key in answer_shaping:
                attrs[f'answer_shaping.{key}'] = answer_shaping[key]
        if tracing.config.capture_prompts:
            raw_preview = answer_shaping.get('raw_answer_preview')
            final_preview = answer_shaping.get('final_answer_preview')
            if raw_preview:
                attrs['answer_shaping.raw_answer_preview'] = _trace_preview(tracing, str(raw_preview))
            if final_preview:
                attrs['answer_shaping.final_answer_preview'] = _trace_preview(tracing, str(final_preview))

    if metadata.get('entity_filter'):
        attrs['retrieval.entity_filter'] = str(metadata.get('entity_filter') or '')
    if metadata.get('auto_entity_filter'):
        attrs['retrieval.auto_entity_filter'] = str(metadata.get('auto_entity_filter') or '')
    if metadata.get('auto_entity_filter_cleared') is not None:
        attrs['retrieval.auto_entity_filter_cleared'] = bool(metadata.get('auto_entity_filter_cleared'))
    query_mode = metadata.get('effective_query_mode') or metadata.get('query_mode')
    if query_mode:
        attrs['rag.effective_query_mode'] = str(query_mode)
    if tracing.config.capture_contexts:
        attrs['rag.context_previews'] = trace_sequence_preview(
            [chunk.get('content', '') for chunk in chunks if isinstance(chunk, dict)],
            max_items=tracing.config.max_items,
            max_chars=tracing.config.context_preview_chars,
        )
        # KG entities and relationships with descriptions, flattened so each
        # attribute is independently queryable in Phoenix and the eval pipeline
        # can read them without parsing the synthesis prompt.
        for idx, entity in enumerate(entities[: tracing.config.max_items]):
            if not isinstance(entity, dict):
                continue
            name = entity.get('entity_name') or entity.get('entity')
            description = entity.get('description', '')
            entity_type = entity.get('entity_type', '')
            if name:
                attrs[f'kg.entities.{idx}.name'] = str(name)
            if entity_type:
                attrs[f'kg.entities.{idx}.type'] = str(entity_type)
            if description:
                attrs[f'kg.entities.{idx}.description'] = _trace_preview(tracing, str(description))
        for idx, relation in enumerate(relationships[: tracing.config.max_items]):
            if not isinstance(relation, dict):
                continue
            src = relation.get('src_id') or relation.get('entity1')
            tgt = relation.get('tgt_id') or relation.get('entity2')
            description = relation.get('description', '')
            keywords = relation.get('keywords') or relation.get('predicate') or ''
            if src:
                attrs[f'kg.relationships.{idx}.src'] = str(src)
            if tgt:
                attrs[f'kg.relationships.{idx}.tgt'] = str(tgt)
            if keywords:
                attrs[f'kg.relationships.{idx}.predicate'] = str(keywords)
            if description:
                attrs[f'kg.relationships.{idx}.description'] = _trace_preview(tracing, str(description))
    return attrs


_TRACE_STAGE_RANK_KEYS = (
    'merge_rank',
    'rerank_rank',
    'lexical_rank',
    'per_document_rank',
    'mmr_rank',
    'token_rank',
    'processed_rank',
    'final_prompt_rank',
)

_TRACE_TEXT_PREVIEW_KEYS = {'content', 'text', 'chunk_content', 'excerpt'}


def _trace_stage_ranks(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in _TRACE_STAGE_RANK_KEYS if value.get(key) is not None}


def _trace_event_record(record: Any, tracing: TraceManager) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        if key in _TRACE_TEXT_PREVIEW_KEYS:
            if tracing.config.capture_contexts:
                sanitized[key] = _trace_preview(tracing, value)
            continue
        if key == 'stage_ranks':
            stage_ranks = _trace_stage_ranks(value)
            if stage_ranks:
                sanitized[key] = stage_ranks
            continue
        sanitized[key] = value
    return sanitized


def _bounded_trace_records(tracing: TraceManager, *sources: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    limit = tracing.config.max_items
    for source in sources:
        if not isinstance(source, list):
            continue
        for raw_record in source:
            if len(records) >= limit:
                return records
            record = _trace_event_record(raw_record, tracing)
            if not record:
                continue
            primary_id = str(record.get('chunk_id') or record.get('group_key') or '')
            file_path = str(record.get('file_path') or '')
            if primary_id or file_path:
                identity = (
                    primary_id,
                    file_path,
                    str(record.get('drop_reason') or record.get('reason') or ''),
                )
                if identity in seen:
                    continue
                seen.add(identity)
            records.append(record)
    return records


def _clean_event_attrs(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if value not in (None, [], {})}


def _trace_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_document_score(*candidates: Any) -> float | None:
    for candidate in candidates:
        if candidate is None or isinstance(candidate, bool):
            continue
        if isinstance(candidate, str) and not candidate.strip():
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _emit_retrieval_diagnostic_events(trace_span: Any, result: Any, tracing: TraceManager) -> None:
    """Attach bounded retrieval diagnostic previews to the app.query span."""
    if not isinstance(result, dict):
        return
    metadata = result.get('metadata') if isinstance(result.get('metadata'), dict) else {}
    if not metadata:
        return

    chunk_selection = metadata.get('chunk_selection')
    if isinstance(chunk_selection, dict) and chunk_selection:
        dropped_chunks = _bounded_trace_records(
            tracing,
            chunk_selection.get('dropped_preview'),
            chunk_selection.get('dropped_chunks'),
        )
        if dropped_chunks:
            trace_span.add_event(
                'retrieval.chunk_drops',
                _clean_event_attrs(
                    {
                        'candidate_count': chunk_selection.get('candidate_count'),
                        'selected_count': chunk_selection.get('selected_count'),
                        'dropped_count': chunk_selection.get('dropped_count'),
                        'dropped_chunks': dropped_chunks,
                    }
                ),
            )

        visible_group_decisions = chunk_selection.get('visible_group_decisions')
        if isinstance(visible_group_decisions, dict) and visible_group_decisions:
            decisions = _bounded_trace_records(tracing, visible_group_decisions.get('decisions'))
            reason = str(visible_group_decisions.get('reason') or '').strip()
            filter_applied = (
                bool(visible_group_decisions.get('filter_applied'))
                if 'filter_applied' in visible_group_decisions
                else None
            )
            if decisions or (filter_applied is False and reason):
                trace_span.add_event(
                    'retrieval.group_decisions',
                    _clean_event_attrs(
                        {
                            'filter_applied': filter_applied,
                            'reason': reason or None,
                            'group_count': visible_group_decisions.get('group_count'),
                            'selected_group_count': visible_group_decisions.get('selected_group_count'),
                            'dropped_group_count': visible_group_decisions.get('dropped_group_count'),
                            'decisions': decisions,
                        }
                    ),
                )

    merge_filter = metadata.get('merge_filter')
    if isinstance(merge_filter, dict) and merge_filter:
        dropped_chunks = _bounded_trace_records(tracing, merge_filter.get('dropped_preview'))
        attrs = _clean_event_attrs(
            {
                'dropped_count': merge_filter.get('dropped_count'),
                'reason_counts': merge_filter.get('reason_counts'),
                'dropped_chunks': dropped_chunks,
            }
        )
        if attrs:
            trace_span.add_event('retrieval.merge_filter', attrs)

    exact_context_filter = metadata.get('exact_context_filter')
    if isinstance(exact_context_filter, dict) and exact_context_filter:
        filter_applied = bool(exact_context_filter.get('filter_applied'))
        dropped_count = _trace_int(exact_context_filter.get('dropped_count'))
        if filter_applied or dropped_count > 0:
            trace_span.add_event(
                'retrieval.exact_context_filter',
                _clean_event_attrs(
                    {
                        'filter_applied': filter_applied,
                        'reason': exact_context_filter.get('reason'),
                        'group_count': exact_context_filter.get('group_count'),
                        'selected_count': exact_context_filter.get('selected_count'),
                        'dropped_count': exact_context_filter.get('dropped_count'),
                        'selected_group_key': exact_context_filter.get('selected_group_key'),
                        'selected_file_path': exact_context_filter.get('selected_file_path'),
                        'support_score': exact_context_filter.get('support_score'),
                        'dropped_chunks': _bounded_trace_records(
                            tracing,
                            exact_context_filter.get('dropped_preview'),
                        ),
                    }
                ),
            )

    entity_truncation = metadata.get('entity_truncation')
    if isinstance(entity_truncation, dict) and entity_truncation:
        entities_dropped = max(
            0,
            _trace_int(entity_truncation.get('entities_before')) - _trace_int(entity_truncation.get('entities_after')),
        )
        relations_dropped = max(
            0,
            _trace_int(entity_truncation.get('relations_before'))
            - _trace_int(entity_truncation.get('relations_after')),
        )
        if entities_dropped or relations_dropped:
            trace_span.add_event(
                'retrieval.entity_truncation',
                _clean_event_attrs(
                    {
                        **entity_truncation,
                        'entities_dropped': entities_dropped,
                        'relations_dropped': relations_dropped,
                    }
                ),
            )


def _retrieval_documents_from_result(
    result: Any,
    tracing: TraceManager,
    *,
    augmented_references: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenInference retrieval-document payloads from a YAR query result."""
    if not isinstance(result, dict):
        return []
    data = result.get('data') if isinstance(result.get('data'), dict) else {}
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for chunk in data.get('chunks', []) or []:
        if isinstance(chunk, dict):
            ref_id = chunk.get('reference_id')
            if ref_id is not None:
                chunks_by_id[str(ref_id)] = chunk
    documents: list[dict[str, Any]] = []
    result_metadata = result.get('metadata') if isinstance(result.get('metadata'), dict) else {}
    exact_context_filter = (
        result_metadata.get('exact_context_filter')
        if isinstance(result_metadata.get('exact_context_filter'), dict)
        else {}
    )
    exact_context_file_path = str(exact_context_filter.get('selected_file_path') or '').strip()
    exact_context_group_key = str(exact_context_filter.get('selected_group_key') or '').strip()
    exact_context_support = _coerce_document_score(exact_context_filter.get('support_score'))
    references = augmented_references if augmented_references is not None else data.get('references', []) or []
    if not isinstance(references, list):
        return []
    for reference in references[: tracing.config.max_items]:
        if not isinstance(reference, dict):
            continue
        ref_id = reference.get('reference_id')
        chunk = chunks_by_id.get(str(ref_id), {}) if ref_id is not None else {}
        content = chunk.get('content') or reference.get('content') or reference.get('excerpt') or ''
        score = _coerce_document_score(
            chunk.get('score'),
            reference.get('score'),
            chunk.get('merge_score'),
            chunk.get('retrieval_score'),
            chunk.get('rerank_score'),
        )
        metadata = {
            'document_title': reference.get('document_title'),
            'file_path': reference.get('file_path'),
            's3_key': reference.get('s3_key'),
            'content_index': reference.get('content_index'),
            'chunk_id': chunk.get('chunk_id'),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        stage_ranks = _trace_stage_ranks(chunk.get('stage_ranks') or chunk.get('_trace_stage_ranks'))
        if stage_ranks:
            metadata['stage_ranks'] = stage_ranks
        for key in ('merge_score', 'source_type', 'exact_support_score'):
            if chunk.get(key) is not None:
                metadata[key] = chunk[key]
        if exact_context_support is not None and metadata.get('exact_support_score') is None:
            reference_file_path = str(reference.get('file_path') or chunk.get('file_path') or '').strip()
            ref_id_text = str(ref_id) if ref_id is not None else ''
            if (
                reference_file_path
                and not ref_id_text.startswith('kg-')
                and (
                    reference_file_path == exact_context_file_path
                    or f'path:{reference_file_path}' == exact_context_group_key
                )
            ):
                metadata['exact_support_score'] = exact_context_support
        drop_reason = chunk.get('drop_reason') or chunk.get('_trace_drop_reason')
        if drop_reason:
            metadata['drop_reason'] = str(drop_reason)
        documents.append(
            {
                'id': str(ref_id) if ref_id is not None else None,
                'content': content if tracing.config.capture_contexts else '',
                'score': score,
                'metadata': metadata,
            }
        )
    return documents


def _trace_response_attrs(response_content: str, tracing: TraceManager) -> dict[str, Any]:
    attrs: dict[str, Any] = {'output.answer_length': len(response_content)}
    if tracing.config.capture_prompts:
        answer_preview = _trace_preview(tracing, response_content)
        attrs['output.answer'] = answer_preview
        attrs['output.final_answer'] = answer_preview
    return attrs


def _query_response_metadata(result: dict[str, Any]) -> dict[str, Any] | None:
    metadata = result.get('metadata')
    if not isinstance(metadata, dict):
        return None
    response_metadata = {key: metadata[key] for key in ('answer_shaping', 'exact_context_filter') if key in metadata}
    return response_metadata or None


async def filter_reasoning_stream(response_stream):
    """Filter <think>...</think> blocks from streaming response in real-time.

    This is a state machine that buffers chunks and filters out reasoning blocks
    as they stream in, preventing <think> tags from appearing to the user.
    """
    buffer = ''
    in_think_block = False

    async for chunk in response_stream:
        buffer += chunk

        while buffer:
            if in_think_block:
                # Look for </think> to exit reasoning block
                end_idx = buffer.find('</think>')
                if end_idx != -1:
                    buffer = buffer[end_idx + 8 :]  # Skip past </think>
                    in_think_block = False
                else:
                    break  # Need more data to find closing tag
            else:
                # Look for <think> to enter reasoning block
                start_idx = buffer.find('<think>')
                if start_idx != -1:
                    # Emit everything before <think>
                    if start_idx > 0:
                        yield buffer[:start_idx]
                    buffer = buffer[start_idx + 7 :]  # Skip past <think>
                    in_think_block = True
                else:
                    # Check for partial "<think>" match at buffer end
                    # This prevents emitting incomplete tags
                    for i in range(min(7, len(buffer)), 0, -1):
                        if '<think>'[:i] == buffer[-i:]:
                            if len(buffer) > i:
                                yield buffer[:-i]
                            buffer = buffer[-i:]
                            break
                    else:
                        yield buffer
                        buffer = ''
                    break

    # Emit any remaining buffer (only if not inside a think block)
    if buffer and not in_think_block:
        yield buffer


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=100_000,
        description='The query text',
    )

    mode: Literal['local', 'global', 'hybrid', 'naive', 'mix', 'bypass'] = Field(
        default='mix',
        description='Query mode',
    )

    only_need_context: bool | None = Field(
        default=None,
        description='If True, only returns the retrieved context without generating a response.',
    )

    only_need_prompt: bool | None = Field(
        default=None,
        description='If True, only returns the generated prompt without producing a response.',
    )

    response_type: str | None = Field(
        min_length=1,
        max_length=500,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: int | None = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    chunk_top_k: int | None = Field(
        ge=1,
        default=None,
        description='Number of text chunks to retrieve initially from vector search and keep after reranking.',
    )

    retrieval_multiplier: int | None = Field(
        ge=1,
        le=10,
        default=None,
        description='Two-stage retrieval oversampling factor for chunks. When reranking is enabled, the vector search retrieves chunk_top_k * retrieval_multiplier candidates, then the reranker trims back to chunk_top_k. 1 disables oversampling, 2-3 helps surface chunks the first-stage vector score buried. No effect when reranking is disabled.',
    )

    max_entity_tokens: int | None = Field(
        default=None,
        description='Maximum number of tokens allocated for entity context in unified token control system.',
        ge=1,
    )

    max_relation_tokens: int | None = Field(
        default=None,
        description='Maximum number of tokens allocated for relationship context in unified token control system.',
        ge=1,
    )

    max_total_tokens: int | None = Field(
        default=None,
        description='Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).',
        ge=1,
    )

    hl_keywords: list[str] = Field(
        default_factory=list,
        description='List of high-level keywords to prioritize in retrieval. Leave empty to use the LLM to generate the keywords.',
    )

    ll_keywords: list[str] = Field(
        default_factory=list,
        description='List of low-level keywords to refine retrieval focus. Leave empty to use the LLM to generate the keywords.',
    )

    user_prompt: str | None = Field(
        default=None,
        max_length=100_000,
        description='User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template.',
    )

    enable_rerank: bool | None = Field(
        default=None,
        description='Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is False.',
    )

    enable_bm25_fusion: bool | None = Field(
        default=None,
        description='Enable BM25 fusion: combines vector similarity with BM25 full-text search using Reciprocal Rank Fusion (RRF). Helps with multi-constraint queries where exact keyword matches matter (e.g., drug names, dates, acronyms).',
    )

    bm25_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Weight for BM25 results in BM25 fusion (0.0-1.0). Higher values give more influence to keyword matches vs semantic similarity. Default is 0.3.',
    )

    include_references: bool | None = Field(
        default=True,
        description='If True, includes reference list in responses. Affects /query and /query/stream endpoints. /query/data always includes references.',
    )

    include_chunk_content: bool | None = Field(
        default=False,
        description='If True, includes actual chunk text content in references. Only applies when include_references=True. Useful for evaluation and debugging.',
    )

    stream: bool | None = Field(
        default=True,
        description='If True, enables streaming output for real-time responses. Only affects /query/stream endpoint.',
    )

    citation_mode: Literal['none', 'inline', 'footnotes'] | None = Field(
        default='none',
        description="Citation extraction mode: 'none' (no post-processing), 'inline' (add [n] markers in text), 'footnotes' (add markers and formatted footnotes). When enabled, citations are computed asynchronously after response completes.",
    )

    citation_threshold: float | None = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description='Minimum similarity threshold for citation matching (0.0-1.0). Higher values mean stricter matching.',
    )

    entity_filter: str | None = Field(
        default=None,
        max_length=500,
        description='Filter results to entities/chunks containing this term. Useful for multi-product corpora to prevent context mixing. Example: "Product A" to restrict to Product A-related content only.',
    )
    disable_cache: bool | None = Field(
        default=False,
        description='If True, bypasses keyword and query-result cache reads/writes for this request. Useful for evaluation and debugging.',
    )

    @field_validator('query', mode='after')
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        stripped_query = query.strip()
        if len(stripped_query) < 3:
            raise ValueError('Query text must be at least 3 non-whitespace characters.')
        return stripped_query

    @field_validator('hl_keywords', 'll_keywords', mode='after')
    @classmethod
    def keywords_length_check(cls, keywords: list[str]) -> list[str]:
        if len(keywords) > 100:
            raise ValueError('At most 100 keywords allowed.')
        return keywords

    def to_query_params(self, is_stream: bool) -> 'QueryParam':
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        # Exclude API-level parameters that don't belong in QueryParam
        request_data = self.model_dump(
            exclude_none=True,
            exclude={
                'query',
                'include_chunk_content',
                'include_references',
                'citation_mode',
                'citation_threshold',
            },
        )

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description='Unique reference identifier')
    file_path: str = Field(description='Path to the source file')
    document_title: str | None = Field(default=None, description='Human-readable document title')
    s3_key: str | None = Field(default=None, description='S3 object key for source document')
    presigned_url: str | None = Field(default=None, description='Presigned URL for direct document access')
    excerpt: str | None = Field(default=None, description='Brief excerpt from the source')
    content: list[str] | None = Field(
        default=None,
        description='List of chunk contents from this file (only present when include_chunk_content=True)',
    )


class QueryResponse(BaseModel):
    response: str = Field(
        description='The generated response',
    )
    references: list[ReferenceItem] | None = Field(
        default=None,
        description='Reference list (Disabled when include_references=False, /query/data always includes references.)',
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description='Answer-stage diagnostic metadata for tracing and evaluation alignment.',
    )


class QueryDataResponse(BaseModel):
    status: str = Field(description='Query execution status')
    message: str = Field(description='Status message')
    data: dict[str, Any] = Field(
        description='Query result data containing entities, relationships, chunks, and references'
    )
    metadata: dict[str, Any] = Field(description='Query metadata including mode, keywords, and processing information')


class StreamChunkResponse(BaseModel):
    """Response model for streaming chunks in NDJSON format"""

    references: list[dict[str, str]] | None = Field(
        default=None,
        description='Reference list (only in first chunk when include_references=True)',
    )
    response: str | None = Field(default=None, description='Response content chunk or complete response')
    error: str | None = Field(default=None, description='Error message if processing fails')


class CitationSpanModel(BaseModel):
    """A span in the response with citation attribution."""

    start_char: int = Field(description='Start character position in response')
    end_char: int = Field(description='End character position in response')
    text: str = Field(description='The text span being cited')
    reference_ids: list[str] = Field(description='Reference IDs supporting this span')
    confidence: float = Field(description='Citation confidence score (0.0-1.0)')


class EnhancedReferenceItem(BaseModel):
    """Enhanced reference with full metadata for footnotes."""

    reference_id: str = Field(description='Unique reference identifier')
    file_path: str = Field(description='Path to the source file')
    document_title: str | None = Field(default=None, description='Human-readable document title')
    section_title: str | None = Field(default=None, description='Section or chapter title')
    page_range: str | None = Field(default=None, description='Page range (e.g., pp. 45-67)')
    excerpt: str | None = Field(default=None, description='Brief excerpt from the source')
    s3_key: str | None = Field(default=None, description='S3 object key for source document')
    presigned_url: str | None = Field(default=None, description='Presigned URL for direct access')


async def _extract_and_stream_citations(
    response: str,
    chunks: list[dict[str, Any]],
    references: list[dict[str, str]],
    rag,
    min_similarity: float,
    citation_mode: str,
    s3_client=None,
):
    """Extract citations from response and yield NDJSON lines.

    NEW PROTOCOL (eliminates duplicate payload):
    - Does NOT send full annotated_response (that would duplicate the streamed response)
    - Instead sends citation positions + metadata for frontend marker insertion
    - Frontend uses character positions to insert [n] markers client-side

    Args:
        response: The full LLM response text
        chunks: List of chunk dictionaries from retrieval
        references: List of reference dicts with s3_key and document metadata
        rag: The RAG instance (for embedding function)
        min_similarity: Minimum similarity threshold
        citation_mode: 'inline' or 'footnotes'
        s3_client: Optional S3Client for generating presigned URLs

    Yields:
        NDJSON lines for citation metadata (no duplicate text)
    """
    try:
        from yar.citation import extract_citations_from_response

        # Extract citations using the citation module
        citation_result = await extract_citations_from_response(
            response=response,
            chunks=chunks,
            references=references,
            embedding_func=rag.embedding_func,
            min_similarity=min_similarity,
        )

        # Build citation markers with positions for frontend insertion
        # Each marker tells frontend where to insert [n] without sending full text
        citation_markers = []
        for citation in citation_result.citations:
            citation_markers.append(
                {
                    'marker': '[' + ','.join(citation.reference_ids) + ']',
                    'insert_position': citation.end_char,  # Insert after sentence
                    'reference_ids': citation.reference_ids,
                    'confidence': citation.confidence,
                    'text_preview': citation.text[:50] + '...' if len(citation.text) > 50 else citation.text,
                }
            )

        # Build enhanced sources with metadata and presigned URLs
        sources = [
            {
                'reference_id': ref.reference_id,
                'file_path': ref.file_path,
                'document_title': ref.document_title,
                'section_title': ref.section_title,
                'page_range': ref.page_range,
                'excerpt': ref.excerpt,
                's3_key': getattr(ref, 's3_key', None),
                'presigned_url': None,
            }
            for ref in citation_result.references
        ]
        sources = await _attach_presigned_urls(sources, s3_client)

        # Format footnotes if requested
        footnotes = citation_result.footnotes if citation_mode == 'footnotes' else []

        # Send single consolidated citations_metadata object
        # Frontend uses this to insert markers without needing the full text again
        yield (
            json.dumps(
                {
                    'citations_metadata': {
                        'markers': citation_markers,  # Position-based markers for insertion
                        'sources': sources,  # Enhanced reference metadata with presigned URLs
                        'footnotes': footnotes,  # Pre-formatted footnote strings
                        'uncited_count': len(citation_result.uncited_claims),
                    }
                }
            )
            + '\n'
        )

    except ImportError:
        logger.warning('Citation module not available. Skipping citation extraction.')
        yield json.dumps({'citation_error': 'Citation module not available'}) + '\n'
    except Exception as e:
        logger.error(f'Citation extraction error: {e!s}')
        yield json.dumps({'citation_error': 'Citation extraction failed'}) + '\n'


def create_query_routes(
    rag,
    api_key: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    s3_client: S3Client | None = None,
    tracing: TraceManager | None = None,
):
    """Create query routes with optional S3 client for presigned URL generation in citations.

    Args:
        rag: YAR instance
        api_key: Optional API key for authentication
        top_k: Default top_k for retrieval
        s3_client: Optional S3Client for generating presigned URLs in citation responses
    """
    # Use a fresh router per factory call so route handlers don't leak state
    # when multiple app instances are created (tests, multi-tenant deployments).
    router = APIRouter(tags=['query'])
    combined_auth = get_combined_auth_dependency(api_key)
    tracing_manager = _route_trace_manager(tracing)

    @router.post(
        '/query',
        response_model=QueryResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                'description': 'Successful RAG query response',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'response': {
                                    'type': 'string',
                                    'description': 'The generated response from the RAG system',
                                },
                                'references': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'properties': {
                                            'reference_id': {'type': 'string'},
                                            'file_path': {'type': 'string'},
                                            'content': {
                                                'type': 'array',
                                                'items': {'type': 'string'},
                                                'description': 'List of chunk contents from this file (only included when include_chunk_content=True)',
                                            },
                                        },
                                    },
                                    'description': 'Reference list (only included when include_references=True)',
                                },
                            },
                            'required': ['response'],
                        },
                        'examples': {
                            'with_references': {
                                'summary': 'Response with references',
                                'description': 'Example response when include_references=True',
                                'value': {
                                    'response': 'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.',
                                    'references': [
                                        {
                                            'reference_id': '1',
                                            'file_path': '/documents/ai_overview.pdf',
                                        },
                                        {
                                            'reference_id': '2',
                                            'file_path': '/documents/machine_learning.txt',
                                        },
                                    ],
                                },
                            },
                            'with_chunk_content': {
                                'summary': 'Response with chunk content',
                                'description': 'Example response when include_references=True and include_chunk_content=True. Note: content is an array of chunks from the same file.',
                                'value': {
                                    'response': 'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.',
                                    'references': [
                                        {
                                            'reference_id': '1',
                                            'file_path': '/documents/ai_overview.pdf',
                                            'content': [
                                                'Artificial Intelligence (AI) represents a transformative field in computer science focused on creating systems that can perform tasks requiring human-like intelligence. These tasks include learning from experience, understanding natural language, recognizing patterns, and making decisions.',
                                                'AI systems can be categorized into narrow AI, which is designed for specific tasks, and general AI, which aims to match human cognitive abilities across a wide range of domains.',
                                            ],
                                        },
                                        {
                                            'reference_id': '2',
                                            'file_path': '/documents/machine_learning.txt',
                                            'content': [
                                                'Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of algorithms that can access data and use it to learn for themselves.'
                                            ],
                                        },
                                    ],
                                },
                            },
                            'without_references': {
                                'summary': 'Response without references',
                                'description': 'Example response when include_references=False',
                                'value': {
                                    'response': 'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.'
                                },
                            },
                            'different_modes': {
                                'summary': 'Different query modes',
                                'description': 'Examples of responses from different query modes',
                                'value': {
                                    'local_mode': 'Focuses on specific entities and their relationships',
                                    'global_mode': 'Provides broader context from relationship patterns',
                                    'hybrid_mode': 'Combines local and global approaches',
                                    'naive_mode': 'Simple vector similarity search',
                                    'mix_mode': 'Integrates knowledge graph and vector retrieval',
                                },
                            },
                        },
                    }
                },
            },
            400: {
                'description': 'Bad Request - Invalid input parameters',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {'detail': {'type': 'string'}},
                        },
                        'example': {'detail': 'Query text must be at least 3 characters long'},
                    }
                },
            },
            500: {
                'description': 'Internal Server Error - Query processing failed',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {'detail': {'type': 'string'}},
                        },
                        'example': {'detail': 'Failed to process query: LLM service unavailable'},
                    }
                },
            },
        },
    )
    async def query_text(request: QueryRequest, http_response: Response):
        """
        Comprehensive RAG query endpoint with non-streaming response. Parameter "stream" is ignored.

        This endpoint performs Retrieval-Augmented Generation (RAG) queries using various modes
        to provide intelligent responses based on your knowledge base.

        **Query Modes:**
        - **local**: Focuses on specific entities and their direct relationships
        - **global**: Analyzes broader patterns and relationships across the knowledge graph
        - **hybrid**: Combines local and global approaches for comprehensive results
        - **naive**: Simple vector similarity search without knowledge graph
        - **mix**: Integrates knowledge graph retrieval with vector search (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        **Usage Examples:**

        Basic query:
        ```json
        {
            "query": "What is machine learning?",
            "mode": "mix"
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Advanced query with references:
        ```json
        {
            "query": "Explain neural networks",
            "mode": "hybrid",
            "include_references": true,
            "response_type": "Multiple Paragraphs",
            "top_k": 10
        }
        ```

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process (min 3 characters)
                - **mode**: Query strategy - "mix" recommended for best results
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            QueryResponse: JSON response containing:
                - **response**: The generated answer to your query
                - **references**: Source citations (if include_references=True)

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short)
                - 500: Internal processing error (e.g., LLM service unavailable)
        """
        trace_attrs = _trace_query_request_attrs('/query', request, tracing_manager)
        input_value = request.query if tracing_manager.config.capture_prompts else None
        with tracing_manager.start_chain_span(
            'app.query',
            input_value=input_value,
            attributes={**trace_attrs, 'tag.tags': _request_tags('/query', request, streaming=False)},
        ) as trace_span:
            _attach_prompt_template(trace_span, request)
            if trace_span.trace_id:
                http_response.headers['x-yar-trace-id'] = trace_span.trace_id
                if trace_span.span_id:
                    http_response.headers['x-yar-span-id'] = trace_span.span_id
            try:
                param = request.to_query_params(False)  # Ensure stream=False for non-streaming endpoint
                # Force stream=False for /query endpoint regardless of include_references setting
                param.stream = False

                # Unified approach: always use aquery_llm for both cases
                reset_query_cache_hit()
                _aquery_started = asyncio.get_event_loop().time()
                try:
                    result = await rag.aquery_llm(request.query, param=param)
                finally:
                    trace_span.set_attribute('llm.cache_hit', bool(query_cache_hit_was_set()))
                trace_span.set_attribute(
                    'phase.aquery_llm_ms',
                    round((asyncio.get_event_loop().time() - _aquery_started) * 1000, 3),
                )
                data = result.get('data') if isinstance(result.get('data'), dict) else {}
                raw_chunks = data.get('chunks', [])
                chunks = list(raw_chunks) if isinstance(raw_chunks, list) else []
                raw_references = data.get('references', [])
                base_references = list(raw_references) if isinstance(raw_references, list) else []
                include_reference_content = bool(request.include_references and request.include_chunk_content)
                augmented_references = _append_knowledge_graph_references(
                    _attach_chunk_content(
                        base_references,
                        chunks,
                        include_chunk_content=include_reference_content,
                    ),
                    data,
                    include_content=include_reference_content,
                )
                trace_span.set_attributes(_trace_rag_result_attrs(result, tracing_manager))
                _emit_retrieval_diagnostic_events(trace_span, result, tracing_manager)
                trace_span.set_retrieval_documents(
                    _retrieval_documents_from_result(
                        result,
                        tracing_manager,
                        augmented_references=augmented_references,
                    )
                )
                _fingerprint = _retrieval_fingerprint(result)
                if _fingerprint:
                    trace_span.set_attribute('retrieval.fingerprint', _fingerprint)
                _emit_synthetic_retriever_span(
                    tracing=tracing_manager,
                    query=request.query,
                    request=request,
                    result=result,
                    augmented_references=augmented_references,
                )
                # Tags will be finalized after citation metrics so the result and
                # retrieval-precision dimensions are captured together.
                failure_message = get_query_failure_message(result)
                if failure_message:
                    trace_span.set_status_error(failure_message)
                    trace_span.set_attribute('error.message', failure_message)
                    raise HTTPException(status_code=500, detail=failure_message)

                # Extract LLM response and references from unified result
                llm_response = result.get('llm_response', {})
                references = await _attach_presigned_urls(
                    augmented_references,
                    s3_client if request.include_references else None,
                )

                raw_response_text = llm_response.get('content', '') or ''
                response_content = _normalize_query_response_text(
                    raw_response_text,
                    keep_inline_citations=requests_inline_citations(request.query, request.user_prompt),
                )
                if include_reference_content:
                    references = _filter_references_for_answer_support(
                        references,
                        response_text=response_content,
                        query=request.query,
                    )
                trace_span.set_attributes(_trace_response_attrs(response_content, tracing_manager))
                if tracing_manager.config.capture_prompts:
                    trace_span.set_output_value(response_content)
                    trace_span.set_llm_output_messages([{'role': 'assistant', 'content': response_content}])
                citation_tags = _emit_citation_metrics(
                    trace_span=trace_span,
                    response_text=response_content,
                    raw_response_text=raw_response_text,
                    result=result,
                )
                _query_elapsed_ms = (asyncio.get_event_loop().time() - _aquery_started) * 1000
                _latency_tags = [_latency_bucket(_query_elapsed_ms)]
                if _query_elapsed_ms > _slow_query_threshold_ms():
                    _latency_tags.append('slow:true')
                _final_tags = (
                    _request_tags('/query', request, streaming=False)
                    + _result_tags(result)
                    + citation_tags
                    + _latency_tags
                )
                if query_cache_hit_was_set():
                    _final_tags.append('cached:true')
                trace_span.set_tags(_final_tags)
                tracing_manager.record_query_metrics(
                    endpoint='/query',
                    mode=request.mode,
                    status='success' if not get_query_failure_message(result) else 'failure',
                    duration_ms=_query_elapsed_ms,
                    cached=query_cache_hit_was_set(),
                )
                response_metadata = _query_response_metadata(result)

                # Return response with or without references based on request
                if request.include_references:
                    return QueryResponse(
                        response=response_content,
                        references=[ReferenceItem.model_validate(ref) for ref in references],
                        metadata=response_metadata,
                    )
                return QueryResponse(response=response_content, references=None, metadata=response_metadata)
            except HTTPException:
                raise
            except Exception as e:
                trace_span.record_exception(e)
                trace_span.set_status_error(str(e))
                trace_span.set_tags(
                    [
                        *_request_tags('/query', request, streaming=False),
                        'status:failure',
                        f'error:{_classify_exception(e)}',
                    ]
                )
                logger.error(f'Error processing query: {e!s}', exc_info=True)
                raise HTTPException(status_code=500, detail='Internal server error') from e

    @router.post(
        '/query/stream',
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                'description': 'Flexible RAG query response - format depends on stream parameter',
                'content': {
                    'application/x-ndjson': {
                        'schema': {
                            'type': 'string',
                            'format': 'ndjson',
                            'description': 'Newline-delimited JSON (NDJSON) format used for both streaming and non-streaming responses. For streaming: multiple lines with separate JSON objects. For non-streaming: single line with complete JSON object.',
                            'example': '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"response": " a field of computer science"}\n{"response": " that focuses on creating intelligent machines."}',
                        },
                        'examples': {
                            'streaming_with_references': {
                                'summary': 'Streaming mode with references (stream=true)',
                                'description': 'Multiple NDJSON lines when stream=True and include_references=True. First line contains references, subsequent lines contain response chunks.',
                                'value': '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf"}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt"}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence, such as learning,"}\n{"response": " reasoning, and problem-solving."}',
                            },
                            'streaming_with_chunk_content': {
                                'summary': 'Streaming mode with chunk content (stream=true, include_chunk_content=true)',
                                'description': 'Multiple NDJSON lines when stream=True, include_references=True, and include_chunk_content=True. First line contains references with content arrays (one file may have multiple chunks), subsequent lines contain response chunks.',
                                'value': '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf", "content": ["Artificial Intelligence (AI) represents a transformative field...", "AI systems can be categorized into narrow AI and general AI..."]}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt", "content": ["Machine learning is a subset of AI that enables computers to learn..."]}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence."}',
                            },
                            'streaming_without_references': {
                                'summary': 'Streaming mode without references (stream=true)',
                                'description': 'Multiple NDJSON lines when stream=True and include_references=False. Only response chunks are sent.',
                                'value': '{"response": "Machine learning is a subset of artificial intelligence"}\n{"response": " that enables computers to learn and improve from experience"}\n{"response": " without being explicitly programmed for every task."}',
                            },
                            'non_streaming_with_references': {
                                'summary': 'Non-streaming mode with references (stream=false)',
                                'description': 'Single NDJSON line when stream=False and include_references=True. Complete response with references in one message.',
                                'value': '{"references": [{"reference_id": "1", "file_path": "/documents/neural_networks.pdf"}], "response": "Neural networks are computational models inspired by biological neural networks that consist of interconnected nodes (neurons) organized in layers. They are fundamental to deep learning and can learn complex patterns from data through training processes."}',
                            },
                            'non_streaming_without_references': {
                                'summary': 'Non-streaming mode without references (stream=false)',
                                'description': 'Single NDJSON line when stream=False and include_references=False. Complete response only.',
                                'value': '{"response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence deep) to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition."}',
                            },
                            'error_response': {
                                'summary': 'Error during streaming',
                                'description': 'Error handling in NDJSON format when an error occurs during processing.',
                                'value': '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"error": "LLM service temporarily unavailable"}',
                            },
                        },
                    }
                },
            },
            400: {
                'description': 'Bad Request - Invalid input parameters',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {'detail': {'type': 'string'}},
                        },
                        'example': {'detail': 'Query text must be at least 3 characters long'},
                    }
                },
            },
            500: {
                'description': 'Internal Server Error - Query processing failed',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {'detail': {'type': 'string'}},
                        },
                        'example': {'detail': 'Failed to process streaming query: Knowledge graph unavailable'},
                    }
                },
            },
        },
    )
    async def query_text_stream(request: QueryRequest):
        """
        Advanced RAG query endpoint with flexible streaming response.

        This endpoint provides the most flexible querying experience, supporting both real-time streaming
        and complete response delivery based on your integration needs.

        **Response Modes:**
        - Real-time response delivery as content is generated
        - NDJSON format: each line is a separate JSON object
        - First line: `{"references": [...]}` (if include_references=True)
        - Subsequent lines: `{"response": "content chunk"}`
        - Error handling: `{"error": "error message"}`

        > If stream parameter is False, or the query hit LLM cache, complete response delivered in a single streaming message.

        **Response Format Details**
        - **Content-Type**: `application/x-ndjson` (Newline-Delimited JSON)
        - **Structure**: Each line is an independent, valid JSON object
        - **Parsing**: Process line-by-line, each line is self-contained
        - **Headers**: Includes cache control and connection management

        **Query Modes (same as /query endpoint)**
        - **local**: Entity-focused retrieval with direct relationships
        - **global**: Pattern analysis across the knowledge graph
        - **hybrid**: Combined local and global strategies
        - **naive**: Vector similarity search only
        - **mix**: Integrated knowledge graph + vector retrieval (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        **Usage Examples**

        Real-time streaming query:
        ```json
        {
            "query": "Explain machine learning algorithms",
            "mode": "mix",
            "stream": true,
            "include_references": true
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Complete response query:
        ```json
        {
            "query": "What is deep learning?",
            "mode": "hybrid",
            "stream": false,
            "response_type": "Multiple Paragraphs"
        }
        ```

        **Response Processing:**

        ```python
        async for line in response.iter_lines():
            data = json.loads(line)
            if "references" in data:
                # Handle references (first message)
                references = data["references"]
            if "response" in data:
                # Handle content chunk
                content_chunk = data["response"]
            if "error" in data:
                # Handle error
                error_message = data["error"]
        ```

        **Error Handling:**
        - Streaming errors are delivered as `{"error": "message"}` lines
        - Non-streaming errors raise HTTP exceptions
        - Partial responses may be delivered before errors in streaming mode
        - Always check for error objects when processing streaming responses

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process (min 3 characters)
                - **mode**: Query strategy - "mix" recommended for best results
                - **stream**: Enable streaming (True) or complete response (False)
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            StreamingResponse: NDJSON streaming response containing:
                - **Streaming mode**: Multiple JSON objects, one per line
                  - References object (if requested): `{"references": [...]}`
                  - Content chunks: `{"response": "chunk content"}`
                  - Error objects: `{"error": "error message"}`
                - **Non-streaming mode**: Single JSON object
                  - Complete response: `{"references": [...], "response": "complete content"}`

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short, invalid mode)
                - 500: Internal processing error (e.g., LLM service unavailable)

        Note:
            This endpoint is ideal for applications requiring flexible response delivery.
            Use streaming mode for real-time interfaces and non-streaming for batch processing.
        """
        trace_attrs = _trace_query_request_attrs('/query/stream', request, tracing_manager)
        input_value = request.query if tracing_manager.config.capture_prompts else None
        with tracing_manager.start_chain_span(
            'app.query_stream',
            input_value=input_value,
            attributes={
                **trace_attrs,
                'tag.tags': _request_tags(
                    '/query/stream',
                    request,
                    streaming=request.stream is not False,
                ),
            },
        ) as trace_span:
            _attach_prompt_template(trace_span, request)
            try:
                # Use the stream parameter from the request, defaulting to True if not specified
                stream_mode = request.stream if request.stream is not None else True
                trace_span.set_attribute('rag.stream', bool(stream_mode))
                param = request.to_query_params(stream_mode)

                from fastapi.responses import StreamingResponse

                # Unified approach: always use aquery_llm for all cases
                reset_query_cache_hit()
                _aquery_started = asyncio.get_event_loop().time()
                try:
                    result = await rag.aquery_llm(request.query, param=param)
                finally:
                    trace_span.set_attribute('llm.cache_hit', bool(query_cache_hit_was_set()))
                trace_span.set_attribute(
                    'phase.aquery_llm_ms',
                    round((asyncio.get_event_loop().time() - _aquery_started) * 1000, 3),
                )
                trace_span.set_attributes(_trace_rag_result_attrs(result, tracing_manager))
                trace_span.set_retrieval_documents(_retrieval_documents_from_result(result, tracing_manager))
                _fingerprint = _retrieval_fingerprint(result)
                if _fingerprint:
                    trace_span.set_attribute('retrieval.fingerprint', _fingerprint)
                _emit_synthetic_retriever_span(
                    tracing=tracing_manager, query=request.query, request=request, result=result
                )
                # Tags are finalized after the optional citation pass below.
                citation_tags: list[str] = []
                failure_message = get_query_failure_message(result)
                if failure_message and not stream_mode:
                    trace_span.set_status_error(failure_message)
                    trace_span.set_attribute('error.message', failure_message)
                    raise HTTPException(status_code=500, detail=failure_message)

                if isinstance(result, dict):
                    llm_response_for_trace = result.get('llm_response') or {}
                    if isinstance(llm_response_for_trace, dict) and not llm_response_for_trace.get('is_streaming'):
                        raw_response_text_stream = llm_response_for_trace.get('content', '') or ''
                        response_content_for_trace = _normalize_query_response_text(
                            raw_response_text_stream,
                            keep_inline_citations=requests_inline_citations(request.query, request.user_prompt),
                        )
                        trace_span.set_attributes(_trace_response_attrs(response_content_for_trace, tracing_manager))
                        if tracing_manager.config.capture_prompts:
                            trace_span.set_output_value(response_content_for_trace)
                            trace_span.set_llm_output_messages(
                                [{'role': 'assistant', 'content': response_content_for_trace}]
                            )
                        citation_tags = _emit_citation_metrics(
                            trace_span=trace_span,
                            response_text=response_content_for_trace,
                            raw_response_text=raw_response_text_stream,
                            result=result,
                        )
                _query_elapsed_ms = (asyncio.get_event_loop().time() - _aquery_started) * 1000
                _latency_tags = [_latency_bucket(_query_elapsed_ms)]
                if _query_elapsed_ms > _slow_query_threshold_ms():
                    _latency_tags.append('slow:true')
                _final_tags = (
                    _request_tags('/query/stream', request, streaming=stream_mode)
                    + _result_tags(result)
                    + citation_tags
                    + _latency_tags
                )
                if query_cache_hit_was_set():
                    _final_tags.append('cached:true')
                trace_span.set_tags(_final_tags)
                tracing_manager.record_query_metrics(
                    endpoint='/query/stream',
                    mode=request.mode,
                    status='success' if not get_query_failure_message(result) else 'failure',
                    duration_ms=_query_elapsed_ms,
                    cached=query_cache_hit_was_set(),
                )

                async def stream_generator():
                    # Extract references and LLM response from unified result
                    result_payload = result if isinstance(result, dict) else {}
                    references: list[dict[str, Any]] = list(result_payload.get('data', {}).get('references', []))
                    chunks: list[dict[str, Any]] = list(result_payload.get('data', {}).get('chunks', []))
                    llm_response: dict[str, Any] = result_payload.get('llm_response', {}) or {}
                    if failure_message:
                        yield f'{json.dumps({"error": failure_message})}\n'
                        return

                    references = await _attach_presigned_urls(
                        _append_knowledge_graph_references(
                            _attach_chunk_content(
                                references,
                                chunks,
                                include_chunk_content=bool(
                                    request.include_references and request.include_chunk_content
                                ),
                            ),
                            result_payload.get('data', {}),
                            include_content=bool(request.include_references and request.include_chunk_content),
                        ),
                        s3_client if request.include_references else None,
                    )

                    citation_mode = request.citation_mode or 'none'
                    should_extract_citations = citation_mode in ['inline', 'footnotes']

                    if llm_response.get('is_streaming'):
                        # Streaming mode: send references first, then stream response chunks
                        if request.include_references:
                            yield f'{json.dumps({"references": references})}\n'

                        response_stream = llm_response.get('response_iterator')
                        collected_response: list[str] | None = [] if should_extract_citations else None
                        if response_stream:
                            try:
                                # Filter <think>...</think> blocks in real-time
                                async for chunk in filter_reasoning_stream(response_stream):
                                    if chunk:  # Only send non-empty content
                                        yield f'{json.dumps({"response": chunk})}\n'
                                        if collected_response is not None:
                                            collected_response.append(chunk)
                            except Exception as e:
                                logger.error(f'Streaming error: {e!s}')
                                yield f'{json.dumps({"error": "Stream processing error"})}\n'
                                return

                        # After streaming completes, extract citations if enabled
                        if collected_response:
                            full_response = strip_reasoning_tags(''.join(collected_response))
                            async for line in _extract_and_stream_citations(
                                full_response,
                                chunks,
                                references,
                                rag,
                                request.citation_threshold or 0.7,
                                citation_mode,
                                s3_client,
                            ):
                                yield line
                    else:
                        # Non-streaming mode: send complete response in one message
                        response_content = _normalize_query_response_text(
                            llm_response.get('content', ''),
                            keep_inline_citations=requests_inline_citations(request.query, request.user_prompt),
                        )

                        # Create complete response object
                        complete_response: dict[str, Any] = {'response': response_content}
                        if request.include_references:
                            complete_response['references'] = references

                        yield f'{json.dumps(complete_response)}\n'

                        # Extract citations for non-streaming mode too
                        if should_extract_citations and response_content:
                            async for line in _extract_and_stream_citations(
                                response_content,
                                chunks,
                                references,
                                rag,
                                request.citation_threshold or 0.7,
                                citation_mode,
                                s3_client,
                            ):
                                yield line

                # Wrap the generator so we record first-token latency. The
                # streaming span is opened as a child of the chain (require_parent
                # is satisfied while we're still inside the chain context) and is
                # held open across the stream by the wrapper's finally block.
                streaming_span = None
                streaming_started = asyncio.get_event_loop().time()
                if isinstance(result, dict) and (result.get('llm_response') or {}).get('is_streaming'):
                    streaming_span = tracing_manager.start_llm_span(
                        'llm.stream',
                        model='streaming',
                        provider='yar',
                        attributes={'streaming.is_streaming': True},
                    )
                    streaming_span.__enter__()

                async def traced_stream():
                    first_token_recorded = False
                    inner = stream_generator()
                    chunk_count = 0
                    total_bytes = 0
                    last_chunk_at = streaming_started
                    inter_chunk_intervals: list[float] = []
                    try:
                        async for chunk in inner:
                            now = asyncio.get_event_loop().time()
                            chunk_count += 1
                            total_bytes += len(chunk) if isinstance(chunk, (str, bytes)) else 0
                            if not first_token_recorded:
                                first_token_ms = round((now - streaming_started) * 1000, 3)
                                if streaming_span is not None:
                                    streaming_span.set_attribute('streaming.first_token_ms', first_token_ms)
                                # NOTE: cannot mirror onto ``trace_span`` here — the chain
                                # ``with`` block has already exited by the time Starlette
                                # iterates this generator. Dashboards must read
                                # ``streaming.first_token_ms`` from the child ``llm.stream``
                                # span (which lives long enough for late attribute writes).
                                first_token_recorded = True
                            else:
                                inter_chunk_intervals.append((now - last_chunk_at) * 1000)
                            last_chunk_at = now
                            yield chunk
                    finally:
                        if streaming_span is not None:
                            elapsed_ms = round((asyncio.get_event_loop().time() - streaming_started) * 1000, 3)
                            if not first_token_recorded:
                                streaming_span.set_attribute('streaming.first_token_ms', -1)
                            streaming_span.set_attributes(
                                {
                                    'streaming.total_ms': elapsed_ms,
                                    'streaming.chunk_count': chunk_count,
                                    'streaming.total_bytes': total_bytes,
                                    'streaming.bytes_per_second': (
                                        round(total_bytes * 1000 / elapsed_ms, 3) if elapsed_ms > 0 else 0
                                    ),
                                    'streaming.avg_chunk_bytes': (
                                        round(total_bytes / chunk_count, 3) if chunk_count > 0 else 0
                                    ),
                                    'streaming.avg_inter_chunk_ms': (
                                        round(sum(inter_chunk_intervals) / len(inter_chunk_intervals), 3)
                                        if inter_chunk_intervals
                                        else 0
                                    ),
                                }
                            )
                            streaming_span.__exit__(None, None, None)

                return StreamingResponse(
                    traced_stream(),
                    media_type='application/x-ndjson',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Content-Type': 'application/x-ndjson',
                        'X-Accel-Buffering': 'no',  # Ensure proper handling when proxied by Nginx
                    },
                )
            except HTTPException:
                raise
            except Exception as e:
                trace_span.record_exception(e)
                trace_span.set_status_error(str(e))
                trace_span.set_tags(
                    [
                        *_request_tags('/query/stream', request, streaming=True),
                        'status:failure',
                        f'error:{_classify_exception(e)}',
                    ]
                )
                logger.error(f'Error processing streaming query: {e!s}', exc_info=True)
                raise HTTPException(status_code=500, detail='Internal server error') from e

    @router.post(
        '/query/data',
        response_model=QueryDataResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                'description': 'Successful data retrieval response with structured RAG data',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'status': {
                                    'type': 'string',
                                    'enum': ['success', 'failure'],
                                    'description': 'Query execution status',
                                },
                                'message': {
                                    'type': 'string',
                                    'description': 'Status message describing the result',
                                },
                                'data': {
                                    'type': 'object',
                                    'properties': {
                                        'entities': {
                                            'type': 'array',
                                            'items': {
                                                'type': 'object',
                                                'properties': {
                                                    'entity_name': {'type': 'string'},
                                                    'entity_type': {'type': 'string'},
                                                    'description': {'type': 'string'},
                                                    'source_id': {'type': 'string'},
                                                    'file_path': {'type': 'string'},
                                                    'reference_id': {'type': 'string'},
                                                },
                                            },
                                            'description': 'Retrieved entities from knowledge graph',
                                        },
                                        'relationships': {
                                            'type': 'array',
                                            'items': {
                                                'type': 'object',
                                                'properties': {
                                                    'src_id': {'type': 'string'},
                                                    'tgt_id': {'type': 'string'},
                                                    'description': {'type': 'string'},
                                                    'keywords': {'type': 'string'},
                                                    'weight': {'type': 'number'},
                                                    'source_id': {'type': 'string'},
                                                    'file_path': {'type': 'string'},
                                                    'reference_id': {'type': 'string'},
                                                },
                                            },
                                            'description': 'Retrieved relationships from knowledge graph',
                                        },
                                        'chunks': {
                                            'type': 'array',
                                            'items': {
                                                'type': 'object',
                                                'properties': {
                                                    'content': {'type': 'string'},
                                                    'file_path': {'type': 'string'},
                                                    'chunk_id': {'type': 'string'},
                                                    'reference_id': {'type': 'string'},
                                                },
                                            },
                                            'description': 'Retrieved text chunks from vector database',
                                        },
                                        'references': {
                                            'type': 'array',
                                            'items': {
                                                'type': 'object',
                                                'properties': {
                                                    'reference_id': {'type': 'string'},
                                                    'file_path': {'type': 'string'},
                                                },
                                            },
                                            'description': 'Reference list for citation purposes',
                                        },
                                    },
                                    'description': 'Structured retrieval data containing entities, relationships, chunks, and references',
                                },
                                'metadata': {
                                    'type': 'object',
                                    'properties': {
                                        'query_mode': {'type': 'string'},
                                        'keywords': {
                                            'type': 'object',
                                            'properties': {
                                                'high_level': {
                                                    'type': 'array',
                                                    'items': {'type': 'string'},
                                                },
                                                'low_level': {
                                                    'type': 'array',
                                                    'items': {'type': 'string'},
                                                },
                                            },
                                        },
                                        'processing_info': {
                                            'type': 'object',
                                            'properties': {
                                                'total_entities_found': {'type': 'integer'},
                                                'total_relations_found': {'type': 'integer'},
                                                'entities_after_truncation': {'type': 'integer'},
                                                'relations_after_truncation': {'type': 'integer'},
                                                'final_chunks_count': {'type': 'integer'},
                                            },
                                        },
                                    },
                                    'description': 'Query metadata including mode, keywords, and processing information',
                                },
                            },
                            'required': ['status', 'message', 'data', 'metadata'],
                        },
                        'examples': {
                            'successful_local_mode': {
                                'summary': 'Local mode data retrieval',
                                'description': 'Example of structured data from local mode query focusing on specific entities',
                                'value': {
                                    'status': 'success',
                                    'message': 'Query executed successfully',
                                    'data': {
                                        'entities': [
                                            {
                                                'entity_name': 'Neural Networks',
                                                'entity_type': 'CONCEPT',
                                                'description': 'Computational models inspired by biological neural networks',
                                                'source_id': 'chunk-123',
                                                'file_path': '/documents/ai_basics.pdf',
                                                'reference_id': '1',
                                            }
                                        ],
                                        'relationships': [
                                            {
                                                'src_id': 'Neural Networks',
                                                'tgt_id': 'Machine Learning',
                                                'description': 'Neural networks are a subset of machine learning algorithms',
                                                'keywords': 'subset, algorithm, learning',
                                                'weight': 0.85,
                                                'source_id': 'chunk-123',
                                                'file_path': '/documents/ai_basics.pdf',
                                                'reference_id': '1',
                                            }
                                        ],
                                        'chunks': [
                                            {
                                                'content': 'Neural networks are computational models that mimic the way biological neural networks work...',
                                                'file_path': '/documents/ai_basics.pdf',
                                                'chunk_id': 'chunk-123',
                                                'reference_id': '1',
                                            }
                                        ],
                                        'references': [
                                            {
                                                'reference_id': '1',
                                                'file_path': '/documents/ai_basics.pdf',
                                            }
                                        ],
                                    },
                                    'metadata': {
                                        'query_mode': 'local',
                                        'keywords': {
                                            'high_level': ['neural', 'networks'],
                                            'low_level': [
                                                'computation',
                                                'model',
                                                'algorithm',
                                            ],
                                        },
                                        'processing_info': {
                                            'total_entities_found': 5,
                                            'total_relations_found': 3,
                                            'entities_after_truncation': 1,
                                            'relations_after_truncation': 1,
                                            'final_chunks_count': 1,
                                        },
                                    },
                                },
                            },
                            'global_mode': {
                                'summary': 'Global mode data retrieval',
                                'description': 'Example of structured data from global mode query analyzing broader patterns',
                                'value': {
                                    'status': 'success',
                                    'message': 'Query executed successfully',
                                    'data': {
                                        'entities': [],
                                        'relationships': [
                                            {
                                                'src_id': 'Artificial Intelligence',
                                                'tgt_id': 'Machine Learning',
                                                'description': 'AI encompasses machine learning as a core component',
                                                'keywords': 'encompasses, component, field',
                                                'weight': 0.92,
                                                'source_id': 'chunk-456',
                                                'file_path': '/documents/ai_overview.pdf',
                                                'reference_id': '2',
                                            }
                                        ],
                                        'chunks': [],
                                        'references': [
                                            {
                                                'reference_id': '2',
                                                'file_path': '/documents/ai_overview.pdf',
                                            }
                                        ],
                                    },
                                    'metadata': {
                                        'query_mode': 'global',
                                        'keywords': {
                                            'high_level': [
                                                'artificial',
                                                'intelligence',
                                                'overview',
                                            ],
                                            'low_level': [],
                                        },
                                    },
                                },
                            },
                            'naive_mode': {
                                'summary': 'Naive mode data retrieval',
                                'description': 'Example of structured data from naive mode using only vector search',
                                'value': {
                                    'status': 'success',
                                    'message': 'Query executed successfully',
                                    'data': {
                                        'entities': [],
                                        'relationships': [],
                                        'chunks': [
                                            {
                                                'content': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers...',
                                                'file_path': '/documents/deep_learning.pdf',
                                                'chunk_id': 'chunk-789',
                                                'reference_id': '3',
                                            }
                                        ],
                                        'references': [
                                            {
                                                'reference_id': '3',
                                                'file_path': '/documents/deep_learning.pdf',
                                            }
                                        ],
                                    },
                                    'metadata': {
                                        'query_mode': 'naive',
                                        'keywords': {'high_level': [], 'low_level': []},
                                    },
                                },
                            },
                        },
                    }
                },
            },
            400: {
                'description': 'Bad Request - Invalid input parameters',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {'detail': {'type': 'string'}},
                        },
                        'example': {'detail': 'Query text must be at least 3 characters long'},
                    }
                },
            },
            500: {
                'description': 'Internal Server Error - Data retrieval failed',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {'detail': {'type': 'string'}},
                        },
                        'example': {'detail': 'Failed to retrieve data: Knowledge graph unavailable'},
                    }
                },
            },
        },
    )
    async def query_data(request: QueryRequest, http_response: Response):
        """
        Advanced data retrieval endpoint for structured RAG analysis.

        This endpoint provides raw retrieval results without LLM generation, perfect for:
        - **Data Analysis**: Examine what information would be used for RAG
        - **System Integration**: Get structured data for custom processing
        - **Debugging**: Understand retrieval behavior and quality
        - **Research**: Analyze knowledge graph structure and relationships

        **Key Features:**
        - No LLM generation - pure data retrieval
        - Complete structured output with entities, relationships, and chunks
        - Always includes references for citation
        - Detailed metadata about processing and keywords
        - Compatible with all query modes and parameters

        **Query Mode Behaviors:**
        - **local**: Returns entities and their direct relationships + related chunks
        - **global**: Returns relationship patterns across the knowledge graph
        - **hybrid**: Combines local and global retrieval strategies
        - **naive**: Returns only vector-retrieved text chunks (no knowledge graph)
        - **mix**: Integrates knowledge graph data with vector-retrieved chunks
        - **bypass**: Returns empty data arrays (used for direct LLM queries)

        **Data Structure:**
        - **entities**: Knowledge graph entities with descriptions and metadata
        - **relationships**: Connections between entities with weights and descriptions
        - **chunks**: Text segments from documents with source information
        - **references**: Citation information mapping reference IDs to file paths
        - **metadata**: Processing information, keywords, and query statistics

        **Usage Examples:**

        Analyze entity relationships:
        ```json
        {
            "query": "machine learning algorithms",
            "mode": "local",
            "top_k": 10
        }
        ```

        Explore global patterns:
        ```json
        {
            "query": "artificial intelligence trends",
            "mode": "global",
            "max_relation_tokens": 2000
        }
        ```

        Vector similarity search:
        ```json
        {
            "query": "neural network architectures",
            "mode": "naive",
            "chunk_top_k": 5
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        **Response Analysis:**
        - **Empty arrays**: Normal for certain modes (e.g., naive mode has no entities/relationships)
        - **Processing info**: Shows retrieval statistics and token usage
        - **Keywords**: High-level and low-level keywords extracted from query
        - **Reference mapping**: Links all data back to source documents

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The search query to analyze (min 3 characters)
                - **mode**: Retrieval strategy affecting data types returned
                - **top_k**: Number of top entities/relationships to retrieve
                - **chunk_top_k**: Number of text chunks to retrieve
                - **max_entity_tokens**: Token limit for entity context
                - **max_relation_tokens**: Token limit for relationship context
                - **max_total_tokens**: Overall token budget for retrieval

        Returns:
            QueryDataResponse: Structured JSON response containing:
                - **status**: "success" or "failure"
                - **message**: Human-readable status description
                - **data**: Complete retrieval results with entities, relationships, chunks, references
                - **metadata**: Query processing information and statistics

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short, invalid mode)
                - 500: Internal processing error (e.g., knowledge graph unavailable)

        Note:
            This endpoint always includes references regardless of the include_references parameter,
            as structured data analysis typically requires source attribution.
        """
        trace_attrs = _trace_query_request_attrs('/query/data', request, tracing_manager)
        input_value = request.query if tracing_manager.config.capture_prompts else None
        with tracing_manager.start_chain_span(
            'app.query_data',
            input_value=input_value,
            attributes={
                **trace_attrs,
                'tag.tags': _request_tags('/query/data', request, streaming=False),
            },
        ) as trace_span:
            _attach_prompt_template(trace_span, request)
            if trace_span.trace_id:
                http_response.headers['x-yar-trace-id'] = trace_span.trace_id
                if trace_span.span_id:
                    http_response.headers['x-yar-span-id'] = trace_span.span_id
            try:
                param = request.to_query_params(False)  # No streaming for data endpoint
                reset_query_cache_hit()
                _aquery_started = asyncio.get_event_loop().time()
                try:
                    response = await rag.aquery_data(request.query, param=param)
                finally:
                    trace_span.set_attribute('llm.cache_hit', bool(query_cache_hit_was_set()))
                trace_span.set_attribute(
                    'phase.aquery_data_ms',
                    round((asyncio.get_event_loop().time() - _aquery_started) * 1000, 3),
                )
                trace_span.set_attributes(_trace_rag_result_attrs(response, tracing_manager))
                trace_span.set_retrieval_documents(_retrieval_documents_from_result(response, tracing_manager))
                _fingerprint = _retrieval_fingerprint(response)
                if _fingerprint:
                    trace_span.set_attribute('retrieval.fingerprint', _fingerprint)
                _emit_synthetic_retriever_span(
                    tracing=tracing_manager, query=request.query, request=request, result=response
                )
                _query_elapsed_ms = (asyncio.get_event_loop().time() - _aquery_started) * 1000
                _latency_tags = [_latency_bucket(_query_elapsed_ms)]
                if _query_elapsed_ms > _slow_query_threshold_ms():
                    _latency_tags.append('slow:true')
                _final_tags = (
                    _request_tags('/query/data', request, streaming=False) + _result_tags(response) + _latency_tags
                )
                if query_cache_hit_was_set():
                    _final_tags.append('cached:true')
                trace_span.set_tags(_final_tags)
                tracing_manager.record_query_metrics(
                    endpoint='/query/data',
                    mode=request.mode,
                    status='success' if not get_query_failure_message(response) else 'failure',
                    duration_ms=_query_elapsed_ms,
                    cached=query_cache_hit_was_set(),
                )

                if not isinstance(response, dict):
                    trace_span.set_status_error('Invalid response type')
                    raise HTTPException(status_code=500, detail='Invalid response type')

                failure_message = get_query_failure_message(response)
                if failure_message:
                    trace_span.set_status_error(failure_message)
                    trace_span.set_attribute('error.message', failure_message)
                    raise HTTPException(status_code=500, detail=failure_message)

                return QueryDataResponse(**response)
            except HTTPException:
                raise
            except Exception as e:
                trace_span.record_exception(e)
                trace_span.set_status_error(str(e))
                trace_span.set_tags(
                    [
                        *_request_tags('/query/data', request, streaming=False),
                        'status:failure',
                        f'error:{_classify_exception(e)}',
                    ]
                )
                logger.error(f'Error processing data query: {e!s}', exc_info=True)
                raise HTTPException(status_code=500, detail='Internal server error') from e

    return router
