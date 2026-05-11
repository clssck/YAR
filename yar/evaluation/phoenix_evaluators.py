"""Phoenix LLM-as-judge auto-evaluation pipeline for YAR query traces.

This module builds a small set of OpenInference-aware evaluators (relevance,
groundedness, hallucination) and a runner that:

1. Pulls recent ``app.query*`` chain spans from a Phoenix project.
2. Materializes a dataframe with ``input``, ``output``, and ``reference``
   (concatenated retrieved documents) columns.
3. Runs the evaluators via ``phoenix.evals.evaluate_dataframe``.
4. Pushes the scores back as span annotations using the Phoenix client.

Usage (CLI)::

    python -m yar.evaluation.phoenix_evaluators \
        --project yar-app --limit 100 --since 24h

Or programmatically::

    from yar.evaluation.phoenix_evaluators import run_query_evaluation
    df = run_query_evaluation(project='yar-app', limit=100)

The whole thing is opt-in: it imports ``phoenix.evals`` lazily and degrades
to a clear error message when the dependency is missing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator templates
# ---------------------------------------------------------------------------


_RELEVANCE_TEMPLATE = """You are evaluating retrieval quality for a RAG system.

Given a user QUERY and the RETRIEVED DOCUMENTS that were surfaced for it,
judge whether the retrieved documents are RELEVANT to answering the query.

Rubric:
* RELEVANT — at least one document directly addresses the query's subject.
  For COMPARISON queries ("compare X with Y", "differences between A and B"),
  RELEVANT also applies when documents from BOTH sides of the comparison are
  present in the context, even if no single document covers the comparison
  exhaustively. The judgment is about whether the retriever surfaced the
  right material, not whether the model perfectly synthesized it.
* PARTIAL — documents are on-topic but miss the specific question (e.g.
  comparison query but only ONE side is in the context).
* UNRELATED — none of the documents help answer the query.

QUERY: {input}

RETRIEVED DOCUMENTS:
{reference}

Respond with one word: RELEVANT, PARTIAL, or UNRELATED.
"""


_GROUNDEDNESS_TEMPLATE = """You are evaluating whether a RAG answer is grounded
in the retrieved documents.

Given the QUERY, the RETRIEVED DOCUMENTS that were used as context, and the
ANSWER produced by the model, judge whether every factual claim in the answer
is supported by the documents.

Reply GROUNDED when every factual claim has support in the documents;
UNSUPPORTED when the answer contains claims not found in the documents;
PARTIAL when most claims are grounded but at least one drifts beyond the
provided context.

QUERY: {input}

RETRIEVED DOCUMENTS:
{reference}

ANSWER: {output}

Respond with one word: GROUNDED, PARTIAL, or UNSUPPORTED.
"""


_HALLUCINATION_TEMPLATE = """Detect hallucinations in a RAG answer.

The model produced an ANSWER for the user's QUERY using the RETRIEVED
DOCUMENTS as its only ground truth.

Mark HALLUCINATED only when the answer asserts a SPECIFIC FACT (a named
entity, number, date, mechanism, target, classification, or relationship)
that is not derivable from the retrieved documents. "Derivable" means a
human reading the documents could agree the answer is supported by what
they say — synonyms, paraphrases, restructuring, or summarization of
grounded content do NOT count as hallucination as long as no NEW
independent fact is introduced.

Mark FACTUAL when:
* All specific facts are supported by the documents.
* The model uses different wording but the underlying claims trace back
  to the documents.
* The model abstracts or summarizes (e.g. "documentation issues" as a
  category for items the documents call out individually).
* The model honestly admits a gap ("the context does not cover X") even
  if it provides grounded info for the rest.

Mark HALLUCINATED only when:
* A named entity / target / number / date appears in the answer that has
  no support in the documents.
* The model invents a mechanism, classification, or causal link that the
  documents do not state.

When in doubt, prefer FACTUAL — only call HALLUCINATED if you can point
at the specific unsupported claim.

QUERY: {input}

RETRIEVED DOCUMENTS:
{reference}

ANSWER: {output}

Respond with one word: FACTUAL or HALLUCINATED.
"""


_DOC_RELEVANCE_TEMPLATE = """You are evaluating whether a single retrieved
document is relevant to a user query in a RAG system.

Mark RELEVANT when the document directly addresses the query subject and
contains material that would help compose a grounded answer; UNRELATED
when the document is on a different topic or only tangentially related.
PARTIAL when the document touches on the topic but lacks the specific
detail the query asks for.

QUERY: {input}

DOCUMENT:
{reference}

Respond with one word: RELEVANT, PARTIAL, or UNRELATED.
"""


_REFUSAL_TEMPLATE = """Detect refusals in a RAG answer.

The model produced an ANSWER for the user's QUERY.

Rubric:
* REFUSAL — the answer is essentially "I can't tell you anything" because
  none of the retrieved documents addressed the question. Phrases like
  "insufficient information", "the documents do not contain that
  information", and a body that provides no concrete factual content
  qualify as REFUSAL.
* ANSWERED — the model gave concrete factual content for the question OR
  for at least one part of a multi-part / comparison question. Important:
  for comparison questions ("compare X with Y") it counts as ANSWERED if
  the model presents one side substantively, even when it notes that the
  other side wasn't in the context. A partial-coverage answer is NOT a
  refusal.
* The metric records the model's *behaviour* (refused vs. attempted), not
  whether the refusal/answer was correct.

QUERY: {input}

ANSWER: {output}

Respond with one word: REFUSAL or ANSWERED.
"""


_DEFAULT_CHOICES = {
    'relevance': {'RELEVANT': 1.0, 'PARTIAL': 0.5, 'UNRELATED': 0.0},
    'groundedness': {'GROUNDED': 1.0, 'PARTIAL': 0.5, 'UNSUPPORTED': 0.0},
    'hallucination': {'FACTUAL': 0.0, 'HALLUCINATED': 1.0},
    'refusal': {'ANSWERED': 0.0, 'REFUSAL': 1.0},
    'doc_relevance': {'RELEVANT': 1.0, 'PARTIAL': 0.5, 'UNRELATED': 0.0},
}


_DIRECTIONS = {
    'relevance': 'maximize',
    'groundedness': 'maximize',
    'hallucination': 'minimize',
    # No fixed direction: refusal is good when context is missing, bad when
    # context is rich. Phoenix UI shows it as a tracked label rather than a
    # score to maximize/minimize.
    'refusal': 'minimize',
    'doc_relevance': 'maximize',
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    project: str = 'yar-app'
    limit: int = 100
    since: timedelta = field(default_factory=lambda: timedelta(hours=24))
    judge_provider: str = 'openai'
    judge_model: str = 'gpt-4o-mini'
    evaluators: list[str] = field(default_factory=lambda: ['relevance', 'groundedness', 'hallucination', 'refusal'])
    push_annotations: bool = True
    sample_size: int | None = None
    api_key: str | None = None
    base_url: str | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _require_phoenix_evals() -> Any:
    try:
        from phoenix import evals as pe
    except Exception as exc:
        raise RuntimeError(
            'phoenix.evals is required for auto-evaluation. Install with `pip install -e .[observability]`.'
        ) from exc
    return pe


def _build_evaluator(name: str, llm: Any) -> Any:
    pe = _require_phoenix_evals()
    template = {
        'relevance': _RELEVANCE_TEMPLATE,
        'groundedness': _GROUNDEDNESS_TEMPLATE,
        'hallucination': _HALLUCINATION_TEMPLATE,
        'refusal': _REFUSAL_TEMPLATE,
        'doc_relevance': _DOC_RELEVANCE_TEMPLATE,
    }.get(name)
    if template is None:
        raise ValueError(f'Unknown evaluator: {name!r}')
    return pe.create_classifier(
        name=name,
        prompt_template=template,
        llm=llm,
        choices=_DEFAULT_CHOICES[name],
        direction=_DIRECTIONS[name],
    )


def _extract_context_section(prompt: str) -> str:
    """Pull the body after ``---Context---`` from a synthesis prompt.

    YAR's RAG prompt template ends with::

        ---Context---
        # Knowledge Graph Data (Entity)
        ...
        # Knowledge Graph Data (Relationship)
        ...
        # Document Chunks
        ...
        # Reference Document List
        ...

    Returning that block gives the judge exactly what the synthesis LLM saw —
    KG entity/relation descriptions plus chunk text — instead of just the
    flattened chunks (which miss KG-only grounding).
    """
    if not prompt:
        return ''
    marker = '---Context---'
    idx = prompt.find(marker)
    if idx < 0:
        return prompt
    return prompt[idx + len(marker) :].strip()


def _fetch_synthesis_contexts(
    *,
    client: Any,
    project: str,
    start_time: Any,
    end_time: Any,
    limit: int,
) -> dict[str, str]:
    """Map trace_id -> synthesis-prompt context section for synthesis LLM children.

    Skips keyword-extraction LLM calls (``llm.keyword_extraction == True``) and
    returns the largest input message from each trace's synthesis span — that
    is the prompt with full KG + chunk context the synthesis LLM actually
    received.
    """
    try:
        import pandas as pd
        from phoenix.client.types.spans import SpanQuery
    except Exception:
        return {}

    query = SpanQuery().where("name == 'llm.openai.complete'")
    spans = client.spans.get_spans_dataframe(
        query=query,
        project_identifier=project,
        start_time=start_time,
        end_time=end_time,
        limit=limit * 4,  # one chain typically has 1-3 LLM children
    )
    if spans is None or spans.empty:
        return {}

    # Phoenix dataframe shape (current): ``attributes.llm`` is a dict per cell
    # carrying flag attributes (e.g. ``keyword_extraction``), and
    # ``attributes.llm.input_messages`` is a list-of-dicts per cell with
    # ``message.role`` / ``message.content`` keys. Some attributes also surface
    # as flat ``attributes.llm.<name>`` columns when they appear often enough.
    def _is_keyword_extraction(row: Any) -> bool:
        meta = row.get('attributes.llm') if hasattr(row, 'get') else None
        if isinstance(meta, dict) and meta.get('keyword_extraction'):
            return True
        flat = row.get('attributes.llm.keyword_extraction') if hasattr(row, 'get') else None
        return bool(flat)

    def _system_message_content(row: Any) -> str:
        msgs = row.get('attributes.llm.input_messages') if hasattr(row, 'get') else None
        if not isinstance(msgs, list):
            return ''
        for m in msgs:
            if isinstance(m, dict) and m.get('message.role') == 'system':
                return str(m.get('message.content') or '')
        # No system message — fall back to first message of any role.
        for m in msgs:
            if isinstance(m, dict):
                return str(m.get('message.content') or '')
        return ''

    out: dict[str, str] = {}
    if 'context.trace_id' not in spans.columns:
        return out

    for _, row in spans.iterrows():
        if _is_keyword_extraction(row):
            continue
        tid = row.get('context.trace_id')
        if tid is None or pd.isna(tid):
            continue
        prompt = _system_message_content(row)
        if not prompt:
            continue
        ctx = _extract_context_section(prompt)
        if not ctx:
            continue
        # Keep the largest synthesis context per trace — usually only one
        # synthesis call but be defensive when reranks/summaries share a trace.
        existing = out.get(str(tid), '')
        if len(ctx) > len(existing):
            out[str(tid)] = ctx
    return out


def _fetch_query_dataframe(*, config: EvalConfig) -> Any:
    """Pull recent query chain spans into a dataframe with input/output/reference."""
    try:
        import pandas as pd
        from phoenix.client import Client
        from phoenix.client.types.spans import SpanQuery
    except Exception as exc:
        raise RuntimeError(
            'pandas and phoenix.client are required. Install via `pip install -e .[observability]`.'
        ) from exc

    client_kwargs: dict[str, Any] = {}
    if config.base_url:
        client_kwargs['base_url'] = config.base_url
    if config.api_key:
        client_kwargs['api_key'] = config.api_key
    client = Client(**client_kwargs)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - config.since

    query = SpanQuery().where("name in ('app.query', 'app.query_stream', 'app.query_data')")
    spans = client.spans.get_spans_dataframe(
        query=query,
        project_identifier=config.project,
        start_time=start_time,
        end_time=end_time,
        limit=config.limit,
    )
    if spans is None or spans.empty:
        logger.warning('No query traces found in project %s for the requested window.', config.project)
        return pd.DataFrame({'span_id': [], 'trace_id': [], 'input': [], 'output': [], 'reference': []})

    # Pull the synthesis-prompt context for these traces so the judge sees the
    # exact KG + chunk grounding the synthesis LLM consumed. This eliminates
    # false-hallucination calls when answers cite KG entities/relations whose
    # descriptions never appear in the chunk text.
    synthesis_contexts = _fetch_synthesis_contexts(
        client=client,
        project=config.project,
        start_time=start_time,
        end_time=end_time,
        limit=config.limit,
    )

    inputs = spans.get('attributes.input.value', pd.Series([''] * len(spans), index=spans.index))
    outputs = spans.get('attributes.output.value', pd.Series([''] * len(spans), index=spans.index))

    references: list[str] = []
    for _, row in spans.iterrows():
        tid = row.get('context.trace_id')
        synth_ref = synthesis_contexts.get(str(tid)) if tid is not None else None
        if synth_ref:
            references.append(synth_ref)
            continue
        # Fallback: rebuild from retrieval.documents when synthesis context is
        # unavailable (older traces, partial captures, or instrumentor disabled).
        docs_attr = row.get('attributes.retrieval.documents')
        docs: list[str] = []
        if isinstance(docs_attr, list):
            for entry in docs_attr:
                if not isinstance(entry, dict):
                    continue
                content = entry.get('document.content') or entry.get('content')
                if content:
                    docs.append(str(content))
        else:
            for idx in range(50):
                content = row.get(f'attributes.retrieval.documents.{idx}.document.content')
                if content is None or (isinstance(content, float) and pd.isna(content)):
                    break
                docs.append(str(content))
        references.append('\n\n---\n\n'.join(docs))

    out_df = pd.DataFrame(
        {
            'span_id': spans.get('context.span_id', spans.index),
            'trace_id': spans.get('context.trace_id', spans.index),
            'input': inputs.fillna('').astype(str),
            'output': outputs.fillna('').astype(str),
            'reference': references,
        }
    )

    if config.sample_size is not None and config.sample_size < len(out_df):
        out_df = out_df.sample(n=config.sample_size, random_state=42).reset_index(drop=True)

    return out_df


def run_query_evaluation(
    *,
    project: str = 'yar-app',
    limit: int = 100,
    since: timedelta | None = None,
    evaluators: list[str] | None = None,
    judge_model: str = 'gpt-4o-mini',
    judge_provider: str = 'openai',
    sample_size: int | None = None,
    push_annotations: bool = True,
) -> Any:
    """Run LLM-as-judge evaluators against recent YAR query traces.

    Returns the resulting dataframe (input/output/reference + per-evaluator
    score columns). When ``push_annotations`` is True the scores are also
    posted back as span annotations on the originating chain spans.
    """
    pe = _require_phoenix_evals()

    config = EvalConfig(
        project=project,
        limit=limit,
        since=since or timedelta(hours=24),
        evaluators=evaluators or ['relevance', 'groundedness', 'hallucination', 'refusal'],
        judge_model=judge_model,
        judge_provider=judge_provider,
        push_annotations=push_annotations,
        sample_size=sample_size,
    )

    df = _fetch_query_dataframe(config=config)
    if df.empty:
        return df

    llm = pe.LLM(provider=config.judge_provider, model=config.judge_model)
    built = [_build_evaluator(name, llm) for name in config.evaluators]

    scored = pe.evaluate_dataframe(df, built)

    if config.push_annotations:
        _push_annotations(df=scored, config=config)

    return scored


def _fetch_document_dataframe(*, config: EvalConfig) -> Any:
    """Build a dataframe with one row per (trace, retrieved document).

    Each row carries ``input`` (query), ``reference`` (single document
    content), and ``span_id`` / ``doc_id`` for annotation push-back. Used by
    ``run_document_evaluation`` to score per-chunk relevance separately from
    the chain-level relevance metric.
    """
    try:
        import pandas as pd
        from phoenix.client import Client
        from phoenix.client.types.spans import SpanQuery
    except Exception as exc:
        raise RuntimeError(
            'pandas and phoenix.client are required. Install via `pip install -e .[observability]`.'
        ) from exc

    client_kwargs: dict[str, Any] = {}
    if config.base_url:
        client_kwargs['base_url'] = config.base_url
    if config.api_key:
        client_kwargs['api_key'] = config.api_key
    client = Client(**client_kwargs)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - config.since

    query = SpanQuery().where("name in ('app.query', 'app.query_stream', 'app.query_data')")
    spans = client.spans.get_spans_dataframe(
        query=query,
        project_identifier=config.project,
        start_time=start_time,
        end_time=end_time,
        limit=config.limit,
    )
    if spans is None or spans.empty:
        logger.warning('No query traces found in project %s for the requested window.', config.project)
        return pd.DataFrame({'span_id': [], 'doc_id': [], 'input': [], 'reference': []})

    rows: list[dict[str, Any]] = []
    for _, span_row in spans.iterrows():
        span_id = str(span_row.get('context.span_id') or '')
        query_text = str(span_row.get('attributes.input.value') or '')
        if not span_id or not query_text:
            continue
        docs_attr = span_row.get('attributes.retrieval.documents')
        if not isinstance(docs_attr, list):
            continue
        for idx, entry in enumerate(docs_attr):
            if not isinstance(entry, dict):
                continue
            content = entry.get('document.content') or entry.get('content')
            if not content:
                continue
            doc_id = entry.get('document.id') or entry.get('id') or str(idx)
            rows.append(
                {
                    'span_id': span_id,
                    'doc_id': str(doc_id),
                    'doc_index': idx,
                    'input': query_text,
                    'reference': str(content),
                }
            )

    out_df = pd.DataFrame(rows)
    if config.sample_size is not None and config.sample_size < len(out_df):
        out_df = out_df.sample(n=config.sample_size, random_state=42).reset_index(drop=True)
    return out_df


def run_document_evaluation(
    *,
    project: str = 'yar-app',
    limit: int = 100,
    since: timedelta | None = None,
    judge_model: str = 'gpt-4o-mini',
    judge_provider: str = 'openai',
    sample_size: int | None = None,
    push_annotations: bool = True,
) -> Any:
    """Score every retrieved document individually for query-relevance.

    Returns a dataframe with one row per ``(span, document_index)`` plus a
    ``doc_relevance_score`` payload column. Pushes per-document annotations
    using the document index as the annotator key when enabled.
    """
    pe = _require_phoenix_evals()

    config = EvalConfig(
        project=project,
        limit=limit,
        since=since or timedelta(hours=24),
        evaluators=['doc_relevance'],
        judge_model=judge_model,
        judge_provider=judge_provider,
        push_annotations=push_annotations,
        sample_size=sample_size,
    )

    df = _fetch_document_dataframe(config=config)
    if df.empty:
        return df

    llm = pe.LLM(provider=config.judge_provider, model=config.judge_model)
    classifier = _build_evaluator('doc_relevance', llm)
    scored = pe.evaluate_dataframe(df, [classifier])

    if config.push_annotations:
        _push_document_annotations(df=scored, config=config)

    return scored


def _push_document_annotations(*, df: Any, config: EvalConfig) -> None:
    """Push per-document relevance scores back as document annotations."""
    try:
        from phoenix.client import Client
    except Exception:  # pragma: no cover
        return
    client_kwargs: dict[str, Any] = {}
    if config.base_url:
        client_kwargs['base_url'] = config.base_url
    if config.api_key:
        client_kwargs['api_key'] = config.api_key
    client = Client(**client_kwargs)

    for _, row in df.iterrows():
        span_id = str(row.get('span_id') or '')
        doc_index = row.get('doc_index')
        if not span_id or doc_index is None:
            continue
        payload = row.get('doc_relevance_score')
        if isinstance(payload, str):
            try:
                import ast

                payload = ast.literal_eval(payload)
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            continue
        try:
            score_value = payload.get('score')
            label_value = payload.get('label')
            explanation_value = payload.get('explanation')
            client.spans.add_document_annotation(
                span_id=span_id,
                document_position=int(doc_index),
                annotation_name='doc_relevance',
                label=str(label_value) if label_value is not None else None,
                score=float(score_value) if score_value is not None else None,
                explanation=str(explanation_value) if explanation_value else None,
                annotator_kind='LLM',
            )
        except Exception as exc:
            logger.warning('Failed to push doc_relevance for span %s doc %s: %s', span_id, doc_index, exc)


def _push_annotations(*, df: Any, config: EvalConfig) -> None:
    try:
        from phoenix.client import Client
    except Exception:  # pragma: no cover
        return
    client_kwargs: dict[str, Any] = {}
    if config.base_url:
        client_kwargs['base_url'] = config.base_url
    if config.api_key:
        client_kwargs['api_key'] = config.api_key
    client = Client(**client_kwargs)

    for _, row in df.iterrows():
        span_id = str(row.get('span_id') or '')
        if not span_id:
            continue
        for evaluator_name in config.evaluators:
            # New phoenix.evals shape: ``{evaluator}_score`` is a dict with keys
            # ``score``, ``label``, ``explanation``. Older releases produced flat
            # numeric/string columns, so we accept either.
            score_col = f'{evaluator_name}_score'
            payload = row.get(score_col)
            if isinstance(payload, str):
                try:
                    import ast

                    payload = ast.literal_eval(payload)
                except Exception:
                    payload = None
            if isinstance(payload, dict):
                score = payload.get('score')
                label = payload.get('label')
                explanation = payload.get('explanation')
            else:
                score = payload
                label = row.get(f'{evaluator_name}_label')
                explanation = row.get(f'{evaluator_name}_explanation')
            if score is None and label is None:
                continue
            try:
                client.spans.add_span_annotation(
                    span_id=span_id,
                    annotation_name=evaluator_name,
                    label=str(label) if label is not None else None,
                    score=float(score) if score is not None else None,
                    explanation=str(explanation) if explanation is not None else None,
                    annotator_kind='LLM',
                )
            except Exception as exc:
                logger.warning('Failed to push %s annotation for span %s: %s', evaluator_name, span_id, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_since(raw: str) -> timedelta:
    raw = raw.strip().lower()
    if raw.endswith('h'):
        return timedelta(hours=float(raw[:-1]))
    if raw.endswith('d'):
        return timedelta(days=float(raw[:-1]))
    if raw.endswith('m'):
        return timedelta(minutes=float(raw[:-1]))
    return timedelta(hours=float(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run LLM-as-judge evaluators on YAR query traces.')
    parser.add_argument('--project', default=os.getenv('YAR_TRACE_PROJECT', 'yar-app'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--since', default='24h', help='Lookback window: 24h, 7d, 30m, etc.')
    parser.add_argument('--evaluators', default='relevance,groundedness,hallucination,refusal')
    parser.add_argument('--judge-model', default='gpt-4o-mini')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--no-annotations', action='store_true', help='Skip pushing annotations to Phoenix.')
    parser.add_argument('--output', default=None, help='Optional CSV path to dump scored dataframe.')
    parser.add_argument(
        '--per-document',
        action='store_true',
        help='Run per-document relevance scoring instead of chain-level evaluators.',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    if args.per_document:
        df = run_document_evaluation(
            project=args.project,
            limit=args.limit,
            since=_parse_since(args.since),
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            sample_size=args.sample_size,
            push_annotations=not args.no_annotations,
        )
        unit = 'documents'
    else:
        df = run_query_evaluation(
            project=args.project,
            limit=args.limit,
            since=_parse_since(args.since),
            evaluators=[name.strip() for name in args.evaluators.split(',') if name.strip()],
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            sample_size=args.sample_size,
            push_annotations=not args.no_annotations,
        )
        unit = 'traces'

    if df.empty:
        logger.info('No %s evaluated.', unit)
        return 0

    logger.info('Evaluated %d %s.', len(df), unit)
    if args.output:
        df.to_csv(args.output, index=False)
        logger.info('Wrote scored dataframe to %s', args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
