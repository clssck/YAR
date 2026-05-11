"""Run Phoenix Experiments against a YAR baseline dataset.

This module gives us regression-style A/B testing for prompt or pipeline
changes. Workflow::

    # 1. Capture a baseline dataset from production traces.
    python -m yar.evaluation.phoenix_dataset_export \
        --project yar-app --name yar-baseline-2026-05-08 --since 24h --limit 50

    # 2. Make a code/prompt change.

    # 3. Re-run the baseline as an experiment to measure the delta.
    python -m yar.evaluation.phoenix_experiments \
        --dataset yar-baseline-2026-05-08 \
        --judge-model tuna --judge-provider openai \
        --experiment-name "after-citation-prompt-tighten"

The experiment runs the baseline questions against the *current* yar-server,
captures the new answers, scores them with the same LLM-as-judge evaluators
as ``phoenix_evaluators``, and persists the results in Phoenix's Experiments
view so the diff vs. previous experiments is visible.

Cache is disabled per request so each experiment exercises the live pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    dataset_name: str
    server_url: str = 'http://localhost:9621'
    api_key: str | None = None  # YAR_API_KEY when the server requires auth
    project: str = 'yar-app'
    judge_provider: str = 'openai'
    judge_model: str = 'gpt-5.4-mini'
    evaluators: list[str] = field(default_factory=lambda: ['relevance', 'groundedness', 'hallucination', 'refusal'])
    experiment_name: str | None = None
    experiment_description: str | None = None
    timeout_seconds: int = 90
    repetitions: int = 1
    rate_limit_retries: int = 3
    phoenix_base_url: str | None = None
    phoenix_api_key: str | None = None
    query_overrides: dict[str, Any] = field(default_factory=dict)
    """Extra body params to merge into every /query call (e.g. ``{'top_k': 60, 'enable_rerank': True}``)."""


# ---------------------------------------------------------------------------
# Imports / capability checks
# ---------------------------------------------------------------------------


def _require_phoenix_client() -> Any:
    try:
        from phoenix.client import Client
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError('phoenix.client is required. Install via `pip install -e .[observability]`.') from exc
    return Client


def _require_phoenix_evals() -> Any:
    try:
        from phoenix import evals as pe
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('phoenix.evals is required. Install via `pip install -e .[observability]`.') from exc
    return pe


# ---------------------------------------------------------------------------
# Task: re-run a dataset example against the live yar-server
# ---------------------------------------------------------------------------


def _build_task(config: ExperimentConfig):
    """Return the task callable Phoenix's experiment runner will invoke per example.

    Each invocation hits ``/query`` on the live yar-server with ``disable_cache``
    so prompt/config tweaks are actually exercised. The task returns the RAG
    answer plus the references the judge will need.
    """

    headers: dict[str, str] = {'Content-Type': 'application/json'}
    if config.api_key:
        headers['X-API-Key'] = config.api_key

    def task(example) -> dict[str, Any]:
        # ``example`` from Phoenix is a ``DatasetExample`` dict with shape
        # ``{id, node_id, input: {query, mode}, output, metadata}``. Read
        # defensively in case Phoenix wraps it in a class instead.
        if isinstance(example, dict):
            input_payload = example.get('input') or {}
        else:
            input_payload = getattr(example, 'input', None) or {}
        if not isinstance(input_payload, dict):
            input_payload = {}
        query = str(input_payload.get('query') or '').strip()
        mode = str(input_payload.get('mode') or 'mix').strip() or 'mix'
        if not query:
            logger.warning('Task received empty query for example: %r', example)
            return {'response': '', 'references': [], 'error': 'empty query'}

        body = {
            'query': query,
            'mode': mode,
            'top_k': 10,
            'disable_cache': True,
        }
        # Merge any retrieval-knob overrides supplied via CLI (e.g. top_k, chunk_top_k,
        # max_total_tokens, enable_rerank, retrieval_multiplier). Only override the
        # specific keys the user passes; defaults stay untouched for everything else.
        if config.query_overrides:
            body.update(config.query_overrides)
        try:
            with httpx.Client(timeout=config.timeout_seconds) as client:
                resp = client.post(f'{config.server_url}/query', json=body, headers=headers)
        except Exception as exc:
            return {'response': '', 'references': [], 'error': f'transport: {exc}'}

        if resp.status_code != 200:
            return {
                'response': '',
                'references': [],
                'error': f'http {resp.status_code}: {resp.text[:200]}',
            }
        try:
            payload = resp.json()
        except Exception as exc:
            return {'response': '', 'references': [], 'error': f'json: {exc}'}

        return {
            'response': str(payload.get('response') or ''),
            'references': payload.get('references') or [],
            'mode': mode,
            'query': query,
            # Surface YAR's chain trace_id so the evaluator wrapper can pull
            # the synthesis-prompt context for richer reference text — closes
            # the gap where the experiment evaluator only saw chunk content
            # (no KG entity/relation descriptions).
            'trace_id': resp.headers.get('x-yar-trace-id') or '',
        }

    return task


# ---------------------------------------------------------------------------
# Evaluators: reuse phoenix.evals classifiers from phoenix_evaluators
# ---------------------------------------------------------------------------


def _build_evaluators(config: ExperimentConfig) -> list[Any]:
    """Build the same evaluator set ``phoenix_evaluators`` uses, but adapted
    so each one extracts ``input``/``output``/``reference`` from the experiment
    example/output shape.
    """
    from yar.evaluation.phoenix_evaluators import (
        _DEFAULT_CHOICES,
        _DIRECTIONS,
        _DOC_RELEVANCE_TEMPLATE,
        _GROUNDEDNESS_TEMPLATE,
        _HALLUCINATION_TEMPLATE,
        _REFUSAL_TEMPLATE,
        _RELEVANCE_TEMPLATE,
        _extract_context_section,
    )

    pe = _require_phoenix_evals()

    templates: dict[str, str] = {
        'relevance': _RELEVANCE_TEMPLATE,
        'groundedness': _GROUNDEDNESS_TEMPLATE,
        'hallucination': _HALLUCINATION_TEMPLATE,
        'refusal': _REFUSAL_TEMPLATE,
        'doc_relevance': _DOC_RELEVANCE_TEMPLATE,
    }

    llm = pe.LLM(provider=config.judge_provider, model=config.judge_model)

    # Per-experiment cache so the synthesis context is fetched once per
    # trace, not once per (trace, evaluator) pair (4 evaluators ⇒ 4× the
    # Phoenix round-trips otherwise).
    synthesis_context_cache: dict[str, str] = {}
    fetch_synthesis_context = _make_synthesis_context_fetcher(
        config=config, cache=synthesis_context_cache, extract=_extract_context_section
    )

    evaluators: list[Any] = []
    for name in config.evaluators:
        template = templates.get(name)
        if template is None:
            raise ValueError(f'Unknown evaluator: {name!r}')
        classifier = pe.create_classifier(
            name=name,
            prompt_template=template,
            llm=llm,
            choices=_DEFAULT_CHOICES[name],
            direction=_DIRECTIONS[name],
        )
        evaluators.append(_wrap_classifier_for_experiment(name, classifier, fetch_synthesis_context))

    return evaluators


def _make_synthesis_context_fetcher(
    *,
    config: ExperimentConfig,
    cache: dict[str, str],
    extract: Any,
) -> Any:
    """Build a per-experiment ``trace_id -> reference text`` lookup.

    Returns a function that, given a trace_id, finds the synthesis LLM child
    span via Phoenix's spans dataframe, extracts the ``---Context---`` section
    from its system prompt, and caches the result for the rest of the
    experiment so each trace is fetched at most once.
    """

    def fetch(trace_id: str | None) -> str:
        if not trace_id:
            return ''
        if trace_id in cache:
            return cache[trace_id]
        try:
            from datetime import datetime, timedelta, timezone

            from phoenix.client import Client
            from phoenix.client.types.spans import SpanQuery
        except Exception:
            cache[trace_id] = ''
            return ''
        client_kwargs: dict[str, Any] = {}
        if config.phoenix_base_url:
            client_kwargs['base_url'] = config.phoenix_base_url
        if config.phoenix_api_key:
            client_kwargs['api_key'] = config.phoenix_api_key
        client = Client(**client_kwargs)

        # Wide window — experiment runs can take minutes; trace clock can
        # be off by a couple seconds vs. host clock too.
        end_time = datetime.now(timezone.utc) + timedelta(minutes=2)
        start_time = end_time - timedelta(hours=2)
        try:
            spans = client.spans.get_spans_dataframe(
                query=SpanQuery().where(f"name == 'llm.openai.complete' and trace_id == '{trace_id}'"),
                project_identifier=config.project,
                start_time=start_time,
                end_time=end_time,
                limit=20,
            )
        except Exception as exc:
            logger.debug('synthesis-context lookup failed for %s: %s', trace_id, exc)
            cache[trace_id] = ''
            return ''

        if spans is None or spans.empty:
            cache[trace_id] = ''
            return ''

        # Drop keyword-extraction calls (they don't carry synthesis context).
        def _is_keyword_extraction(row: Any) -> bool:
            meta = row.get('attributes.llm') if hasattr(row, 'get') else None
            if isinstance(meta, dict) and meta.get('keyword_extraction'):
                return True
            flat = row.get('attributes.llm.keyword_extraction') if hasattr(row, 'get') else None
            return bool(flat)

        def _system_message(row: Any) -> str:
            msgs = row.get('attributes.llm.input_messages') if hasattr(row, 'get') else None
            if not isinstance(msgs, list):
                return ''
            for m in msgs:
                if isinstance(m, dict) and m.get('message.role') == 'system':
                    return str(m.get('message.content') or '')
            for m in msgs:
                if isinstance(m, dict):
                    return str(m.get('message.content') or '')
            return ''

        biggest_prompt = ''
        for _, row in spans.iterrows():
            if _is_keyword_extraction(row):
                continue
            prompt = _system_message(row)
            if len(prompt) > len(biggest_prompt):
                biggest_prompt = prompt

        ctx = extract(biggest_prompt) if biggest_prompt else ''
        cache[trace_id] = ctx
        return ctx

    return fetch


def _wrap_classifier_for_experiment(name: str, classifier: Any, fetch_synthesis_context: Any):
    """Adapt a phoenix.evals classifier to the experiment evaluator signature.

    The classifier wants ``{input, output, reference}``; our experiment
    output is ``{response, references, trace_id}`` and the example carries
    the query. We assemble ``reference`` by preferring the synthesis-prompt
    KG context fetched by ``trace_id``, falling back to the chunks list when
    no synthesis context is available (older traces, missing trace_id, span
    not yet exported).

    Phoenix identifies evaluators by the function's ``__name__``; we use
    ``exec`` here to mint a fresh function per call so each evaluator has
    a distinct name in the experiment view (``relevance``, ``groundedness``,
    ``hallucination``, ``refusal``) instead of all collapsing to ``evaluator``.
    """

    def _impl(output: dict[str, Any]) -> dict[str, Any]:
        query = str(output.get('query') or '')
        answer = str(output.get('response') or '')
        trace_id = str(output.get('trace_id') or '')
        # Prefer synthesis prompt KG context — that's exactly what the
        # synthesis LLM saw, including KG entity/relation descriptions and
        # chunk text. Fall back to bare references when unavailable so the
        # eval still produces a verdict (just a noisier one).
        synthesis_context = fetch_synthesis_context(trace_id) if trace_id else ''
        if synthesis_context:
            reference_text = synthesis_context
        else:
            refs = output.get('references') or []
            reference_blocks: list[str] = []
            for ref in refs[:50]:
                if not isinstance(ref, dict):
                    continue
                content = ref.get('content') or ref.get('excerpt') or ''
                if content:
                    reference_blocks.append(str(content))
            reference_text = '\n\n---\n\n'.join(reference_blocks)
        try:
            scores = classifier.evaluate({'input': query, 'output': answer, 'reference': reference_text})
        except Exception as exc:
            return {'label': 'ERROR', 'score': None, 'explanation': f'evaluator: {exc}'}
        if isinstance(scores, list) and scores:
            result: Any = scores[0]
        else:
            result = scores
        if hasattr(result, 'score') and hasattr(result, 'label'):
            return {
                'label': str(result.label) if result.label is not None else None,
                'score': result.score,
                'explanation': str(getattr(result, 'explanation', '') or ''),
            }
        if isinstance(result, dict):
            return result
        return {'label': str(result), 'score': None, 'explanation': ''}

    # Phoenix's experiment view identifies evaluator rows by the callable's
    # ``__name__``; if we just set the attribute on a closure they all share
    # name ``evaluator``. ``exec`` mints a new function with the correct
    # ``co_name`` so each metric (relevance / groundedness / …) lands in its
    # own column.
    namespace: dict[str, Any] = {'_impl': _impl}
    exec(  # nosec
        f'def {name}(*, output, expected=None, **_):\n    return _impl(output)\n',
        namespace,
    )
    return namespace[name]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _build_client(config: ExperimentConfig) -> Any:
    Client = _require_phoenix_client()
    kwargs: dict[str, Any] = {}
    if config.phoenix_base_url:
        kwargs['base_url'] = config.phoenix_base_url
    if config.phoenix_api_key:
        kwargs['api_key'] = config.phoenix_api_key
    return Client(**kwargs)


def run_experiment(*, config: ExperimentConfig) -> Any:
    """Run an experiment over the named dataset and return Phoenix's RanExperiment."""
    client = _build_client(config)

    dataset = client.datasets.get_dataset(dataset=config.dataset_name)
    if dataset is None:
        raise RuntimeError(f'Dataset {config.dataset_name!r} not found in Phoenix.')

    task = _build_task(config)
    evaluators = _build_evaluators(config)

    return client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=config.experiment_name,
        experiment_description=config.experiment_description,
        repetitions=config.repetitions,
        retries=config.rate_limit_retries,
        timeout=config.timeout_seconds,
        print_summary=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run a Phoenix experiment over a YAR baseline dataset.')
    parser.add_argument('--dataset', required=True, help='Phoenix dataset name (e.g. yar-baseline-2026-05-08).')
    parser.add_argument(
        '--server-url',
        default=os.getenv('YAR_SERVER_URL', 'http://localhost:9621'),
        help='YAR server URL the experiment will hit per example.',
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('YAR_API_KEY') or None,
        help='Optional YAR API key (sent as X-API-Key).',
    )
    parser.add_argument('--project', default=os.getenv('YAR_TRACE_PROJECT', 'yar-app'))
    parser.add_argument('--judge-model', default='gpt-5.4-mini')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--evaluators', default='relevance,groundedness,hallucination,refusal')
    parser.add_argument(
        '--query-overrides',
        default='',
        help='JSON object of /query body params to merge into every task call. '
        'Example: \'{"top_k":60,"chunk_top_k":40,"enable_rerank":true,"retrieval_multiplier":2}\'',
    )
    parser.add_argument('--experiment-name', default=None)
    parser.add_argument('--experiment-description', default=None)
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--phoenix-base-url', default=os.getenv('PHOENIX_BASE_URL') or None)
    parser.add_argument('--phoenix-api-key', default=os.getenv('PHOENIX_API_KEY') or None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    config = ExperimentConfig(
        dataset_name=args.dataset,
        server_url=args.server_url,
        api_key=args.api_key,
        project=args.project,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        evaluators=[name.strip() for name in args.evaluators.split(',') if name.strip()],
        experiment_name=args.experiment_name,
        experiment_description=args.experiment_description,
        timeout_seconds=args.timeout,
        repetitions=args.repetitions,
        phoenix_base_url=args.phoenix_base_url,
        phoenix_api_key=args.phoenix_api_key,
        query_overrides=json.loads(args.query_overrides) if args.query_overrides.strip() else {},
    )

    experiment = run_experiment(config=config)

    # Pull the experiment id wherever it lives across phoenix-client versions.
    experiment_id = ''
    if isinstance(experiment, dict):
        experiment_id = str(experiment.get('id') or experiment.get('experiment_id') or '')
    else:
        for attr in ('id', 'experiment_id'):
            value = getattr(experiment, attr, None)
            if value:
                experiment_id = str(value)
                break
    logger.info('Experiment id: %s', experiment_id or '?')

    # Print intent-aware example-level rollup. Phoenix's per-evaluator
    # summary penalizes correct refusals on out-of-scope / mechanism-bait
    # queries (a textbook refusal answer is genuinely UNRELATED and has
    # nothing factual to ground), so the raw aggregate undersells real
    # quality. The intent-aware view ignores those artifact-prone
    # evaluators on refusal-expected examples.
    if experiment_id:
        try:
            from yar.evaluation.phoenix_aggregate import (
                aggregate_experiment,
                render_summary,
            )

            summary = aggregate_experiment(
                experiment_id=experiment_id,
                dataset_name=config.dataset_name,
                phoenix_base_url=config.phoenix_base_url,
                phoenix_api_key=config.phoenix_api_key,
            )
            print()
            print(render_summary(summary))
        except Exception as exc:
            logger.warning('intent-aware aggregation failed: %s', exc)

    return 0


if __name__ == '__main__':
    sys.exit(main())
