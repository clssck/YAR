"""DSPy-driven optimization of the YAR synthesis instructions.

We hand-coded several rounds of negative examples into ``yar/prompt.py`` to
suppress specific failure modes. That's overfitting to the eval corpus.
This module replaces that loop with a DSPy MIPROv2 / GEPA optimizer that:

1. Takes a held-out sample of baseline queries from the auto-generated
   evaluation set.
2. For each query, hits the live yar-server, captures the synthesis-prompt
   *context block* (KG entities + relationships + chunks + reference
   list) from the resulting Phoenix trace, and saves it as the input to
   the DSPy module.
3. Iterates the system instructions for a generic RAG signature
   (``query, context -> answer``) using MIPROv2 / GEPA.
4. Scores each candidate answer with a composite metric built on the
   same judge templates Phoenix uses (refusal correctness + groundedness
   + hallucination + relevance for answer-expected; refusal compliance
   + grounded refusal for refusal-expected).
5. Outputs the optimized instruction string to a JSON artifact and to
   stdout. The operator copies it back into ``PROMPTS['rag_response']``
   in ``yar/prompt.py`` (we don't touch the file automatically because
   the structure has named placeholders the optimizer doesn't know
   about — section numbering, ``{response_type}``, ``{user_prompt}``,
   ``{context_data}``).

Run::

    OPENAI_BASE_URL=http://localhost:4000/v1 \
    OPENAI_API_KEY=sk-litellm-master-key \
      .venv/bin/python -m yar.evaluation.dspy_optimize \
        --baseline /tmp/yar_generated_v3.json \
        --train-size 15 --val-size 10 \
        --task-model tuna --reflection-model tuna \
        --output /tmp/dspy_optimized_instructions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

# gpt-5 / o1 series reject temperature != 1; let LiteLLM silently drop the
# unsupported parameter so judge/reflection calls still go through.
try:
    import litellm  # type: ignore[import-not-found]

    litellm.drop_params = True
except Exception:  # pragma: no cover - litellm ships alongside dspy
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    baseline_path: str
    server_url: str = 'http://localhost:9621'
    yar_api_key: str | None = None
    train_size: int = 15
    val_size: int = 10
    task_model: str = 'tuna'
    task_provider: str = 'openai'
    task_base_url: str | None = None
    task_api_key: str | None = None
    judge_model: str = 'tuna'
    judge_provider: str = 'openai'
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    reflection_model: str = 'tuna'
    reflection_provider: str = 'openai'
    reflection_base_url: str | None = None
    reflection_api_key: str | None = None
    base_url: str = 'http://localhost:4000/v1'
    api_key: str = 'sk-litellm-master-key'
    optimizer: str = 'mipro'  # or 'gepa'
    auto_setting: str = 'light'  # 'light' | 'medium' | 'heavy'
    output_path: str | None = None
    seed: int = 17
    phoenix_project: str = 'yar-app'
    # When non-empty, every metric call inspects the current candidate
    # instruction (``module.predict.signature.instructions``) and applies
    # ``forbid_terms_penalty`` if any forbidden substring appears. Used to
    # block GEPA from "winning" by encoding corpus-specific entity names,
    # role names, or quoted phrases into the prompt itself.
    forbid_terms: tuple[str, ...] = ()
    forbid_terms_penalty: float = 0.0  # final score multiplier when leak detected (0 = hard veto)


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------


def _yar_query(config: OptimizerConfig, query: str) -> tuple[str, str]:
    """Hit /query, return (trace_id, response_text)."""
    headers = {'Content-Type': 'application/json'}
    if config.yar_api_key:
        headers['X-API-Key'] = config.yar_api_key
    body = {'query': query, 'mode': 'mix', 'top_k': 10, 'disable_cache': True}
    with httpx.Client(timeout=120) as client:
        resp = client.post(f'{config.server_url}/query', json=body, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    trace_id = resp.headers.get('x-yar-trace-id', '')
    answer = str(payload.get('response') or '')
    return trace_id, answer


def _fetch_synthesis_context(trace_id: str, project: str) -> str:
    """Pull the synthesis-prompt context block from a Phoenix trace.

    Reuses the helper logic from ``phoenix_evaluators`` so the evaluator
    and the optimizer always see the same input to grade against.
    """
    if not trace_id:
        return ''
    try:
        from phoenix.client import Client
        from phoenix.client.types.spans import SpanQuery
    except Exception:
        return ''
    client = Client()
    end_time = datetime.now(timezone.utc) + timedelta(minutes=2)
    start_time = end_time - timedelta(hours=2)
    try:
        spans = client.spans.get_spans_dataframe(
            query=SpanQuery().where(f"name == 'llm.openai.complete' and trace_id == '{trace_id}'"),
            project_identifier=project,
            start_time=start_time,
            end_time=end_time,
            limit=20,
        )
    except Exception:
        return ''
    if spans is None or spans.empty:
        return ''

    # Pick the largest non-keyword-extraction system message — that's the synthesis prompt.
    biggest = ''
    for _, row in spans.iterrows():
        meta = row.get('attributes.llm') if hasattr(row, 'get') else None
        if isinstance(meta, dict) and meta.get('keyword_extraction'):
            continue
        msgs = row.get('attributes.llm.input_messages')
        if not isinstance(msgs, list):
            continue
        sys_msg = ''
        for m in msgs:
            if isinstance(m, dict) and m.get('message.role') == 'system':
                sys_msg = str(m.get('message.content') or '')
                break
        if len(sys_msg) > len(biggest):
            biggest = sys_msg

    if not biggest:
        return ''
    marker = '---Context---'
    idx = biggest.find(marker)
    if idx < 0:
        return biggest
    return biggest[idx + len(marker) :].strip()


def _build_examples(config: OptimizerConfig) -> list[Any]:
    """For each baseline query: call /query (collect trace_ids), wait for span
    ingestion, then batch-fetch synthesis context per trace_id.

    Phoenix's spans dataframe API is timing-sensitive — querying by
    ``trace_id`` immediately after the /query response often hits the index
    before the OTel span has propagated. We collect all trace_ids first,
    sleep for ingestion, then resolve contexts in a second pass.
    """
    import time

    import dspy

    with open(config.baseline_path) as f:
        baseline = json.load(f)

    rng = random.Random(config.seed)
    rng.shuffle(baseline)

    pending: list[tuple[dict[str, Any], str]] = []
    for entry in baseline:
        query = entry.get('query', '').strip()
        if not query:
            continue
        try:
            trace_id, _ = _yar_query(config, query)
        except Exception as exc:
            logger.warning('yar /query failed for %r: %s', query[:60], exc)
            continue
        if not trace_id:
            logger.warning('no x-yar-trace-id header for %r', query[:60])
            continue
        pending.append((entry, trace_id))
        logger.info('  collected trace %s for %r', trace_id[:12], query[:60])

    flush_wait = 30
    logger.info('Sleeping %ds for Phoenix ingestion...', flush_wait)
    time.sleep(flush_wait)

    examples: list[Any] = []
    for entry, trace_id in pending:
        context = _fetch_synthesis_context(trace_id, config.phoenix_project)
        if not context:
            logger.warning('no synthesis context for trace_id=%s; skipping', trace_id)
            continue
        ex = dspy.Example(
            query=entry['query'],
            context=context,
            intent=entry.get('intent', ''),
            should_refuse=bool(entry.get('should_refuse', False)),
        ).with_inputs('query', 'context')
        examples.append(ex)

    return examples


# ---------------------------------------------------------------------------
# Signature + module
# ---------------------------------------------------------------------------


def _build_signature():
    import dspy

    class RAGAnswer(dspy.Signature):
        """Answer a question grounded in the retrieved context.

        The context is a structured block containing knowledge-graph entities,
        knowledge-graph relationships, document chunks, and a reference list.

        Rules:
        * Every factual claim must be supported by something in the context.
        * Do not pull facts from world knowledge or training data; if the
          context does not address what the question asks, say so explicitly
          and stop.
        * When the context has the answer, cite supporting chunks inline as
          ``[n]`` using the reference_id from the reference list. Skip
          citations when the reference list is empty.
        """

        query: str = dspy.InputField(desc='The user question.')
        context: str = dspy.InputField(desc='The retrieved KG + chunks + reference list.')
        answer: str = dspy.OutputField(desc='The grounded answer, with inline [n] citations when supported.')

    return RAGAnswer


def _build_module():
    import dspy

    class RAGModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(_build_signature())

        def forward(self, query: str, context: str):  # type: ignore[override]
            return self.predict(query=query, context=context)

    return RAGModule()


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------


_REFUSAL_PHRASES = (
    'insufficient information',
    'the documents do not',
    'the available context does not',
    'the retrieved context does not',
    'not addressed in the retrieved',
    'the context does not state',
    'the context does not specify',
    'the context does not contain',
    'no information about',
    'is not described in the retrieved',
    'is not mentioned in the retrieved',
)


def _looks_like_refusal(text: str) -> bool:
    lowered = (text or '').casefold()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


_GROUNDEDNESS_TEMPLATE = """You are evaluating whether a RAG answer is grounded in the retrieved context.

CONTEXT:
{context}

QUERY: {query}

ANSWER: {answer}

Reply GROUNDED if every factual claim in the answer is supported by the context (paraphrases ok).
Reply UNSUPPORTED if any specific fact (named entity, number, date, mechanism, target, classification, relationship) is not supported by the context.
Reply with one word: GROUNDED or UNSUPPORTED.
"""


_HALLUCINATION_TEMPLATE = """You are detecting hallucinations in a RAG answer.

CONTEXT:
{context}

QUERY: {query}

ANSWER: {answer}

Reply HALLUCINATED if the answer asserts a specific fact (named entity, number, date, mechanism, target, classification, relationship) that does not appear in the context.
Reply FACTUAL if every specific fact traces back to the context. Paraphrases and abstractions over grounded content are FACTUAL.
Reply with one word: FACTUAL or HALLUCINATED.
"""


def _judge_call(judge_lm: Any, prompt: str) -> str:
    try:
        resp = judge_lm(prompt)
    except Exception as exc:
        logger.debug('judge LM call failed: %s', exc)
        return ''
    if isinstance(resp, list) and resp:
        resp = resp[0]
    return str(resp).strip().split()[0] if resp else ''


def _composite_metric(
    judge_lm: Any,
    *,
    example: Any,
    prediction: Any,
    trace: Any = None,
    return_detail: bool = False,
) -> Any:
    """Score an answer on refusal correctness + groundedness + hallucination.

    When ``return_detail`` is True, returns a ``(score, detail_dict)`` tuple
    so the GEPA wrapper can surface labels + reasons to reflection.
    """
    del trace
    answer = str(getattr(prediction, 'answer', '') or '').strip()
    query = str(getattr(example, 'query', '') or '')
    context = str(getattr(example, 'context', '') or '')
    should_refuse = bool(getattr(example, 'should_refuse', False))
    intent = str(getattr(example, 'intent', '') or '')

    detail: dict[str, Any] = {
        'intent': intent,
        'should_refuse': should_refuse,
        'refused': False,
        'grounded_label': None,
        'hallucination_label': None,
    }

    if not answer:
        return (0.0, {**detail, 'reason': 'empty_answer'}) if return_detail else 0.0

    refused = _looks_like_refusal(answer)
    detail['refused'] = refused
    refusal_score = 1.0 if (should_refuse == refused) else 0.0

    if should_refuse and refused:
        detail['reason'] = 'correct_refusal'
        return (1.0, detail) if return_detail else 1.0
    if not should_refuse and refused:
        detail['reason'] = 'incorrect_refusal_on_answerable'
        return (0.0, detail) if return_detail else 0.0

    grounded_resp = _judge_call(
        judge_lm,
        _GROUNDEDNESS_TEMPLATE.format(context=context[:18000], query=query, answer=answer[:4000]),
    )
    grounded_label = 'GROUNDED' if 'GROUND' in grounded_resp.upper() else 'UNSUPPORTED'
    grounded_score = 1.0 if grounded_label == 'GROUNDED' else 0.0
    detail['grounded_label'] = grounded_label

    hal_resp = _judge_call(
        judge_lm,
        _HALLUCINATION_TEMPLATE.format(context=context[:18000], query=query, answer=answer[:4000]),
    )
    hallucination_label = 'HALLUCINATED' if 'HALLUCIN' in hal_resp.upper() else 'FACTUAL'
    hal_score = 0.0 if hallucination_label == 'HALLUCINATED' else 1.0
    detail['hallucination_label'] = hallucination_label

    score = round(0.4 * grounded_score + 0.4 * hal_score + 0.2 * refusal_score, 3)
    detail['reason'] = 'all_pass' if score >= 0.99 else f'grounded={grounded_label}, hal={hallucination_label}'
    return (score, detail) if return_detail else score


# ---------------------------------------------------------------------------
# Optimizer driver
# ---------------------------------------------------------------------------


def run_optimization(config: OptimizerConfig) -> dict[str, Any]:
    import dspy
    from dspy.teleprompt import MIPROv2

    # Wire DSPy LM (task model the optimized module uses to generate answers).
    task_lm = dspy.LM(
        f'{config.task_provider}/{config.task_model}',
        api_base=config.task_base_url or config.base_url,
        api_key=config.task_api_key or config.api_key,
        max_tokens=2000,
    )
    judge_lm = dspy.LM(
        f'{config.judge_provider}/{config.judge_model}',
        api_base=config.judge_base_url or config.base_url,
        api_key=config.judge_api_key or config.api_key,
        max_tokens=20,
        temperature=0.0,
    )
    dspy.settings.configure(lm=task_lm)

    examples = _build_examples(config)
    if len(examples) < config.train_size + 1:
        raise RuntimeError(
            f'Not enough examples after dataset prep: got {len(examples)}, need at least {config.train_size + 1}.'
        )

    trainset = examples[: config.train_size]
    valset = examples[config.train_size : config.train_size + config.val_size]
    logger.info('Train: %d examples, Val: %d examples', len(trainset), len(valset))

    def _current_instruction() -> str:
        """Read the candidate instruction GEPA has just written into the
        module's signature. Captured in closure over ``module`` so each
        metric call inspects the *current* mutation, not the baseline."""
        try:
            sig = getattr(module.predict, 'signature', None)
            return str(getattr(sig, 'instructions', '') or '')
        except Exception:
            return ''

    def _detect_leakage(instruction: str) -> list[str]:
        if not config.forbid_terms or not instruction:
            return []
        lowered = instruction.casefold()
        return [t for t in config.forbid_terms if t and t.casefold() in lowered]

    def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float:
        leaks = _detect_leakage(_current_instruction())
        if leaks:
            return float(config.forbid_terms_penalty)
        return _composite_metric(judge_lm, example=example, prediction=prediction, trace=trace)

    def metric_with_detail(example: Any, prediction: Any, trace: Any = None) -> tuple[float, dict[str, Any]]:
        leaks = _detect_leakage(_current_instruction())
        if leaks:
            detail = {
                'reason': 'corpus_leakage',
                'leaked_terms': leaks,
            }
            return float(config.forbid_terms_penalty), detail
        result = _composite_metric(judge_lm, example=example, prediction=prediction, trace=trace, return_detail=True)
        if isinstance(result, tuple):
            return result
        return float(result), {}

    module = _build_module()

    if config.optimizer == 'gepa':
        # GEPA path — uses reflection-based instruction proposal.
        from dspy.teleprompt import GEPA

        reflection_lm = dspy.LM(
            f'{config.reflection_provider}/{config.reflection_model}',
            api_base=config.reflection_base_url or config.base_url,
            api_key=config.reflection_api_key or config.api_key,
            max_tokens=4000,
            temperature=1.0,
        )

        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del pred_name, pred_trace
            score, detail = metric_with_detail(gold, pred, trace)
            # Hand reflection a structured feedback line so it can target the
            # specific failure axis instead of guessing what the composite
            # score punished. Per-axis labels (`grounded`, `hallucination`,
            # `refused`) plus the example's intent give the reflection LM
            # enough signal to propose targeted instruction tweaks.
            feedback_parts = [f'score={score:.2f}', f'reason={detail.get("reason", "?")}']
            leaked = detail.get('leaked_terms') or []
            if leaked:
                feedback_parts.append(
                    'CORPUS_LEAKAGE: prompt instruction contains forbidden corpus-specific terms '
                    f'{leaked}. Rewrite the instruction with NO references to specific entity '
                    'names, role names, document titles, or quoted phrases from any training '
                    'example. Use only generic class-level rules. Score is hard-vetoed to 0 '
                    'until this is fixed.'
                )
            for key in ('intent', 'should_refuse', 'refused', 'grounded_label', 'hallucination_label'):
                value = detail.get(key)
                if value is not None:
                    feedback_parts.append(f'{key}={value}')
            return dspy.Prediction(score=score, feedback='; '.join(feedback_parts))

        compiler = GEPA(
            metric=gepa_metric,
            reflection_lm=reflection_lm,
            auto=config.auto_setting,
            track_best_outputs=True,
            track_stats=True,
        )
        optimized = compiler.compile(module, trainset=trainset, valset=valset or None)
    else:
        compiler = MIPROv2(
            metric=metric_fn,
            auto=config.auto_setting,
            num_threads=4,
        )
        optimized = compiler.compile(
            module,
            trainset=trainset,
            valset=valset or None,
            requires_permission_to_run=False,
        )

    # Extract the optimized signature instructions (and any few-shot demos).
    predict_attr: Any = getattr(optimized, 'predict', optimized)
    sig = getattr(predict_attr, 'signature', None)
    instructions = getattr(sig, 'instructions', '') or ''
    demos = getattr(predict_attr, 'demos', None) or []

    payload: dict[str, Any] = {
        'optimizer': config.optimizer,
        'auto_setting': config.auto_setting,
        'train_size': len(trainset),
        'val_size': len(valset),
        'instructions': instructions,
        'demos': [
            {
                k: v
                for k, v in (demo.toDict() if hasattr(demo, 'toDict') else dict(demo)).items()
                if not k.startswith('_')
            }
            for demo in demos
        ],
    }

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Optimize the YAR synthesis instructions with DSPy.')
    parser.add_argument('--baseline', required=True, help='Path to the auto-generated baseline JSON dump.')
    parser.add_argument('--server-url', default=os.getenv('YAR_SERVER_URL', 'http://localhost:9621'))
    parser.add_argument('--yar-api-key', default=os.getenv('YAR_API_KEY'))
    parser.add_argument('--train-size', type=int, default=15)
    parser.add_argument('--val-size', type=int, default=10)
    parser.add_argument('--task-model', default='tuna')
    parser.add_argument('--task-provider', default='openai')
    parser.add_argument('--task-base-url', default=None, help='Override base URL for task LM.')
    parser.add_argument('--task-api-key', default=None, help='Override API key for task LM.')
    parser.add_argument('--judge-model', default='tuna')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--judge-base-url', default=None, help='Override base URL for judge LM.')
    parser.add_argument('--judge-api-key', default=None, help='Override API key for judge LM.')
    parser.add_argument('--reflection-model', default='tuna')
    parser.add_argument('--reflection-provider', default='openai')
    parser.add_argument('--reflection-base-url', default=None, help='Override base URL for reflection LM.')
    parser.add_argument('--reflection-api-key', default=None, help='Override API key for reflection LM.')
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'http://localhost:4000/v1'))
    parser.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', 'sk-litellm-master-key'))
    parser.add_argument('--optimizer', choices=['mipro', 'gepa'], default='mipro')
    parser.add_argument('--auto', dest='auto_setting', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--output', default=None, help='JSON artifact path for optimized instructions.')
    parser.add_argument('--phoenix-project', default=os.getenv('YAR_TRACE_PROJECT', 'yar-app'))
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument(
        '--forbid-terms',
        default='',
        help='Comma-separated forbidden substrings; if any appear in the candidate '
        'instruction, that proposal is hard-vetoed (score = --forbid-penalty).',
    )
    parser.add_argument(
        '--forbid-penalty',
        type=float,
        default=0.0,
        help='Score returned when the candidate instruction leaks a forbidden term (default 0.0).',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    config = OptimizerConfig(
        baseline_path=args.baseline,
        server_url=args.server_url,
        yar_api_key=args.yar_api_key,
        train_size=args.train_size,
        val_size=args.val_size,
        task_model=args.task_model,
        task_provider=args.task_provider,
        task_base_url=args.task_base_url,
        task_api_key=args.task_api_key,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        reflection_model=args.reflection_model,
        reflection_provider=args.reflection_provider,
        reflection_base_url=args.reflection_base_url,
        reflection_api_key=args.reflection_api_key,
        base_url=args.base_url,
        api_key=args.api_key,
        optimizer=args.optimizer,
        auto_setting=args.auto_setting,
        output_path=args.output,
        seed=args.seed,
        phoenix_project=args.phoenix_project,
        forbid_terms=tuple(t.strip() for t in args.forbid_terms.split(',') if t.strip()),
        forbid_terms_penalty=args.forbid_penalty,
    )

    payload = run_optimization(config)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info('Wrote optimized instructions to %s', args.output)

    print('\n=== OPTIMIZED INSTRUCTIONS ===\n')
    print(payload['instructions'])
    if payload['demos']:
        print(f'\n=== {len(payload["demos"])} few-shot demos selected ===\n')
        for i, demo in enumerate(payload['demos'], 1):
            preview = json.dumps(demo, ensure_ascii=False, default=str)
            preview = re.sub(r'\s+', ' ', preview)[:300]
            print(f'  [{i}] {preview}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
