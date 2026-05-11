"""DSPy-driven optimization of the YAR keyword-extraction prompt.

The synthesis-prompt optimizer (``dspy_optimize.py``) already covers downstream
answer quality. Empirically the synthesis prompt has plateaued; the remaining
gap moves with retrieval quality, which is gated by which keywords the
``keywords_extraction`` step pulls out of the user query.

This module optimizes a DSPy keyword-extraction module via GEPA / MIPROv2 by
measuring **end-to-end synthesis quality** when those keywords are fed back
into yar's ``/query`` (yar accepts pre-extracted ``hl_keywords`` /
``ll_keywords`` and skips its own extraction step when they are non-empty).

Pipeline per example::

    query -> DSPy extractor -> (hl_keywords, ll_keywords)
                              \\-> POST /query{hl_keywords, ll_keywords}
                                   -> yar synthesizes -> judge
                                                        -> composite score

Run::

    OPENAI_BASE_URL=http://localhost:4000/v1 \\
    OPENAI_API_KEY=sk-litellm-master-key \\
      .venv/bin/python -m yar.evaluation.dspy_optimize_keywords \\
        --baseline /tmp/yar_baseline_curated.json \\
        --train-size 12 --val-size 6 \\
        --task-model tuna --reflection-model tuna --judge-model tuna \\
        --optimizer gepa --auto light \\
        --output /tmp/dspy_keywords.json
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
from typing import Any

import httpx

# gpt-5 / o1 series reject temperature != 1; let LiteLLM silently drop the
# unsupported parameter so judge/reflection calls still go through.
try:
    import litellm  # type: ignore[import-not-found]

    litellm.drop_params = True
except Exception:  # pragma: no cover
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
    train_size: int = 12
    val_size: int = 6
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
    optimizer: str = 'gepa'
    auto_setting: str = 'light'
    output_path: str | None = None
    seed: int = 17


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------


def _load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # accept Phoenix-export shape {"examples": [...]}
        for key in ('examples', 'queries', 'rows'):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError('Could not find a list of examples in dataset payload.')
    if isinstance(data, list):
        return data
    raise ValueError(f'Unexpected dataset shape: {type(data).__name__}')


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce a baseline record to ``{query, intent, should_refuse}``."""
    inp = rec.get('input') if isinstance(rec.get('input'), dict) else None
    md = rec.get('metadata') if isinstance(rec.get('metadata'), dict) else None
    query = (inp or {}).get('query') or rec.get('query') or (md or {}).get('query') or ''
    query = str(query).strip()
    if not query:
        return None
    intent = rec.get('intent') or (md or {}).get('intent') or (md or {}).get('intent_type') or ''
    should_refuse = bool(rec.get('should_refuse') or (md or {}).get('should_refuse'))
    return {'query': query, 'intent': str(intent), 'should_refuse': should_refuse}


# ---------------------------------------------------------------------------
# Yar /query helper (pre-extracted keywords path)
# ---------------------------------------------------------------------------


def _yar_query_with_keywords(
    config: OptimizerConfig,
    *,
    query: str,
    hl_keywords: list[str],
    ll_keywords: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    """Hit /query passing pre-extracted keywords; return (response, references)."""
    headers = {'Content-Type': 'application/json'}
    if config.yar_api_key:
        headers['X-API-Key'] = config.yar_api_key
    body = {
        'query': query,
        'mode': 'mix',
        'top_k': 10,
        'disable_cache': True,
        'hl_keywords': hl_keywords,
        'll_keywords': ll_keywords,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f'{config.server_url}/query', json=body, headers=headers)
    if resp.status_code != 200:
        return '', []
    payload = resp.json()
    return str(payload.get('response') or ''), list(payload.get('references') or [])


def _references_to_context(refs: list[dict[str, Any]]) -> str:
    """Flatten the references payload into a context-style string for the judge."""
    parts: list[str] = []
    for ref in refs[:30]:
        if not isinstance(ref, dict):
            continue
        body = ref.get('content') or ref.get('text') or ref.get('snippet') or ref.get('chunk') or ''
        if body:
            parts.append(str(body))
    return '\n\n'.join(parts)[:18000]


# ---------------------------------------------------------------------------
# Signature + module
# ---------------------------------------------------------------------------


def _build_signature():
    import dspy

    class KeywordExtraction(dspy.Signature):
        """Extract retrieval keywords from a user question for a graph + vector RAG.

        Output two parallel keyword lists, comma-separated:

        * ``high_level_keywords`` (2–4): broad topical themes — intent
          ("comparison", "mechanism", "overview"), domain ("manufacturing",
          "regulatory submission"), and information type ("risk assessment",
          "quality controls"). These drive the knowledge-graph traversal.
        * ``low_level_keywords`` (1–4): specific entities **explicit** in the
          question — proper nouns, code names, identifiers, dates,
          technical terms. Preserve names verbatim; never substitute a
          generic term for a specific one. These drive vector lookup.

        Rules:
        * Derive every keyword from the query itself; do not invent related
          concepts that the user did not name.
        * Vague or off-topic queries (greetings, meta questions) return
          empty lists.
        * Keep keywords language-agnostic when proper nouns are involved.
        """

        query: str = dspy.InputField(desc='The user question.')
        high_level_keywords: str = dspy.OutputField(
            desc='Comma-separated high-level themes (2-4 items, no JSON).',
        )
        low_level_keywords: str = dspy.OutputField(
            desc='Comma-separated low-level entities (1-4 items, no JSON).',
        )

    return KeywordExtraction


def _parse_csv(value: str) -> list[str]:
    items = [chunk.strip().strip('"').strip("'") for chunk in str(value or '').split(',')]
    return [s for s in items if s]


def _build_module():
    import dspy

    class KeywordModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(_build_signature())

        def forward(self, query: str):  # type: ignore[override]
            return self.predict(query=query)

    return KeywordModule()


# ---------------------------------------------------------------------------
# Composite metric (end-to-end synthesis quality)
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


_GROUNDEDNESS_TEMPLATE = (
    'You are evaluating whether a RAG answer is grounded in the retrieved context.\n\n'
    'CONTEXT:\n{context}\n\nQUERY: {query}\n\nANSWER: {answer}\n\n'
    'Reply GROUNDED if every factual claim in the answer is supported by the context.\n'
    'Reply UNSUPPORTED if any specific fact (named entity, number, date, mechanism, target, '
    'classification, relationship) is not supported by the context.\n'
    'Reply with one word: GROUNDED or UNSUPPORTED.'
)

_HALLUCINATION_TEMPLATE = (
    'You are detecting hallucinations in a RAG answer.\n\n'
    'CONTEXT:\n{context}\n\nQUERY: {query}\n\nANSWER: {answer}\n\n'
    'Reply HALLUCINATED if the answer asserts a specific fact that does not appear in the context.\n'
    'Reply FACTUAL if every specific fact traces back to the context.\n'
    'Reply with one word: FACTUAL or HALLUCINATED.'
)


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
    config: OptimizerConfig,
    judge_lm: Any,
    *,
    example: Any,
    prediction: Any,
    trace: Any = None,
    return_detail: bool = False,
) -> Any:
    """Run /query with the predicted keywords and judge the synthesized answer."""
    del trace
    query = str(getattr(example, 'query', '') or '')
    intent = str(getattr(example, 'intent', '') or '')
    should_refuse = bool(getattr(example, 'should_refuse', False))

    hl = _parse_csv(getattr(prediction, 'high_level_keywords', ''))
    ll = _parse_csv(getattr(prediction, 'low_level_keywords', ''))

    detail: dict[str, Any] = {
        'intent': intent,
        'should_refuse': should_refuse,
        'hl_keywords': hl,
        'll_keywords': ll,
        'refused': False,
        'grounded_label': None,
        'hallucination_label': None,
    }

    if not hl and not ll:
        detail['reason'] = 'empty_keywords'
        return (0.0, detail) if return_detail else 0.0

    try:
        answer, refs = _yar_query_with_keywords(config, query=query, hl_keywords=hl, ll_keywords=ll)
    except Exception as exc:
        detail['reason'] = f'yar_call_failed: {exc!r}'
        return (0.0, detail) if return_detail else 0.0

    answer = (answer or '').strip()
    if not answer:
        detail['reason'] = 'empty_answer'
        return (0.0, detail) if return_detail else 0.0

    refused = _looks_like_refusal(answer)
    detail['refused'] = refused
    refusal_score = 1.0 if (should_refuse == refused) else 0.0

    if should_refuse and refused:
        detail['reason'] = 'correct_refusal'
        return (1.0, detail) if return_detail else 1.0
    if not should_refuse and refused:
        detail['reason'] = 'incorrect_refusal_on_answerable'
        return (0.0, detail) if return_detail else 0.0

    context = _references_to_context(refs)

    grounded_resp = _judge_call(
        judge_lm,
        _GROUNDEDNESS_TEMPLATE.format(context=context, query=query, answer=answer[:4000]),
    )
    grounded_label = 'GROUNDED' if 'GROUND' in grounded_resp.upper() else 'UNSUPPORTED'
    grounded_score = 1.0 if grounded_label == 'GROUNDED' else 0.0
    detail['grounded_label'] = grounded_label

    hal_resp = _judge_call(
        judge_lm,
        _HALLUCINATION_TEMPLATE.format(context=context, query=query, answer=answer[:4000]),
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


def _build_examples(config: OptimizerConfig) -> list[Any]:
    import dspy

    rng = random.Random(config.seed)
    raw = _load_dataset(config.baseline_path)
    rng.shuffle(raw)

    examples: list[Any] = []
    for rec in raw:
        norm = _normalize_record(rec)
        if not norm:
            continue
        ex = dspy.Example(
            query=norm['query'],
            intent=norm['intent'],
            should_refuse=norm['should_refuse'],
        ).with_inputs('query')
        examples.append(ex)
    return examples


def run_optimization(config: OptimizerConfig) -> dict[str, Any]:
    import dspy
    from dspy.teleprompt import MIPROv2

    task_lm = dspy.LM(
        f'{config.task_provider}/{config.task_model}',
        api_base=config.task_base_url or config.base_url,
        api_key=config.task_api_key or config.api_key,
        max_tokens=600,
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

    def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float:
        return _composite_metric(config, judge_lm, example=example, prediction=prediction, trace=trace)

    def metric_with_detail(example: Any, prediction: Any, trace: Any = None) -> tuple[float, dict[str, Any]]:
        result = _composite_metric(
            config, judge_lm, example=example, prediction=prediction, trace=trace, return_detail=True
        )
        if isinstance(result, tuple):
            return result
        return float(result), {}

    module = _build_module()

    if config.optimizer == 'gepa':
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
            feedback_parts = [f'score={score:.2f}', f'reason={detail.get("reason", "?")}']
            for key in (
                'intent',
                'should_refuse',
                'refused',
                'grounded_label',
                'hallucination_label',
                'hl_keywords',
                'll_keywords',
            ):
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
            num_threads=2,
        )
        optimized = compiler.compile(
            module,
            trainset=trainset,
            valset=valset or None,
            requires_permission_to_run=False,
        )

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
    parser = argparse.ArgumentParser(description='Optimize the YAR keyword-extraction prompt with DSPy.')
    parser.add_argument('--baseline', required=True, help='Path to JSON dump (Phoenix dataset export or list).')
    parser.add_argument('--server-url', default=os.getenv('YAR_SERVER_URL', 'http://localhost:9621'))
    parser.add_argument('--yar-api-key', default=os.getenv('YAR_API_KEY'))
    parser.add_argument('--train-size', type=int, default=12)
    parser.add_argument('--val-size', type=int, default=6)
    parser.add_argument('--task-model', default='tuna')
    parser.add_argument('--task-provider', default='openai')
    parser.add_argument('--task-base-url', default=None)
    parser.add_argument('--task-api-key', default=None)
    parser.add_argument('--judge-model', default='tuna')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--judge-base-url', default=None)
    parser.add_argument('--judge-api-key', default=None)
    parser.add_argument('--reflection-model', default='tuna')
    parser.add_argument('--reflection-provider', default='openai')
    parser.add_argument('--reflection-base-url', default=None)
    parser.add_argument('--reflection-api-key', default=None)
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'http://localhost:4000/v1'))
    parser.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', 'sk-litellm-master-key'))
    parser.add_argument('--optimizer', choices=['mipro', 'gepa'], default='gepa')
    parser.add_argument('--auto', dest='auto_setting', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=17)
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
    )

    payload = run_optimization(config)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info('Wrote optimized instructions to %s', args.output)

    print('\n=== OPTIMIZED KEYWORD-EXTRACTION INSTRUCTIONS ===\n')
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
