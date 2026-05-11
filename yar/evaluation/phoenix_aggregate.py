"""Intent-aware aggregation for YAR Phoenix experiments.

Phoenix surfaces per-evaluator scores (relevance, groundedness, hallucination,
refusal). Aggregating those naively into a "perfect cell" count silently
penalizes correct refusals on out-of-scope / mechanism-bait queries: an ideal
refusal answer ("the context does not describe X") is genuinely UNRELATED to
the question and contains no factual claims to ground, so the relevance and
groundedness judges fire negatively even though the system did exactly the
right thing. That noise drowns the signal optimizers (DSPy / GEPA) need.

This module re-scores an experiment with intent-aware example-level rules:

* answer-expected (factual_lookup, enumeration, comparison): the example is
  perfect iff all four verdicts are positive (RELEVANT, GROUNDED, FACTUAL,
  ANSWERED).
* refusal-expected (out_of_scope, mechanism_bait): the example is perfect
  iff ``refusal=REFUSAL``. Other verdicts on a refusal answer are noise.

Usage::

    .venv/bin/python -m yar.evaluation.phoenix_aggregate \\
        --experiment-id RXhwZXJpbWVudDoyNQ== \\
        --dataset yar-baseline-curated-2026-05-08

Programmatic::

    from yar.evaluation.phoenix_aggregate import aggregate_experiment
    summary = aggregate_experiment(experiment_id='RXhwZX...', dataset_name='...')
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

ANSWER_EXPECTED_INTENTS = {'factual_lookup', 'enumeration', 'comparison'}
REFUSAL_EXPECTED_INTENTS = {'out_of_scope', 'mechanism_bait'}

ANSWER_EXPECTED_REQUIREMENTS = {
    'relevance': 'RELEVANT',
    'groundedness': 'GROUNDED',
    'hallucination': 'FACTUAL',
    'refusal': 'ANSWERED',
}


@dataclass
class ExampleResult:
    example_id: str
    intent: str
    verdicts: dict[str, str]
    perfect: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class IntentSummary:
    perfect: int = 0
    total: int = 0
    failures: list[ExampleResult] = field(default_factory=list)


def example_perfect(intent: str, verdicts: dict[str, str]) -> tuple[bool, list[str]]:
    """Return (is_perfect, failure_reasons) for one example."""
    if intent in REFUSAL_EXPECTED_INTENTS:
        actual = verdicts.get('refusal') or '<missing>'
        if actual == 'REFUSAL':
            return True, []
        return False, [f'refusal={actual} (expected REFUSAL)']
    if intent in ANSWER_EXPECTED_INTENTS:
        reasons: list[str] = []
        for evaluator, expected in ANSWER_EXPECTED_REQUIREMENTS.items():
            actual = verdicts.get(evaluator) or '<missing>'
            if actual != expected:
                reasons.append(f'{evaluator}={actual} (expected {expected})')
        return (not reasons), reasons
    return False, [f'unknown intent: {intent!r}']


# ---------------------------------------------------------------------------
# Phoenix lookups
# ---------------------------------------------------------------------------


def _require_phoenix_client() -> Any:
    try:
        from phoenix.client import Client
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError('phoenix.client is required. Install via `pip install -e .[observability]`.') from exc
    return Client


def _intent_for_examples(client: Any, dataset_name: str) -> dict[str, str]:
    ds = client.datasets.get_dataset(dataset=dataset_name)
    intent_for: dict[str, str] = {}
    for ex in ds.examples:
        if isinstance(ex, dict):
            md = ex.get('metadata') or {}
            ex_id = ex.get('id')
        else:
            md = getattr(ex, 'metadata', None) or {}
            ex_id = getattr(ex, 'id', None)
        intent = ''
        if isinstance(md, dict):
            intent = str(md.get('intent') or md.get('intent_type') or '')
        if ex_id is not None:
            intent_for[str(ex_id)] = intent
    return intent_for


def _verdicts_for_experiment(client: Any, experiment_id: str) -> dict[str, dict[str, str]]:
    exp = client.experiments.get_experiment(experiment_id=experiment_id)
    run_to_example: dict[str, str] = {}
    for tr in exp.get('task_runs', []):
        run_to_example[str(tr['id'])] = str(tr['dataset_example_id'])

    per_example: dict[str, dict[str, str]] = defaultdict(dict)
    for er in exp.get('evaluation_runs', []):
        run_id = getattr(er, 'experiment_run_id', None)
        if run_id is None and isinstance(er, dict):
            run_id = er.get('experiment_run_id')
        ex_id = run_to_example.get(str(run_id))
        if not ex_id:
            continue
        result = getattr(er, 'result', None) or (er.get('result') if isinstance(er, dict) else {})
        evaluator_name = getattr(er, 'name', None) or (er.get('name') if isinstance(er, dict) else '')
        verdict = ''
        if isinstance(result, dict):
            verdict = str(result.get('label') or '')
        per_example[ex_id][str(evaluator_name)] = verdict
    return dict(per_example)


# ---------------------------------------------------------------------------
# Public aggregation entry point
# ---------------------------------------------------------------------------


def aggregate_experiment(
    *,
    experiment_id: str,
    dataset_name: str,
    phoenix_base_url: str | None = None,
    phoenix_api_key: str | None = None,
) -> dict[str, Any]:
    """Pull verdicts + dataset metadata, return intent-aware summary."""
    Client = _require_phoenix_client()
    kwargs: dict[str, Any] = {}
    if phoenix_base_url:
        kwargs['base_url'] = phoenix_base_url
    if phoenix_api_key:
        kwargs['api_key'] = phoenix_api_key
    client = Client(**kwargs)

    intent_for = _intent_for_examples(client, dataset_name)
    per_example = _verdicts_for_experiment(client, experiment_id)

    by_intent: dict[str, IntentSummary] = defaultdict(IntentSummary)
    examples: list[ExampleResult] = []
    for ex_id, verdicts in per_example.items():
        intent = intent_for.get(ex_id, '')
        is_perfect, reasons = example_perfect(intent, verdicts)
        result = ExampleResult(
            example_id=ex_id,
            intent=intent,
            verdicts=verdicts,
            perfect=is_perfect,
            failure_reasons=reasons,
        )
        examples.append(result)
        bucket = by_intent[intent]
        bucket.total += 1
        if is_perfect:
            bucket.perfect += 1
        else:
            bucket.failures.append(result)

    total_perfect = sum(bucket.perfect for bucket in by_intent.values())
    total_examples = sum(bucket.total for bucket in by_intent.values())

    return {
        'experiment_id': experiment_id,
        'dataset': dataset_name,
        'total_examples': total_examples,
        'total_perfect': total_perfect,
        'rate': (total_perfect / total_examples) if total_examples else 0.0,
        'by_intent': {
            intent: {
                'perfect': bucket.perfect,
                'total': bucket.total,
                'failures': [
                    {
                        'example_id': f.example_id,
                        'verdicts': f.verdicts,
                        'reasons': f.failure_reasons,
                    }
                    for f in bucket.failures
                ],
            }
            for intent, bucket in by_intent.items()
        },
        'examples': [
            {
                'example_id': r.example_id,
                'intent': r.intent,
                'verdicts': r.verdicts,
                'perfect': r.perfect,
                'failure_reasons': r.failure_reasons,
            }
            for r in examples
        ],
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def render_summary(summary: dict[str, Any]) -> str:
    intents_order = ['factual_lookup', 'enumeration', 'comparison', 'out_of_scope', 'mechanism_bait']
    lines: list[str] = []
    lines.append(f'Experiment {summary["experiment_id"]} on {summary["dataset"]}')
    lines.append('Intent-aware example-level perfection:')
    lines.append(
        '  answer-expected: relevance=RELEVANT & groundedness=GROUNDED & hallucination=FACTUAL & refusal=ANSWERED'
    )
    lines.append('  refusal-expected: refusal=REFUSAL (other verdicts ignored)')
    lines.append('')
    lines.append(f'{"intent":<18}{"perfect":>10}{"total":>10}')
    lines.append('-' * 38)
    for intent in intents_order:
        bucket = summary['by_intent'].get(intent)
        if bucket is None:
            continue
        lines.append(f'{intent:<18}{bucket["perfect"]:>10}{bucket["total"]:>10}')
    lines.append('-' * 38)
    lines.append(f'{"TOTAL":<18}{summary["total_perfect"]:>10}{summary["total_examples"]:>10}')
    lines.append(f'rate: {summary["rate"]:.1%}')
    lines.append('')
    lines.append('Failures by intent:')
    for intent in intents_order:
        bucket = summary['by_intent'].get(intent)
        if bucket is None or not bucket['failures']:
            continue
        lines.append(f'  {intent}: {len(bucket["failures"])}')
        for fail in bucket['failures']:
            v = fail['verdicts']
            short_id = fail['example_id'][:8]
            lines.append(f'    {short_id} verdicts={v} reasons={fail["reasons"]}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Intent-aware aggregation of a YAR Phoenix experiment.')
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--phoenix-base-url', default=os.getenv('PHOENIX_BASE_URL') or None)
    parser.add_argument('--phoenix-api-key', default=os.getenv('PHOENIX_API_KEY') or None)
    parser.add_argument('--json-out', default=None, help='Optional path for full summary JSON.')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    summary = aggregate_experiment(
        experiment_id=args.experiment_id,
        dataset_name=args.dataset,
        phoenix_base_url=args.phoenix_base_url,
        phoenix_api_key=args.phoenix_api_key,
    )

    print(render_summary(summary))

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
        print(f'\nWrote full summary to {args.json_out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
