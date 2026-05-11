"""DSPy-driven optimization of yar's summarize_entity_descriptions prompt.

When yar ingests new chunks for an existing entity, it merges multiple
descriptions into a single concise summary via the
``summarize_entity_descriptions`` prompt. Quality of the merge governs how
well downstream retrieval surfaces an entity (its single canonical
description is what feeds the synthesis context).

This module runs an offline harness:

* Sample N entities that have multi-source descriptions in postgres
  (descriptions joined with ``<SEP>``).
* Treat each ``<SEP>``-split fragment as one input description.
* Run the candidate merge prompt → get summary.
* Judge with gpt-5.4-mini on:
  - FACT_PRESERVATION: every distinct fact from the inputs survives.
  - GROUNDING: no facts beyond what the inputs say.
  - CONCISENESS: no redundancy, compact phrasing.
  - VOICE: third-person, starts with entity name.

The optimized instruction can be folded into
``yar/prompt.py:PROMPTS['summarize_entity_descriptions']``.
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

import asyncpg

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
    train_size: int = 8
    val_size: int = 4
    task_model: str = 'tuna'
    task_provider: str = 'openai'
    task_base_url: str | None = None
    task_api_key: str | None = None
    judge_model: str = 'gpt-5.4-mini'
    judge_provider: str = 'openai'
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    reflection_model: str = 'gpt-5.4-mini'
    reflection_provider: str = 'openai'
    reflection_base_url: str | None = None
    reflection_api_key: str | None = None
    base_url: str = 'http://localhost:4000/v1'
    api_key: str = 'sk-litellm-master-key'
    optimizer: str = 'gepa'
    auto_setting: str = 'light'
    output_path: str | None = None
    seed: int = 23
    pg_dsn: str = 'postgresql://yar:yar_pass@localhost:5432/yar'
    workspace: str = 'default'
    summary_length: int = 200
    language: str = 'English'
    description_type: str = 'Entity'
    min_descriptions: int = 3


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------


def _load_entities(config: OptimizerConfig) -> list[dict[str, Any]]:
    """Pull entities with at least N <SEP>-merged descriptions."""
    import asyncio

    async def _fetch() -> list[dict[str, Any]]:
        conn = await asyncpg.connect(config.pg_dsn)
        try:
            rows = await conn.fetch(
                """
                SELECT entity_name, entity_type, content
                  FROM yar_vdb_entity
                 WHERE workspace = $1
                   AND content LIKE '%<SEP>%'
                """,
                config.workspace,
            )
        finally:
            await conn.close()
        out = []
        for row in rows:
            content = str(row['content'] or '')
            # First newline separates entity name header from descriptions
            parts = content.split('\n', 1)
            descriptions_blob = parts[1] if len(parts) > 1 else parts[0]
            fragments = [f.strip() for f in descriptions_blob.split('<SEP>') if f.strip()]
            if len(fragments) >= config.min_descriptions:
                out.append(
                    {
                        'entity_name': str(row['entity_name'] or ''),
                        'entity_type': str(row['entity_type'] or ''),
                        'descriptions': fragments,
                    }
                )
        return out

    entities = asyncio.run(_fetch())
    rng = random.Random(config.seed)
    rng.shuffle(entities)
    return entities


# ---------------------------------------------------------------------------
# Signature + module
# ---------------------------------------------------------------------------


def _build_signature():
    import dspy

    class SummarizeDescriptions(dspy.Signature):
        """Synthesize multiple descriptions of one entity into a single summary.

        Output a plain-text summary that:

        * Begins with the entity name in third person.
        * Preserves every distinct fact from the input descriptions; if two
          descriptions assert the same fact, include it once.
        * Asserts no fact that does not appear in at least one input
          description (no world knowledge).
        * Is compact and free of filler. No headings, bullets, or markdown.
        * Stays under the requested ``summary_length`` token budget.
        """

        entity_name: str = dspy.InputField(desc='The entity being summarized.')
        entity_type: str = dspy.InputField(desc='Type label of the entity.')
        descriptions: str = dspy.InputField(desc='One description per line; lines come from independent extractions.')
        summary_length: str = dspy.InputField(desc='Approximate token budget for the summary.')
        summary: str = dspy.OutputField(desc='Plain-text consolidated summary, third-person, starts with entity name.')

    return SummarizeDescriptions


def _build_module():
    import dspy

    class SummaryModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(_build_signature())

        def forward(self, entity_name: str, entity_type: str, descriptions: str, summary_length: str):  # type: ignore[override]
            return self.predict(
                entity_name=entity_name,
                entity_type=entity_type,
                descriptions=descriptions,
                summary_length=summary_length,
            )

    return SummaryModule()


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------


_QUALITY_TEMPLATE = (
    'You are evaluating an entity-description merge for a knowledge graph.\n\n'
    'ENTITY: {name} ({type})\n\n'
    'INPUT DESCRIPTIONS (one per line):\n{descriptions}\n\n'
    'MERGED SUMMARY:\n{summary}\n\n'
    'Rate on these axes (each 0-2: 0=fails, 1=partial, 2=fully):\n'
    '* FACT_PRESERVATION: every distinct fact present in the inputs is reflected '
    'in the summary (paraphrasing is fine). 2 if no facts are dropped.\n'
    '* GROUNDING: every claim in the summary traces to at least one input. '
    '0 if any claim is invented (world knowledge, fabrication).\n'
    '* CONCISENESS: no redundant repetition; no filler boilerplate. 2 if compact.\n'
    '* VOICE: third-person, starts with the entity name, plain text (no markdown). '
    '2 if all three hold.\n\n'
    'Reply with only four lines, each "<AXIS>:<integer 0-2>". Example:\n'
    'FACT_PRESERVATION:2\nGROUNDING:2\nCONCISENESS:1\nVOICE:2'
)


def _judge_call(judge_lm: Any, prompt: str) -> str:
    try:
        resp = judge_lm(prompt)
    except Exception as exc:
        logger.debug('judge LM call failed: %s', exc)
        return ''
    if isinstance(resp, list) and resp:
        resp = resp[0]
    return str(resp).strip()


def _parse_axis_scores(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (text or '').splitlines():
        m = re.match(r'\s*(FACT_PRESERVATION|GROUNDING|CONCISENESS|VOICE)\s*:\s*(\d)', line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def _composite_metric(
    judge_lm: Any,
    *,
    example: Any,
    prediction: Any,
    return_detail: bool = False,
) -> Any:
    name = str(getattr(example, 'entity_name', '') or '')
    etype = str(getattr(example, 'entity_type', '') or '')
    descriptions = str(getattr(example, 'descriptions', '') or '')
    summary = str(getattr(prediction, 'summary', '') or '').strip()

    detail: dict[str, Any] = {
        'summary_length_chars': len(summary),
        'starts_with_name': summary.casefold().startswith(name.casefold()) if name else False,
        'axis_scores': {},
    }

    if not summary:
        detail['reason'] = 'empty_summary'
        return (0.0, detail) if return_detail else 0.0

    raw = _judge_call(
        judge_lm,
        _QUALITY_TEMPLATE.format(
            name=name,
            type=etype,
            descriptions=descriptions[:6000],
            summary=summary[:2000],
        ),
    )
    detail['judge_raw'] = raw[:200]
    axes = _parse_axis_scores(raw)
    detail['axis_scores'] = axes

    if not axes:
        detail['reason'] = 'judge_unparseable'
        return (0.2, detail) if return_detail else 0.2

    quality = sum(axes.values()) / (2 * max(len(axes), 1))
    score = round(quality, 3)
    detail['reason'] = 'all_pass' if score >= 0.95 else f'axes={axes}'
    return (score, detail) if return_detail else score


# ---------------------------------------------------------------------------
# Optimizer driver
# ---------------------------------------------------------------------------


def _build_examples(config: OptimizerConfig) -> list[Any]:
    import dspy

    entities = _load_entities(config)
    needed = config.train_size + config.val_size
    if len(entities) < needed:
        raise RuntimeError(
            f'Not enough entities with >= {config.min_descriptions} descriptions in postgres '
            f'({len(entities)}); need at least {needed}.'
        )

    examples: list[Any] = []
    for ent in entities[: needed * 2]:
        ex = dspy.Example(
            entity_name=ent['entity_name'],
            entity_type=ent['entity_type'],
            descriptions='\n'.join(ent['descriptions']),
            summary_length=str(config.summary_length),
        ).with_inputs('entity_name', 'entity_type', 'descriptions', 'summary_length')
        examples.append(ex)
        if len(examples) >= needed:
            break
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
        max_tokens=80,
        temperature=0.0,
    )
    dspy.settings.configure(lm=task_lm)

    examples = _build_examples(config)
    trainset = examples[: config.train_size]
    valset = examples[config.train_size : config.train_size + config.val_size]
    logger.info('Train: %d examples, Val: %d examples', len(trainset), len(valset))

    def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float:
        del trace
        return _composite_metric(judge_lm, example=example, prediction=prediction)

    def metric_with_detail(example: Any, prediction: Any, trace: Any = None) -> tuple[float, dict[str, Any]]:
        del trace
        result = _composite_metric(judge_lm, example=example, prediction=prediction, return_detail=True)
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
            for key in ('summary_length_chars', 'starts_with_name', 'axis_scores'):
                value = detail.get(key)
                if value is not None and value != []:
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
        compiler = MIPROv2(metric=metric_fn, auto=config.auto_setting, num_threads=2)
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

    return {
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Offline DSPy optimization of yar summarize_entity_descriptions prompt.'
    )
    parser.add_argument('--train-size', type=int, default=8)
    parser.add_argument('--val-size', type=int, default=4)
    parser.add_argument('--task-model', default='tuna')
    parser.add_argument('--task-provider', default='openai')
    parser.add_argument('--task-base-url', default=None)
    parser.add_argument('--task-api-key', default=None)
    parser.add_argument('--judge-model', default='gpt-5.4-mini')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--judge-base-url', default=None)
    parser.add_argument('--judge-api-key', default=None)
    parser.add_argument('--reflection-model', default='gpt-5.4-mini')
    parser.add_argument('--reflection-provider', default='openai')
    parser.add_argument('--reflection-base-url', default=None)
    parser.add_argument('--reflection-api-key', default=None)
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'http://localhost:4000/v1'))
    parser.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY', 'sk-litellm-master-key'))
    parser.add_argument('--optimizer', choices=['mipro', 'gepa'], default='gepa')
    parser.add_argument('--auto', dest='auto_setting', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--pg-dsn', default=os.getenv('PG_DSN', 'postgresql://yar:yar_pass@localhost:5432/yar'))
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', 'default'))
    parser.add_argument('--summary-length', type=int, default=200)
    parser.add_argument('--min-descriptions', type=int, default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    config = OptimizerConfig(
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
        pg_dsn=args.pg_dsn,
        workspace=args.workspace,
        summary_length=args.summary_length,
        min_descriptions=args.min_descriptions,
    )

    payload = run_optimization(config)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info('Wrote optimized instructions to %s', args.output)

    print('\n=== OPTIMIZED SUMMARIZE-DESCRIPTIONS INSTRUCTIONS ===\n')
    print(payload['instructions'])
    if payload['demos']:
        print(f'\n=== {len(payload["demos"])} few-shot demos selected ===')
        for i, demo in enumerate(payload['demos'], 1):
            preview = json.dumps(demo, ensure_ascii=False, default=str)
            preview = re.sub(r'\s+', ' ', preview)[:300]
            print(f'  [{i}] {preview}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
