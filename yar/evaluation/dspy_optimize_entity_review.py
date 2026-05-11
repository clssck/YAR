"""DSPy-driven optimization of yar's entity-resolution review prompt.

When yar ingests new entities and decides whether they match existing ones
(`entity_review_system_prompt` + `entity_review_user_prompt`), errors here
either fragment the graph (false negatives) or merge distinct concepts
(false positives). Both hurt downstream retrieval.

This offline harness:

* Builds gold-labeled entity pairs from the running database:
  - **same** pairs from ``yar_entity_aliases`` (alias -> canonical).
  - **different** pairs by sampling random distinct entities that are
    *not* in the alias table.
* Asks the candidate prompt to judge each pair.
* Scores each pair as 1.0 if the prediction matches the gold label,
  0.0 otherwise. Aggregate is accuracy.

The optimized instruction can be folded into
``yar/prompt.py:PROMPTS['entity_review_system_prompt']``.
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
    train_size: int = 12
    val_size: int = 6
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
    seed: int = 29
    pg_dsn: str = 'postgresql://yar:yar_pass@localhost:5432/yar'
    workspace: str = 'default'


# ---------------------------------------------------------------------------
# Dataset prep: build gold-labeled pairs from postgres
# ---------------------------------------------------------------------------


def _load_pairs(config: OptimizerConfig) -> list[dict[str, Any]]:
    """Build {alias, canonical, gold_label, types} pair list."""
    import asyncio

    async def _fetch() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        conn = await asyncpg.connect(config.pg_dsn)
        try:
            same_rows = await conn.fetch(
                """
                SELECT alias, canonical_entity, entity_type
                  FROM yar_entity_aliases
                 WHERE workspace = $1
                   AND alias IS NOT NULL
                   AND canonical_entity IS NOT NULL
                """,
                config.workspace,
            )
            entity_rows = await conn.fetch(
                """
                SELECT entity_name, entity_type
                  FROM yar_vdb_entity
                 WHERE workspace = $1
                   AND entity_name IS NOT NULL
                """,
                config.workspace,
            )
        finally:
            await conn.close()
        return list(same_rows), list(entity_rows)

    same_rows, entity_rows = asyncio.run(_fetch())

    rng = random.Random(config.seed)

    same_pairs: list[dict[str, Any]] = []
    alias_set: set[tuple[str, str]] = set()
    for r in same_rows:
        a = str(r['alias'] or '').strip()
        c = str(r['canonical_entity'] or '').strip()
        t = str(r['entity_type'] or '').strip() or 'unknown'
        if not a or not c or a.casefold() == c.casefold():
            continue
        same_pairs.append(
            {
                'a': a,
                'b': c,
                'a_type': t,
                'b_type': t,
                'gold': 'same',
            }
        )
        alias_set.add((a.casefold(), c.casefold()))
        alias_set.add((c.casefold(), a.casefold()))

    # Build different pairs by random sampling of distinct entities not in aliases.
    entities = [
        (str(r['entity_name'] or '').strip(), str(r['entity_type'] or '').strip() or 'unknown')
        for r in entity_rows
        if r['entity_name']
    ]
    rng.shuffle(entities)
    different_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, (a, ta) in enumerate(entities):
        if len(different_pairs) >= len(same_pairs) * 2:
            break
        for j in range(i + 1, min(i + 8, len(entities))):
            b, tb = entities[j]
            key = (a.casefold(), b.casefold())
            if key in alias_set or key in seen_pairs:
                continue
            # Skip suspiciously similar names (substring) -- avoid mislabeling
            # an actual alias as a "different" pair.
            if a.casefold() in b.casefold() or b.casefold() in a.casefold():
                continue
            different_pairs.append(
                {
                    'a': a,
                    'b': b,
                    'a_type': ta,
                    'b_type': tb,
                    'gold': 'different',
                }
            )
            seen_pairs.add(key)
            seen_pairs.add((b.casefold(), a.casefold()))
            if len(different_pairs) >= len(same_pairs) * 2:
                break

    pairs = same_pairs + different_pairs
    rng.shuffle(pairs)
    logger.info(
        'Loaded %d pairs (same=%d, different=%d).',
        len(pairs),
        len(same_pairs),
        len(different_pairs),
    )
    return pairs


# ---------------------------------------------------------------------------
# Signature + module
# ---------------------------------------------------------------------------


def _build_signature():
    import dspy

    class EntityResolution(dspy.Signature):
        """Decide whether two entity names refer to the same real-world entity.

        Output exactly one word: ``SAME`` or ``DIFFERENT``.

        Mark SAME for: abbreviations, alternate names, translations, typos,
        legal-suffix variants ("Apple" vs "Apple Inc."), case differences,
        unit-symbol case ("mL" vs "ml"), initial-prefix variants
        ("J.Smith" vs "Smith") **only when types are compatible**.

        Mark DIFFERENT for: parent/child concepts, related-but-distinct
        instances ("Method 1" vs "Method 2"), type-mismatched homographs
        ("apple" the fruit vs "Apple" the company), or anything where you
        cannot confidently establish referential identity.
        """

        a_name: str = dspy.InputField(desc='First entity name.')
        a_type: str = dspy.InputField(desc='First entity type label.')
        b_name: str = dspy.InputField(desc='Second entity name.')
        b_type: str = dspy.InputField(desc='Second entity type label.')
        verdict: str = dspy.OutputField(desc='Exactly one word: SAME or DIFFERENT.')

    return EntityResolution


def _build_module():
    import dspy

    class ReviewModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(_build_signature())

        def forward(self, a_name: str, a_type: str, b_name: str, b_type: str):  # type: ignore[override]
            return self.predict(a_name=a_name, a_type=a_type, b_name=b_name, b_type=b_type)

    return ReviewModule()


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def _normalize_verdict(text: str) -> str:
    upper = (text or '').strip().upper()
    if 'SAME' in upper.split() or upper.startswith('SAME'):
        return 'same'
    if 'DIFFERENT' in upper.split() or upper.startswith('DIFFERENT'):
        return 'different'
    # secondary heuristics — accept yes/no, true/false
    if upper in {'YES', 'TRUE', 'MATCH'}:
        return 'same'
    if upper in {'NO', 'FALSE', 'DISTINCT'}:
        return 'different'
    return ''


def _composite_metric(
    *,
    example: Any,
    prediction: Any,
    return_detail: bool = False,
) -> Any:
    gold = str(getattr(example, 'gold', '') or '').strip().lower()
    raw = str(getattr(prediction, 'verdict', '') or '').strip()
    pred = _normalize_verdict(raw)
    correct = int(pred == gold)
    detail = {
        'gold': gold,
        'predicted': pred,
        'raw_verdict': raw[:80],
        'correct': bool(correct),
    }
    if not pred:
        detail['reason'] = 'unparseable_verdict'
        return (0.0, detail) if return_detail else 0.0
    detail['reason'] = 'correct' if correct else f'wrong (predicted {pred}, gold {gold})'
    return (float(correct), detail) if return_detail else float(correct)


# ---------------------------------------------------------------------------
# Optimizer driver
# ---------------------------------------------------------------------------


def _build_examples(config: OptimizerConfig) -> list[Any]:
    import dspy

    pairs = _load_pairs(config)
    needed = config.train_size + config.val_size
    if len(pairs) < needed:
        raise RuntimeError(f'Not enough labeled pairs ({len(pairs)}) for train+val ({needed}).')

    examples: list[Any] = []
    for p in pairs[: needed * 2]:
        ex = dspy.Example(
            a_name=p['a'],
            a_type=p['a_type'],
            b_name=p['b'],
            b_type=p['b_type'],
            gold=p['gold'],
        ).with_inputs('a_name', 'a_type', 'b_name', 'b_type')
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
        max_tokens=20,
    )
    dspy.settings.configure(lm=task_lm)

    examples = _build_examples(config)
    trainset = examples[: config.train_size]
    valset = examples[config.train_size : config.train_size + config.val_size]
    logger.info('Train: %d examples, Val: %d examples', len(trainset), len(valset))

    def metric_fn(example: Any, prediction: Any, trace: Any = None) -> float:
        del trace
        return _composite_metric(example=example, prediction=prediction)

    def metric_with_detail(example: Any, prediction: Any, trace: Any = None) -> tuple[float, dict[str, Any]]:
        del trace
        result = _composite_metric(example=example, prediction=prediction, return_detail=True)
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
            for key in ('gold', 'predicted', 'raw_verdict'):
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
    parser = argparse.ArgumentParser(description='Offline DSPy optimization of yar entity-resolution prompt.')
    parser.add_argument('--train-size', type=int, default=12)
    parser.add_argument('--val-size', type=int, default=6)
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
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--pg-dsn', default=os.getenv('PG_DSN', 'postgresql://yar:yar_pass@localhost:5432/yar'))
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', 'default'))
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
    )

    payload = run_optimization(config)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info('Wrote optimized instructions to %s', args.output)

    print('\n=== OPTIMIZED ENTITY-RESOLUTION INSTRUCTIONS ===\n')
    print(payload['instructions'])
    if payload['demos']:
        print(f'\n=== {len(payload["demos"])} few-shot demos selected ===')
        for i, demo in enumerate(payload['demos'], 1):
            preview = json.dumps(demo, ensure_ascii=False, default=str)
            preview = re.sub(r'\s+', ' ', preview)[:200]
            print(f'  [{i}] {preview}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
