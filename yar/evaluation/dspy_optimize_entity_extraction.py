"""DSPy-driven optimization of yar's entity-extraction prompt.

The entity-extraction step runs at INGEST time and carves a chunk into
``(entity, entity_type, description)`` rows plus ``(source, target,
predicate, description)`` rows that yar's KG ingestion parses. Re-running
GEPA against a live ingest pipeline costs hours per rollout, so we run an
offline harness instead:

* Sample N real chunks from postgres ``yar_doc_chunks``.
* Run the candidate extraction prompt on each chunk via DSPy (no yar
  pipeline in the loop).
* Score with a gpt-5.4-mini rubric: salience (no generic terms), grounding
  (every claim traces to chunk text), structural validity (every relation
  endpoint is also an entity), and coverage (no obvious omissions).

The optimizer's winning instruction can then be folded into
``yar/prompt.py:PROMPTS['entity_extraction_system_prompt']`` ahead of the
next ingest.

Run::

    OPENAI_BASE_URL=https://api.openai.com/v1 \\
    OPENAI_API_KEY="$OPENAI_API_KEY" \\
      .venv/bin/python -m yar.evaluation.dspy_optimize_entity_extraction \\
        --train-size 8 --val-size 4 \\
        --task-model tuna \\
          --task-base-url http://localhost:4000/v1 \\
          --task-api-key sk-litellm-master-key \\
        --judge-model gpt-5.4-mini \\
        --reflection-model gpt-5.4-mini \\
        --optimizer gepa --auto light \\
        --output /tmp/dspy_entity_extraction.json
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
except Exception:  # pragma: no cover - litellm always present alongside dspy
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_TYPES = (
    'Person, Organization, Location, Event, Concept, Method, Technology, Product, Document, Data, Artifact'
)


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
    seed: int = 19
    pg_dsn: str = 'postgresql://yar:yar_pass@localhost:5432/yar'
    workspace: str = 'default'
    chunk_min_chars: int = 800
    chunk_max_chars: int = 4000
    entity_types: str = DEFAULT_ENTITY_TYPES


# ---------------------------------------------------------------------------
# Dataset prep: sample real chunks from postgres
# ---------------------------------------------------------------------------


def _load_chunks(config: OptimizerConfig) -> list[dict[str, Any]]:
    """Pull random chunks within size bounds from ``yar_doc_chunks``."""
    import asyncio

    async def _fetch() -> list[dict[str, Any]]:
        # asyncpg uses postgres:// URI form; psycopg-style postgresql:// is fine.
        conn = await asyncpg.connect(config.pg_dsn)
        try:
            rows = await conn.fetch(
                """
                SELECT id, file_path, content
                  FROM yar_doc_chunks
                 WHERE workspace = $1
                   AND length(content) BETWEEN $2 AND $3
                 ORDER BY id
                """,
                config.workspace,
                config.chunk_min_chars,
                config.chunk_max_chars,
            )
        finally:
            await conn.close()
        return [
            {
                'id': str(row['id']),
                'file_path': str(row['file_path'] or ''),
                'content': str(row['content'] or ''),
            }
            for row in rows
        ]

    chunks = asyncio.run(_fetch())
    rng = random.Random(config.seed)
    rng.shuffle(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Signature + module
# ---------------------------------------------------------------------------


def _build_signature():
    import dspy

    class EntityExtraction(dspy.Signature):
        """Extract knowledge-graph entities and relationships from a document chunk.

        Output two structured lists:

        * ``entities``: one line per entity, formatted as
          ``ENTITY||name||type||description``. Type is one of the listed
          entity_types. Description is one to three sentences, grounded
          verbatim in the chunk text.
        * ``relationships``: one line per relation, formatted as
          ``RELATION||source||target||predicate||description``. ``source``
          and ``target`` MUST exactly match one of the emitted entity
          ``name`` values.

        Rules:
        * Extract only specifically-named entities. No generic categories
          (e.g. "annual report", "research initiative", "AI", "ML",
          "data analytics"). No bare dates, no percentages, no adjectives.
        * Every entity must appear as the source or target of at least one
          relation; otherwise omit it.
        * Person names are canonical (no honorifics like Dr. / Prof.).
        * Relationships are explicit, action-oriented (developed, approved,
          targets, leads). Co-occurrence is not a relationship.
        * Never include the column separator ``||`` inside a name, type,
          predicate, or description.
        """

        chunk: str = dspy.InputField(desc='The document chunk to extract from.')
        entity_types: str = dspy.InputField(desc='Comma-separated list of allowed entity types.')
        entities: str = dspy.OutputField(desc='One ENTITY||... line per entity, separated by newlines.')
        relationships: str = dspy.OutputField(desc='One RELATION||... line per relationship, separated by newlines.')

    return EntityExtraction


def _build_module():
    import dspy

    class ExtractionModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.Predict(_build_signature())

        def forward(self, chunk: str, entity_types: str):  # type: ignore[override]
            return self.predict(chunk=chunk, entity_types=entity_types)

    return ExtractionModule()


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------


_ENTITY_LINE = re.compile(r'^\s*ENTITY\s*\|\|\s*(.+?)\s*\|\|\s*(.+?)\s*\|\|\s*(.+?)\s*$')
_RELATION_LINE = re.compile(r'^\s*RELATION\s*\|\|\s*(.+?)\s*\|\|\s*(.+?)\s*\|\|\s*(.+?)\s*\|\|\s*(.+?)\s*$')


def _parse_entities(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line in (text or '').splitlines():
        m = _ENTITY_LINE.match(line)
        if not m:
            continue
        name, etype, desc = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
        if not name or not etype or not desc:
            continue
        out.append({'name': name, 'type': etype, 'description': desc})
    return out


def _parse_relations(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line in (text or '').splitlines():
        m = _RELATION_LINE.match(line)
        if not m:
            continue
        src, tgt, pred, desc = (g.strip() for g in m.groups())
        if not src or not tgt or not pred:
            continue
        out.append({'source': src, 'target': tgt, 'predicate': pred, 'description': desc})
    return out


def _structural_score(entities: list[dict[str, str]], relations: list[dict[str, str]]) -> tuple[float, list[str]]:
    """Cheap structural validation - every relation endpoint must match an entity."""
    issues: list[str] = []
    if not entities:
        issues.append('no entities extracted')
        return 0.0, issues
    if not relations:
        issues.append('no relations extracted')
    names = {e['name'].casefold() for e in entities}
    orphan_entities = sum(
        1
        for e in entities
        if not any(
            r['source'].casefold() == e['name'].casefold() or r['target'].casefold() == e['name'].casefold()
            for r in relations
        )
    )
    bad_endpoints = sum(
        1 for r in relations if r['source'].casefold() not in names or r['target'].casefold() not in names
    )
    if orphan_entities:
        issues.append(f'{orphan_entities} orphan entities (no incident relation)')
    if bad_endpoints:
        issues.append(f'{bad_endpoints} relations with endpoints not in entity list')
    if not issues:
        return 1.0, []
    penalties = (orphan_entities + bad_endpoints) / max(len(entities) + len(relations), 1)
    return max(0.0, 1.0 - penalties), issues


# ---------------------------------------------------------------------------
# Composite metric
# ---------------------------------------------------------------------------


_QUALITY_TEMPLATE = (
    'You are evaluating the quality of knowledge-graph extraction from a document chunk.\n\n'
    'CHUNK:\n{chunk}\n\n'
    'EXTRACTED ENTITIES (one per line, "name | type | description"):\n{entities}\n\n'
    'EXTRACTED RELATIONS (one per line, "source -> target [predicate]: description"):\n{relations}\n\n'
    'Rate the extraction on these axes (each 0-2: 0=fails, 1=partial, 2=fully):\n'
    '* SALIENCE: every entity is a specifically-named, meaningful thing (not a generic category, '
    'bare date, percentage, or adjective).\n'
    '* GROUNDING: every entity name and description traces to phrases actually in the chunk; '
    'every relation reflects an explicit statement in the chunk (not co-occurrence).\n'
    '* COVERAGE: the major entities and relationships in the chunk are captured (not exhaustive '
    'minutiae, but the core facts are present).\n'
    '* TYPING: entity_type is one of the allowed types and is a reasonable choice for the entity.\n\n'
    'Reply with only four lines, each "<AXIS>:<integer 0-2>". '
    'Example:\nSALIENCE:2\nGROUNDING:2\nCOVERAGE:1\nTYPING:2'
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
        m = re.match(r'\s*(SALIENCE|GROUNDING|COVERAGE|TYPING)\s*:\s*(\d)', line)
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
    chunk = str(getattr(example, 'chunk', '') or '')
    raw_entities = str(getattr(prediction, 'entities', '') or '')
    raw_relations = str(getattr(prediction, 'relationships', '') or '')

    entities = _parse_entities(raw_entities)
    relations = _parse_relations(raw_relations)

    detail: dict[str, Any] = {
        'n_entities': len(entities),
        'n_relations': len(relations),
        'structural_issues': [],
        'axis_scores': {},
    }

    structural, issues = _structural_score(entities, relations)
    detail['structural_issues'] = issues

    if not entities:
        detail['reason'] = 'no_entities_parsed'
        return (0.0, detail) if return_detail else 0.0

    judge_prompt = _QUALITY_TEMPLATE.format(
        chunk=chunk[:6000],
        entities='\n'.join(f'{e["name"]} | {e["type"]} | {e["description"]}' for e in entities[:40]),
        relations='\n'.join(
            f'{r["source"]} -> {r["target"]} [{r["predicate"]}]: {r["description"]}' for r in relations[:40]
        )
        or '(none)',
    )
    raw = _judge_call(judge_lm, judge_prompt)
    axes = _parse_axis_scores(raw)
    detail['axis_scores'] = axes
    detail['judge_raw'] = raw[:200]

    if not axes:
        detail['reason'] = 'judge_unparseable'
        return (0.3 * structural, detail) if return_detail else 0.3 * structural

    quality = sum(axes.values()) / (2 * max(len(axes), 1))
    score = round(0.7 * quality + 0.3 * structural, 3)
    detail['reason'] = 'all_pass' if score >= 0.95 else f'quality={quality:.2f}, structural={structural:.2f}'
    return (score, detail) if return_detail else score


# ---------------------------------------------------------------------------
# Optimizer driver
# ---------------------------------------------------------------------------


def _build_examples(config: OptimizerConfig) -> list[Any]:
    import dspy

    chunks = _load_chunks(config)
    needed = config.train_size + config.val_size
    if len(chunks) < needed:
        raise RuntimeError(f'Not enough chunks in postgres ({len(chunks)}) for train+val ({needed}).')
    examples: list[Any] = []
    for chunk in chunks[: needed * 2]:  # take a buffer in case some are useless
        ex = dspy.Example(
            chunk=chunk['content'],
            entity_types=config.entity_types,
            chunk_id=chunk['id'],
            file_path=chunk['file_path'],
        ).with_inputs('chunk', 'entity_types')
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
        max_tokens=4000,
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
            for key in ('n_entities', 'n_relations', 'structural_issues', 'axis_scores'):
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
    parser = argparse.ArgumentParser(description='Offline DSPy optimization of yar entity-extraction prompt.')
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
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--pg-dsn', default=os.getenv('PG_DSN', 'postgresql://yar:yar_pass@localhost:5432/yar'))
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', 'default'))
    parser.add_argument('--entity-types', default=DEFAULT_ENTITY_TYPES)
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
        entity_types=args.entity_types,
    )

    payload = run_optimization(config)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info('Wrote optimized instructions to %s', args.output)

    print('\n=== OPTIMIZED ENTITY-EXTRACTION INSTRUCTIONS ===\n')
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
