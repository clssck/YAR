"""Chunk-level extraction QA: judge whether ingested entities/relations are
specific enough to ground answers.

Why we need this: the trace-level evaluators tell us if an answer is grounded
in the retrieved context. They cannot tell us whether the retrieved entity or
relation *descriptions* are usable in the first place. If our extractor emits
"AAV-PCL IND Mock Dossier — discussed in the IND filing", a judge can confirm
or refute that against the chunks it came from but not whether downstream
queries about "AAV-PCL Mock Dossier" will get a useful answer.

This evaluator samples N random entities (and N random relations) from the KG
storage, runs a classifier that asks "is this description specific enough to
ground a citation in an answer?", and writes a CSV plus aggregate stats. Run
periodically; spikes in the LOW-QUALITY rate are the signal that ingestion
quality is regressing.

CLI::

    python -m yar.evaluation.phoenix_extraction_qa \
        --workspace default --sample-entities 30 --sample-relations 30 \
        --judge-model tuna --judge-provider openai \
        --output /tmp/extraction_qa.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_ENTITY_QUALITY_TEMPLATE = """You are auditing the quality of an extracted
knowledge-graph entity description. The entity will be surfaced to a RAG
synthesis LLM as primary grounding context.

Mark USEFUL when the description is specific enough that downstream answers
can cite verifiable facts from it (e.g. names, dates, attributes, roles,
identifiers tied to the entity). Mark VAGUE when the description is generic,
near-empty, or only restates the entity name. Mark MISLEADING when the
description introduces concrete claims that would clearly be hallucinated if
cited.

ENTITY NAME: {entity_name}
ENTITY TYPE: {entity_type}
DESCRIPTION:
{description}

Respond with one word: USEFUL, VAGUE, or MISLEADING.
"""


_RELATION_QUALITY_TEMPLATE = """You are auditing the quality of an extracted
knowledge-graph relation description.

Mark USEFUL when the description anchors a specific factual link between the
two entities (e.g. dates, scope, conditions, mechanisms). Mark VAGUE when it
just restates the predicate (e.g. "X is related to Y", "X involves Y"). Mark
MISLEADING when it introduces concrete claims unsupported by typical source
documents.

SOURCE: {src}
TARGET: {tgt}
PREDICATE: {predicate}
DESCRIPTION:
{description}

Respond with one word: USEFUL, VAGUE, or MISLEADING.
"""


_QUALITY_CHOICES = {'USEFUL': 1.0, 'VAGUE': 0.5, 'MISLEADING': 0.0}


@dataclass
class ExtractionQAConfig:
    workspace: str = 'default'
    sample_entities: int = 30
    sample_relations: int = 30
    judge_provider: str = 'openai'
    judge_model: str = 'gpt-4o-mini'
    pg_host: str = field(default_factory=lambda: os.getenv('POSTGRES_HOST', 'localhost'))
    pg_port: int = field(default_factory=lambda: int(os.getenv('POSTGRES_PORT', '5432')))
    pg_user: str = field(default_factory=lambda: os.getenv('POSTGRES_USER', 'yar'))
    pg_password: str = field(default_factory=lambda: os.getenv('POSTGRES_PASSWORD', 'yar_pass'))
    pg_database: str = field(default_factory=lambda: os.getenv('POSTGRES_DATABASE', 'yar'))


def _require_phoenix_evals() -> Any:
    try:
        from phoenix import evals as pe
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('phoenix.evals is required. Install via `pip install -e .[observability]`.') from exc
    return pe


def _parse_relation_content(content: str) -> dict[str, str]:
    """``yar_vdb_relation.content`` stores ``predicate`` plus ``description``
    in a flat blob; this teases apart the most common shape so the judge has
    real fields to evaluate. Falls back to dumping the whole blob into the
    description slot.
    """
    if not isinstance(content, str):
        return {'predicate': '', 'description': str(content or '')}
    text = content.strip()
    # Common shapes: "predicate: ... description: ..." or pipe-delimited.
    predicate = ''
    description = text
    lower = text.lower()
    pred_idx = lower.find('predicate')
    desc_idx = lower.find('description')
    if pred_idx >= 0 and desc_idx > pred_idx:
        predicate = text[pred_idx:desc_idx].split(':', 1)[-1].strip()
        description = text[desc_idx:].split(':', 1)[-1].strip()
    elif desc_idx >= 0:
        description = text[desc_idx:].split(':', 1)[-1].strip()
    return {'predicate': predicate, 'description': description}


def _fetch_entity_sample(config: ExtractionQAConfig) -> Any:
    """Pull a random sample of entities into a dataframe."""
    try:
        import pandas as pd
        import psycopg
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            'psycopg and pandas are required. Install via `pip install -e .[api]` and `[observability]`.'
        ) from exc

    conn_str = (
        f'host={config.pg_host} port={config.pg_port} dbname={config.pg_database} '
        f'user={config.pg_user} password={config.pg_password}'
    )
    sql = """
    SELECT entity_name, entity_type, content
    FROM yar_vdb_entity
    WHERE workspace = %s AND content IS NOT NULL AND length(content) > 0
    ORDER BY random()
    LIMIT %s
    """
    rows: list[dict[str, Any]] = []
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(sql, (config.workspace, config.sample_entities))
        for entity_name, entity_type, content in cur.fetchall():
            rows.append(
                {
                    'kind': 'entity',
                    'entity_name': str(entity_name or ''),
                    'entity_type': str(entity_type or ''),
                    'description': str(content or ''),
                }
            )
    return pd.DataFrame(rows)


def _fetch_relation_sample(config: ExtractionQAConfig) -> Any:
    """Pull a random sample of relations into a dataframe."""
    try:
        import pandas as pd
        import psycopg
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            'psycopg and pandas are required. Install via `pip install -e .[api]` and `[observability]`.'
        ) from exc

    conn_str = (
        f'host={config.pg_host} port={config.pg_port} dbname={config.pg_database} '
        f'user={config.pg_user} password={config.pg_password}'
    )
    # Schema: yar_vdb_relation has columns id, workspace, src_id, tgt_id, content, ...
    sql = """
    SELECT source_id, target_id, content
    FROM yar_vdb_relation
    WHERE workspace = %s AND content IS NOT NULL AND length(content) > 0
    ORDER BY random()
    LIMIT %s
    """
    rows: list[dict[str, Any]] = []
    try:
        with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
            cur.execute(sql, (config.workspace, config.sample_relations))
            for src, tgt, content in cur.fetchall():
                parsed = _parse_relation_content(str(content or ''))
                rows.append(
                    {
                        'kind': 'relation',
                        'src': str(src or ''),
                        'tgt': str(tgt or ''),
                        'predicate': parsed['predicate'],
                        'description': parsed['description'],
                    }
                )
    except Exception as exc:
        logger.warning('Relation sample failed (table may differ): %s', exc)
    return pd.DataFrame(rows)


def run_extraction_qa(config: ExtractionQAConfig) -> Any:
    """Run extraction-QA classifiers over a random sample of entities/relations."""
    pe = _require_phoenix_evals()

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError('pandas is required.') from exc

    entity_df = _fetch_entity_sample(config)
    relation_df = _fetch_relation_sample(config)

    llm = pe.LLM(provider=config.judge_provider, model=config.judge_model)

    entity_eval = pe.create_classifier(
        name='entity_quality',
        prompt_template=_ENTITY_QUALITY_TEMPLATE,
        llm=llm,
        choices=_QUALITY_CHOICES,
        direction='maximize',
    )
    relation_eval = pe.create_classifier(
        name='relation_quality',
        prompt_template=_RELATION_QUALITY_TEMPLATE,
        llm=llm,
        choices=_QUALITY_CHOICES,
        direction='maximize',
    )

    scored_entities = pe.evaluate_dataframe(entity_df, [entity_eval]) if not entity_df.empty else entity_df
    scored_relations = pe.evaluate_dataframe(relation_df, [relation_eval]) if not relation_df.empty else relation_df

    return pd.concat([scored_entities, scored_relations], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit YAR extraction quality with an LLM judge.')
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', 'default'))
    parser.add_argument('--sample-entities', type=int, default=30)
    parser.add_argument('--sample-relations', type=int, default=30)
    parser.add_argument('--judge-model', default='gpt-4o-mini')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--output', default=None, help='CSV path to dump scored sample.')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    config = ExtractionQAConfig(
        workspace=args.workspace,
        sample_entities=args.sample_entities,
        sample_relations=args.sample_relations,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )
    df = run_extraction_qa(config)

    if df.empty:
        logger.info('No entities/relations sampled.')
        return 0

    logger.info('Sampled %d items.', len(df))

    # Aggregate score by kind for a quick stdout summary.
    if 'entity_quality_score' in df.columns or 'relation_quality_score' in df.columns:
        for kind in ('entity', 'relation'):
            sub = df[df['kind'] == kind]
            score_col = f'{kind}_quality_score'
            if not sub.empty and score_col in sub.columns:
                # phoenix.evals returns dict payloads under {col}_score; take score.
                def _score(payload: Any) -> float:
                    if isinstance(payload, dict):
                        score = payload.get('score')
                        return float(score) if score is not None else 0.0
                    if isinstance(payload, str):
                        try:
                            import ast

                            parsed = ast.literal_eval(payload)
                            if isinstance(parsed, dict):
                                value = parsed.get('score')
                                return float(value) if value is not None else 0.0
                        except Exception:
                            return 0.0
                    return 0.0

                mean = sub[score_col].apply(_score).mean()
                logger.info('%s quality mean: %.3f over %d samples', kind, mean, len(sub))

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info('Wrote scored dataframe to %s', args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
