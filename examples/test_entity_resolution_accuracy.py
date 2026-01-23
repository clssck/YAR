#!/usr/bin/env python
"""Standalone Entity Resolution Accuracy Evaluation.

Tests how the entity resolution system performs on real data.
Measures:
1. Type mismatch detection (should NOT match different types)
2. Abbreviation/variant detection (SHOULD match same entities)
3. Confidence distribution

Usage:
    POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_USER=yar \
    POSTGRES_PASSWORD=yar_pass POSTGRES_DATABASE=yar \
    python examples/test_entity_resolution_accuracy.py

    # With custom parameters:
    python examples/test_entity_resolution_accuracy.py --workspace default --sample-size 50
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_type_from_content(content: str | None) -> str:
    """Extract entity type from content string."""
    if not content:
        return 'Unknown'
    lines = content.split('\n', 2)
    if len(lines) >= 2:
        second_line = lines[1].strip()
        if ':' in second_line:
            potential_type = second_line.split(':')[0].strip()
            if potential_type and len(potential_type) < 50 and ' ' not in potential_type:
                return potential_type
    return 'Unknown'


async def run_evaluation(
    workspace: str,
    sample_size: int,
) -> None:
    """Run entity resolution accuracy evaluation."""
    from yar.kg.postgres_impl import PostgreSQLDB

    # Database config
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'yar'),
        'password': os.getenv('POSTGRES_PASSWORD', ''),
        'database': os.getenv('POSTGRES_DATABASE', 'yar'),
        'workspace': workspace,
        'max_connections': 10,
        'connection_retry_attempts': 3,
        'connection_retry_backoff': 0.5,
        'connection_retry_backoff_max': 5.0,
        'pool_close_timeout': 5.0,
    }

    db = PostgreSQLDB(config=db_config)

    try:
        await db.initdb()
        print(f'\n{"="*70}')
        print('ENTITY RESOLUTION ACCURACY EVALUATION')
        print(f'{"="*70}')
        print(f'Workspace: {workspace}')
        print(f'Sample size: {sample_size}')
        print(f'{"="*70}\n')

        # Step 1: Get all entities with their types
        print('Fetching entities with type information...\n')

        entity_sql = """
            SELECT entity_name, entity_type, content
            FROM LIGHTRAG_VDB_ENTITY
            WHERE workspace = $1
              AND content IS NOT NULL
            LIMIT $2
        """

        entities = await db.query(entity_sql, params=[workspace, sample_size * 10], multirows=True)

        if not entities:
            print('No entities found in the database.')
            return

        # Use stored entity_type, fall back to extracting from content
        entities_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in entities:
            etype = e.get('entity_type')
            if not etype:
                etype = extract_type_from_content(e['content'])
            e['entity_type'] = etype
            entities_by_type[etype].append(e)

        print(f'Found {len(entities)} entities across {len(entities_by_type)} types:\n')
        for etype, elist in sorted(entities_by_type.items(), key=lambda x: -len(x[1])):
            print(f'  {etype}: {len(elist)} entities')

        # Step 2: Check existing alias cache
        print(f'\n{"="*70}')
        print('EXISTING ALIAS CACHE ANALYSIS')
        print(f'{"="*70}\n')

        alias_sql = """
            SELECT alias, canonical_entity, method, confidence, entity_type
            FROM LIGHTRAG_ENTITY_ALIASES
            WHERE workspace = $1
            ORDER BY confidence DESC
            LIMIT $2
        """

        aliases = await db.query(alias_sql, params=[workspace, 100], multirows=True)

        if aliases:
            print(f'Found {len(aliases)} existing aliases:\n')

            # Analyze alias quality
            high_confidence = [a for a in aliases if a['confidence'] >= 0.85]
            medium_confidence = [a for a in aliases if 0.70 <= a['confidence'] < 0.85]
            low_confidence = [a for a in aliases if a['confidence'] < 0.70]

            print(f'  High confidence (>=0.85): {len(high_confidence)}')
            print(f'  Medium confidence (0.70-0.85): {len(medium_confidence)}')
            print(f'  Low confidence (<0.70): {len(low_confidence)}')

            # Show sample aliases
            print('\nSample high-confidence aliases:')
            for a in high_confidence[:5]:
                print(f'  "{a["alias"]}" → "{a["canonical_entity"]}" ({a["confidence"]:.2f})')

            if medium_confidence:
                print('\nSample medium-confidence aliases (would benefit from soft threshold):')
                for a in medium_confidence[:5]:
                    print(f'  "{a["alias"]}" → "{a["canonical_entity"]}" ({a["confidence"]:.2f})')
        else:
            print('No existing aliases found.')

        # Step 3: Find potential type mismatch candidates
        print(f'\n{"="*70}')
        print('TYPE MISMATCH RISK ANALYSIS')
        print(f'{"="*70}\n')

        # Find entities with similar names but different types
        # This is where type-aware resolution would help

        # Group by normalized name (lowercase, no special chars)
        import re
        name_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in entities:
            normalized = re.sub(r'[^a-z0-9]', '', e['entity_name'].lower())
            if len(normalized) >= 3:  # Skip very short names
                name_groups[normalized].append(e)

        # Find groups with multiple types
        type_conflicts = []
        for norm_name, group in name_groups.items():
            types_in_group = {e['entity_type'] for e in group}
            # Exclude Unknown from conflict detection
            known_types = types_in_group - {'Unknown'}
            if len(known_types) > 1:
                type_conflicts.append({
                    'normalized_name': norm_name,
                    'entities': group,
                    'types': known_types,
                })

        if type_conflicts:
            print(f'Found {len(type_conflicts)} potential type conflicts:\n')
            for conflict in type_conflicts[:10]:
                print(f'  Name pattern: "{conflict["normalized_name"]}"')
                print(f'  Types: {conflict["types"]}')
                for e in conflict['entities'][:3]:
                    print(f'    - "{e["entity_name"]}" ({e["entity_type"]})')
                print()

            print('→ Type-aware resolution would prevent merging these different entities.')
        else:
            print('No obvious type conflicts found in entity names.')
            print('(This is good - your data may not need type-aware resolution)')

        # Step 4: Analyze abbreviation/variant patterns
        print(f'\n{"="*70}')
        print('ABBREVIATION/VARIANT PATTERN ANALYSIS')
        print(f'{"="*70}\n')

        # Find potential abbreviations (short names that might expand)
        short_entities = [e for e in entities if len(e['entity_name']) <= 5]
        long_entities = [e for e in entities if len(e['entity_name']) > 20]

        print(f'Short entities (<=5 chars, potential abbreviations): {len(short_entities)}')
        print(f'Long entities (>20 chars, potential expansions): {len(long_entities)}')

        if short_entities:
            print('\nSample short entities:')
            for e in short_entities[:10]:
                print(f'  "{e["entity_name"]}" ({e["entity_type"]})')

        # Step 5: Summary and recommendations
        print(f'\n{"="*70}')
        print('EVALUATION SUMMARY')
        print(f'{"="*70}\n')

        print('Data characteristics:')
        print(f'  Total entities: {len(entities)}')
        print(f'  Entity types: {len(entities_by_type)}')
        print(f'  Existing aliases: {len(aliases) if aliases else 0}')
        print(f'  Type conflicts: {len(type_conflicts)}')
        print(f'  Short entities (abbrev candidates): {len(short_entities)}')

        print('\nAccuracy improvement potential:')

        if type_conflicts:
            print(f'  ✓ Type-aware resolution: HIGH value ({len(type_conflicts)} conflicts to prevent)')
        else:
            print('  ✗ Type-aware resolution: LOW value (no type conflicts found)')

        if medium_confidence:
            print(f'  ✓ Soft match threshold: HIGH value ({len(medium_confidence)} medium-confidence matches)')
        else:
            print('  ✗ Soft match threshold: LOW value (no medium-confidence matches)')

        if len(short_entities) > 10:
            print(f'  ✓ Abbreviation handling: RELEVANT ({len(short_entities)} short entities)')
        else:
            print('  ✗ Abbreviation handling: LOW relevance (few short entities)')

    finally:
        if db.pool:
            await db.pool.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate entity resolution accuracy')
    parser.add_argument('--workspace', default='default', help='Workspace name (default: default)')
    parser.add_argument('--sample-size', type=int, default=100, help='Sample size (default: 100)')

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        workspace=args.workspace,
        sample_size=args.sample_size,
    ))


if __name__ == '__main__':
    main()
