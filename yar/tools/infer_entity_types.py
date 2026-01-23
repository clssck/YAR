#!/usr/bin/env python
"""Post-processing tool to infer types for UNKNOWN entities.

Entities discovered through relationships get UNKNOWN type. This tool
uses the LLM to classify them based on entity name and description.

Usage:
    POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_USER=yar \
    POSTGRES_PASSWORD=yar_pass POSTGRES_DATABASE=yar \
    python -m yar.tools.infer_entity_types --workspace default
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load .env file
from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, '.env'))

from yar.constants import DEFAULT_ENTITY_TYPES
from yar.utils import logger

INFERENCE_PROMPT = """You are classifying entities into types. For each entity, determine the most appropriate type from this list:
{entity_types}

If none fit well, use "other".

Entities to classify:
{entities}

Respond with a JSON array of objects, each with "entity_name" and "inferred_type":
```json
[
  {{"entity_name": "Example Corp", "inferred_type": "organization"}},
  {{"entity_name": "John Smith", "inferred_type": "person"}}
]
```

Only output the JSON array, no other text."""


async def get_unknown_entities(db: Any, workspace: str) -> list[dict[str, str]]:
    """Query entities with UNKNOWN type from the graph."""
    sql = """
    SELECT
      (props::text)::jsonb->>'entity_id' as entity_name,
      (props::text)::jsonb->>'description' as description
    FROM ag_catalog.cypher('chunk_entity_relation', $$
      MATCH (n)
      RETURN properties(n) as props
    $$) as (props ag_catalog.agtype)
    WHERE (props::text)::jsonb->>'entity_type' = 'UNKNOWN'
    """

    results = await db.query(sql, multirows=True)
    return results or []


async def infer_types_with_llm(
    entities: list[dict[str, str]],
    llm_func: Any,
) -> dict[str, str]:
    """Ask LLM to classify entities into types."""
    if not entities:
        return {}

    # Format entities for prompt
    entity_lines = []
    for e in entities:
        name = e.get('entity_name', '')
        desc = e.get('description', '')[:200]  # Truncate long descriptions
        entity_lines.append(f"- {name}: {desc}")

    prompt = INFERENCE_PROMPT.format(
        entity_types=', '.join([*DEFAULT_ENTITY_TYPES, 'other']),
        entities='\n'.join(entity_lines),
    )

    try:
        response = await llm_func(prompt)

        # Parse JSON from response
        # Handle markdown code blocks
        text = response
        if '```' in text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1)

        data = json.loads(text.strip())

        # Build name -> type mapping
        result = {}
        for item in data:
            name = item.get('entity_name', '')
            inferred = item.get('inferred_type', '').lower().replace(' ', '')
            if name and inferred:
                result[name] = inferred

        return result

    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        return {}


async def update_entity_type_in_graph(db: Any, workspace: str, entity_name: str, new_type: str) -> bool:
    """Update entity type in the AGE graph."""
    # AGE requires special handling - use cypher SET
    try:
        # Escape single quotes in entity name
        safe_name = entity_name.replace("'", "\\'")

        sql = f"""
        SELECT * FROM ag_catalog.cypher('chunk_entity_relation', $$
          MATCH (n {{entity_id: '{safe_name}'}})
          SET n.entity_type = '{new_type}'
          RETURN n.entity_id
        $$) as (entity_id ag_catalog.agtype)
        """

        await db.execute(sql)
        return True
    except Exception as e:
        logger.error(f"Failed to update graph for '{entity_name}': {e}")
        return False


async def update_entity_type_in_vdb(db: Any, workspace: str, entity_name: str, new_type: str) -> bool:
    """Update entity type in the VDB."""
    try:
        sql = """
        UPDATE yar_vdb_entity
        SET entity_type = $1
        WHERE workspace = $2 AND LOWER(entity_name) = LOWER($3)
        """
        await db.execute(sql, [new_type, workspace, entity_name])
        return True
    except Exception as e:
        logger.error(f"Failed to update VDB for '{entity_name}': {e}")
        return False


async def run_inference(
    workspace: str,
    dry_run: bool = False,
    batch_size: int = 20,
) -> None:
    """Run type inference on UNKNOWN entities."""
    from yar.kg.postgres_impl import PostgreSQLDB
    from yar.llm.openai import openai_complete_if_cache

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

        print(f'\n{"="*60}')
        print('ENTITY TYPE INFERENCE')
        print(f'{"="*60}')
        print(f'Workspace: {workspace}')
        print(f'Dry run: {dry_run}')
        print(f'{"="*60}\n')

        # Get UNKNOWN entities
        unknown_entities = await get_unknown_entities(db, workspace)

        if not unknown_entities:
            print('No UNKNOWN entities found.')
            return

        print(f'Found {len(unknown_entities)} UNKNOWN entities:\n')
        for e in unknown_entities[:10]:
            print(f"  - {e['entity_name']}")
        if len(unknown_entities) > 10:
            print(f"  ... and {len(unknown_entities) - 10} more\n")

        # Create LLM function using the direct API
        # Use YAR's env var conventions
        model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        api_key = os.getenv('LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('LLM_BINDING_HOST') or os.getenv('OPENAI_API_BASE')

        print(f'Using LLM: {model}')
        print(f'Base URL: {base_url}\n')

        async def llm_func(prompt: str) -> str:
            return await openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt="You are a helpful assistant that classifies entities.",
                api_key=api_key,
                base_url=base_url,
            )

        # Process in batches
        all_inferences: dict[str, str] = {}

        for i in range(0, len(unknown_entities), batch_size):
            batch = unknown_entities[i:i + batch_size]
            print(f'Processing batch {i // batch_size + 1} ({len(batch)} entities)...')

            inferences = await infer_types_with_llm(batch, llm_func)
            all_inferences.update(inferences)

        print(f'\nInferred types for {len(all_inferences)} entities:\n')

        # Group by inferred type for display
        by_type: dict[str, list[str]] = {}
        for name, inferred_type in all_inferences.items():
            by_type.setdefault(inferred_type, []).append(name)

        for type_name, entities in sorted(by_type.items()):
            print(f'  {type_name}:')
            for e in entities:
                print(f'    - {e}')

        if dry_run:
            print('\n[DRY RUN] No changes applied.')
            return

        # Apply updates
        print('\nApplying updates...')

        success_count = 0
        for entity_name, new_type in all_inferences.items():
            graph_ok = await update_entity_type_in_graph(db, workspace, entity_name, new_type)
            vdb_ok = await update_entity_type_in_vdb(db, workspace, entity_name, new_type)

            if graph_ok and vdb_ok:
                success_count += 1
                print(f'  ✓ {entity_name} → {new_type}')
            else:
                print(f'  ✗ {entity_name} (graph={graph_ok}, vdb={vdb_ok})')

        print(f'\nUpdated {success_count}/{len(all_inferences)} entities.')

    finally:
        if db.pool:
            await db.pool.close()


def main():
    parser = argparse.ArgumentParser(description='Infer types for UNKNOWN entities')
    parser.add_argument('--workspace', default='default', help='Workspace name')
    parser.add_argument('--dry-run', action='store_true', help='Show inferences without applying')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for LLM calls')

    args = parser.parse_args()

    asyncio.run(run_inference(
        workspace=args.workspace,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    ))


if __name__ == '__main__':
    main()
