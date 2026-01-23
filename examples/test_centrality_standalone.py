#!/usr/bin/env python
"""Standalone Entity Centrality Evaluation.

Tests how entity importance (measured by graph connections) would affect ranking.
Run this BEFORE deciding whether to integrate centrality scoring into the core.

Usage:
    POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_USER=yar \
    POSTGRES_PASSWORD=yar_pass POSTGRES_DATABASE=yar \
    python examples/test_centrality_standalone.py "your search query"

    # With custom parameters:
    python examples/test_centrality_standalone.py "query" --top-k 20 --alpha 0.3 --workspace default
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def centrality_boost(
    results: list[dict[str, Any]],
    centrality_scores: dict[str, float],
    alpha: float = 0.3,
) -> list[dict[str, Any]]:
    """Boost entity scores by their centrality (connection count).

    Formula: final_score = similarity * (1 + alpha * normalized_centrality)

    Args:
        results: List of entity search results
        centrality_scores: Dict mapping entity_name -> normalized centrality (0-1)
        alpha: Boost strength (0 = no boost, 1 = strong boost)

    Returns:
        Results with boosted scores
    """
    boosted_results = []

    for r in results:
        entity_name = r.get('entity_name', r.get('id', ''))
        centrality = centrality_scores.get(entity_name, 0.0)
        original_score = r.get('similarity_score', 0.0)

        # Apply centrality boost
        boost_factor = 1 + alpha * centrality
        boosted_score = original_score * boost_factor

        result = r.copy()
        result['original_score'] = original_score
        result['centrality'] = centrality
        result['boost_factor'] = boost_factor
        result['boosted_score'] = boosted_score
        boosted_results.append(result)

    # Sort by boosted score
    boosted_results.sort(key=lambda x: x['boosted_score'], reverse=True)

    return boosted_results


async def run_evaluation(
    query: str,
    workspace: str,
    top_k: int,
    alpha: float,
) -> None:
    """Run centrality evaluation on entities."""
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
        print('ENTITY CENTRALITY EVALUATION')
        print(f'{"="*70}')
        print(f'Query: "{query}"')
        print(f'Workspace: {workspace}')
        print(f'Top-K: {top_k}, Alpha (boost strength): {alpha}')
        print(f'{"="*70}\n')

        # Step 1: Calculate degree centrality for all entities
        print('Calculating entity centrality (connection counts)...\n')

        # Count connections for each entity (as source or target in relations)
        centrality_sql = """
            WITH entity_degrees AS (
                SELECT entity_name, COUNT(*) as total_connections
                FROM (
                    SELECT source_id as entity_name FROM LIGHTRAG_VDB_RELATION WHERE workspace = $1
                    UNION ALL
                    SELECT target_id as entity_name FROM LIGHTRAG_VDB_RELATION WHERE workspace = $1
                ) connections
                GROUP BY entity_name
            )
            SELECT entity_name, total_connections,
                   total_connections::float / NULLIF(MAX(total_connections) OVER (), 0) as normalized_centrality
            FROM entity_degrees
            ORDER BY total_connections DESC
        """

        centrality_results = await db.query(centrality_sql, params=[workspace], multirows=True)

        if not centrality_results:
            print('No relations found. Checking if entities exist...')
            # Check entity count
            entity_count = await db.query(
                'SELECT COUNT(*) as cnt FROM LIGHTRAG_VDB_ENTITY WHERE workspace = $1',
                params=[workspace],
                multirows=False,
            )
            print(f'Found {entity_count["cnt"] if entity_count else 0} entities but no relations.')
            print('Centrality scoring requires a connected graph. Exiting.')
            return

        # Build centrality lookup
        centrality_scores = {
            r['entity_name']: r['normalized_centrality']
            for r in centrality_results
        }

        # Show top hub entities
        print('=== TOP HUB ENTITIES (most connected) ===\n')
        for i, r in enumerate(centrality_results[:10], 1):
            print(f'{i:2}. {r["entity_name"][:50]:50} | {r["total_connections"]:4} connections | centrality: {r["normalized_centrality"]:.3f}')

        total_entities = len(centrality_results)
        avg_connections = sum(r['total_connections'] for r in centrality_results) / total_entities if total_entities else 0
        print(f'\nTotal entities with connections: {total_entities}')
        print(f'Average connections per entity: {avg_connections:.1f}')

        # Step 2: Get entities with their embeddings for similarity search
        print(f'\n{"="*70}')
        print('Running entity similarity search...')
        print(f'{"="*70}\n')

        # Get sample entities with vectors
        entity_sql = """
            SELECT id, entity_name, content, content_vector
            FROM LIGHTRAG_VDB_ENTITY
            WHERE workspace = $1
              AND content_vector IS NOT NULL
            LIMIT $2
        """

        entities = await db.query(entity_sql, params=[workspace, top_k * 3], multirows=True)

        if not entities:
            print('No entities with embeddings found.')
            return

        # Use first entity as pseudo-query embedding (for demonstration)
        query_embedding = np.array(entities[0]['content_vector'])

        # Calculate similarity scores
        for e in entities:
            vec = np.array(e['content_vector'])
            e['similarity_score'] = cosine_similarity(vec, query_embedding)

        # Standard ranking (similarity only)
        standard_results = sorted(entities, key=lambda x: x['similarity_score'], reverse=True)[:top_k]

        # Add original rank
        for i, r in enumerate(standard_results, 1):
            r['original_rank'] = i

        # Apply centrality boost
        boosted_results = centrality_boost(entities, centrality_scores, alpha)[:top_k]

        # Add boosted rank
        for i, r in enumerate(boosted_results, 1):
            r['boosted_rank'] = i

        # Print comparison
        print('=== STANDARD RANKING (similarity only) ===\n')
        for i, r in enumerate(standard_results[:10], 1):
            entity_name = r['entity_name'][:40]
            centrality = centrality_scores.get(r['entity_name'], 0)
            print(f'{i:2}. {entity_name:40} | sim: {r["similarity_score"]:.4f} | centrality: {centrality:.3f}')

        print(f'\n=== CENTRALITY-BOOSTED RANKING (α={alpha}) ===\n')

        # Create lookup for rank changes
        standard_rank_lookup = {r['entity_name']: r['original_rank'] for r in standard_results}

        for r in boosted_results[:10]:
            entity_name = r['entity_name'][:40]
            old_rank = standard_rank_lookup.get(r['entity_name'], top_k + 1)
            new_rank = r['boosted_rank']
            rank_change = old_rank - new_rank

            if rank_change > 0:
                change_str = f'[↑{rank_change}]'
            elif rank_change < 0:
                change_str = f'[↓{-rank_change}]'
            else:
                change_str = '[=]'

            print(f'{new_rank:2}. {entity_name:40} | boosted: {r["boosted_score"]:.4f} | centrality: {r["centrality"]:.3f} {change_str}')

        # Calculate ranking changes
        print(f'\n{"="*70}')
        print('RANKING IMPACT ANALYSIS')
        print(f'{"="*70}')

        # Count how many entities changed position
        standard_top_set = {r['entity_name'] for r in standard_results}
        boosted_top_set = {r['entity_name'] for r in boosted_results}

        new_entries = boosted_top_set - standard_top_set
        removed_entries = standard_top_set - boosted_top_set

        print(f'\nEntities promoted into top-{top_k}: {len(new_entries)}')
        for name in list(new_entries)[:5]:
            centrality = centrality_scores.get(name, 0)
            print(f'  + {name[:50]} (centrality: {centrality:.3f})')

        print(f'\nEntities demoted out of top-{top_k}: {len(removed_entries)}')
        for name in list(removed_entries)[:5]:
            centrality = centrality_scores.get(name, 0)
            print(f'  - {name[:50]} (centrality: {centrality:.3f})')

        # Recommendation
        change_pct = (len(new_entries) / top_k) * 100 if top_k else 0

        print(f'\n→ Ranking changed {change_pct:.0f}% of results')

        if change_pct > 30:
            print('\n→ RECOMMENDATION: Centrality significantly reshuffles rankings.')
            print('   If hub entities are more authoritative in your domain, consider integrating.')
        elif change_pct > 10:
            print('\n→ RECOMMENDATION: Moderate ranking changes. Worth testing on real queries.')
        else:
            print('\n→ RECOMMENDATION: Minimal impact. Your similarity ranking may already favor connected entities.')

    finally:
        if db.pool:
            await db.pool.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate entity centrality scoring on your data')
    parser.add_argument('query', nargs='?', default='test query', help='Search query to evaluate')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Centrality boost strength 0-1 (default: 0.3)')
    parser.add_argument('--workspace', default='default', help='Workspace name (default: default)')

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        query=args.query,
        workspace=args.workspace,
        top_k=args.top_k,
        alpha=args.alpha,
    ))


if __name__ == '__main__':
    main()
