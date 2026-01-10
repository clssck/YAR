#!/usr/bin/env python
"""Standalone MMR (Maximal Marginal Relevance) Diversity Evaluation.

Tests how MMR re-ranking would improve result diversity on your actual data.
Run this BEFORE deciding whether to integrate MMR into the core codebase.

Usage:
    POSTGRES_HOST=localhost POSTGRES_PORT=5433 POSTGRES_USER=lightrag \
    POSTGRES_PASSWORD=lightrag_pass POSTGRES_DATABASE=lightrag \
    python examples/test_mmr_standalone.py "your search query"

    # With custom parameters:
    python examples/test_mmr_standalone.py "query" --top-k 20 --lambda 0.7 --workspace default
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


def mmr_rerank(
    results: list[dict[str, Any]],
    query_embedding: np.ndarray,
    lambda_param: float = 0.5,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Apply Maximal Marginal Relevance re-ranking for diversity.

    MMR balances relevance to query with diversity from already-selected results.
    Formula: MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected))

    Args:
        results: List of search results with 'content_vector' field
        query_embedding: Query embedding vector
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        top_k: Number of results to return

    Returns:
        Re-ranked results with MMR scores
    """
    if not results:
        return []

    # Extract embeddings
    embeddings = []
    for r in results:
        vec = r.get('content_vector')
        if vec is None:
            embeddings.append(np.zeros(len(query_embedding)))
        elif isinstance(vec, (list, tuple)):
            embeddings.append(np.array(vec))
        else:
            embeddings.append(vec)

    embeddings = np.array(embeddings)

    # Calculate relevance scores (similarity to query)
    relevance_scores = np.array([
        cosine_similarity(emb, query_embedding) for emb in embeddings
    ])

    selected_indices: list[int] = []
    remaining_indices = list(range(len(results)))

    while len(selected_indices) < min(top_k, len(results)) and remaining_indices:
        mmr_scores = []

        for idx in remaining_indices:
            relevance = relevance_scores[idx]

            # Max similarity to already selected documents
            if selected_indices:
                max_sim_to_selected = max(
                    cosine_similarity(embeddings[idx], embeddings[sel_idx])
                    for sel_idx in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            # MMR formula
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append((idx, mmr, relevance, max_sim_to_selected))

        # Select document with highest MMR score
        best = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best[0])
        remaining_indices.remove(best[0])

    # Build result list with MMR info
    mmr_results = []
    for rank, idx in enumerate(selected_indices):
        result = results[idx].copy()
        result['mmr_rank'] = rank + 1
        result['original_rank'] = idx + 1
        result['relevance_score'] = float(relevance_scores[idx])
        mmr_results.append(result)

    return mmr_results


def calculate_diversity_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate diversity metrics for a result set."""
    if len(results) < 2:
        return {'avg_pairwise_similarity': 0.0, 'min_pairwise_similarity': 0.0}

    embeddings = []
    for r in results:
        vec = r.get('content_vector')
        if vec is not None:
            if isinstance(vec, (list, tuple)):
                embeddings.append(np.array(vec))
            else:
                embeddings.append(vec)

    if len(embeddings) < 2:
        return {'avg_pairwise_similarity': 0.0, 'min_pairwise_similarity': 0.0}

    # Calculate all pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    return {
        'avg_pairwise_similarity': float(np.mean(similarities)),
        'min_pairwise_similarity': float(np.min(similarities)),
        'max_pairwise_similarity': float(np.max(similarities)),
    }


async def run_evaluation(
    query: str,
    workspace: str,
    top_k: int,
    lambda_param: float,
) -> None:
    """Run MMR evaluation on a query."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    # Database config
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'lightrag'),
        'password': os.getenv('POSTGRES_PASSWORD', ''),
        'database': os.getenv('POSTGRES_DATABASE', 'lightrag'),
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
        print('MMR DIVERSITY EVALUATION')
        print(f'{"="*70}')
        print(f'Query: "{query}"')
        print(f'Workspace: {workspace}')
        print(f'Top-K: {top_k}, Lambda: {lambda_param}')
        print(f'{"="*70}\n')

        # Get query embedding using existing embedding function
        # For this standalone test, we'll query chunks and use their embeddings
        # In production, you'd embed the query first

        # Fetch chunks with embeddings
        fetch_k = top_k * 3  # Fetch more for MMR to work with

        # Get sample chunks with their vectors
        sample_sql = """
            SELECT id, content, content_vector, file_path
            FROM LIGHTRAG_VDB_CHUNKS
            WHERE workspace = $1
              AND content_vector IS NOT NULL
            LIMIT $2
        """

        results = await db.query(sample_sql, params=[workspace, fetch_k], multirows=True)

        if not results:
            print('No chunks found in the database. Please ensure data is loaded.')
            return

        print(f'Fetched {len(results)} chunks for evaluation\n')

        # Use first chunk's embedding as pseudo-query (for demonstration)
        # In real usage, you'd embed the actual query
        query_embedding = np.array(results[0]['content_vector'])

        # Calculate relevance scores for standard ranking
        for r in results:
            vec = np.array(r['content_vector'])
            r['relevance_score'] = cosine_similarity(vec, query_embedding)

        # Sort by relevance (standard ranking)
        standard_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:top_k]

        # Apply MMR
        mmr_results = mmr_rerank(results, query_embedding, lambda_param, top_k)

        # Calculate diversity metrics
        standard_diversity = calculate_diversity_metrics(standard_results)
        mmr_diversity = calculate_diversity_metrics(mmr_results)

        # Print comparison
        print('=== STANDARD RANKING (by relevance only) ===\n')
        for i, r in enumerate(standard_results[:10], 1):
            content_preview = r['content'][:80].replace('\n', ' ') + '...'
            print(f'{i:2}. [score: {r["relevance_score"]:.4f}] {content_preview}')

        print(f'\n=== MMR RANKING (λ={lambda_param}) ===\n')
        for r in mmr_results[:10]:
            content_preview = r['content'][:80].replace('\n', ' ') + '...'
            rank_change = r['original_rank'] - r['mmr_rank']
            change_str = f'[+{rank_change}]' if rank_change > 0 else f'[{rank_change}]' if rank_change < 0 else '[=]'
            print(f'{r["mmr_rank"]:2}. [score: {r["relevance_score"]:.4f}] {change_str:5} {content_preview}')

        # Print metrics
        print(f'\n{"="*70}')
        print('DIVERSITY METRICS')
        print(f'{"="*70}')
        print('\nStandard Ranking:')
        print(f'  Avg pairwise similarity: {standard_diversity["avg_pairwise_similarity"]:.4f}')
        print(f'  Max pairwise similarity: {standard_diversity.get("max_pairwise_similarity", 0):.4f}')

        print('\nMMR Ranking:')
        print(f'  Avg pairwise similarity: {mmr_diversity["avg_pairwise_similarity"]:.4f}')
        print(f'  Max pairwise similarity: {mmr_diversity.get("max_pairwise_similarity", 0):.4f}')

        improvement = standard_diversity['avg_pairwise_similarity'] - mmr_diversity['avg_pairwise_similarity']
        if standard_diversity['avg_pairwise_similarity'] > 0:
            improvement_pct = (improvement / standard_diversity['avg_pairwise_similarity']) * 100
        else:
            improvement_pct = 0

        print(f'\n✓ Diversity Improvement: {improvement_pct:.1f}% (lower similarity = more diverse)')

        if improvement_pct > 10:
            print('\n→ RECOMMENDATION: MMR shows significant diversity improvement. Consider integrating.')
        elif improvement_pct > 5:
            print('\n→ RECOMMENDATION: MMR shows moderate improvement. May be worth integrating for diverse queries.')
        else:
            print('\n→ RECOMMENDATION: MMR shows minimal improvement on this data. May not be needed.')

    finally:
        if db.pool:
            await db.pool.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate MMR diversity on your data')
    parser.add_argument('query', nargs='?', default='test query', help='Search query to evaluate')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results (default: 10)')
    parser.add_argument('--lambda', dest='lambda_param', type=float, default=0.5,
                        help='MMR lambda parameter 0-1 (default: 0.5, higher=more relevance)')
    parser.add_argument('--workspace', default='default', help='Workspace name (default: default)')

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        query=args.query,
        workspace=args.workspace,
        top_k=args.top_k,
        lambda_param=args.lambda_param,
    ))


if __name__ == '__main__':
    main()
