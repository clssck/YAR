"""BM25 SQL builder, factored out of postgres_impl so it can be unit-tested without asyncpg.

This module is a pure-Python helper. It produces parameterized SQL strings for the BM25 leg of
the hybrid (vector + BM25) chunk search. Database execution lives in postgres_impl.
"""

from __future__ import annotations

import os

# TS_RANK_CD_FLAG: PostgreSQL ts_rank_cd normalization flags.
# 32 = normalize by document length (fairer across variable-length chunks).
# Mirror the postgres_impl constant so the SQL stays consistent if either is updated.
TS_RANK_CD_FLAG = int(os.getenv('YAR_TS_RANK_FLAG', '32'))


def build_bm25_sql(
    *,
    query: str,
    language: str,
    phrase_terms: list[str] | None = None,
    heading_weight_enabled: bool = False,
) -> tuple[str, list[str]]:
    """Build the parameterized BM25 SQL fragment for hybrid_search.

    Returns ``(sql, normalized_phrases)`` so callers can pass the phrases through to db.query as
    additional positional parameters ($4 ... $N). The SQL itself uses $1=query, $2=workspace,
    $3=limit, with phrase parameters appended in order.

    Args:
        query: Search query text. Drives plainto_tsquery / websearch_to_tsquery selection.
        language: PostgreSQL FTS configuration name (e.g. 'english').
        phrase_terms: Optional list of multi-word phrases. Single-word terms are ignored because
            phraseto_tsquery on a single token is identical to plainto_tsquery.
        heading_weight_enabled: When True, ts_rank_cd uses a setweight'd tsvector (markdown
            heading lines weighted 'A', body 'B') so heading matches outrank body matches at
            equal coverage. The @@ filter still uses the indexed expression so the GIN index is
            honored.
    """
    advanced_syntax_chars = ('"', ' OR ', ' AND ', ' NOT ')
    if any(c in query for c in advanced_syntax_chars) or query.startswith('-'):
        ts_query_func = 'websearch_to_tsquery'
    else:
        ts_query_func = 'plainto_tsquery'

    normalized_phrases = [
        phrase.strip()
        for phrase in (phrase_terms or [])
        if phrase and isinstance(phrase, str) and ' ' in phrase.strip()
    ]
    if normalized_phrases:
        phrase_clauses = ' || '.join(
            f"phraseto_tsquery('{language}', ${i + 4})" for i in range(len(normalized_phrases))
        )
        ts_query_expr = f"({ts_query_func}('{language}', $1) || {phrase_clauses})"
    else:
        ts_query_expr = f"{ts_query_func}('{language}', $1)"

    if heading_weight_enabled:
        rank_tsvector_expr = (
            f"setweight(to_tsvector('{language}', "
            "COALESCE(array_to_string(ARRAY(SELECT m[1] FROM regexp_matches(content, '^#+\\s+(.*)$', 'gn') AS m), ' '), '')), 'A')"
            f" || setweight(to_tsvector('{language}', regexp_replace(content, '^#+\\s+.*$', '', 'gn')), 'B')"
        )
    else:
        rank_tsvector_expr = f"to_tsvector('{language}', content)"

    bm25_sql = f"""
        SELECT
            id,
            full_doc_id,
            chunk_order_index,
            tokens,
            content,
            file_path,
            s3_key,
            ts_rank_cd(
                {rank_tsvector_expr},
                {ts_query_expr},
                {TS_RANK_CD_FLAG}
            ) AS bm25_score
        FROM YAR_DOC_CHUNKS
        WHERE workspace = $2
          AND to_tsvector('{language}', content) @@ {ts_query_expr}
        ORDER BY bm25_score DESC
        LIMIT $3
    """
    return bm25_sql, normalized_phrases


__all__ = ['TS_RANK_CD_FLAG', 'build_bm25_sql']
