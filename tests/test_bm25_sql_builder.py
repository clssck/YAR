"""Unit tests for the BM25 SQL builder.

The builder is the seam between query keywords and the on-disk BM25 index. Pinning its output
at the SQL-string level catches accidental regressions in:
- parameter numbering (asyncpg is positional; a missing $N silently breaks queries)
- tsquery parser selection (plainto vs websearch)
- phrase support (multi-word ll_keywords)
- heading-weighted ranking opt-in

Every test asserts on the literal SQL string. The tests do not need a database.
"""

from __future__ import annotations

import pytest

from yar.kg.bm25_sql import build_bm25_sql


class TestBuildBm25Sql:
    def test_simple_query_uses_plainto_tsquery(self) -> None:
        sql, phrases = build_bm25_sql(query='hello world', language='english')
        assert phrases == []
        assert "plainto_tsquery('english', $1)" in sql
        assert 'websearch_to_tsquery' not in sql

    def test_quoted_query_uses_websearch_tsquery(self) -> None:
        sql, _ = build_bm25_sql(query='"closed system"', language='english')
        assert "websearch_to_tsquery('english', $1)" in sql
        assert 'plainto_tsquery' not in sql

    def test_boolean_operator_uses_websearch_tsquery(self) -> None:
        sql, _ = build_bm25_sql(query='alpha AND beta', language='english')
        assert "websearch_to_tsquery('english', $1)" in sql

    def test_negation_query_uses_websearch_tsquery(self) -> None:
        sql, _ = build_bm25_sql(query='-stopword', language='english')
        assert "websearch_to_tsquery('english', $1)" in sql

    def test_multi_word_phrase_appends_phraseto_clause(self) -> None:
        sql, phrases = build_bm25_sql(
            query='manufacturing strategy',
            language='english',
            phrase_terms=['closed system drug transfer device'],
        )
        assert phrases == ['closed system drug transfer device']
        # Phrase parameter is bound at $4 (after query, workspace, limit).
        assert "phraseto_tsquery('english', $4)" in sql
        # The combined tsquery wraps both clauses with OR (||).
        assert "plainto_tsquery('english', $1) || phraseto_tsquery('english', $4)" in sql

    def test_multiple_phrases_get_sequential_parameter_indexes(self) -> None:
        sql, phrases = build_bm25_sql(
            query='strategy',
            language='english',
            phrase_terms=['closed system drug transfer device', 'risk evaluation strategy'],
        )
        assert phrases == ['closed system drug transfer device', 'risk evaluation strategy']
        assert "phraseto_tsquery('english', $4)" in sql
        assert "phraseto_tsquery('english', $5)" in sql

    def test_single_word_phrase_terms_are_ignored(self) -> None:
        # phraseto_tsquery on a single token is equivalent to plainto_tsquery; skip the noise.
        sql, phrases = build_bm25_sql(
            query='strategy',
            language='english',
            phrase_terms=['device', 'risk'],
        )
        assert phrases == []
        assert 'phraseto_tsquery' not in sql

    def test_phrase_term_whitespace_is_trimmed(self) -> None:
        sql, phrases = build_bm25_sql(
            query='strategy',
            language='english',
            phrase_terms=['  closed system drug transfer device  '],
        )
        assert phrases == ['closed system drug transfer device']
        assert 'phraseto_tsquery' in sql

    def test_empty_phrase_terms_produces_no_phrase_clauses(self) -> None:
        sql, phrases = build_bm25_sql(query='strategy', language='english', phrase_terms=[])
        assert phrases == []
        assert 'phraseto_tsquery' not in sql

    def test_none_phrase_terms_produces_no_phrase_clauses(self) -> None:
        sql, phrases = build_bm25_sql(query='strategy', language='english', phrase_terms=None)
        assert phrases == []
        assert 'phraseto_tsquery' not in sql

    def test_default_rank_tsvector_uses_indexed_expression(self) -> None:
        # When heading boost is OFF, rank uses to_tsvector(content) so the GIN index serves both
        # the WHERE @@ filter and the ORDER BY ts_rank_cd computation.
        sql, _ = build_bm25_sql(query='strategy', language='english', heading_weight_enabled=False)
        assert "to_tsvector('english', content)" in sql
        assert 'setweight' not in sql

    def test_heading_weight_enabled_uses_setweight(self) -> None:
        sql, _ = build_bm25_sql(
            query='strategy',
            language='english',
            heading_weight_enabled=True,
        )
        # With heading boost ON, ts_rank_cd uses a weighted tsvector. The @@ filter still uses
        # the indexed plain expression so the GIN index is honored.
        assert "setweight(to_tsvector('english'," in sql
        assert "'A')" in sql
        assert "'B')" in sql
        # Filter expression remains unchanged so the GIN index continues to apply.
        assert "to_tsvector('english', content) @@" in sql

    def test_required_parameter_placeholders_present(self) -> None:
        sql, _ = build_bm25_sql(query='x', language='english')
        # $1 = query text, $2 = workspace, $3 = limit
        assert '$1' in sql
        assert '$2' in sql
        assert '$3' in sql

    @pytest.mark.parametrize('language', ['english', 'german', 'french'])
    def test_language_is_threaded_through(self, language: str) -> None:
        sql, _ = build_bm25_sql(query='hello', language=language)
        assert f"to_tsvector('{language}', content)" in sql
        assert f"plainto_tsquery('{language}', $1)" in sql
