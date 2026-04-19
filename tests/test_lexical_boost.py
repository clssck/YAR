from __future__ import annotations

import asyncio

from yar.base import QueryParam
from yar.utils import process_chunks_unified, score_chunk_lexical


def _run_process_chunks(
	query: str,
	chunks: list[dict],
	query_param: QueryParam,
	topic_terms: list[str] | None = None,
	facet_terms: list[str] | None = None,
) -> list[dict]:
	return asyncio.run(
		process_chunks_unified(
			query=query,
			unique_chunks=chunks,
			query_param=query_param,
			global_config={},
			topic_terms=topic_terms,
			facet_terms=facet_terms,
		)
	)


def test_process_chunks_unified_reorders_by_rare_terms(monkeypatch) -> None:
	monkeypatch.delenv('ENABLE_LEXICAL_BOOST', raising=False)
	monkeypatch.delenv('LEXICAL_BOOST_WEIGHT', raising=False)
	query_param = QueryParam(chunk_top_k=2, enable_rerank=False)
	chunks = [
		{
			'chunk_id': 'generic',
			'content': 'The treatment protocol requires a cold start checklist.',
			'file_path': 'generic.md',
			'retrieval_score': 0.9,
		},
		{
			'chunk_id': 'rare',
			'content': 'Gamma-42 treatment requires a cold start checklist.',
			'file_path': 'rare.md',
			'retrieval_score': 0.9,
		},
	]

	processed = _run_process_chunks(
		query='What does gamma-42 treatment require?',
		chunks=chunks,
		query_param=query_param,
		topic_terms=['gamma-42'],
	)

	assert [chunk['chunk_id'] for chunk in processed] == ['rare', 'generic']


def test_process_chunks_unified_keeps_stopword_queries_stable(monkeypatch) -> None:
	monkeypatch.delenv('ENABLE_LEXICAL_BOOST', raising=False)
	monkeypatch.delenv('LEXICAL_BOOST_WEIGHT', raising=False)
	query_param = QueryParam(chunk_top_k=3, enable_rerank=False)
	chunks = [
		{'chunk_id': 'first', 'content': 'Alpha note', 'file_path': 'a.md', 'retrieval_score': 0.5},
		{'chunk_id': 'second', 'content': 'Beta note', 'file_path': 'b.md', 'retrieval_score': 0.5},
		{'chunk_id': 'third', 'content': 'Gamma note', 'file_path': 'c.md', 'retrieval_score': 0.5},
	]

	processed = _run_process_chunks(
		query='the and what is',
		chunks=chunks,
		query_param=query_param,
	)

	assert [chunk['chunk_id'] for chunk in processed] == ['first', 'second', 'third']


def test_score_chunk_lexical_returns_zero_for_empty_rare_terms() -> None:
	assert score_chunk_lexical('gamma-42 treatment', 'Gamma-42 treatment requires calibration.', []) == 0.0
	assert score_chunk_lexical('gamma-42 treatment', '', ['gamma-42']) == 0.0


def test_process_chunks_unified_respects_lexical_toggle(monkeypatch) -> None:
	monkeypatch.setenv('ENABLE_LEXICAL_BOOST', 'false')
	monkeypatch.delenv('LEXICAL_BOOST_WEIGHT', raising=False)
	query_param = QueryParam(chunk_top_k=2, enable_rerank=False)
	chunks = [
		{
			'chunk_id': 'generic',
			'content': 'The treatment protocol requires a cold start checklist.',
			'file_path': 'generic.md',
			'retrieval_score': 0.9,
		},
		{
			'chunk_id': 'rare',
			'content': 'Gamma-42 treatment requires a cold start checklist.',
			'file_path': 'rare.md',
			'retrieval_score': 0.9,
		},
	]

	processed = _run_process_chunks(
		query='What does gamma-42 treatment require?',
		chunks=chunks,
		query_param=query_param,
		topic_terms=['gamma-42'],
	)

	assert [chunk['chunk_id'] for chunk in processed] == ['generic', 'rare']
