"""Unit tests for retrieval-only metric helpers."""

from __future__ import annotations

import pytest

from yar.evaluation.retrieval_eval import doc_hit_at_k, hit_at_k, mrr, recall_at_k


def test_hit_at_k_returns_true_when_gold_in_top_k() -> None:
    retrieved = ['a', 'b', 'c']
    gold = {'b'}

    assert hit_at_k(retrieved, gold, 2) is True


def test_hit_at_k_returns_false_when_gold_not_in_top_k() -> None:
    retrieved = ['a', 'b', 'c']
    gold = {'d'}

    assert hit_at_k(retrieved, gold, 3) is False


def test_hit_at_k_handles_empty_retrieved() -> None:
    assert hit_at_k([], {'a'}, 1) is False


def test_hit_at_k_handles_empty_gold() -> None:
    assert hit_at_k(['a'], set(), 1) is False


def test_recall_at_k_partial_match() -> None:
    retrieved = ['a', 'b']
    gold = {'a', 'c'}

    assert recall_at_k(retrieved, gold, 2) == pytest.approx(0.5)


def test_recall_at_k_full_match() -> None:
    retrieved = ['a', 'b']
    gold = {'a', 'b'}

    assert recall_at_k(retrieved, gold, 2) == pytest.approx(1.0)


def test_recall_at_k_zero_gold_returns_zero() -> None:
    assert recall_at_k(['a'], set(), 1) == pytest.approx(0.0)


def test_mrr_first_position() -> None:
    retrieved = ['a', 'b']
    gold = {'a'}

    assert mrr(retrieved, gold) == pytest.approx(1.0)


def test_mrr_third_position() -> None:
    retrieved = ['x', 'y', 'a', 'b']
    gold = {'a'}

    assert mrr(retrieved, gold) == pytest.approx(1 / 3)


def test_mrr_no_match_returns_zero() -> None:
    assert mrr(['x', 'y'], {'a'}) == pytest.approx(0.0)


def test_doc_hit_at_k_substring_match() -> None:
    retrieved_paths = ['s3://bucket/docs/alpha.md']
    gold = {'alpha.md'}

    assert doc_hit_at_k(retrieved_paths, gold, 1) is True


def test_metric_handles_duplicate_retrieved_ids() -> None:
    retrieved = ['a', 'a', 'b']
    gold = {'a'}

    assert hit_at_k(retrieved, gold, 1) is True
