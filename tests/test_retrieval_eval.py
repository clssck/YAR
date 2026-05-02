"""Unit tests for retrieval-only metric helpers."""

from __future__ import annotations

import json

import pytest

from yar.evaluation.eval_rag_quality import RAGEvaluator
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


def test_case_mode_override_takes_precedence_over_dataset_retrieval_mode(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': 'What are the japan-specific activities?',
                        'answer': 'Expected answer',
                        'project': 'Sophie',
                        'source_documents': ['Japanese iCMC Operations Managers Dairy life Handbook_V1.0.pdf'],
                        'retrieval_mode': 'naive',
                    }
                ]
            }
        )
    )

    evaluator = RAGEvaluator(
        test_dataset_path=dataset_path,
        selected_case_numbers=[1],
        case_mode_overrides={1: 'hybrid'},
    )

    assert evaluator._load_test_dataset()[0]['mode'] == 'hybrid'
    assert 'retrieval_mode' not in evaluator._load_test_dataset()[0]

    payload = evaluator._build_query_payload(
        'question text', evaluator._load_test_dataset()[0], include_response_type=False
    )
    assert payload['mode'] == 'hybrid'


def test_dataset_retrieval_mode_applies_without_case_mode_override(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': 'What are the japan-specific activities?',
                        'answer': 'Expected answer',
                        'project': 'Sophie',
                        'source_documents': ['Japanese iCMC Operations Managers Dairy life Handbook_V1.0.pdf'],
                        'retrieval_mode': 'naive',
                    }
                ]
            }
        )
    )

    evaluator = RAGEvaluator(test_dataset_path=dataset_path, selected_case_numbers=[1])

    assert evaluator._load_test_dataset()[0]['retrieval_mode'] == 'naive'


def test_build_query_payload_uses_dataset_retrieval_mode_without_case_override(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': 'What are the japan-specific activities?',
                        'answer': 'Expected answer',
                        'project': 'Sophie',
                        'source_documents': ['Japanese iCMC Operations Managers Dairy life Handbook_V1.0.pdf'],
                        'retrieval_mode': 'naive',
                    }
                ]
            }
        )
    )

    evaluator = RAGEvaluator(test_dataset_path=dataset_path, selected_case_numbers=[1])

    payload = evaluator._build_query_payload(
        'question text', evaluator._load_test_dataset()[0], include_response_type=False
    )
    assert payload['mode'] == 'naive'
