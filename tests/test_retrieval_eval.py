"""Unit tests for retrieval-only metric helpers."""

from __future__ import annotations

import json

import pytest

from yar.evaluation.eval_rag_quality import RAGEvaluator, _references_from_chunks
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
        retrieval_only=True,
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

    evaluator = RAGEvaluator(test_dataset_path=dataset_path, retrieval_only=True, selected_case_numbers=[1])

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

    evaluator = RAGEvaluator(test_dataset_path=dataset_path, retrieval_only=True, selected_case_numbers=[1])

    payload = evaluator._build_query_payload(
        'question text', evaluator._load_test_dataset()[0], include_response_type=False
    )
    assert payload['mode'] == 'naive'


def test_build_query_payload_adds_evidence_span_grounding_to_generation_prompt(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': 'Is the plan acceptable?',
                        'answer': 'Expected answer',
                        'project': 'Thomas',
                        'source_documents': ['source.pdf'],
                    }
                ]
            }
        )
    )
    evaluator = RAGEvaluator(test_dataset_path=dataset_path, retrieval_only=True, selected_case_numbers=[1])
    test_case = evaluator._load_test_dataset()[0]

    payload = evaluator._build_query_payload(
        test_case['question'],
        test_case,
        include_response_type=True,
    )

    assert payload['response_type'] == 'Single Paragraph'
    assert 'extractive evidence spans' in payload['user_prompt']
    assert 'shortest exact context span(s)' in payload['user_prompt']
    assert 'preserve numbers, dates, table cells, row labels' in payload['user_prompt']
    grounding_phrase = (
        'answer only from claims that those spans or the surrounding chunk text directly support.'
    )
    assert grounding_phrase in payload['user_prompt']


def test_build_query_payload_keeps_retrieval_query_but_prompts_original_question(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    original_question = 'What are the japan-specific activities?'
    retrieval_query = 'Japanese iCMC Operations Managers handbook FMA J-CTD'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': original_question,
                        'answer': 'Expected answer',
                        'project': 'Sophie',
                        'source_documents': ['Japanese iCMC Operations Managers Dairy life Handbook_V1.0.pdf'],
                        'retrieval_query': retrieval_query,
                        'retrieval_mode': 'naive',
                    }
                ]
            }
        )
    )

    evaluator = RAGEvaluator(test_dataset_path=dataset_path, retrieval_only=True, selected_case_numbers=[1])
    test_case = evaluator._load_test_dataset()[0]

    payload = evaluator._build_query_payload(
        retrieval_query,
        test_case,
        include_response_type=True,
    )

    assert payload['query'] == retrieval_query
    original_question_prompt = (
        f'Answer the original user question, not the retrieval keywords: {original_question}'
    )
    assert original_question_prompt in payload['user_prompt']
    assert 'retrieval keywords' in payload['user_prompt']
    assert 'extractive evidence spans' in payload['user_prompt']


def test_build_query_payload_does_not_leak_ground_truth_or_context_reference(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': 'Is the strategy feasible?',
                        'answer': 'Secret expected answer',
                        'ground_truth': 'Secret ground truth',
                        'context_reference': 'Secret context reference',
                        'project': 'Thomas',
                        'source_documents': ['source.pdf'],
                    }
                ]
            }
        )
    )
    evaluator = RAGEvaluator(test_dataset_path=dataset_path, retrieval_only=True, selected_case_numbers=[1])
    test_case = evaluator._load_test_dataset()[0]

    payload = evaluator._build_query_payload(
        test_case['question'],
        test_case,
        include_response_type=True,
    )

    assert 'Secret expected answer' not in payload['user_prompt']
    assert 'Secret ground truth' not in payload['user_prompt']
    assert 'Secret context reference' not in payload['user_prompt']


def test_build_query_payload_does_not_add_generation_prompt_to_query_data_payload(tmp_path) -> None:
    dataset_path = tmp_path / 'qa_pairs.json'
    dataset_path.write_text(
        json.dumps(
            {
                'qa_pairs': [
                    {
                        'question': 'What is the shipment duration?',
                        'answer': 'Expected answer',
                        'project': 'Thomas',
                        'source_documents': ['source.pdf'],
                    }
                ]
            }
        )
    )
    evaluator = RAGEvaluator(test_dataset_path=dataset_path, retrieval_only=True, selected_case_numbers=[1])
    test_case = evaluator._load_test_dataset()[0]

    payload = evaluator._build_query_payload(
        test_case['question'],
        test_case,
        include_response_type=False,
    )

    assert 'user_prompt' not in payload
    assert 'response_type' not in payload


def test_references_from_chunks_prefers_partial_retrieval_query_evidence() -> None:
    references = _references_from_chunks(
        [
            {
                'file_path': 'unrelated.pdf',
                'content': 'Drug substance manufacturing takes ten months and release takes one month.',
            },
            {
                'file_path': 'workflow.pdf',
                'content': 'The standard duration is 1-3 months before Start packaging.',
            },
        ],
        focus_terms=['shipment to depot 1-3 months before Start packaging'],
        limit=2,
    )

    assert references[0]['file_path'] == 'workflow.pdf'
    assert set(references[0]) == {'reference_id', 'document_title', 'file_path', 'content'}


def test_references_from_chunks_preserves_schema_and_order_for_ties() -> None:
    references = _references_from_chunks(
        [
            {'file_path': 'first.pdf', 'content': 'same score'},
            {'file_path': 'second.pdf', 'content': 'same score'},
        ],
        focus_terms=['missing phrase'],
        limit=2,
    )

    assert set(references[0]) == {'reference_id', 'document_title', 'file_path', 'content'}
    assert [reference['file_path'] for reference in references] == ['first.pdf', 'second.pdf']
