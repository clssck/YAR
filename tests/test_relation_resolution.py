from __future__ import annotations

import json

import pytest

from yar.relation_resolution import llm_review_relation_predicates_batch


@pytest.mark.offline
@pytest.mark.asyncio
async def test_canonical_compound_predicate_with_mock_llm() -> None:
    captured_prompt = ''

    async def mock_llm(prompt: str, system_prompt: str | None = None) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        assert system_prompt is None
        return json.dumps(
            [
                {
                    'src': 'Clinical protocol',
                    'tgt': 'FDA',
                    'canonical_keywords': ['approves', 'reviews'],
                    'primary': 'approves',
                    'confidence': 0.91,
                    'reasoning': 'Evidence says the protocol was reviewed and approved.',
                }
            ]
        )

    results = await llm_review_relation_predicates_batch(
        [
            {
                'src': 'Clinical protocol',
                'tgt': 'FDA',
                'candidate_keywords': ['reviewed and approved', 'sent to'],
                'evidence_spans': ['The clinical protocol was reviewed and approved by FDA.'],
            }
        ],
        llm_func=mock_llm,
    )

    assert len(results) == 1
    result = results[0]
    assert result.src == 'Clinical protocol'
    assert result.tgt == 'FDA'
    assert result.original_keywords == ('reviews', 'approves', 'sends to')
    assert result.canonical_keywords == ('approves', 'reviews')
    assert result.primary == 'approves'
    assert result.confidence == 0.91
    assert 'The clinical protocol was reviewed and approved by FDA.' in captured_prompt
    assert '"allowed_keywords"' in captured_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_low_confidence_leaves_keywords_unchanged() -> None:
    async def mock_llm(prompt: str, system_prompt: str | None = None) -> str:
        return json.dumps(
            [
                {
                    'src': 'Clinical protocol',
                    'tgt': 'FDA',
                    'canonical_keywords': ['approves'],
                    'primary': 'approves',
                    'confidence': 0.41,
                    'reasoning': 'Approval is ambiguous.',
                }
            ]
        )

    results = await llm_review_relation_predicates_batch(
        [
            {
                'src': 'Clinical protocol',
                'tgt': 'FDA',
                'candidate_keywords': ['reviews', 'approves'],
                'evidence_spans': ['FDA reviewed the clinical protocol.'],
            }
        ],
        llm_func=mock_llm,
    )

    assert results[0].original_keywords == ('reviews', 'approves')
    assert results[0].canonical_keywords == ('reviews', 'approves')
    assert results[0].primary == 'reviews'
    assert results[0].confidence == 0.41


@pytest.mark.offline
@pytest.mark.asyncio
async def test_changed_endpoints_are_rejected_and_evidence_is_preserved_in_prompt() -> None:
    captured_prompt = ''

    async def mock_llm(prompt: str, system_prompt: str | None = None) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps(
            [
                {
                    'src': 'FDA',
                    'tgt': 'Clinical protocol',
                    'canonical_keywords': ['approves'],
                    'primary': 'approves',
                    'confidence': 0.99,
                    'reasoning': 'Changed endpoints should be rejected.',
                }
            ]
        )

    results = await llm_review_relation_predicates_batch(
        [
            {
                'src': 'Clinical protocol',
                'tgt': 'FDA',
                'candidate_keywords': ['reviews', 'approves'],
                'evidence_spans': ['Original evidence span must remain context only.'],
            }
        ],
        llm_func=mock_llm,
    )

    assert results[0].src == 'Clinical protocol'
    assert results[0].tgt == 'FDA'
    assert results[0].original_keywords == ('reviews', 'approves')
    assert results[0].canonical_keywords == ('reviews', 'approves')
    assert results[0].primary == 'reviews'
    assert 'Original evidence span must remain context only.' in captured_prompt
