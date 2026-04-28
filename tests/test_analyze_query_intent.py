"""Behavioral tests for analyze_query_intent's regex cascade.

The classifier is a 200-line conditional with corpus-specific patterns. Without these
fixtures, an editor can change a single regex and silently shift mode routing for whole
classes of queries. Each test pins one decision branch.

We assert on the returned ``kind`` key. The recommended_chunk_limit / per_document_limit
fields are intentionally not asserted here; tests/test_lexical_boost.py and
tests/test_retrieval_scoring.py cover their downstream effect.
"""

from __future__ import annotations

import pytest

from yar.utils import analyze_query_intent


class TestAnalyzeQueryIntent:
    @pytest.mark.parametrize(
        ('query', 'expected_kind'),
        [
            # Consequence: causal/impact phrasing.
            ('What are the consequences of the failure?', 'consequence'),
            ('What is the impact on patient safety?', 'consequence'),
            # Mitigation: how-to-prevent phrasing.
            ('How do we mitigate the risk of contamination?', 'mitigation'),
            # Risk format: explicit risk template trigger.
            ('What is the phrasing of the risk in this template?', 'risk_format'),
            # Material lookup: literal 'link to the material' phrasing.
            ('Provide the link to the material for compound X', 'material_lookup'),
            # Document completeness: pharma-corpus completeness phrasing.
            (
                'Provide the full detail covering each section of the submission',
                'document_completeness',
            ),
            # Comparison: explicit compare phrasing.
            ('Compare drug A and drug B for efficacy and safety profiles in clinical use', 'comparison'),
            # Enumeration: list/items phrasing.
            ('List all the steps in the manufacturing process and how they connect', 'enumeration'),
        ],
    )
    def test_known_intent_branches(self, query: str, expected_kind: str) -> None:
        intent = analyze_query_intent(query)
        # Non-strict: the cascade may merge neighboring patterns. We only require that the
        # expected branch matches; more specific tests pin the harder boundaries.
        assert intent['kind'] == expected_kind, (
            f'query={query!r} expected kind={expected_kind!r} got intent={intent!r}'
        )

    def test_short_factual_query_falls_to_single_fact(self) -> None:
        # Short queries (<=7 tokens) without any other branch firing should land in single_fact,
        # not the default profile. This is what gives small queries a per-document cap.
        intent = analyze_query_intent('What is mRNA?')
        assert intent['kind'] == 'single_fact'
        assert intent['per_document_limit'] == 1

    def test_long_unmatched_query_falls_to_default(self) -> None:
        # A long query that matches no specific pattern should hit the default profile.
        # The default profile must always be returned even for free-form prose so that the
        # downstream per-document fallback (utils.process_chunks_unified) has a stable shape.
        intent = analyze_query_intent(
            'The quick brown fox jumps over the lazy dog and then continues running through '
            'the meadow toward the distant forest under the warm summer sun'
        )
        assert intent['kind'] == 'default'
        # Default profile fields the downstream code relies on.
        assert intent['per_document_limit'] == 0
        assert intent['recommended_chunk_limit'] == 0
        assert intent['allow_single_document_expansion'] is False
        assert intent['recommended_mode'] == 'mix'

    def test_choice_query_returns_single_fact(self) -> None:
        # 'A or B' style choice questions are handled by the single_fact branch, not enumeration.
        intent = analyze_query_intent('Which method should we use, method A or method B?')
        assert intent['kind'] == 'single_fact'

    def test_intent_profile_always_has_required_keys(self) -> None:
        # Every branch must return the same shape; downstream code reads these unconditionally.
        intent = analyze_query_intent('arbitrary query phrase')
        assert set(intent.keys()) >= {
            'kind',
            'recommended_chunk_limit',
            'per_document_limit',
            'allow_single_document_expansion',
            'recommended_mode',
        }

    def test_empty_query_returns_default_or_single_fact_shape(self) -> None:
        # An empty query must not crash; behavior may be either default or single_fact, but
        # the shape contract holds.
        intent = analyze_query_intent('')
        assert 'kind' in intent
        assert 'per_document_limit' in intent
