"""Behavioral tests for analyze_query_intent's regex cascade.

The classifier routes broad query shapes to retrieval modes. Each fixture pins
one generic decision branch so regex changes do not silently shift mode routing.
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
        assert intent['kind'] == expected_kind, f'query={query!r} expected kind={expected_kind!r} got intent={intent!r}'

    def test_short_factual_query_falls_to_single_fact(self) -> None:
        # Very short queries without any other branch firing should land in single_fact,
        # but still allow two chunks from one document for split table/header evidence.
        intent = analyze_query_intent('What is mRNA?')
        assert intent['kind'] == 'single_fact'
        assert intent['per_document_limit'] == 2

    def test_six_token_query_falls_to_default(self) -> None:
        intent = analyze_query_intent('alpha beta gamma delta epsilon zeta')

        assert intent['kind'] == 'default'

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

    def test_lessons_learned_domain_phrase_does_not_force_enumeration(self) -> None:
        intent = analyze_query_intent(
            'How can retrospectives be applied to process strategy development '
            'based on lessons learned from the collaboration?'
        )

        assert intent['kind'] == 'source_reference'
        assert intent['recommended_mode'] == 'hybrid'

    def test_lessons_learned_category_query_still_enumerates(self) -> None:
        intent = analyze_query_intent('What are the 3 categories of lessons learned about chemistry?')

        assert intent['kind'] == 'enumeration'
        assert intent['recommended_mode'] == 'hybrid'

    def test_mabel_dose_range_query_routes_to_mix_evidence(self) -> None:
        intent = analyze_query_intent('What is the dose-ranging recommended by the MABEL approach?')

        assert intent['kind'] == 'single_fact'
        assert intent['recommended_mode'] == 'mix'

    def test_project_technology_issue_recommended_step_keeps_mix_mode(self) -> None:
        intent = analyze_query_intent('What is the first recommended step for resolving a technology issue?')

        assert intent['kind'] == 'mitigation'
        assert intent['recommended_mode'] == 'mix'

    def test_main_driver_query_routes_to_exact_chunk_lookup(self) -> None:
        intent = analyze_query_intent('What are the main drivers behind the device strategy proposal?')

        assert intent['kind'] == 'enumeration'
        assert intent['recommended_mode'] == 'naive'

    def test_domain_queries_follow_generic_intent_branches(self) -> None:
        issue_intent = analyze_query_intent(
            'What are the key issues identified in the implementation review, '
            'and how does the tool contribute to these challenges?'
        )
        source_intent = analyze_query_intent(
            'What is the significance of the approval timeline according to the product presentation?'
        )

        assert issue_intent['kind'] == 'enumeration'
        assert issue_intent['recommended_mode'] == 'naive'
        assert source_intent['kind'] == 'source_reference'
        assert source_intent['recommended_mode'] == 'hybrid'

    def test_project_impact_queries_keep_extra_chunks_per_document(self) -> None:
        intent = analyze_query_intent(
            'What is the significance of the approval timeline for Product X, '
            'and how does it impact project management?'
        )

        assert intent['kind'] == 'consequence'
        assert intent['recommended_chunk_limit'] == 8
        assert intent['per_document_limit'] == 3
        assert intent['allow_single_document_expansion'] is True
        assert intent['recommended_mode'] == 'mix'

    def test_significance_project_management_queries_expand_retrieval_budget(self) -> None:
        intent = analyze_query_intent(
            'What is the significance of the approval timeline for Product X in project management?'
        )

        assert intent['kind'] == 'consequence'
        assert intent['recommended_chunk_limit'] == 8
        assert intent['per_document_limit'] == 3
        assert intent['allow_single_document_expansion'] is True
        assert intent['recommended_mode'] == 'mix'

    def test_standalone_significance_queries_expand_retrieval_budget(self) -> None:
        intent = analyze_query_intent('What is the significance of the approval decision?')

        assert intent['kind'] == 'consequence'
        assert intent['recommended_chunk_limit'] == 6
        assert intent['per_document_limit'] == 2
        assert intent['allow_single_document_expansion'] is True
        assert intent['recommended_mode'] == 'mix'

    def test_prerequisite_before_and_since_do_not_force_comparison(self) -> None:
        prerequisite_intent = analyze_query_intent('What must be completed before releasing the package?')
        causal_intent = analyze_query_intent('What risks changed since the supplier review failed?')

        assert prerequisite_intent['kind'] != 'comparison'
        assert causal_intent['kind'] != 'comparison'

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
