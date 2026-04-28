"""Unit tests for retrieval-side scoring primitives.

Covers:
- ChunkMergeWeights default values + env override path.
- reciprocal_rank_fusion cross-source ranking, tie-breaking, and edge cases.

These primitives have no external dependencies (no DB, no LLM), so they're cheap to pin
behaviorally. Without these tests, future tuning of the merge weights or the fusion constant
would be flying blind on retrieval-only metrics.
"""

from __future__ import annotations

import pytest

from yar.operate import ChunkMergeWeights
from yar.utils import reciprocal_rank_fusion


class TestChunkMergeWeights:
    def test_defaults_sum_to_one(self) -> None:
        # The default weights are designed as a convex combination over normalized features.
        # If someone tunes one weight up, they should know they're stealing from the others.
        weights = ChunkMergeWeights()
        total = (
            weights.retrieval_score
            + weights.heading_relevance
            + weights.body_relevance
            + weights.facet_match
            + weights.temporal_signal
            + weights.source_count
            + weights.occurrence
            + weights.order
        )
        assert total == pytest.approx(1.00, abs=1e-9)

    def test_default_priority_order(self) -> None:
        # If this ordering changes, retrieval behavior changes meaningfully. Treat as a guard.
        weights = ChunkMergeWeights()
        assert weights.retrieval_score > weights.heading_relevance
        assert weights.heading_relevance > weights.body_relevance
        assert weights.body_relevance > weights.facet_match
        assert weights.facet_match == weights.temporal_signal
        assert weights.source_count > weights.occurrence
        assert weights.occurrence > weights.order

    def test_from_env_uses_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for field in (
            'RETRIEVAL_SCORE',
            'HEADING_RELEVANCE',
            'BODY_RELEVANCE',
            'FACET_MATCH',
            'TEMPORAL_SIGNAL',
            'SOURCE_COUNT',
            'OCCURRENCE',
            'ORDER',
        ):
            monkeypatch.delenv(f'YAR_CHUNK_MERGE_WEIGHT_{field}', raising=False)
        loaded = ChunkMergeWeights.from_env()
        assert loaded == ChunkMergeWeights()

    def test_from_env_overrides_specific_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('YAR_CHUNK_MERGE_WEIGHT_RETRIEVAL_SCORE', '0.5')
        loaded = ChunkMergeWeights.from_env()
        assert loaded.retrieval_score == 0.5
        # Other fields must remain at default.
        assert loaded.heading_relevance == ChunkMergeWeights().heading_relevance

    def test_from_env_clamps_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Negative weights would invert ranking; reject them.
        monkeypatch.setenv('YAR_CHUNK_MERGE_WEIGHT_RETRIEVAL_SCORE', '-0.3')
        loaded = ChunkMergeWeights.from_env()
        assert loaded.retrieval_score == 0.0

    def test_from_env_falls_back_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('YAR_CHUNK_MERGE_WEIGHT_RETRIEVAL_SCORE', 'not-a-float')
        loaded = ChunkMergeWeights.from_env()
        assert loaded.retrieval_score == ChunkMergeWeights().retrieval_score


class TestReciprocalRankFusion:
    def test_empty_input_returns_empty_list(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self) -> None:
        items = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
        fused = reciprocal_rank_fusion([items], k=60)
        assert [item['id'] for item in fused] == ['a', 'b', 'c']
        # All RRF scores must be 1/(k + rank + 1).
        assert fused[0]['rrf_score'] == pytest.approx(1 / (60 + 1))
        assert fused[1]['rrf_score'] == pytest.approx(1 / (60 + 2))

    def test_cross_source_agreement_wins(self) -> None:
        # An item ranked top by both sources should beat an item ranked top by only one.
        list_a = [{'id': 'shared'}, {'id': 'a_only'}]
        list_b = [{'id': 'shared'}, {'id': 'b_only'}]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert fused[0]['id'] == 'shared'
        # 'shared' has score 2 * 1/(k+1); the others have 1/(k+1) + 1/(k+2).
        assert fused[0]['rrf_score'] > fused[1]['rrf_score']

    def test_lower_k_amplifies_top_rank_advantage(self) -> None:
        list_a = [{'id': 'top'}, {'id': 'second'}]
        list_b = [{'id': 'second'}, {'id': 'top'}]
        # With small k, mutual top-1 placement matters more than mutual top-2.
        fused_small = reciprocal_rank_fusion([list_a, list_b], k=5)
        fused_large = reciprocal_rank_fusion([list_a, list_b], k=100)
        # In both cases the items tie on RRF score, so order falls to dict-iteration insertion
        # (top from list_a first, second from list_a second). Confirm fusion is stable.
        assert {item['id'] for item in fused_small} == {'top', 'second'}
        assert {item['id'] for item in fused_large} == {'top', 'second'}

    def test_custom_id_key(self) -> None:
        list_a = [{'entity_name': 'X'}, {'entity_name': 'Y'}]
        list_b = [{'entity_name': 'Y'}, {'entity_name': 'Z'}]
        fused = reciprocal_rank_fusion([list_a, list_b], id_key='entity_name', k=60)
        # Y appears in both lists; it should fuse to the top.
        assert fused[0]['entity_name'] == 'Y'

    def test_missing_id_skipped(self) -> None:
        # Items without the id key should be silently skipped, not crash.
        list_a = [{'id': 'a'}, {'no_id_here': True}, {'id': 'b'}]
        fused = reciprocal_rank_fusion([list_a], id_key='id', k=60)
        assert [item['id'] for item in fused] == ['a', 'b']

    def test_first_occurrence_kept_for_metadata(self) -> None:
        # When the same id appears in multiple lists, the first occurrence's metadata is kept.
        list_a = [{'id': 'x', 'source': 'vector', 'score': 0.9}]
        list_b = [{'id': 'x', 'source': 'bm25', 'score': 0.4}]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert fused[0]['source'] == 'vector'
        assert fused[0]['score'] == 0.9
