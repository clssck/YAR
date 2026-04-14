"""Tests for yar.entity_resolution.config."""

from __future__ import annotations

import pytest

from yar.entity_resolution.config import EntityResolutionConfig


@pytest.mark.offline
class TestEntityResolutionConfig:
    """Tests for EntityResolutionConfig defaults and validation."""

    def test_default_batch_size_is_200(self) -> None:
        assert EntityResolutionConfig().batch_size == 200

    def test_default_candidates_per_entity_is_15(self) -> None:
        assert EntityResolutionConfig().candidates_per_entity == 15

    def test_custom_batch_size_overrides_default(self) -> None:
        assert EntityResolutionConfig(batch_size=50).batch_size == 50

    def test_validation_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValueError):
            EntityResolutionConfig(batch_size=0)

    def test_validation_rejects_negative_batch_size(self) -> None:
        with pytest.raises(ValueError):
            EntityResolutionConfig(batch_size=-1)

    def test_validation_rejects_zero_candidates(self) -> None:
        with pytest.raises(ValueError):
            EntityResolutionConfig(candidates_per_entity=0)

    def test_default_min_confidence_is_0_80(self) -> None:
        assert EntityResolutionConfig().min_confidence == 0.80

    def test_default_soft_match_threshold_is_0_70(self) -> None:
        assert EntityResolutionConfig().soft_match_threshold == 0.70

    def test_validation_rejects_soft_match_threshold_above_min_confidence(self) -> None:
        with pytest.raises(ValueError):
            EntityResolutionConfig(min_confidence=0.80, soft_match_threshold=0.81)
