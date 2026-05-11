"""Configuration for relation predicate review."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelationResolutionConfig:
    """Configuration for LLM relation predicate review."""

    enabled: bool = True
    max_predicates_per_pair: int = 3
    min_keywords_for_review: int = 2
    confidence_threshold: float = 0.6

    def __post_init__(self) -> None:
        """Validate relation resolution configuration values."""
        _validate_positive_int('max_predicates_per_pair', self.max_predicates_per_pair)
        _validate_positive_int('min_keywords_for_review', self.min_keywords_for_review)
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f'confidence_threshold must be between 0 and 1, got {self.confidence_threshold}'
            )


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value}')


DEFAULT_CONFIG = RelationResolutionConfig()
