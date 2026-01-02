"""Configuration for Entity Resolution

Uses the same LLM that LightRAG is configured with - no separate model config needed.

All entity resolution is LLM-based:
- Cache check first (instant, free)
- VDB similarity search for candidates
- LLM batch review for decisions
"""

from dataclasses import dataclass


@dataclass
class EntityResolutionConfig:
    """Configuration for the LLM-based entity resolution system.

    Raises:
        ValueError: If batch_size <= 0, candidates_per_entity <= 0,
                    or min_confidence is not in [0, 1].
    """

    # Whether entity resolution is enabled
    enabled: bool = True

    # Auto-resolve during extraction: When enabled, automatically resolve
    # entity aliases during document extraction/indexing
    auto_resolve_on_extraction: bool = True

    # Number of entities to review in a single LLM call
    # Larger = more efficient but may hit context limits
    batch_size: int = 20

    # Number of VDB candidates to retrieve per entity for LLM review
    # More candidates = better recall but more tokens
    # Increased from 5 to 10 to catch abbreviation-expansion pairs
    candidates_per_entity: int = 10

    # Minimum confidence for LLM to auto-apply an alias
    # Below this: alias is suggested but not auto-applied
    min_confidence: float = 0.85

    # Automatically apply LLM alias decisions
    # When True: Matching entities are merged automatically
    # When False: Aliases are stored but require manual verification
    auto_apply: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {self.batch_size}')
        if self.candidates_per_entity <= 0:
            raise ValueError(
                f'candidates_per_entity must be positive, got {self.candidates_per_entity}'
            )
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(
                f'min_confidence must be between 0 and 1, got {self.min_confidence}'
            )


# Default configuration
DEFAULT_CONFIG = EntityResolutionConfig()
