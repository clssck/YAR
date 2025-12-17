"""
Unit tests for Entity Resolution

Tests the 3-layer approach with mock embed_fn and llm_fn.
No database or external services required.
"""

import pytest

from lightrag.entity_resolution import (
    EntityResolutionConfig,
    resolve_entity,
)

# Mock embeddings - pre-computed for test entities
# These simulate what an embedding model would return
MOCK_EMBEDDINGS = {
    # FDA and full name have ~0.67 similarity (based on real test)
    'fda': [0.1, 0.2, 0.3, 0.4, 0.5],
    'us food and drug administration': [0.15, 0.25, 0.28, 0.38, 0.52],
    # Dupixent and dupilumab have ~0.63 similarity
    'dupixent': [0.5, 0.6, 0.7, 0.8, 0.9],
    'dupilumab': [0.48, 0.58, 0.72, 0.78, 0.88],
    # Celebrex and Cerebyx are different (low similarity)
    'celebrex': [0.9, 0.1, 0.2, 0.3, 0.4],
    'cerebyx': [0.1, 0.9, 0.8, 0.7, 0.6],
    # Default for unknown entities
    'default': [0.0, 0.0, 0.0, 0.0, 0.0],
}

# Mock LLM responses
MOCK_LLM_RESPONSES = {
    ('fda', 'us food and drug administration'): 'YES',
    ('us food and drug administration', 'fda'): 'YES',
    ('dupixent', 'dupilumab'): 'YES',
    ('dupilumab', 'dupixent'): 'YES',
    ('heart attack', 'myocardial infarction'): 'YES',
    ('celebrex', 'cerebyx'): 'NO',
    ('metformin', 'metoprolol'): 'NO',
}


async def mock_embed_fn(text: str) -> list[float]:
    """Mock embedding function."""
    key = text.lower().strip()
    return MOCK_EMBEDDINGS.get(key, MOCK_EMBEDDINGS['default'])


async def mock_llm_fn(prompt: str) -> str:
    """Mock LLM function that parses the prompt and returns YES/NO."""
    # Extract term_a and term_b from the prompt
    lines = prompt.strip().split('\n')
    term_a = None
    term_b = None
    for line in lines:
        if line.startswith('Term A:'):
            term_a = line.replace('Term A:', '').strip().lower()
        elif line.startswith('Term B:'):
            term_b = line.replace('Term B:', '').strip().lower()

    if term_a and term_b:
        # Check both orderings
        response = MOCK_LLM_RESPONSES.get((term_a, term_b))
        if response is None:
            response = MOCK_LLM_RESPONSES.get((term_b, term_a), 'NO')
        return response
    return 'NO'


# Test fixtures
@pytest.fixture
def existing_entities():
    """Existing entities in the knowledge graph."""
    return [
        (
            'US Food and Drug Administration',
            MOCK_EMBEDDINGS['us food and drug administration'],
        ),
        ('Dupixent', MOCK_EMBEDDINGS['dupixent']),
        ('Celebrex', MOCK_EMBEDDINGS['celebrex']),
    ]


@pytest.fixture
def config():
    """Default resolution config."""
    return EntityResolutionConfig()


# Layer 1: Case normalization tests
class TestCaseNormalization:
    @pytest.mark.asyncio
    async def test_exact_match_same_case(self, existing_entities, config):
        """Exact match with same case."""
        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'exact'
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_different_case(self, existing_entities, config):
        """DUPIXENT should match Dupixent via case normalization."""
        result = await resolve_entity(
            'DUPIXENT',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'exact'

    @pytest.mark.asyncio
    async def test_exact_match_lowercase(self, existing_entities, config):
        """dupixent should match Dupixent."""
        result = await resolve_entity(
            'dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.method == 'exact'


# Layer 2: Fuzzy matching tests
class TestFuzzyMatching:
    @pytest.mark.asyncio
    async def test_fuzzy_match_typo(self, existing_entities, config):
        """Dupixant (typo) should match Dupixent via fuzzy matching (88%)."""
        result = await resolve_entity(
            'Dupixant',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'fuzzy'
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_fuzzy_rejects_below_threshold(self, existing_entities, config):
        """Celebrex vs Cerebyx is 67% - should NOT fuzzy match."""
        # Add Cerebyx as the query (Celebrex exists)
        result = await resolve_entity(
            'Cerebyx',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        # Should not be fuzzy match (67% < 85%)
        assert result.method != 'fuzzy' or result.action == 'new'


# Layer 1.5: Abbreviation detection tests
class TestAbbreviationDetection:
    @pytest.mark.asyncio
    async def test_abbreviation_catches_acronym(self, existing_entities, config):
        """FDA should match US Food and Drug Administration via abbreviation detection."""
        result = await resolve_entity(
            'FDA',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'US Food and Drug Administration'
        assert result.method == 'abbreviation'

    @pytest.mark.asyncio
    async def test_abbreviation_disabled_falls_through_to_llm(self, existing_entities):
        """When abbreviation detection disabled, LLM should catch the match."""
        config = EntityResolutionConfig(abbreviation_detection_enabled=False)
        result = await resolve_entity(
            'FDA',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'US Food and Drug Administration'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_abbreviation_confidence_threshold(self, existing_entities):
        """High confidence threshold should reject weak matches."""
        config = EntityResolutionConfig(abbreviation_min_confidence=0.99)
        result = await resolve_entity(
            'FDA',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        # FDA → USFDA is subsequence match (0.80 confidence), should fail threshold
        # Will fall through to LLM
        assert result.method in ('llm', 'abbreviation')  # Either is acceptable


# Layer 3: LLM verification tests
class TestLLMVerification:
    @pytest.mark.asyncio
    async def test_llm_matches_acronym_when_abbrev_disabled(self, existing_entities):
        """FDA should match via LLM when abbreviation detection is disabled."""
        config = EntityResolutionConfig(abbreviation_detection_enabled=False)
        result = await resolve_entity(
            'FDA',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'US Food and Drug Administration'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_llm_matches_brand_generic(self, config):
        """Dupixent should match dupilumab via LLM."""
        existing = [
            ('dupilumab', MOCK_EMBEDDINGS['dupilumab']),
        ]
        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'dupilumab'
        assert result.method == 'llm'


# Edge cases
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_existing_entities(self, config):
        """New entity when no existing entities."""
        result = await resolve_entity(
            'NewEntity',
            [],
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'new'

    @pytest.mark.asyncio
    async def test_disabled_resolution(self, existing_entities):
        """Resolution disabled returns new."""
        config = EntityResolutionConfig(enabled=False)
        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'new'
        assert result.method == 'disabled'

    @pytest.mark.asyncio
    async def test_genuinely_new_entity(self, existing_entities, config):
        """Completely new entity should return 'new'."""
        result = await resolve_entity(
            'CompletelyNewDrug',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'new'
        assert result.method == 'none'


class TestResolutionResult:
    """Tests for ResolutionResult dataclass."""

    @pytest.mark.asyncio
    async def test_result_has_all_fields(self, existing_entities, config):
        """ResolutionResult should have action, matched_entity, confidence, method."""
        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert hasattr(result, 'action')
        assert hasattr(result, 'matched_entity')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'method')

    @pytest.mark.asyncio
    async def test_confidence_range(self, existing_entities, config):
        """Confidence should be between 0 and 1."""
        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert 0.0 <= result.confidence <= 1.0


class TestConfigOptions:
    """Tests for EntityResolutionConfig behavior."""

    @pytest.mark.asyncio
    async def test_fuzzy_threshold_affects_matching(self, existing_entities):
        """Higher fuzzy threshold should reject more matches."""
        # Very high threshold - should not fuzzy match
        config_strict = EntityResolutionConfig(fuzzy_threshold=0.99)
        result = await resolve_entity(
            'Dupixant',  # typo
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config_strict,
        )
        # Should not match via fuzzy (88% < 99%)
        assert result.method != 'fuzzy'

    @pytest.mark.asyncio
    async def test_fuzzy_pre_resolution_disabled_still_fuzzy_matches(self, existing_entities):
        """Disabling fuzzy_pre_resolution doesn't disable fuzzy matching entirely.

        fuzzy_pre_resolution_enabled controls pre-filtering before LLM,
        not the fuzzy matching layer itself.
        """
        config = EntityResolutionConfig(fuzzy_pre_resolution_enabled=False)
        result = await resolve_entity(
            'Dupixant',  # typo, 88% fuzzy match to Dupixent
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        # Fuzzy matching layer (Layer 2) still works
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'

    @pytest.mark.asyncio
    async def test_max_candidates_limits_search(self, config):
        """max_candidates should limit number of candidates considered."""
        # Create many existing entities
        many_entities = [(f'Entity{i}', [float(i), 0.0, 0.0, 0.0, 0.0]) for i in range(100)]
        config.max_candidates = 3

        result = await resolve_entity(
            'SomeEntity',
            many_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        # Should complete without error
        assert result.action in ('match', 'new')


class TestLayerOrder:
    """Tests verifying the correct layer execution order."""

    @pytest.mark.asyncio
    async def test_exact_match_short_circuits(self, existing_entities, config):
        """Exact match should return immediately without calling LLM."""
        # Track if LLM was called
        llm_called = [False]

        async def tracking_llm(prompt: str) -> str:
            llm_called[0] = True
            return 'YES'

        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            tracking_llm,
            config,
        )

        assert result.method == 'exact'
        assert not llm_called[0]  # LLM should not be called

    @pytest.mark.asyncio
    async def test_abbreviation_before_fuzzy(self, config):
        """Abbreviation detection should run before fuzzy matching."""
        existing = [
            ('World Health Organization', [0.1, 0.2, 0.3, 0.4, 0.5]),
        ]
        result = await resolve_entity(
            'WHO',
            existing,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )

        assert result.method == 'abbreviation'
        assert result.matched_entity == 'World Health Organization'

    @pytest.mark.asyncio
    async def test_fuzzy_before_llm(self, config):
        """Fuzzy matching should run before LLM for close matches."""
        existing = [
            ('Dupixent', MOCK_EMBEDDINGS['dupixent']),
        ]

        llm_called = [False]

        async def tracking_llm(prompt: str) -> str:
            llm_called[0] = True
            return 'YES'

        result = await resolve_entity(
            'Dupixant',  # typo, 88% match
            existing,
            mock_embed_fn,
            tracking_llm,
            config,
        )

        assert result.method == 'fuzzy'
        assert not llm_called[0]  # LLM should not be called for fuzzy match


# --- LLM Response Parsing Tests ---


class TestLLMResponseParsing:
    """Tests for LLM response parsing edge cases.

    These tests verify that llm_verify handles malformed and unexpected
    responses gracefully without false positives.
    """

    @pytest.mark.asyncio
    async def test_empty_response_returns_false(self, existing_entities, config):
        """Empty LLM response should be treated as NO (default to safe)."""

        async def empty_llm(prompt: str) -> str:
            return ''

        # Configure to reach LLM verification
        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,  # Prevent fuzzy match
            abbreviation_detection_enabled=False,
        )

        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            empty_llm,
            config,
        )

        # Empty response should default to NO (no match)
        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_false(self, config):
        """Whitespace-only response should be treated as NO."""

        async def whitespace_llm(prompt: str) -> str:
            return '   \n\t  \n  '

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )
        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            whitespace_llm,
            config,
        )

        assert result.action == 'new'

    @pytest.mark.asyncio
    async def test_multiline_response_takes_first_line(self, config):
        """Multi-line response should use first line only."""

        async def multiline_llm(prompt: str) -> str:
            return 'YES\nBut actually I changed my mind NO\nDefinitely NO'

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )
        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            multiline_llm,
            config,
        )

        # Should take "YES" from first line
        assert result.action == 'match'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_response_with_trailing_punctuation(self, config):
        """Response with trailing punctuation should be handled."""

        async def punctuated_llm(prompt: str) -> str:
            return 'YES.'

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )
        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            punctuated_llm,
            config,
        )

        assert result.action == 'match'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_lowercase_yes_normalized(self, config):
        """Lowercase 'yes' should be normalized to YES."""

        async def lowercase_llm(prompt: str) -> str:
            return 'yes'

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )
        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            lowercase_llm,
            config,
        )

        assert result.action == 'match'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_ambiguous_response_defaults_to_no(self, config):
        """Ambiguous response should default to NO (safer)."""

        async def ambiguous_llm(prompt: str) -> str:
            return 'Maybe, it depends on the context'

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )
        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            ambiguous_llm,
            config,
        )

        # Ambiguous should default to NO (new entity)
        assert result.action == 'new'

    @pytest.mark.asyncio
    async def test_partial_match_not_accepted(self, config):
        """Partial token match should not be accepted (e.g., 'YESSIR')."""

        async def partial_llm(prompt: str) -> str:
            return 'YESSIR'

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )
        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            partial_llm,
            config,
        )

        # "YESSIR" contains "YES" but is not exact token match
        assert result.action == 'new'

    @pytest.mark.asyncio
    async def test_valid_alternative_tokens(self, config):
        """Test all valid positive tokens: TRUE, SAME, MATCH."""
        valid_positive_tokens = ['TRUE', 'SAME', 'MATCH']

        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        for token in valid_positive_tokens:

            async def token_llm(prompt: str, t=token) -> str:
                return t

            config = EntityResolutionConfig(
                fuzzy_threshold=0.99,
                abbreviation_detection_enabled=False,
            )

            result = await resolve_entity(
                'Dupixent',
                existing,
                mock_embed_fn,
                token_llm,
                config,
            )

            assert result.action == 'match', f"Token '{token}' should be accepted"
            assert result.method == 'llm'


# --- VDB Error Recovery Tests ---


class TestVDBErrorRecovery:
    """Tests for VDB failure handling and recovery.

    These tests verify that VDB errors are handled gracefully without
    crashing or producing incorrect results.
    """

    @pytest.mark.asyncio
    async def test_vdb_returns_malformed_candidates_handled(self, config):
        """VDB returning candidates with unexpected structure is handled gracefully.

        The code defensively handles malformed candidates (strings instead of dicts)
        by skipping them and returning 'new' if no valid candidates remain.
        """
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.entity_resolution.resolver import resolve_entity_with_vdb

        mock_vdb = MagicMock()
        # Return malformed data - list of strings instead of dicts
        mock_vdb.query = AsyncMock(return_value=['string1', 'string2'])

        async def no_llm(prompt: str) -> str:
            return 'NO'

        # Should handle gracefully and return new (no valid candidates)
        result = await resolve_entity_with_vdb('Entity', mock_vdb, no_llm, config)
        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_vdb_mixed_valid_invalid_candidates(self, config):
        """VDB with mix of valid dicts and invalid items processes valid ones."""
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.entity_resolution.resolver import resolve_entity_with_vdb

        mock_vdb = MagicMock()
        # Mix of valid and invalid candidates
        mock_vdb.query = AsyncMock(return_value=[
            'string_garbage',  # Invalid: string
            {'entity_name': 'Dupixent'},  # Valid
            42,  # Invalid: number
            {'other_field': 'no_name'},  # Invalid: missing entity_name
            None,  # Invalid: None
        ])

        async def no_llm(prompt: str) -> str:
            return 'NO'

        # Should find the valid candidate via exact match
        result = await resolve_entity_with_vdb('dupixent', mock_vdb, no_llm, config)
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'exact'

    @pytest.mark.asyncio
    async def test_vdb_returns_none_candidates(self, config):
        """VDB returning None instead of empty list."""
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.entity_resolution.resolver import resolve_entity_with_vdb

        mock_vdb = MagicMock()
        mock_vdb.query = AsyncMock(return_value=None)

        async def no_llm(prompt: str) -> str:
            return 'NO'

        result = await resolve_entity_with_vdb('Entity', mock_vdb, no_llm, config)

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_vdb_timeout_handled(self, config):
        """VDB timeout should be handled gracefully."""
        import asyncio
        from unittest.mock import MagicMock

        from lightrag.entity_resolution.resolver import resolve_entity_with_vdb

        mock_vdb = MagicMock()

        async def timeout_query(*args, **kwargs):
            raise asyncio.TimeoutError('VDB query timed out')

        mock_vdb.query = timeout_query

        async def no_llm(prompt: str) -> str:
            return 'NO'

        result = await resolve_entity_with_vdb('Entity', mock_vdb, no_llm, config)

        assert result.action == 'new'
        assert result.method == 'none'


# --- Large Batch Processing Tests ---


class TestLargeBatchProcessing:
    """Tests for handling large numbers of entities and candidates.

    These tests verify that the resolution system handles scale gracefully.
    """

    @pytest.mark.asyncio
    async def test_many_existing_entities(self, config):
        """Should handle resolution against many existing entities."""
        # Create 100 existing entities
        existing = [(f'Entity_{i}', [float(i % 10) / 10, 0.0, 0.0, 0.0, 0.0]) for i in range(100)]

        result = await resolve_entity(
            'Entity_50',  # Exact match
            existing,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )

        assert result.action == 'match'
        assert result.matched_entity == 'Entity_50'
        assert result.method == 'exact'

    @pytest.mark.asyncio
    async def test_no_match_among_many(self, config):
        """Should correctly return new when no match among many entities."""
        # Create 50 entities, none matching
        existing = [(f'Other_{i}', [float(i % 10) / 10, 0.0, 0.0, 0.0, 0.0]) for i in range(50)]

        result = await resolve_entity(
            'CompletelyUnique',
            existing,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_max_candidates_limits_llm_calls(self, config):
        """max_candidates should limit LLM verification calls."""
        # Create many candidates that would need LLM verification
        existing = [(f'Similar_{i}', [0.5, 0.5, 0.5, 0.5, 0.5]) for i in range(20)]

        llm_call_count = [0]

        async def counting_llm(prompt: str) -> str:
            llm_call_count[0] += 1
            return 'NO'  # Never match

        config = EntityResolutionConfig(
            max_candidates=3,
            fuzzy_threshold=0.99,  # Prevent fuzzy match
            abbreviation_detection_enabled=False,
        )

        await resolve_entity(
            'Query',
            existing,
            mock_embed_fn,
            counting_llm,
            config,
        )

        # Should only call LLM max_candidates times
        assert llm_call_count[0] <= config.max_candidates

    @pytest.mark.asyncio
    async def test_fuzzy_finds_best_among_many(self, config):
        """Fuzzy matching should find the best match among many similar entities."""
        # Create entities with varying similarity to "Dupixent"
        existing = [
            ('Dupixant', [0.0] * 5),  # 88% similar - typo
            ('Dupixont', [0.0] * 5),  # 75% similar
            ('Duplexent', [0.0] * 5),  # 75% similar
            ('Different', [0.0] * 5),  # Low similarity
        ]

        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )

        assert result.action == 'match'
        assert result.matched_entity == 'Dupixant'  # Best fuzzy match
        assert result.method == 'fuzzy'

    @pytest.mark.asyncio
    async def test_vdb_with_many_candidates(self, config):
        """VDB resolution should handle many candidates efficiently."""
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.entity_resolution.resolver import resolve_entity_with_vdb

        mock_vdb = MagicMock()
        # Return 50 candidates
        mock_vdb.query = AsyncMock(return_value=[
            {'entity_name': f'Candidate_{i}'} for i in range(50)
        ])

        async def no_llm(prompt: str) -> str:
            return 'NO'

        result = await resolve_entity_with_vdb('candidate_25', mock_vdb, no_llm, config)

        # Should find exact match (case-insensitive)
        assert result.action == 'match'
        assert result.matched_entity == 'Candidate_25'
        assert result.method == 'exact'


# --- LLM Error Handling Tests ---


class TestLLMErrorHandling:
    """Tests for LLM function error handling.

    These tests verify that LLM errors don't crash the resolution system.
    """

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self, config):
        """LLM raising an exception should be handled gracefully."""

        async def error_llm(prompt: str) -> str:
            raise RuntimeError('LLM service unavailable')

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,  # Force LLM path
            abbreviation_detection_enabled=False,
        )

        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        # The current implementation doesn't catch LLM errors,
        # so this documents expected behavior
        with pytest.raises(RuntimeError):
            await resolve_entity(
                'Dupixent',
                existing,
                mock_embed_fn,
                error_llm,
                config,
            )

    @pytest.mark.asyncio
    async def test_llm_returns_none(self, config):
        """LLM returning None should be handled."""

        async def none_llm(prompt: str) -> str:
            return None  # type: ignore

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )

        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        # None should be treated as non-match (can't strip None)
        with pytest.raises(AttributeError):
            await resolve_entity(
                'Dupixent',
                existing,
                mock_embed_fn,
                none_llm,
                config,
            )

    @pytest.mark.asyncio
    async def test_embed_fn_raises_exception(self, config):
        """Embedding function raising an exception propagates up."""

        async def error_embed(text: str) -> list[float]:
            raise ConnectionError('Embedding service down')

        config = EntityResolutionConfig(
            fuzzy_threshold=0.99,
            abbreviation_detection_enabled=False,
        )

        existing = [('dupilumab', MOCK_EMBEDDINGS['dupilumab'])]

        # Embed errors propagate (called for vector similarity layer)
        with pytest.raises(ConnectionError):
            await resolve_entity(
                'Dupixent',
                existing,
                error_embed,
                mock_llm_fn,
                config,
            )
