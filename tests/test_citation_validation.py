"""Tests for citation validation utility functions in lightrag/utils.py.

This module tests the citation validation and auto-correction functions:
- has_citation: Check for citation markers [n]
- is_factual_sentence: Determine if a sentence should have citations
- find_best_reference: Match sentences to relevant references
- insert_citation: Insert citation markers into sentences
- validate_and_fix_citations: Full validation and correction pipeline
"""

import pytest

from lightrag.utils import (
    find_best_reference,
    has_citation,
    insert_citation,
    is_factual_sentence,
    validate_and_fix_citations,
)

# ============================================================================
# Tests for has_citation()
# ============================================================================


class TestHasCitation:
    """Tests for has_citation function."""

    @pytest.mark.offline
    def test_single_citation(self):
        """Test detection of single citation."""
        assert has_citation('LightRAG uses PostgreSQL for storage [1].')

    @pytest.mark.offline
    def test_multiple_citations(self):
        """Test detection of multiple citations."""
        assert has_citation('Feature A [1] and Feature B [2] work together.')

    @pytest.mark.offline
    def test_double_digit_citation(self):
        """Test detection of double-digit citations."""
        assert has_citation('This is supported by source [15].')

    @pytest.mark.offline
    def test_no_citation(self):
        """Test text without citation returns False."""
        assert not has_citation('LightRAG is a RAG framework.')

    @pytest.mark.offline
    def test_similar_but_not_citation(self):
        """Test that similar patterns are not false positives."""
        assert not has_citation('Arrays use [index] notation.')
        assert not has_citation('The value is [unknown].')

    @pytest.mark.offline
    def test_empty_string(self):
        """Test empty string returns False."""
        assert not has_citation('')


# ============================================================================
# Tests for is_factual_sentence()
# ============================================================================


class TestIsFactualSentence:
    """Tests for is_factual_sentence function."""

    @pytest.mark.offline
    def test_factual_claim(self):
        """Test that factual claims are identified."""
        assert is_factual_sentence('LightRAG uses PostgreSQL for graph storage.')
        assert is_factual_sentence('The system supports dual-level retrieval.')

    @pytest.mark.offline
    def test_question_not_factual(self):
        """Test that questions are not considered factual."""
        assert not is_factual_sentence('What storage does LightRAG use?')
        assert not is_factual_sentence('Is this the correct approach?')

    @pytest.mark.offline
    def test_short_sentence_not_factual(self):
        """Test that very short sentences are not considered factual."""
        assert not is_factual_sentence('Yes.')
        assert not is_factual_sentence('See above.')
        assert not is_factual_sentence('It is true.')

    @pytest.mark.offline
    def test_meta_statements_not_factual(self):
        """Test that meta-statements are not considered factual."""
        assert not is_factual_sentence('Based on the context, LightRAG uses PostgreSQL.')
        assert not is_factual_sentence('According to the context, it supports graphs.')
        assert not is_factual_sentence('Here is the information you requested.')
        assert not is_factual_sentence("I can help you with that question.")
        assert not is_factual_sentence("In summary, the system works well.")

    @pytest.mark.offline
    def test_headers_not_factual(self):
        """Test that headers and list markers are not considered factual."""
        assert not is_factual_sentence('# Introduction to LightRAG')
        assert not is_factual_sentence('- LightRAG is a framework')
        assert not is_factual_sentence('* Another bullet point item')
        assert not is_factual_sentence('> This is a quote about RAG')

    @pytest.mark.offline
    def test_empty_and_whitespace(self):
        """Test handling of empty and whitespace strings."""
        assert not is_factual_sentence('')
        assert not is_factual_sentence('   ')
        assert not is_factual_sentence('\n\t')


# ============================================================================
# Tests for find_best_reference()
# ============================================================================


class TestFindBestReference:
    """Tests for find_best_reference function."""

    @pytest.fixture
    def sample_references(self):
        """Sample reference list for testing."""
        return [
            {
                'reference_id': '1',
                'document_title': 'LightRAG Architecture Guide',
                'excerpt': 'LightRAG uses PostgreSQL for graph storage and Neo4j for visualization.',
                'file_path': '/docs/architecture.pdf',
            },
            {
                'reference_id': '2',
                'document_title': 'Vector Database Integration',
                'excerpt': 'Vector similarity search enables semantic retrieval.',
                'file_path': '/docs/vectors.pdf',
            },
            {
                'reference_id': '3',
                'document_title': 'Query Processing',
                'excerpt': 'Query processing involves keyword extraction and context building.',
                'file_path': '/docs/queries.pdf',
            },
        ]

    @pytest.mark.offline
    def test_matches_by_keyword(self, sample_references):
        """Test that best reference matches by keyword overlap."""
        # Should match reference 1 (PostgreSQL, graph, storage)
        ref = find_best_reference(
            'The system stores graph data in PostgreSQL.',
            sample_references,
        )
        assert ref is not None
        assert ref['reference_id'] == '1'

    @pytest.mark.offline
    def test_matches_vector_content(self, sample_references):
        """Test matching vector-related content."""
        # Should match reference 2 (vector, similarity, semantic)
        ref = find_best_reference(
            'Semantic search uses vector similarity.',
            sample_references,
        )
        assert ref is not None
        assert ref['reference_id'] == '2'

    @pytest.mark.offline
    def test_empty_references(self):
        """Test with empty reference list."""
        ref = find_best_reference('Some sentence.', [])
        assert ref is None

    @pytest.mark.offline
    def test_no_meaningful_words(self, sample_references):
        """Test sentence with only stopwords returns first reference."""
        ref = find_best_reference('It is and the a.', sample_references)
        # Falls back to first reference when no meaningful words
        assert ref is not None
        assert ref['reference_id'] == '1'


# ============================================================================
# Tests for insert_citation()
# ============================================================================


class TestInsertCitation:
    """Tests for insert_citation function."""

    @pytest.mark.offline
    def test_insert_before_period(self):
        """Test citation is inserted before period."""
        result = insert_citation('LightRAG is a framework.', '1')
        assert result == 'LightRAG is a framework [1].'

    @pytest.mark.offline
    def test_insert_before_exclamation(self):
        """Test citation is inserted before exclamation mark."""
        result = insert_citation('This is amazing!', '2')
        assert result == 'This is amazing [2]!'

    @pytest.mark.offline
    def test_insert_before_question(self):
        """Test citation is inserted before question mark."""
        result = insert_citation('Is this correct?', '3')
        assert result == 'Is this correct [3]?'

    @pytest.mark.offline
    def test_no_punctuation(self):
        """Test citation is appended when no punctuation."""
        result = insert_citation('No punctuation here', '1')
        assert result == 'No punctuation here [1]'

    @pytest.mark.offline
    def test_trailing_whitespace(self):
        """Test handling of trailing whitespace."""
        result = insert_citation('Sentence with spaces.  ', '1')
        assert result == 'Sentence with spaces [1].'


# ============================================================================
# Tests for validate_and_fix_citations()
# ============================================================================


class TestValidateAndFixCitations:
    """Tests for validate_and_fix_citations function."""

    @pytest.fixture
    def sample_references(self):
        """Sample reference list for testing."""
        return [
            {
                'reference_id': '1',
                'document_title': 'LightRAG Documentation',
                'excerpt': 'LightRAG is a RAG framework with graph-based retrieval.',
                'file_path': '/docs/main.pdf',
            },
            {
                'reference_id': '2',
                'document_title': 'PostgreSQL Guide',
                'excerpt': 'PostgreSQL storage provides reliable data management.',
                'file_path': '/docs/postgres.pdf',
            },
        ]

    @pytest.mark.offline
    def test_adequate_citations_unchanged(self, sample_references):
        """Test that text with adequate citations is unchanged."""
        text = 'LightRAG is a RAG framework [1]. It uses PostgreSQL [2].'
        result, was_modified = validate_and_fix_citations(text, sample_references)
        assert not was_modified
        assert result == text

    @pytest.mark.offline
    def test_short_answer_gets_citation(self, sample_references):
        """Test that short uncited answers get citations. This is the key failure mode."""
        text = 'LightRAG is a RAG framework for graph-based retrieval.'
        result, was_modified = validate_and_fix_citations(text, sample_references)
        assert was_modified
        assert '[1]' in result

    @pytest.mark.offline
    def test_partial_citations_fixed(self, sample_references):
        """Test that partially cited text gets additional citations."""
        text = 'LightRAG uses graphs [1]. PostgreSQL handles data storage.'
        result, was_modified = validate_and_fix_citations(
            text, sample_references, min_coverage=0.9
        )
        # Should try to add citation to second sentence
        assert was_modified
        assert result.count('[') >= 2

    @pytest.mark.offline
    def test_preserves_references_section(self, sample_references):
        """Test that the References section is preserved unchanged."""
        text = '''LightRAG is a framework.

### References

- [1] LightRAG Documentation
- [2] PostgreSQL Guide'''

        result, _was_modified = validate_and_fix_citations(text, sample_references)
        # References section should be preserved exactly
        assert '### References' in result
        assert '- [1] LightRAG Documentation' in result
        assert '- [2] PostgreSQL Guide' in result

    @pytest.mark.offline
    def test_empty_response(self, sample_references):
        """Test handling of empty response."""
        result, was_modified = validate_and_fix_citations('', sample_references)
        assert not was_modified
        assert result == ''

    @pytest.mark.offline
    def test_empty_references(self):
        """Test handling of empty references list."""
        text = 'LightRAG is a framework.'
        result, was_modified = validate_and_fix_citations(text, [])
        assert not was_modified
        assert result == text

    @pytest.mark.offline
    def test_auto_fix_disabled(self, sample_references):
        """Test that auto-fix can be disabled."""
        text = 'LightRAG is a RAG framework for graph-based retrieval.'
        result, was_modified = validate_and_fix_citations(
            text, sample_references, enable_auto_fix=False
        )
        assert not was_modified
        assert result == text  # Unchanged

    @pytest.mark.offline
    def test_questions_not_cited(self, sample_references):
        """Test that questions are not auto-cited."""
        text = 'What does LightRAG do?'
        _result, was_modified = validate_and_fix_citations(text, sample_references)
        # Questions should not be considered factual sentences
        assert not was_modified

    @pytest.mark.offline
    def test_coverage_threshold(self, sample_references):
        """Test coverage threshold affects behavior."""
        text = 'LightRAG is a framework [1]. It supports graphs. PostgreSQL stores data.'
        # 1/3 = 33% coverage

        # With high threshold, should try to fix
        _result, was_modified = validate_and_fix_citations(
            text, sample_references, min_coverage=0.9
        )
        assert was_modified

        # With low threshold, should not fix
        _result, was_modified = validate_and_fix_citations(
            text, sample_references, min_coverage=0.3
        )
        assert not was_modified
