"""Tests for chunking configuration and preset handling.

This module tests:
- CHUNKING_PRESET validation in config.py
- create_chunker factory function in operate.py
- Chunking metadata storage in document processing
- API endpoint integration with chunking_preset parameter
- Error handling and edge cases
"""

import os
from typing import cast
from unittest.mock import patch

import pytest

from yar.document.semantic_chunker import count_tokens
from yar.operate import create_chunker
from yar.utils import TiktokenTokenizer


@pytest.mark.offline
class TestCreateChunker:
    """Tests for the semantic chunker factory."""

    def test_create_chunker_takes_no_preset(self):
        """The built-in chunker has no legacy preset selector."""
        signature = inspect.signature(create_chunker)

        assert list(signature.parameters) == []

    def test_returns_callable_adapter(self):
        """Factory returns a YAR-compatible chunking function."""
        chunker = create_chunker()

        assert callable(chunker)

    def test_chunks_text_with_required_fields(self):
        """Semantic chunks expose the metadata the pipeline consumes."""
        tokenizer = TiktokenTokenizer()
        chunker = create_chunker()
        content = '# Title\n\nThis is a paragraph.\n\n## Details\n\nMore content here.'

        result = chunker(tokenizer, content, None, False, 100, 1200)

        assert len(result) >= 1
        for index, chunk in enumerate(result):
            assert chunk['content'].strip()
            assert chunk['tokens'] == len(tokenizer.encode(chunk['content']))
            assert chunk['chunk_order_index'] == index
            assert 0 <= chunk['char_start'] <= chunk['char_end'] <= len(content)

    def test_empty_content_returns_single_empty_chunk(self):
        """Empty input preserves the pipeline's empty-chunk contract."""
        chunker = create_chunker()

        assert chunker(None, '', None, False, 100, 1200) == [
            {
                'tokens': 0,
                'content': '',
                'chunk_order_index': 0,
                'char_start': 0,
                'char_end': 0,
            }
        ]

    def test_respects_chunk_token_size(self):
        """Small chunk budgets split large documents."""
        chunker = create_chunker()
        content = ' '.join(['word'] * 1000)

        result = chunker(None, content, None, False, 50, 100)

        assert len(result) > 1
        assert all(chunk['tokens'] <= 200 for chunk in result)

    def test_exact_token_count_is_not_character_estimate(self):
        """Token counts use tiktoken, not character approximations."""
        tokenizer = TiktokenTokenizer()
        content = '😀' * 10 + ' This is a test sentence with exactly counted tokens.'

        assert count_tokens(content, tokenizer=tokenizer) == len(tokenizer.encode(content))
        assert count_tokens(content, tokenizer=tokenizer) != (len(content) + 3) // 4
