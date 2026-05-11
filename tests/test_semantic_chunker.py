from __future__ import annotations

import pytest

from yar.document.semantic_chunker import chunk_markdown, count_tokens
from yar.document.vision_adapter import (
    NO_TEXT_DETECTED_SENTINEL,
    should_split_batch,
    split_batch_response,
)
from yar.utils import TiktokenTokenizer


class CountingWhitespaceTokenizer:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, content: str) -> list[int]:
        self.encode_calls += 1
        return list(range(len(content.split())))

    def decode(self, tokens: list[int]) -> str:
        return ' '.join(str(token) for token in tokens)


class TiktokenWrapper:
    def __init__(self):
        self.inner = TiktokenTokenizer()

    def encode(self, content: str) -> list[int]:
        return self.inner.encode(content)

    def decode(self, tokens: list[int]) -> str:
        return self.inner.decode(tokens)


def _token_count(text: str) -> int:
    return count_tokens(text)


@pytest.mark.offline
class TestSemanticChunker:
    @pytest.mark.parametrize('markdown', ['', '   \n\n  '])
    def test_empty_input_returns_no_chunks(self, markdown: str):
        assert chunk_markdown(markdown) == []

    def test_single_heading_body_returns_one_chunk_with_context(self):
        chunks = chunk_markdown('# Title\n\nSome body text')

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.heading_context == 'Title'
        assert chunk.page_number is None
        assert chunk.page_start is None
        assert chunk.page_end is None
        assert chunk.page_numbers == ()
        assert chunk.chunk_index == 0
        assert chunk.content.startswith('# Title')
        assert 'Some body text' in chunk.content

    def test_page_markers_are_preserved_and_extracted_from_marker_bearing_chunks(self):
        markdown = '<!-- PAGE 1 -->\n\n# Title\n\nBody\n\n<!-- PAGE 3 -->\n\n## Subtitle\n\nMore'

        chunks = chunk_markdown(markdown, join_threshold=10)

        assert [chunk.page_number for chunk in chunks] == [1, 3]
        assert [chunk.page_start for chunk in chunks] == [1, 3]
        assert [chunk.page_end for chunk in chunks] == [1, 3]
        assert [chunk.page_numbers for chunk in chunks] == [(1,), (3,)]
        assert '<!-- PAGE 1 -->' in chunks[0].content
        assert '<!-- PAGE 3 -->' in chunks[1].content
        assert chunks[1].heading_context == 'Title > Subtitle'

    def test_heading_hierarchy_builds_context_chain(self):
        chunks = chunk_markdown('# Top\n\nA\n\n## Sub\n\nB\n\n### Deep\n\nC')

        assert [chunk.heading_context for chunk in chunks] == [
            'Top',
            'Top > Sub',
            'Top > Sub > Deep',
        ]

    def test_distinct_same_level_headings_remain_separate(self):
        markdown = '# One\n\nA\n\n# Two\n\nB\n\n# Three\n\nC'

        chunks = chunk_markdown(markdown)

        assert [chunk.heading_context for chunk in chunks] == ['One', 'Two', 'Three']
        assert chunks[0].content == '# One\n\nA'
        assert chunks[1].content == '# Two\n\nB'
        assert chunks[2].content == '# Three\n\nC'

    def test_tiny_child_absorbs_into_substantial_parent_only(self):
        parent_body = ' '.join(['parent context'] * 70)
        markdown = f'# Topic\n\n{parent_body}\n\n## Note\n\nbrief child\n\n# Other\n\nshort sibling'

        chunks = chunk_markdown(markdown, join_threshold=180)

        assert len(chunks) == 2
        assert chunks[0].heading_context == 'Topic'
        assert '## Note' in chunks[0].content
        assert chunks[1].heading_context == 'Other'
        assert chunks[1].content == '# Other\n\nshort sibling'

    def test_oversized_section_splits_into_chunks_under_budget(self):
        body = ' '.join(['sentence'] * 1500)

        chunks = chunk_markdown(f'# Big\n\n{body}', join_threshold=200)

        assert len(chunks) > 1
        assert all(chunk.heading_context == 'Big' for chunk in chunks)
        assert all(chunk.content.startswith('# Big') for chunk in chunks)
        assert max(_token_count(chunk.content) for chunk in chunks) <= 400

    def test_reuses_token_counts_for_unchanged_sections(self):
        tokenizer = CountingWhitespaceTokenizer()
        markdown = '\n\n'.join(f'## Heading {index}\n\nsame repeated paragraph words' for index in range(30))

        chunks = chunk_markdown(markdown, join_threshold=12, tokenizer=tokenizer)

        assert len(chunks) == 30
        assert tokenizer.encode_calls <= len(chunks)

    def test_tiktoken_fast_path_matches_generic_tokenizer(self):
        markdown = '\n\n'.join(
            f'<!-- PAGE {index} -->\n\n## Page {index}\n\n' + ('vision extracted text ' * 120) for index in range(1, 20)
        )

        fast_chunks = chunk_markdown(markdown, join_threshold=180, tokenizer=TiktokenTokenizer())
        reference_chunks = chunk_markdown(markdown, join_threshold=180, tokenizer=TiktokenWrapper())

        assert [chunk.content for chunk in fast_chunks] == [chunk.content for chunk in reference_chunks]
        assert [chunk.page_number for chunk in fast_chunks] == [chunk.page_number for chunk in reference_chunks]

    def test_markdown_table_stays_in_one_chunk_when_under_budget(self):
        markdown = '# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |'

        chunks = chunk_markdown(markdown)

        assert len(chunks) == 1
        assert '| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |' in chunks[0].content

    def test_chunk_markdown_keeps_short_table_atomic(self):
        markdown = '\n'.join(
            [
                '| Name | Value |',
                '|---|---|',
                '| Alpha | 1 |',
                '| Beta | 2 |',
                '| Gamma | 3 |',
                '| Delta | 4 |',
                '| Epsilon | 5 |',
            ]
        )

        chunks = chunk_markdown(markdown)

        assert len(chunks) == 1
        assert chunks[0].content == markdown

    def test_chunk_markdown_replicates_header_when_table_splits(self):
        header = '| ID | Description |'
        separator = '|---|---|'
        cell_text = 'cell ' * 20
        rows = [f'| {index} | {cell_text}{index} |' for index in range(100)]
        markdown = '\n'.join([header, separator, *rows])

        chunks = chunk_markdown(markdown, join_threshold=30)

        assert len(chunks) > 1
        assert all(chunk.content.startswith(f'{header}\n{separator}\n') for chunk in chunks)

    def test_chunk_markdown_preserves_table_when_size_under_2x_max(self):
        header = '| ID | Value |'
        separator = '|---|---|'
        rows = [f'| {index} | compact value {index:02d} |' for index in range(8)]
        markdown = '\n'.join([header, separator, *rows])

        chunks = chunk_markdown(markdown, join_threshold=80)

        assert len(chunks) == 1
        assert chunks[0].content == markdown
        assert _token_count(chunks[0].content) <= 160

    def test_chunk_markdown_handles_malformed_table_without_separator(self):
        malformed_text = 'text ' * 12
        rows = [f'| row {index} | malformed {malformed_text}{index} |' for index in range(12)]
        markdown = '\n'.join(rows)

        chunks = chunk_markdown(markdown, join_threshold=20)

        assert len(chunks) > 1
        assert all('|---' not in chunk.content for chunk in chunks)
        assert rows[0] in chunks[0].content

    def test_markdown_list_stays_in_one_chunk_when_under_budget(self):
        markdown = '# Steps\n\n- one\n- two\n- three'

        chunks = chunk_markdown(markdown)

        assert len(chunks) == 1
        assert '- one\n- two\n- three' in chunks[0].content

    def test_plain_text_without_headings_still_produces_level_zero_chunk(self):
        markdown = 'Para 1\n\nPara 2\n\nPara 3'

        chunks = chunk_markdown(markdown)

        assert len(chunks) == 1
        assert chunks[0].heading_context is None
        assert chunks[0].page_number is None
        assert chunks[0].content == markdown

    def test_chunk_with_multiple_page_markers_exposes_page_range(self):
        markdown = '# Report\n\n<!-- PAGE 1 -->\n\nAlpha\n\n<!-- PAGE 2 -->\n\nBeta\n\n<!-- PAGE 3 -->\n\nGamma'

        chunks = chunk_markdown(markdown, join_threshold=100)

        assert len(chunks) == 1
        assert chunks[0].page_number == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 3
        assert chunks[0].page_numbers == (1, 2, 3)

    def test_chunk_indices_are_sequential_starting_at_zero(self):
        body = ' '.join(['sentence'] * 1500)

        chunks = chunk_markdown(f'# Big\n\n{body}', join_threshold=200)

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_large_multi_page_document_keeps_page_numbers_for_each_chunk(self):
        markdown = '\n\n'.join(
            f'# Page {page_number}\n\n<!-- PAGE {page_number} -->\n\nBody for page {page_number}.'
            for page_number in range(1, 13)
        )

        chunks = chunk_markdown(markdown, join_threshold=20)

        assert len(chunks) == 12
        assert [chunk.page_number for chunk in chunks] == list(range(1, 13))
        assert [chunk.page_start for chunk in chunks] == list(range(1, 13))
        assert [chunk.page_end for chunk in chunks] == list(range(1, 13))
        assert [chunk.page_numbers for chunk in chunks] == [(page_number,) for page_number in range(1, 13)]
        assert [chunk.heading_context for chunk in chunks] == [f'Page {page_number}' for page_number in range(1, 13)]


@pytest.mark.offline
class TestVisionBatchHelpers:
    def test_split_batch_response_with_full_markers(self):
        raw = (
            '<!-- PAGE 1 -->\n\n# First\n\nAlpha\n\n'
            f'<!-- PAGE 2 -->\n\n{NO_TEXT_DETECTED_SENTINEL}\n\n'
            '<!-- PAGE 3 -->\n\nGamma'
        )

        results, marker_mode = split_batch_response(raw, [1, 2, 3])

        assert marker_mode == 'full'
        assert [(result.page_number, result.content) for result in results] == [
            (1, '# First\n\nAlpha'),
            (2, ''),
            (3, 'Gamma'),
        ]

    def test_split_batch_response_without_markers_assigns_content_to_first_page(self):
        multi_page_results, multi_page_mode = split_batch_response('Raw text only', [1, 2, 3])
        single_page_results, single_page_mode = split_batch_response('Raw text only', [7])

        assert multi_page_mode == 'none'
        assert [(result.page_number, result.content) for result in multi_page_results] == [
            (1, 'Raw text only'),
            (2, ''),
            (3, ''),
        ]
        assert [(result.page_number, result.content) for result in single_page_results] == [(7, 'Raw text only')]
        assert single_page_mode in {'none', 'full'}

    def test_split_batch_response_with_partial_markers_reports_partial_mode(self):
        raw = '<!-- PAGE 1 -->\n\nOne\n\n<!-- PAGE 3 -->\n\nThree'

        results, marker_mode = split_batch_response(raw, [1, 2, 3])

        assert marker_mode == 'partial'
        assert [(result.page_number, result.content) for result in results] == [
            (1, 'One'),
            (2, ''),
            (3, 'Three'),
        ]

    @pytest.mark.parametrize(
        ('expected_page_count', 'marker_mode', 'finish_reason', 'expected'),
        [
            (3, 'full', 'length', True),
            (3, 'full', 'stop', False),
            (3, 'partial', 'stop', True),
            (1, 'none', 'length', False),
        ],
    )
    def test_should_split_batch_matches_finish_reason_and_marker_quality(
        self,
        expected_page_count: int,
        marker_mode: str,
        finish_reason: str,
        expected: bool,
    ):
        assert should_split_batch(expected_page_count, marker_mode, finish_reason) is expected


class TestChunkMarkdownBoilerplate:
    """Tests for the repeating-line boilerplate stripper applied at the end of chunk_markdown."""

    def test_chunk_markdown_strips_repeating_footer(self):
        sections = [
            f'# Section {i}\n\nUnique content for section {i} discussing topic {i}.\n\nConfidential -- Internal Use'
            for i in range(6)
        ]
        markdown = '\n\n'.join(sections)

        chunks = chunk_markdown(markdown, join_threshold=20)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert 'Confidential -- Internal Use' not in chunk.content
        combined = '\n'.join(chunk.content for chunk in chunks)
        assert 'Unique content for section 0' in combined
        assert 'Unique content for section 5' in combined

    def test_chunk_markdown_strips_repeating_page_number_pattern(self):
        sections = [
            f'# Section {i}\n\nDistinct prose for section {i} expanding on details.\n\nPage 1 of 12' for i in range(6)
        ]
        markdown = '\n\n'.join(sections)

        chunks = chunk_markdown(markdown, join_threshold=20)

        for chunk in chunks:
            assert 'Page 1 of 12' not in chunk.content

    def test_chunk_markdown_keeps_repeating_heading(self):
        # A repeated section heading is NOT boilerplate; headings carry semantic context.
        sections = [f'## Conclusions\n\nUnique closing paragraph {i} for chapter {i}.' for i in range(6)]
        markdown = '# Document\n\n' + '\n\n'.join(sections)

        chunks = chunk_markdown(markdown, join_threshold=20)

        heading_chunks = [c for c in chunks if '## Conclusions' in c.content]
        assert heading_chunks, 'Expected the repeated heading to survive in at least one chunk'

    def test_chunk_markdown_drops_chunk_emptied_by_stripping(self):
        # Build sections where one chunk is ONLY the boilerplate line and unique content elsewhere.
        sections = [
            f'# Section {i}\n\nUnique paragraph {i} with substantial detail.\n\nFooter line A' for i in range(5)
        ]
        # Add a section whose body is essentially only the boilerplate so it gets emptied.
        sections.append('# Empty Section\n\nFooter line A')
        markdown = '\n\n'.join(sections)

        chunks = chunk_markdown(markdown, join_threshold=20)

        for chunk in chunks:
            assert 'Footer line A' not in chunk.content
        # chunk_index must be sequential after dropping.
        indexes = [chunk.chunk_index for chunk in chunks]
        assert indexes == list(range(len(chunks)))

    def test_chunk_markdown_keeps_unique_lines_intact(self):
        # No line repeats across chunks -> stripper is a no-op; content unchanged.
        sections = [f'# Section {i}\n\nWholly unique paragraph {i} with details {i * 7}.' for i in range(5)]
        markdown = '\n\n'.join(sections)

        chunks = chunk_markdown(markdown, join_threshold=20)

        for i, chunk in enumerate(chunks):
            assert f'unique paragraph {i}' in chunk.content.lower()

    def test_chunk_markdown_skips_stripper_when_few_chunks(self):
        # With only 2 chunks, even a shared line is not stripped (insufficient signal).
        markdown = (
            '# A\n\nFirst section content.\n\nShared boilerplate line\n\n'
            '# B\n\nSecond section content.\n\nShared boilerplate line'
        )

        chunks = chunk_markdown(markdown, join_threshold=20)

        assert len(chunks) == 2
        for chunk in chunks:
            assert 'Shared boilerplate line' in chunk.content
