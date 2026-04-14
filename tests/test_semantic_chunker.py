from __future__ import annotations

import pytest

from yar.document.semantic_chunker import chunk_markdown
from yar.document.vision_adapter import (
	NO_TEXT_DETECTED_SENTINEL,
	should_split_batch,
	split_batch_response,
)


def _approx_tokens(text: str) -> int:
	return (len(text) + 3) // 4


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
		assert chunk.chunk_index == 0
		assert chunk.content.startswith('# Title')
		assert 'Some body text' in chunk.content

	def test_page_markers_are_preserved_and_extracted_from_marker_bearing_chunks(self):
		markdown = (
			'<!-- PAGE 1 -->\n\n'
			'# Title\n\n'
			'Body\n\n'
			'<!-- PAGE 3 -->\n\n'
			'## Subtitle\n\n'
			'More'
		)

		chunks = chunk_markdown(markdown, join_threshold=10)

		assert [chunk.page_number for chunk in chunks] == [1, 3, None]
		assert '<!-- PAGE 1 -->' in chunks[0].content
		assert '<!-- PAGE 3 -->' in chunks[1].content
		assert chunks[2].heading_context == 'Title > Subtitle'

	def test_heading_hierarchy_builds_context_chain(self):
		chunks = chunk_markdown('# Top\n\nA\n\n## Sub\n\nB\n\n### Deep\n\nC')

		assert [chunk.heading_context for chunk in chunks] == [
			'Top',
			'Top > Sub',
			'Top > Sub > Deep',
		]

	def test_small_same_level_sections_merge_when_under_threshold(self):
		markdown = '# One\n\nA\n\n# Two\n\nB\n\n# Three\n\nC'

		chunks = chunk_markdown(markdown)

		assert len(chunks) == 1
		assert '# One' in chunks[0].content
		assert '# Two' in chunks[0].content
		assert '# Three' in chunks[0].content

	def test_oversized_section_splits_into_chunks_under_budget(self):
		body = ' '.join(['sentence'] * 1500)

		chunks = chunk_markdown(f'# Big\n\n{body}', join_threshold=200)

		assert len(chunks) > 1
		assert all(chunk.heading_context == 'Big' for chunk in chunks)
		assert all(chunk.content.startswith('# Big') for chunk in chunks)
		assert max(_approx_tokens(chunk.content) for chunk in chunks) <= 400

	def test_markdown_table_stays_in_one_chunk_when_under_budget(self):
		markdown = '# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |'

		chunks = chunk_markdown(markdown)

		assert len(chunks) == 1
		assert '| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |' in chunks[0].content

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
		assert [chunk.heading_context for chunk in chunks] == [
			f'Page {page_number}' for page_number in range(1, 13)
		]


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
