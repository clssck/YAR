from __future__ import annotations

import base64
import os
from unittest.mock import patch

import pytest

from yar.document.vision_adapter import (
	ExtractionWarning,
	PageResult,
	RenderedPage,
	_build_batch_vision_messages,
    _compute_pages_per_call,
	_has_usable_content,
	_summarize_extraction_quality,
	stitch_extracted_pages,
)


def _page_result(
	page_number: int,
	content: str,
	warning: ExtractionWarning | None = None,
) -> PageResult:
	return PageResult(page_number=page_number, content=content, warning=warning)


def _rendered_page(
	page_number: int,
	*,
	total_pages: int = 2,
	media_type: str = 'image/png',
	image_bytes: bytes | None = None,
) -> RenderedPage:
	return RenderedPage(
		page_number=page_number,
		total_pages=total_pages,
		media_type=media_type,
		image_bytes=image_bytes or f'page-{page_number}'.encode(),
	)


@pytest.mark.offline
class TestStitchExtractedPages:
	def test_basic_stitching_adds_page_markers_without_warnings(self):
		stitched, warnings = stitch_extracted_pages(
			[
				_page_result(1, 'Alpha'),
				_page_result(2, '# Second\n\nBeta'),
				_page_result(3, 'Gamma'),
			]
		)

		assert stitched == (
			'<!-- PAGE 1 -->\n\nAlpha\n\n'
			'<!-- PAGE 2 -->\n\n# Second\n\nBeta\n\n'
			'<!-- PAGE 3 -->\n\nGamma'
		)
		assert warnings == []

	def test_filters_pages_without_usable_content_from_output(self):
		stitched, warnings = stitch_extracted_pages(
			[
				_page_result(1, 'Alpha'),
				_page_result(2, ''),
				_page_result(3, '<!-- comment only -->'),
				_page_result(4, '<p>Beta</p>'),
			]
		)

		assert stitched == '<!-- PAGE 1 -->\n\nAlpha\n\n<!-- PAGE 4 -->\n\n<p>Beta</p>'
		assert all(f'<!-- PAGE {page_number} -->' not in stitched for page_number in (2, 3))
		assert warnings == [
			ExtractionWarning(
				source='vision_quality',
				message='Pages with no usable extracted content: 2, 3',
			)
		]

	def test_adds_quality_warning_when_several_pages_are_empty(self):
		stitched, warnings = stitch_extracted_pages(
			[
				_page_result(1, 'Alpha'),
				_page_result(2, ''),
				_page_result(3, '<!-- PAGE 3 -->'),
				_page_result(4, '<!-- comment only -->'),
			]
		)

		assert stitched == '<!-- PAGE 1 -->\n\nAlpha'
		assert warnings == [
			ExtractionWarning(
				source='vision_quality',
				message='Pages with no usable extracted content: 2, 3, 4',
			)
		]

	def test_orders_results_by_page_number_before_stitching(self):
		stitched, warnings = stitch_extracted_pages(
			[
				_page_result(3, 'Third'),
				_page_result(1, 'First'),
				_page_result(2, 'Second'),
			]
		)

		assert stitched == (
			'<!-- PAGE 1 -->\n\nFirst\n\n'
			'<!-- PAGE 2 -->\n\nSecond\n\n'
			'<!-- PAGE 3 -->\n\nThird'
		)
		assert warnings == []

	def test_preserves_extra_warnings_in_returned_warning_list(self):
		extra_warnings = [ExtractionWarning(source='upstream', message='Vision fallback was used')]

		stitched, warnings = stitch_extracted_pages([_page_result(1, 'Alpha')], extra_warnings)

		assert stitched == '<!-- PAGE 1 -->\n\nAlpha'
		assert warnings == extra_warnings


@pytest.mark.offline
class TestExtractionQualityHelpers:
	@pytest.mark.parametrize(
		('content', 'expected'),
		[
			('Plain text content', True),
			('<!-- comment only -->', False),
			('<!-- PAGE 7 -->', False),
			('', False),
			('<div><strong>Alpha 123</strong></div>', True),
		],
	)
	def test_has_usable_content_handles_text_markers_comments_and_html(self, content: str, expected: bool):
		assert _has_usable_content(content) is expected

	def test_summarize_extraction_quality_returns_none_when_all_pages_have_content(self):
		warning = _summarize_extraction_quality(
			[
				_page_result(1, 'Alpha'),
				_page_result(2, 'Beta'),
			],
			[],
		)

		assert warning is None

	def test_summarize_extraction_quality_reports_empty_page_numbers(self):
		warning = _summarize_extraction_quality(
			[
				_page_result(1, 'Alpha'),
				_page_result(2, ''),
				_page_result(3, 'Gamma'),
				_page_result(4, '<!-- PAGE 4 -->'),
			],
			[],
		)

		assert warning == ExtractionWarning(
			source='vision_quality',
			message='Pages with no usable extracted content: 2, 4',
		)

	def test_summarize_extraction_quality_reports_truncation_count(self):
		warning = _summarize_extraction_quality(
			[
				_page_result(1, 'Alpha'),
				_page_result(2, 'Beta'),
			],
			[
				ExtractionWarning(source='vision_truncation', message='Page 1 may be truncated'),
				ExtractionWarning(source='vision_truncation', message='Page 2 may be truncated'),
				ExtractionWarning(source='other', message='Ignore me'),
			],
		)

		assert warning == ExtractionWarning(
			source='vision_quality',
			message='Pages flagged as potentially truncated: 2',
		)


@pytest.mark.offline
class TestBuildBatchVisionMessages:
	def test_build_batch_vision_messages_uses_single_user_message_with_page_range_and_images(self):
		page_one = _rendered_page(1, total_pages=5, image_bytes=b'page-one')
		page_two = _rendered_page(2, total_pages=5, image_bytes=b'page-two')

		messages = _build_batch_vision_messages([page_one, page_two])

		assert len(messages) == 1
		assert messages[0]['role'] == 'user'

		content = messages[0]['content']
		assert [part['type'] for part in content] == ['text', 'image_url', 'image_url']

		prompt = content[0]['text']
		assert 'This batch covers pages 1-2 of 5.' in prompt
		assert 'Return exactly one section for each requested page marker: <!-- PAGE 1 -->, <!-- PAGE 2 -->.' in prompt

		assert content[1]['image_url']['url'] == 'data:image/png;base64,' + base64.b64encode(b'page-one').decode()
		assert content[2]['image_url']['url'] == 'data:image/png;base64,' + base64.b64encode(b'page-two').decode()

@pytest.mark.offline
class TestComputePagesPerCall:
    @staticmethod
    def _clear_batching_env(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('VISION_PAGES_PER_CALL', raising=False)
        monkeypatch.delenv('VISION_MAX_OUTPUT_TOKENS', raising=False)

    def test_default_returns_12_pages_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {}, clear=False):
            assert _compute_pages_per_call() == 12

    def test_explicit_override_via_vision_pages_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_PAGES_PER_CALL': '8'}, clear=False):
            assert _compute_pages_per_call() == 8

    def test_explicit_override_of_1_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_PAGES_PER_CALL': '1'}, clear=False):
            assert _compute_pages_per_call() == 1

    def test_explicit_override_of_0_clamps_to_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_PAGES_PER_CALL': '0'}, clear=False):
            assert _compute_pages_per_call() == 1

    def test_small_output_budget_yields_1_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_MAX_OUTPUT_TOKENS': '4096'}, clear=False):
            assert _compute_pages_per_call() == 1

    def test_large_output_budget_caps_at_default_maximum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_MAX_OUTPUT_TOKENS': '999999'}, clear=False):
            assert _compute_pages_per_call() == 15

    def test_custom_output_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_MAX_OUTPUT_TOKENS': '32000'}, clear=False):
            assert _compute_pages_per_call() == 5

    def test_explicit_override_takes_precedence_over_output_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(
            os.environ,
            {'VISION_PAGES_PER_CALL': '3', 'VISION_MAX_OUTPUT_TOKENS': '999999'},
            clear=False,
        ):
            assert _compute_pages_per_call() == 3
