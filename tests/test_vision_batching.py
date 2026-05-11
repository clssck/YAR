from __future__ import annotations

import base64
import os
from unittest.mock import patch

import pytest

import yar.document.vision_adapter as vision_adapter
from yar.document.vision_adapter import (
    ExtractionWarning,
    PageResult,
    RenderedPage,
    _build_batch_vision_messages,
    _compute_pages_per_call,
    _get_vision_concurrency_limit,
    _has_usable_content,
    _summarize_extraction_quality,
    build_extraction_artifacts,
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
            '<!-- PAGE 1 -->\n\nAlpha\n\n<!-- PAGE 2 -->\n\n# Second\n\nBeta\n\n<!-- PAGE 3 -->\n\nGamma'
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

        assert stitched == '<!-- PAGE 1 -->\n\nAlpha\n\n<!-- PAGE 4 -->\n\nBeta'
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

        assert stitched == ('<!-- PAGE 1 -->\n\nFirst\n\n<!-- PAGE 2 -->\n\nSecond\n\n<!-- PAGE 3 -->\n\nThird')
        assert warnings == []

    def test_preserves_extra_warnings_in_returned_warning_list(self):
        extra_warnings = [ExtractionWarning(source='upstream', message='Vision fallback was used')]

        stitched, warnings = stitch_extracted_pages([_page_result(1, 'Alpha')], extra_warnings)

        assert stitched == '<!-- PAGE 1 -->\n\nAlpha'
        assert warnings == extra_warnings

    def test_normalizes_simple_html_blocks_outside_tables(self):
        stitched, warnings = stitch_extracted_pages(
            [
                _page_result(
                    1,
                    '<p><strong>Objective</strong></p><ul><li>Alpha&nbsp;ready</li><li>Beta<br>review</li></ul>',
                )
            ]
        )

        assert '<p>' not in stitched
        assert '<li>' not in stitched
        assert 'Objective' in stitched
        assert '- Alpha ready' in stitched
        assert '- Beta review' in stitched
        assert warnings == []

    def test_converts_simple_html_tables_to_markdown(self):
        stitched, warnings = stitch_extracted_pages(
            [
                _page_result(
                    1,
                    (
                        '## Risk Table\n\n'
                        '<table><thead><tr><th>Risk</th><th>Owner</th></tr></thead>'
                        '<tbody><tr><td>Late batch</td><td>CMC</td></tr></tbody></table>'
                    ),
                )
            ]
        )

        assert '<table' not in stitched
        assert '| Risk | Owner |' in stitched
        assert '| --- | --- |' in stitched
        assert '| Late batch | CMC |' in stitched
        assert warnings == []

    def test_preserves_complex_html_tables(self):
        html_table = '<table><tr><th colspan="2">Risk</th></tr><tr><td>A</td><td>B</td></tr></table>'

        stitched, warnings = stitch_extracted_pages([_page_result(1, html_table)])

        assert html_table in stitched
        assert warnings == []

    def test_removes_repeated_page_boilerplate_without_dropping_unique_content(self):
        stitched, warnings = stitch_extracted_pages(
            [
                _page_result(1, '# Page One\n\nAlpha\n\nsanofi\n\nPage 1 of 3'),
                _page_result(2, '# Page Two\n\nBeta\n\nsanofi\n\nPage 2 of 3'),
                _page_result(3, '# Page Three\n\nGamma\n\nsanofi\n\nPage 3 of 3'),
            ]
        )

        assert '\nsanofi' not in stitched.lower()
        assert 'Page 1 of 3' not in stitched
        assert 'Page 2 of 3' not in stitched
        assert 'Page 3 of 3' not in stitched
        assert 'Alpha' in stitched
        assert 'Beta' in stitched
        assert 'Gamma' in stitched
        assert warnings == []

    def test_removes_isolated_brand_boilerplate_on_single_page(self):
        stitched, warnings = stitch_extracted_pages([_page_result(1, 'sanofi\n\n# sanofi\n\n# Title\n\nAlpha')])

        assert 'sanofi' not in stitched.lower()
        assert '# Title' in stitched
        assert 'Alpha' in stitched
        assert warnings == []

    def test_preserves_repeated_meaningful_headings(self):
        stitched, warnings = stitch_extracted_pages(
            [
                _page_result(1, '## Objective\n\nAlpha'),
                _page_result(2, '## Objective\n\nBeta'),
                _page_result(3, '## Objective\n\nGamma'),
            ]
        )

        assert stitched.count('## Objective') == 3
        assert warnings == []

    def test_artifacts_keep_canonical_markers_for_dropped_pages(self):
        artifacts = build_extraction_artifacts(
            [
                _page_result(1, 'Alpha content long enough for retrieval ' * 4),
                _page_result(2, ''),
                _page_result(3, 'sanofi'),
            ],
            page_count=3,
            model='test-model',
        )

        assert artifacts.retrieval_content.count('<!-- PAGE') == 1
        assert artifacts.canonical_content.count('<!-- PAGE') == 3
        assert '<!-- YAR_PAGE_STATUS: empty -->' in artifacts.canonical_content
        assert artifacts.quality_report.dropped_retrieval_pages == [2, 3]
        assert artifacts.manifest['quality_report']['pages_emitted_canonical'] == 3

    def test_artifacts_preserve_canonical_boilerplate_but_strip_retrieval(self):
        artifacts = build_extraction_artifacts(
            [
                _page_result(1, '# Page One\n\nAlpha\n\nsanofi\n\nPage 1 of 3'),
                _page_result(2, '# Page Two\n\nBeta\n\nsanofi\n\nPage 2 of 3'),
                _page_result(3, '# Page Three\n\nGamma\n\nsanofi\n\nPage 3 of 3'),
            ],
            page_count=3,
        )

        assert 'sanofi' not in artifacts.retrieval_content.lower()
        assert 'sanofi' in artifacts.canonical_content.lower()
        assert artifacts.quality_report.pages_emitted_canonical == 3

    def test_artifacts_use_native_fallback_only_for_tiny_pages(self):
        artifacts = build_extraction_artifacts(
            [
                _page_result(1, 'Tiny'),
                _page_result(2, 'Vision content is already sufficiently detailed ' * 4),
            ],
            page_count=2,
            native_text_by_page={
                1: 'Native page text with source-backed details and enough length for retrieval.',
                2: 'Native text should not replace a healthy vision extraction.',
            },
        )

        records = {record.page_number: record for record in artifacts.page_records}
        assert records[1].source_method == 'vision_plus_native'
        assert records[1].status == 'native_fallback'
        assert records[2].source_method == 'vision'
        assert 'Native text should not replace' not in records[2].retrieval_content
        assert artifacts.quality_report.native_fallback_pages == [1]

    def test_artifacts_count_and_preserve_complex_tables(self):
        complex_table = '<table><tr><th colspan="2">Risk</th></tr><tr><td>A</td><td>B</td></tr></table>'

        artifacts = build_extraction_artifacts([_page_result(1, complex_table)], page_count=1)

        assert complex_table in artifacts.canonical_content
        assert complex_table in artifacts.retrieval_content
        assert artifacts.quality_report.table_counts['complex_html_tables'] == 1
        assert artifacts.quality_report.pages_containing_tables == [1]

    def test_page_cache_reuses_same_input_records(self):
        first = PageResult(
            page_number=1,
            content='Cached page content with enough substance for retrieval.',
            image_sha256='image-hash',
            cache_key='cache-key',
        )
        first_artifacts = build_extraction_artifacts([first], page_count=1)
        second = PageResult(page_number=1, content='', image_sha256='image-hash', cache_key='cache-key', cached=True)
        second.content = first_artifacts.manifest['page_records'][0]['vision_content_raw']

        second_artifacts = build_extraction_artifacts([second], page_count=1)

        assert first_artifacts.retrieval_content == second_artifacts.retrieval_content
        assert second_artifacts.page_records[0].cached is True


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

    def test_default_batch_prompt_requests_canonical_page_scoped_markdown(self):
        messages = _build_batch_vision_messages([_rendered_page(1)])

        prompt = messages[0]['content'][0]['text']
        assert 'canonical, page-scoped Markdown' in prompt
        assert "Preserve the page's reading order" in prompt
        assert 'Use Markdown tables for simple tables' in prompt
        assert 'use HTML <table> only when merged cells, multiline cells' in prompt
        assert 'Strip repeated page footers, headers, and boilerplate' in prompt
        assert 'Represent diagrams, timelines, flow charts, and infographics as structured sections' in prompt
        assert 'Use [unclear] for unreadable text' in prompt
        assert 'Do not invent missing text, values, labels, or relationships' in prompt


@pytest.mark.offline
class TestComputePagesPerCall:
    @staticmethod
    def _clear_batching_env(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('VISION_PAGES_PER_CALL', raising=False)
        monkeypatch.delenv('VISION_MAX_OUTPUT_TOKENS', raising=False)
        monkeypatch.delenv('VISION_MAX_CONTEXT_TOKENS', raising=False)

    @staticmethod
    def _context_budget_for_pages(page_count: int) -> int:
        return (
            vision_adapter.VISION_PROMPT_TOKENS_ESTIMATE
            + (vision_adapter.VISION_INPUT_TOKENS_PER_PAGE_ESTIMATE + vision_adapter.MAX_TOKENS_PER_PAGE) * page_count
        )

    def test_default_returns_15_pages_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {}, clear=False):
            assert _compute_pages_per_call() == 15

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
            assert _compute_pages_per_call() == 7

    def test_explicit_override_is_clamped_by_output_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_PAGES_PER_CALL': '50'}, clear=False):
            assert _compute_pages_per_call() == 16

    def test_context_cap_reduces_default_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(
            os.environ,
            {'VISION_MAX_CONTEXT_TOKENS': str(self._context_budget_for_pages(2))},
            clear=False,
        ):
            assert _compute_pages_per_call() == 2

    def test_tiny_context_cap_still_yields_1_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(os.environ, {'VISION_MAX_CONTEXT_TOKENS': '1'}, clear=False):
            assert _compute_pages_per_call() == 1

    def test_explicit_override_is_clamped_by_context_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_batching_env(monkeypatch)

        with patch.dict(
            os.environ,
            {
                'VISION_PAGES_PER_CALL': '50',
                'VISION_MAX_OUTPUT_TOKENS': '999999',
                'VISION_MAX_CONTEXT_TOKENS': str(self._context_budget_for_pages(3)),
            },
            clear=False,
        ):
            assert _compute_pages_per_call() == 3


@pytest.mark.offline
class TestVisionConcurrencyLimit:
    @staticmethod
    def _reset_concurrency_state(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('VISION_CONCURRENCY', raising=False)
        monkeypatch.setattr(vision_adapter, '_vision_concurrency_limit', None)
        monkeypatch.setattr(vision_adapter, '_vision_semaphore', None)

    def test_default_matches_configured_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._reset_concurrency_state(monkeypatch)

        assert _get_vision_concurrency_limit() == vision_adapter.VISION_CONCURRENCY_DEFAULT

    def test_env_override_initializes_helper_and_semaphore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._reset_concurrency_state(monkeypatch)

        with patch.dict(os.environ, {'VISION_CONCURRENCY': '7'}, clear=False):
            semaphore = vision_adapter._get_vision_semaphore()

            assert _get_vision_concurrency_limit() == 7
            assert semaphore._value == 7

    def test_zero_concurrency_clamps_to_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._reset_concurrency_state(monkeypatch)

        with patch.dict(os.environ, {'VISION_CONCURRENCY': '0'}, clear=False):
            assert _get_vision_concurrency_limit() == 1
