from __future__ import annotations

import json

import pytest

from yar.evaluation.extraction_quality import count_page_markers, format_markdown_summary, summarize_artifacts


@pytest.mark.offline
def test_count_page_markers_handles_html_comments():
    assert count_page_markers('<!-- PAGE 1 -->\nAlpha\n<!-- page 2 -->\nBeta') == 2


@pytest.mark.offline
def test_summarize_artifacts_reports_page_quality_tables_boilerplate_and_hashes():
    processed = '<!-- PAGE 1 -->\n\nAlpha\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n'
    canonical = (
        '<!-- PAGE 1 -->\n\nAlpha\n\n'
        '<!-- PAGE 2 -->\n\n<!-- YAR_PAGE_STATUS: empty -->\n\n'
        '<table><tr><th colspan="2">Risk</th></tr><tr><td>A</td><td>B</td></tr></table>\n'
        '# sanofi\n'
    )
    manifest = {
        'extractor_version': 'vision-adapter-v1',
        'prompt_version': 'vision-batch-v1',
        'model': 'test-model',
        'quality_report': {
            'page_count': 2,
            'dropped_retrieval_pages': [2],
            'empty_pages': [2],
            'tiny_pages': [1],
            'unexplained_tiny_pages': [],
            'boilerplate_only_pages': [],
            'native_fallback_pages': [1],
            'warning_counts': {'native_fallback': 1},
            'table_counts': {'complex_html_tables': 1, 'markdown_table_lines': 3},
        },
        'page_records': [{'page_number': 1}, {'page_number': 2}],
    }

    summary = summarize_artifacts(
        name='doc',
        processed_text=processed,
        canonical_text=canonical,
        manifest=manifest,
    )

    assert summary['expected_page_count'] == 2
    assert summary['canonical_page_marker_count'] == 2
    assert summary['retrieval_page_marker_count'] == 1
    assert summary['dropped_retrieval_pages'] == [2]
    assert summary['native_fallback_pages'] == [1]
    assert summary['manifest_table_counts']['complex_html_tables'] == 1
    assert summary['processed_table_counts']['markdown_table_lines'] == 3
    assert summary['canonical_table_counts']['html_tables'] == 1
    assert summary['processed_boilerplate_counts'] == {'exact_sanofi_lines': 0, 'heading_sanofi_lines': 0}
    assert summary['canonical_boilerplate_counts']['heading_sanofi_lines'] == 1
    assert len(summary['hashes']['processed_sha256']) == 64
    assert len(summary['hashes']['canonical_sha256']) == 64
    assert len(summary['hashes']['manifest_sha256']) == 64


@pytest.mark.offline
def test_format_markdown_summary_includes_counts_and_hashes():
    summary = {
        'name': 'doc',
        'expected_page_count': 2,
        'canonical_page_marker_count': 2,
        'retrieval_page_marker_count': 1,
        'dropped_retrieval_pages': [2],
        'empty_pages': [2],
        'tiny_pages': [1],
        'unexplained_tiny_pages': [],
        'native_fallback_pages': [1],
        'hashes': {
            'processed_sha256': 'a' * 64,
            'canonical_sha256': 'b' * 64,
            'manifest_sha256': 'c' * 64,
        },
    }

    markdown = format_markdown_summary([summary])

    assert '| doc | 2 | 2 | 1 | 1 | 1 | 1 | 0 | 1 |' in markdown
    assert 'processed_sha256' in markdown


@pytest.mark.offline
def test_manifest_hash_is_stable_for_key_order():
    processed = '<!-- PAGE 1 -->\n\nAlpha'
    canonical = '<!-- PAGE 1 -->\n\nAlpha'
    manifest_a = {'quality_report': {'page_count': 1, 'tiny_pages': []}, 'page_records': []}
    manifest_b = json.loads(json.dumps({'page_records': [], 'quality_report': {'tiny_pages': [], 'page_count': 1}}))

    hash_a = summarize_artifacts(name='a', processed_text=processed, canonical_text=canonical, manifest=manifest_a)[
        'hashes'
    ]['manifest_sha256']
    hash_b = summarize_artifacts(name='b', processed_text=processed, canonical_text=canonical, manifest=manifest_b)[
        'hashes'
    ]['manifest_sha256']

    assert hash_a == hash_b
