"""
Tests for yar/utils.py - Comprehensive utility function tests.

This module tests:
- Hash computation functions (compute_args_hash, compute_mdhash_id)
- Cache key generation and parsing
- JSON handling (load_json, write_json, _sanitize_string_for_json)
- Text processing (split_string_by_multi_markers, is_float_regex)
- Unicode normalization and security (normalize_unicode_for_entity_matching, sanitize_text_for_encoding)
- Entity/relation name normalization (normalize_extracted_info)
- OpenAI message formatting (pack_user_ass_to_openai_messages)
- Vector operations (cosine_similarity, reciprocal_rank_fusion)
- Source ID management (merge_source_ids, apply_source_ids_limit, subtract_source_ids)
- Track ID generation (generate_track_id)
- Relation key operations (make_relation_chunk_key, parse_relation_chunk_key)
- Content summarization (get_content_summary, remove_think_tags)
- Tuple delimiter fixing (fix_tuple_delimiter_corruption)
- Environment value parsing (get_env_value)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import yar.utils as utils_module
from yar.base import QueryParam
from yar.constants import (
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    GRAPH_FIELD_SEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
)
from yar.utils import (
    _mmr_reorder,
    _normalize_math_alphanumerics,
    _sanitize_string_for_json,
    apply_source_ids_limit,
    compute_args_hash,
    compute_incremental_chunk_ids,
    compute_mdhash_id,
    convert_to_user_format,
    cosine_similarity,
    create_prefixed_exception,
    fix_tuple_delimiter_corruption,
    generate_cache_key,
    generate_track_id,
    get_content_summary,
    get_env_value,
    get_pinyin_sort_key,
    is_float_regex,
    load_json,
    make_relation_chunk_key,
    merge_source_ids,
    normalize_extracted_info,
    normalize_source_ids_limit_method,
    normalize_unicode_for_entity_matching,
    pack_user_ass_to_openai_messages,
    parse_cache_key,
    parse_relation_chunk_key,
    process_chunks_unified,
    reciprocal_rank_fusion,
    remove_think_tags,
    sanitize_and_normalize_extracted_text,
    sanitize_text_for_encoding,
    split_string_by_multi_markers,
    subtract_source_ids,
    write_json,
)


def test_convert_to_user_format_prefers_context_content_and_preserves_raw_chunk() -> None:
    result = convert_to_user_format(
        [],
        [],
        [
            {
                'reference_id': '1',
                'content': 'Raw chunk text',
                'context_content': 'Extractive evidence spans:\n- Raw chunk text',
                'evidence_spans': ['Raw chunk text'],
                'file_path': 'source.md',
                'chunk_id': 'chunk-1',
            }
        ],
        [{'reference_id': '1', 'file_path': 'source.md'}],
        'naive',
    )

    chunk = result['data']['chunks'][0]
    assert chunk['content'] == 'Extractive evidence spans:\n- Raw chunk text'
    assert chunk['raw_content'] == 'Raw chunk text'
    assert chunk['evidence_spans'] == ['Raw chunk text']
    assert chunk['chunk_id'] == 'chunk-1'
    assert 'page_start' not in chunk


def test_convert_to_user_format_preserves_page_range_metadata() -> None:
    result = convert_to_user_format(
        [],
        [],
        [
            {
                'reference_id': '1',
                'content': 'Chunk with page range',
                'file_path': 'source.md',
                'chunk_id': 'chunk-1',
                'page_number': 2,
                'page_start': 2,
                'page_end': 4,
                'page_numbers': [2, 3, 4],
            }
        ],
        [{'reference_id': '1', 'file_path': 'source.md'}],
        'naive',
    )

    chunk = result['data']['chunks'][0]
    assert chunk['page_number'] == 2
    assert chunk['page_start'] == 2
    assert chunk['page_end'] == 4
    assert chunk['page_numbers'] == [2, 3, 4]


class TestHashFunctions:
    """Tests for hash computation functions."""

    def test_compute_args_hash_basic(self):
        """Test basic hash computation."""
        hash1 = compute_args_hash('test')
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash is 32 hex characters

    def test_compute_args_hash_multiple_args(self):
        """Test hash computation with multiple arguments."""
        hash_result = compute_args_hash('arg1', 'arg2', 'arg3')
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_compute_args_hash_consistency(self):
        """Test that same inputs produce same hash."""
        hash1 = compute_args_hash('test', 123, 'data')
        hash2 = compute_args_hash('test', 123, 'data')
        assert hash1 == hash2

    def test_compute_args_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = compute_args_hash('test1')
        hash2 = compute_args_hash('test2')
        assert hash1 != hash2

    def test_compute_args_hash_unicode(self):
        """Test hash computation with unicode characters."""
        hash_result = compute_args_hash('测试', '日本語', 'Café')
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_compute_mdhash_id_with_string(self):
        """Test MD hash ID generation with string."""
        hash_id = compute_mdhash_id('test content')
        assert hash_id.startswith('')
        assert len(hash_id) == 32

    def test_compute_mdhash_id_with_prefix(self):
        """Test MD hash ID generation with prefix."""
        hash_id = compute_mdhash_id('test content', prefix='ent-')
        assert hash_id.startswith('ent-')
        assert len(hash_id) == 32 + 4  # 32 hash + 4 prefix

    def test_compute_mdhash_id_with_bytes(self):
        """Test MD hash ID generation with bytes."""
        content_bytes = b'test content'
        hash_id = compute_mdhash_id(content_bytes, prefix='rel-')
        assert hash_id.startswith('rel-')
        assert isinstance(hash_id, str)


class TestCacheKeyFunctions:
    """Tests for cache key generation and parsing."""

    def test_generate_cache_key(self):
        """Test cache key generation."""
        key = generate_cache_key('default', 'extract', 'abc123')
        assert key == 'default:extract:abc123'

    def test_generate_cache_key_different_modes(self):
        """Test cache key generation with different modes."""
        key1 = generate_cache_key('local', 'query', 'hash1')
        key2 = generate_cache_key('global', 'query', 'hash1')
        assert key1 == 'local:query:hash1'
        assert key2 == 'global:query:hash1'
        assert key1 != key2

    def test_parse_cache_key_valid(self):
        """Test parsing valid cache key."""
        result = parse_cache_key('default:extract:abc123')
        assert result == ('default', 'extract', 'abc123')

    def test_parse_cache_key_invalid(self):
        """Test parsing invalid cache key."""
        result = parse_cache_key('invalid_key')
        assert result is None

    def test_parse_cache_key_with_colons_in_hash(self):
        """Test parsing cache key with colons in hash part."""
        result = parse_cache_key('mode:type:hash:with:colons')
        assert result == ('mode', 'type', 'hash:with:colons')

    def test_cache_key_roundtrip(self):
        """Test cache key generation and parsing roundtrip."""
        original_mode = 'default'
        original_type = 'extract'
        original_hash = 'test_hash_123'

        key = generate_cache_key(original_mode, original_type, original_hash)
        parsed = parse_cache_key(key)

        assert parsed is not None
        mode, cache_type, hash_value = parsed
        assert mode == original_mode
        assert cache_type == original_type
        assert hash_value == original_hash


class TestJSONHandling:
    """Tests for JSON loading, writing, and sanitization."""

    def test_load_json_nonexistent(self):
        """Test loading non-existent JSON file."""
        result = load_json('/nonexistent/file.json')
        assert result is None

    def test_load_json_valid(self, tmp_path):
        """Test loading valid JSON file."""
        json_file = tmp_path / 'test.json'
        data = {'key': 'value', 'number': 42}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        result = load_json(str(json_file))
        assert result == data

    def test_load_json_with_bom(self, tmp_path):
        """Test loading JSON file with UTF-8 BOM."""
        json_file = tmp_path / 'test_bom.json'
        data = {'test': 'data'}
        with open(json_file, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f)

        result = load_json(str(json_file))
        assert result == data

    def test_write_json_basic(self, tmp_path):
        """Test basic JSON writing."""
        json_file = tmp_path / 'output.json'
        data = {'key': 'value', 'list': [1, 2, 3]}

        sanitized = write_json(data, str(json_file))
        assert sanitized is False  # No sanitization needed

        # Verify file contents
        with open(json_file, encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data

    def test_write_json_unicode(self, tmp_path):
        """Test JSON writing with Unicode characters."""
        json_file = tmp_path / 'unicode.json'
        data = {'chinese': '测试', 'japanese': '日本語', 'emoji': '😀'}

        sanitized = write_json(data, str(json_file))
        assert sanitized is False

        with open(json_file, encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data

    def test_sanitize_string_for_json_clean(self):
        """Test sanitizing clean string."""
        clean_text = 'This is a clean string'
        result = _sanitize_string_for_json(clean_text)
        assert result == clean_text  # Should return same string (zero-copy)

    def test_sanitize_string_for_json_empty(self):
        """Test sanitizing empty string."""
        assert _sanitize_string_for_json('') == ''
        assert _sanitize_string_for_json(None) is None

    def test_sanitize_string_for_json_with_surrogates(self):
        """Test sanitizing string with surrogate characters."""
        # Create a string with surrogate characters
        dirty_text = 'Test\ud800string'
        result = _sanitize_string_for_json(dirty_text)
        assert '\ud800' not in result
        assert result == 'Teststring'


class TestTextProcessing:
    """Tests for text processing helper functions."""

    def test_split_string_by_multi_markers_single(self):
        """Test splitting by single marker."""
        result = split_string_by_multi_markers('a,b,c', [','])
        assert result == ['a', 'b', 'c']

    def test_split_string_by_multi_markers_multiple(self):
        """Test splitting by multiple markers."""
        text = 'a|b;c|d'
        result = split_string_by_multi_markers(text, ['|', ';'])
        assert result == ['a', 'b', 'c', 'd']

    def test_split_string_by_multi_markers_empty_markers(self):
        """Test splitting with no markers."""
        text = 'hello world'
        result = split_string_by_multi_markers(text, [])
        assert result == ['hello world']

    def test_split_string_by_multi_markers_strips_whitespace(self):
        """Test that results are stripped."""
        result = split_string_by_multi_markers('  a  ,  b  ,  c  ', [','])
        assert result == ['a', 'b', 'c']

    def test_split_string_by_multi_markers_none_content(self):
        """Test splitting None content."""
        result = split_string_by_multi_markers(None, [','])
        assert result == []

    def test_is_float_regex_valid_floats(self):
        """Test float regex with valid floats."""
        assert is_float_regex('3.14') is True
        assert is_float_regex('0.5') is True
        assert is_float_regex('-2.5') is True
        assert is_float_regex('+1.0') is True
        assert is_float_regex('42') is True
        assert is_float_regex('-10') is True

    def test_is_float_regex_invalid(self):
        """Test float regex with invalid values."""
        assert is_float_regex('abc') is False
        assert is_float_regex('1.2.3') is False
        assert is_float_regex('') is False
        assert is_float_regex('1e5') is False  # Scientific notation not supported

    def test_pack_user_ass_to_openai_messages(self):
        """Test OpenAI message formatting."""
        messages = pack_user_ass_to_openai_messages('Hello', 'Hi there', 'How are you?')
        assert len(messages) == 3
        assert messages[0] == {'role': 'user', 'content': 'Hello'}
        assert messages[1] == {'role': 'assistant', 'content': 'Hi there'}
        assert messages[2] == {'role': 'user', 'content': 'How are you?'}

    def test_pack_user_ass_to_openai_messages_empty(self):
        """Test OpenAI message formatting with no arguments."""
        messages = pack_user_ass_to_openai_messages()
        assert messages == []


class TestUnicodeNormalization:
    """Tests for Unicode normalization and security functions."""

    def test_normalize_unicode_for_entity_matching_basic(self):
        """Test basic Unicode normalization."""
        result = normalize_unicode_for_entity_matching('Café')
        assert isinstance(result, str)
        assert 'Café' in result or 'Cafe' in result

    def test_normalize_unicode_for_entity_matching_zero_width(self):
        """Test removing zero-width characters."""
        # Unicode zero-width space
        text = 'Micro\u200bsoft'
        result = normalize_unicode_for_entity_matching(text)
        assert '\u200b' not in result
        assert result == 'Microsoft'

    def test_normalize_unicode_for_entity_matching_bidirectional(self):
        """Test removing bidirectional control characters."""
        text = 'Test\u202atext\u202c'
        result = normalize_unicode_for_entity_matching(text)
        assert '\u202a' not in result
        assert '\u202c' not in result

    def test_normalize_unicode_for_entity_matching_empty(self):
        """Test normalizing empty string."""
        assert normalize_unicode_for_entity_matching('') == ''
        assert normalize_unicode_for_entity_matching(None) is None

    def test_normalize_math_alphanumerics(self):
        """Test normalizing mathematical alphanumeric symbols."""
        # Mathematical bold A (U+1D400)
        text = '\U0001d400pple'
        result = _normalize_math_alphanumerics(text)
        # Should normalize to regular A
        assert result == 'Apple'

    def test_sanitize_text_for_encoding_basic(self):
        """Test basic text sanitization."""
        text = '  Hello World  '
        result = sanitize_text_for_encoding(text)
        assert result == 'Hello World'

    def test_sanitize_text_for_encoding_empty(self):
        """Test sanitizing empty text."""
        assert sanitize_text_for_encoding('') == ''
        assert sanitize_text_for_encoding('   ') == ''

    def test_sanitize_text_for_encoding_surrogate_removal(self):
        """Test removal of surrogate characters."""
        # String with surrogate character
        text = 'Test\ud800data'
        # Should raise ValueError for uncleanable encoding issues
        with pytest.raises(ValueError) as exc_info:
            sanitize_text_for_encoding(text)
        assert 'uncleanable UTF-8 encoding issues' in str(exc_info.value)

    def test_sanitize_text_for_encoding_html_unescape(self):
        """Test HTML entity unescaping."""
        text = '&lt;div&gt;Test&lt;/div&gt;'
        result = sanitize_text_for_encoding(text)
        assert result == '<div>Test</div>'

    def test_sanitize_text_for_encoding_control_chars(self):
        """Test removal of control characters."""
        text = 'Test\x00\x01data'
        result = sanitize_text_for_encoding(text)
        assert '\x00' not in result
        assert '\x01' not in result


class TestNormalizeExtractedInfo:
    """Tests for normalize_extracted_info function."""

    def test_normalize_extracted_info_basic(self):
        """Test basic normalization."""
        result = normalize_extracted_info('Test Entity')
        assert result == 'Test Entity'

    def test_normalize_extracted_info_html_tags(self):
        """Test HTML tag removal."""
        result = normalize_extracted_info('<p>Entity</p>')
        assert result == 'Entity'

        result = normalize_extracted_info('Text<br>NewLine')
        assert result == 'TextNewLine'

    def test_normalize_extracted_info_chinese_fullwidth(self):
        """Test Chinese full-width character conversion."""
        result = normalize_extracted_info('ＡＢＣ１２３')
        assert result == 'ABC123'

    def test_normalize_extracted_info_chinese_parentheses(self):
        """Test Chinese parentheses conversion."""
        result = normalize_extracted_info('Test（data）')
        assert result == 'Test(data)'

    def test_normalize_extracted_info_remove_outer_quotes(self):
        """Test removal of outer quotes."""
        result = normalize_extracted_info('"Entity Name"')
        assert result == 'Entity Name'

        result = normalize_extracted_info("'Entity Name'")
        assert result == 'Entity Name'

    def test_normalize_extracted_info_keep_inner_quotes(self):
        """Test keeping inner quotes when remove_inner_quotes=False."""
        result = normalize_extracted_info('"Entity "Name""', remove_inner_quotes=False)
        # Should keep outer quotes because inner quotes exist
        assert '"' in result

    def test_normalize_extracted_info_filter_short_numeric(self):
        """Test filtering short numeric strings."""
        assert normalize_extracted_info('12') == ''
        assert normalize_extracted_info('1') == ''
        assert normalize_extracted_info('123') == '123'  # Length 3 is OK

    def test_normalize_extracted_info_filter_dots(self):
        """Test filtering dot-number combinations."""
        assert normalize_extracted_info('1.2.3') == ''
        assert normalize_extracted_info('12.3') == ''
        # 123.456 is 7 chars, so it's NOT filtered (only < 6 chars are filtered)
        assert normalize_extracted_info('123.456') == '123.456'
        assert normalize_extracted_info('12.34') == ''  # 5 chars, filtered

    def test_normalize_extracted_info_chinese_spaces(self):
        """Test removal of spaces between Chinese characters."""
        result = normalize_extracted_info('中 文 测 试')
        assert result == '中文测试'

    def test_sanitize_and_normalize_extracted_text(self):
        """Test combined sanitization and normalization."""
        text = '  <p>"Test Entity"</p>  '
        result = sanitize_and_normalize_extracted_text(text)
        assert result == 'Test Entity'

    def test_sanitize_and_normalize_extracted_text_empty(self):
        """Test with empty input."""
        result = sanitize_and_normalize_extracted_text('')
        assert result == ''


class TestVectorOperations:
    """Tests for vector operation functions."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2, 3])
        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = cosine_similarity(v1, v2)
        assert abs(result) < 1e-10

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([-1, -2, -3])
        result = cosine_similarity(v1, v2)
        assert abs(result - (-1.0)) < 1e-10

    def test_cosine_similarity_different(self):
        """Test cosine similarity of different vectors."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result = cosine_similarity(v1, v2)
        assert 0 < result < 1

    def test_reciprocal_rank_fusion_empty(self):
        """Test RRF with empty result lists."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_reciprocal_rank_fusion_single_list(self):
        """Test RRF with single result list."""
        results = [[{'id': 'doc1'}, {'id': 'doc2'}, {'id': 'doc3'}]]
        merged = reciprocal_rank_fusion(results)
        assert len(merged) == 3
        assert merged[0]['id'] == 'doc1'
        assert 'rrf_score' in merged[0]

    def test_reciprocal_rank_fusion_multiple_lists(self):
        """Test RRF with multiple result lists."""
        results = [
            [{'id': 'doc1'}, {'id': 'doc2'}],
            [{'id': 'doc2'}, {'id': 'doc3'}],
        ]
        merged = reciprocal_rank_fusion(results)
        assert len(merged) == 3
        # doc2 appears in both lists, should have highest score
        assert merged[0]['id'] == 'doc2'

    def test_reciprocal_rank_fusion_custom_id_key(self):
        """Test RRF with custom ID key."""
        results = [
            [{'item_id': 'a'}, {'item_id': 'b'}],
            [{'item_id': 'b'}, {'item_id': 'c'}],
        ]
        merged = reciprocal_rank_fusion(results, id_key='item_id')
        assert len(merged) == 3
        assert merged[0]['item_id'] == 'b'

    def test_reciprocal_rank_fusion_custom_k(self):
        """Test RRF with custom k parameter."""
        results = [
            [{'id': 'doc1'}, {'id': 'doc2'}],
        ]
        merged1 = reciprocal_rank_fusion(results, k=60)
        merged2 = reciprocal_rank_fusion(results, k=100)
        # Different k values should produce different scores
        assert merged1[0]['rrf_score'] != merged2[0]['rrf_score']


class TestSourceIDManagement:
    """Tests for source ID management functions."""

    def test_merge_source_ids_both_present(self):
        """Test merging with both source ID lists present."""
        result = merge_source_ids(['a', 'b'], ['c', 'd'])
        assert result == ['a', 'b', 'c', 'd']

    def test_merge_source_ids_with_duplicates(self):
        """Test merging with duplicate source IDs."""
        result = merge_source_ids(['a', 'b', 'c'], ['b', 'c', 'd'])
        assert result == ['a', 'b', 'c', 'd']

    def test_merge_source_ids_none_values(self):
        """Test merging with None values."""
        result = merge_source_ids(None, ['a', 'b'])
        assert result == ['a', 'b']

        result = merge_source_ids(['a', 'b'], None)
        assert result == ['a', 'b']

        result = merge_source_ids(None, None)
        assert result == []

    def test_merge_source_ids_empty_strings(self):
        """Test merging filters out empty strings."""
        result = merge_source_ids(['a', '', 'b'], ['c', '', 'd'])
        assert result == ['a', 'b', 'c', 'd']

    def test_normalize_source_ids_limit_method_valid(self):
        """Test normalizing valid limit methods."""
        assert normalize_source_ids_limit_method('fifo') == 'FIFO'
        assert normalize_source_ids_limit_method('FIFO') == 'FIFO'
        assert normalize_source_ids_limit_method('keep') == 'KEEP'
        assert normalize_source_ids_limit_method('KEEP') == 'KEEP'

    def test_normalize_source_ids_limit_method_invalid(self):
        """Test normalizing invalid limit method."""
        result = normalize_source_ids_limit_method('invalid_method')
        assert result == DEFAULT_SOURCE_IDS_LIMIT_METHOD

    def test_normalize_source_ids_limit_method_none(self):
        """Test normalizing None limit method."""
        result = normalize_source_ids_limit_method(None)
        assert result == DEFAULT_SOURCE_IDS_LIMIT_METHOD

    def test_apply_source_ids_limit_no_limit(self):
        """Test applying limit when list is within limit."""
        source_ids = ['a', 'b', 'c']
        result = apply_source_ids_limit(source_ids, 5, 'FIFO')
        assert result == ['a', 'b', 'c']

    def test_apply_source_ids_limit_fifo(self):
        """Test applying FIFO limit."""
        source_ids = ['a', 'b', 'c', 'd', 'e']
        result = apply_source_ids_limit(source_ids, 3, SOURCE_IDS_LIMIT_METHOD_FIFO)
        assert result == ['c', 'd', 'e']  # Keep last 3

    def test_apply_source_ids_limit_keep(self):
        """Test applying KEEP limit."""
        source_ids = ['a', 'b', 'c', 'd', 'e']
        result = apply_source_ids_limit(source_ids, 3, 'KEEP')
        assert result == ['a', 'b', 'c']  # Keep first 3

    def test_apply_source_ids_limit_zero(self):
        """Test applying zero limit."""
        source_ids = ['a', 'b', 'c']
        result = apply_source_ids_limit(source_ids, 0, 'FIFO')
        assert result == []

    def test_subtract_source_ids_basic(self):
        """Test basic source ID subtraction."""
        source_ids = ['a', 'b', 'c', 'd']
        to_remove = ['b', 'd']
        result = subtract_source_ids(source_ids, to_remove)
        assert result == ['a', 'c']

    def test_subtract_source_ids_empty_removal(self):
        """Test subtraction with empty removal set."""
        source_ids = ['a', 'b', 'c']
        result = subtract_source_ids(source_ids, [])
        assert result == ['a', 'b', 'c']

    def test_subtract_source_ids_filters_empty(self):
        """Test subtraction filters empty strings."""
        source_ids = ['a', '', 'b', 'c']
        result = subtract_source_ids(source_ids, ['a'])
        assert result == ['b', 'c']

    def test_compute_incremental_chunk_ids_basic(self):
        """Test basic incremental chunk ID computation."""
        existing = ['chunk-1', 'chunk-2', 'chunk-3']
        old = ['chunk-1', 'chunk-2']
        new = ['chunk-2', 'chunk-4']
        result = compute_incremental_chunk_ids(existing, old, new)
        # chunk-1 is removed (in old but not new)
        # chunk-3 remains (was in existing but not in old)
        # chunk-2 remains (in both old and new)
        # chunk-4 is added (in new but not old)
        assert result == ['chunk-2', 'chunk-3', 'chunk-4']

    def test_compute_incremental_chunk_ids_no_changes(self):
        """Test incremental computation with no changes."""
        existing = ['chunk-1', 'chunk-2']
        old = ['chunk-1']
        new = ['chunk-1']
        result = compute_incremental_chunk_ids(existing, old, new)
        # chunk-1 stays (in both old and new)
        # chunk-2 stays (in existing but not in old)
        assert result == ['chunk-1', 'chunk-2']

    def test_compute_incremental_chunk_ids_all_new(self):
        """Test incremental computation with all new chunks."""
        existing = []
        old = []
        new = ['chunk-1', 'chunk-2']
        result = compute_incremental_chunk_ids(existing, old, new)
        assert result == ['chunk-1', 'chunk-2']


class TestRelationKeyOperations:
    """Tests for relation key creation and parsing."""

    def test_make_relation_chunk_key(self):
        """Test creating relation chunk key."""
        key = make_relation_chunk_key('entity1', 'entity2')
        assert GRAPH_FIELD_SEP in key
        # Key should be sorted
        parts = key.split(GRAPH_FIELD_SEP)
        assert parts == sorted(parts)

    def test_make_relation_chunk_key_deterministic(self):
        """Test that relation key is deterministic regardless of order."""
        key1 = make_relation_chunk_key('entity1', 'entity2')
        key2 = make_relation_chunk_key('entity2', 'entity1')
        assert key1 == key2

    def test_parse_relation_chunk_key_valid(self):
        """Test parsing valid relation chunk key."""
        key = f'entity1{GRAPH_FIELD_SEP}entity2'
        src, tgt = parse_relation_chunk_key(key)
        assert {src, tgt} == {'entity1', 'entity2'}

    def test_parse_relation_chunk_key_invalid(self):
        """Test parsing invalid relation chunk key."""
        with pytest.raises(ValueError) as exc_info:
            parse_relation_chunk_key('invalid_key')
        assert 'Invalid relation chunk key' in str(exc_info.value)

    def test_relation_key_roundtrip(self):
        """Test creating and parsing relation key roundtrip."""
        original_src = 'source_entity'
        original_tgt = 'target_entity'

        key = make_relation_chunk_key(original_src, original_tgt)
        parsed_src, parsed_tgt = parse_relation_chunk_key(key)

        # Should get back both entities (order doesn't matter)
        assert {parsed_src, parsed_tgt} == {original_src, original_tgt}


class TestTrackIDGeneration:
    """Tests for track ID generation."""

    def test_generate_track_id_default_prefix(self):
        """Test generating track ID with default prefix."""
        track_id = generate_track_id()
        assert track_id.startswith('upload_')
        # Format: prefix_YYYYMMDD_HHMMSS_uuid (4 parts)
        assert len(track_id.split('_')) == 4

    def test_generate_track_id_custom_prefix(self):
        """Test generating track ID with custom prefix."""
        track_id = generate_track_id(prefix='insert')
        assert track_id.startswith('insert_')

    def test_generate_track_id_uniqueness(self):
        """Test that track IDs are unique."""
        id1 = generate_track_id()
        id2 = generate_track_id()
        assert id1 != id2

    def test_generate_track_id_format(self):
        """Test track ID format."""
        track_id = generate_track_id(prefix='test')
        parts = track_id.split('_')
        # Format: prefix_YYYYMMDD_HHMMSS_uuid (4 parts)
        assert len(parts) == 4
        assert parts[0] == 'test'
        # Date part should be 8 digits (YYYYMMDD)
        assert len(parts[1]) == 8
        assert parts[1].isdigit()
        # Time part should be 6 digits (HHMMSS)
        assert len(parts[2]) == 6
        assert parts[2].isdigit()
        # UUID part should be 8 characters
        assert len(parts[3]) == 8


class TestContentSummarization:
    """Tests for content summarization functions."""

    def test_get_content_summary_short(self):
        """Test summary of short content."""
        content = 'Short content'
        result = get_content_summary(content)
        assert result == 'Short content'

    def test_get_content_summary_long(self):
        """Test summary of long content."""
        content = 'a' * 500
        result = get_content_summary(content, max_length=100)
        assert len(result) == 103  # 100 + '...'
        assert result.endswith('...')

    def test_get_content_summary_exact_length(self):
        """Test summary at exact max length."""
        content = 'a' * 250
        result = get_content_summary(content, max_length=250)
        assert result == content  # No truncation

    def test_get_content_summary_custom_length(self):
        """Test summary with custom max length."""
        content = 'a' * 100
        result = get_content_summary(content, max_length=50)
        assert len(result) == 53  # 50 + '...'

    def test_remove_think_tags_basic(self):
        """Test removing think tags."""
        text = '<think>internal thought</think>Actual response'
        result = remove_think_tags(text)
        assert result == 'Actual response'

    def test_remove_think_tags_no_tags(self):
        """Test text without think tags."""
        text = 'Just normal text'
        result = remove_think_tags(text)
        assert result == 'Just normal text'

    def test_remove_think_tags_orphan_closing(self):
        """Test removing orphan closing tag."""
        text = 'Some text</think>Remaining text'
        result = remove_think_tags(text)
        assert result == 'Remaining text'

    def test_remove_think_tags_multiline(self):
        """Test removing multiline think tags."""
        text = '<think>\nMultiple\nlines\n</think>Output'
        result = remove_think_tags(text)
        assert result == 'Output'

    def test_remove_think_tags_mid_text(self):
        """Bug: <think> tags in mid-text must not truncate surrounding content."""
        text = 'Answer about xxx<think>reasoning</think>xxx more content'
        result = remove_think_tags(text)
        assert result == 'Answer about xxxxxx more content'

    def test_remove_think_tags_multiple_blocks(self):
        """Multiple <think> blocks are all removed."""
        text = '<think>r1</think>Answer<think>r2</think> more'
        result = remove_think_tags(text)
        assert result == 'Answer more'

    def test_remove_think_tags_orphan_with_angle_brackets(self):
        """Orphaned </think> prefix containing '<' chars is still removed."""
        text = '2 < 3 reasoning</think>final answer'
        result = remove_think_tags(text)
        assert result == 'final answer'

    def test_remove_think_tags_empty_block(self):
        """Empty think block is removed."""
        text = '<think></think>Content.'
        result = remove_think_tags(text)
        assert result == 'Content.'

    def test_remove_think_tags_only_think(self):
        """Text that is only a think block returns empty string."""
        text = '<think>only reasoning</think>'
        result = remove_think_tags(text)
        assert result == ''


class TestTupleDelimiterFix:
    """Tests for tuple delimiter corruption fixing."""

    def test_fix_tuple_delimiter_corruption_clean(self):
        """Test with clean delimiter."""
        record = 'entity1<|#|>entity2<|#|>relation'
        result = fix_tuple_delimiter_corruption(record, '#', '<|#|>')
        assert result == record  # Should remain unchanged

    def test_fix_tuple_delimiter_corruption_double_delimiter(self):
        """Test fixing double delimiter core."""
        record = 'entity1<|##|>entity2'
        result = fix_tuple_delimiter_corruption(record, '#', '<|#|>')
        assert result == 'entity1<|#|>entity2'

    def test_fix_tuple_delimiter_corruption_missing_pipes(self):
        """Test fixing missing pipes."""
        record = 'entity1<#>entity2'
        result = fix_tuple_delimiter_corruption(record, '#', '<|#|>')
        assert result == 'entity1<|#|>entity2'

    def test_fix_tuple_delimiter_corruption_escaped(self):
        """Test fixing escaped delimiter."""
        record = 'entity1<|\\#|>entity2'
        result = fix_tuple_delimiter_corruption(record, '#', '<|#|>')
        assert result == 'entity1<|#|>entity2'

    def test_fix_tuple_delimiter_corruption_empty_pipes(self):
        """Test fixing empty pipes."""
        record = 'entity1<|>entity2'
        result = fix_tuple_delimiter_corruption(record, '#', '<|#|>')
        assert result == 'entity1<|#|>entity2'

    def test_fix_tuple_delimiter_corruption_missing_bracket(self):
        """Test fixing missing closing bracket."""
        record = 'entity1<|#|entity2'
        result = fix_tuple_delimiter_corruption(record, '#', '<|#|>')
        assert result == 'entity1<|#|>entity2'

    def test_fix_tuple_delimiter_corruption_empty_input(self):
        """Test with empty inputs."""
        assert fix_tuple_delimiter_corruption('', '#', '<|#|>') == ''
        assert fix_tuple_delimiter_corruption('test', '', '<|#|>') == 'test'


class TestEnvironmentValueParsing:
    """Tests for environment value parsing."""

    def test_get_env_value_missing(self, monkeypatch):
        """Test getting missing environment variable."""
        monkeypatch.delenv('TEST_VAR', raising=False)
        result = get_env_value('TEST_VAR', 'default_value')
        assert result == 'default_value'

    def test_get_env_value_string(self, monkeypatch):
        """Test getting string environment variable."""
        monkeypatch.setenv('TEST_VAR', 'test_value')
        result = get_env_value('TEST_VAR', 'default')
        assert result == 'test_value'

    def test_get_env_value_int(self, monkeypatch):
        """Test getting integer environment variable."""
        monkeypatch.setenv('TEST_VAR', '42')
        result = get_env_value('TEST_VAR', 0, value_type=int)
        assert result == 42

    def test_get_env_value_bool_true(self, monkeypatch):
        """Test getting boolean environment variable (true values)."""
        for value in ['true', 'True', '1', 'yes', 't', 'on']:
            monkeypatch.setenv('TEST_VAR', value)
            result = get_env_value('TEST_VAR', False, value_type=bool)
            assert result is True

    def test_get_env_value_bool_false(self, monkeypatch):
        """Test getting boolean environment variable (false values)."""
        for value in ['false', 'False', '0', 'no', 'off']:
            monkeypatch.setenv('TEST_VAR', value)
            result = get_env_value('TEST_VAR', True, value_type=bool)
            assert result is False

    def test_get_env_value_list(self, monkeypatch):
        """Test getting list environment variable."""
        monkeypatch.setenv('TEST_VAR', '["a", "b", "c"]')
        result = get_env_value('TEST_VAR', [], value_type=list)
        assert result == ['a', 'b', 'c']

    def test_get_env_value_list_invalid_json(self, monkeypatch):
        """Test getting invalid JSON list."""
        monkeypatch.setenv('TEST_VAR', 'not a json list')
        result = get_env_value('TEST_VAR', ['default'], value_type=list)
        assert result == ['default']

    def test_get_env_value_special_none(self, monkeypatch):
        """Test special None handling."""
        monkeypatch.setenv('TEST_VAR', 'None')
        result = get_env_value('TEST_VAR', 'default', special_none=True)
        assert result is None

    def test_get_env_value_type_conversion_failure(self, monkeypatch):
        """Test type conversion failure."""
        monkeypatch.setenv('TEST_VAR', 'not_a_number')
        result = get_env_value('TEST_VAR', 42, value_type=int)
        assert result == 42  # Should return default


class TestPrefixedException:
    """Tests for create_prefixed_exception function."""

    def test_create_prefixed_exception_basic(self):
        """Test creating prefixed exception."""
        original = ValueError('original message')
        prefixed = create_prefixed_exception(original, 'PREFIX')
        assert isinstance(prefixed, ValueError)
        assert 'PREFIX' in str(prefixed)
        assert 'original message' in str(prefixed)

    def test_create_prefixed_exception_multiple_args(self):
        """Test with exception having multiple args."""
        original = ValueError('arg1', 'arg2')
        prefixed = create_prefixed_exception(original, 'PREFIX')
        assert 'PREFIX' in str(prefixed)

    def test_create_prefixed_exception_no_args(self):
        """Test with exception having no args."""
        original = ValueError()
        prefixed = create_prefixed_exception(original, 'PREFIX')
        assert 'PREFIX' in str(prefixed)

    def test_create_prefixed_exception_complex_type(self):
        """Test with complex exception type (OSError)."""
        original = OSError(2, 'No such file')
        prefixed = create_prefixed_exception(original, 'PREFIX')
        # Should still work even if reconstruction fails
        assert 'PREFIX' in str(prefixed)


class TestPinyinSort:
    """Tests for pinyin sort key generation."""

    def test_get_pinyin_sort_key_empty(self):
        """Test pinyin sort with empty string."""
        result = get_pinyin_sort_key('')
        assert result == ''

    def test_get_pinyin_sort_key_english(self):
        """Test pinyin sort with English text."""
        result = get_pinyin_sort_key('Hello World')
        assert result == 'hello world'

    def test_get_pinyin_sort_key_numbers(self):
        """Test pinyin sort with numbers."""
        result = get_pinyin_sort_key('Test123')
        assert '123' in result.lower()

    def test_get_pinyin_sort_key_chinese(self):
        """Test pinyin sort with Chinese characters."""
        # This test will work differently depending on whether pypinyin is available
        result = get_pinyin_sort_key('测试')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_pinyin_sort_key_mixed(self):
        """Test pinyin sort with mixed content."""
        result = get_pinyin_sort_key('Test测试123')
        assert isinstance(result, str)
        # Should preserve numbers
        assert '123' in result


class _FixedTokenTokenizer:
    def __init__(self, tokens_per_chunk: int) -> None:
        self.tokens_per_chunk = tokens_per_chunk

    def encode(self, _text: str) -> list[int]:
        return [0] * self.tokens_per_chunk


def _make_budget_chunks(count: int = 40) -> list[dict]:
    return [
        {
            'chunk_id': f'chunk-{index}',
            'content': f'Budget test content {index}',
            'file_path': f'document-{index}.md',
            'retrieval_score': 1.0 - (index / 1000),
        }
        for index in range(count)
    ]


def _run_process_chunks_for_budget(
    chunks: list[dict],
    query_param: QueryParam,
    *,
    tokenizer: _FixedTokenTokenizer,
    global_config: dict[str, object] | None = None,
    chunk_token_limit: int | None = None,
) -> list[dict]:
    config: dict[str, object] = {'tokenizer': tokenizer}
    if global_config:
        config.update(global_config)

    return asyncio.run(
        process_chunks_unified(
            query='Explain the budget allocation across many retrieved documents in detail',
            unique_chunks=chunks,
            query_param=query_param,
            global_config=config,
            source_type='mixed',
            chunk_token_limit=chunk_token_limit,
        )
    )


def test_process_chunks_unified_subtracts_entity_relation_tokens_from_budget() -> None:
    chunks = _make_budget_chunks()
    full_budget_param = QueryParam(
        chunk_top_k=len(chunks),
        enable_rerank=False,
        max_total_tokens=30_000,
    )
    full_budget_chunks = _run_process_chunks_for_budget(
        _make_budget_chunks(),
        full_budget_param,
        tokenizer=_FixedTokenTokenizer(tokens_per_chunk=1000),
    )

    reduced_budget_param = QueryParam(
        chunk_top_k=len(chunks),
        enable_rerank=False,
        max_total_tokens=30_000,
    )
    reduced_budget_param.entities_tokens_used = 10_000
    reduced_budget_param.relations_tokens_used = 5_000

    reduced_budget_chunks = _run_process_chunks_for_budget(
        chunks,
        reduced_budget_param,
        tokenizer=_FixedTokenTokenizer(tokens_per_chunk=1000),
    )

    assert len(reduced_budget_chunks) < len(full_budget_chunks)


def test_process_chunks_unified_floors_at_minimum_chunk_budget_when_entities_dominate(monkeypatch) -> None:
    captured_token_limits: list[int] = []
    original_truncate = utils_module.truncate_list_by_token_size

    def capture_truncate(
        list_data: list[Any],
        key: Callable[[Any], str],
        max_token_size: int,
        tokenizer: Any,
    ) -> list[Any]:
        captured_token_limits.append(max_token_size)
        return original_truncate(
            list_data,
            key=key,
            max_token_size=max_token_size,
            tokenizer=tokenizer,
        )

    monkeypatch.setattr(utils_module, 'truncate_list_by_token_size', capture_truncate)

    processed_chunks = _run_process_chunks_for_budget(
        _make_budget_chunks(),
        QueryParam(
            chunk_top_k=40,
            enable_rerank=False,
            max_total_tokens=30_000,
        ),
        tokenizer=_FixedTokenTokenizer(tokens_per_chunk=512),
        global_config={'entities_tokens_used': 29_000, 'relations_tokens_used': 0},
    )

    assert captured_token_limits == [1024]
    assert processed_chunks


def test_process_chunks_unified_falls_back_to_full_budget_when_keys_missing() -> None:
    missing_keys_param = QueryParam(
        chunk_top_k=40,
        enable_rerank=False,
        max_total_tokens=30_000,
    )
    missing_keys_chunks = _run_process_chunks_for_budget(
        _make_budget_chunks(),
        missing_keys_param,
        tokenizer=_FixedTokenTokenizer(tokens_per_chunk=1000),
    )

    explicit_limit_chunks = _run_process_chunks_for_budget(
        _make_budget_chunks(),
        QueryParam(
            chunk_top_k=40,
            enable_rerank=False,
            max_total_tokens=30_000,
        ),
        tokenizer=_FixedTokenTokenizer(tokens_per_chunk=1000),
        chunk_token_limit=30_000,
    )

    assert [chunk['chunk_id'] for chunk in missing_keys_chunks] == [
        chunk['chunk_id'] for chunk in explicit_limit_chunks
    ]


def _make_mmr_chunks() -> list[dict[str, Any]]:
    return [
        {
            'chunk_id': 'duplicate-a',
            'content': (
                'Process validation batch release protocol requires sterility assurance '
                'controls and stability data review.'
            ),
            'file_path': 'doc-a.md',
            'retrieval_score': 0.9,
        },
        {
            'chunk_id': 'duplicate-b',
            'content': (
                'Process validation batch release protocol requires sterility assurance '
                'controls and stability data review update.'
            ),
            'file_path': 'doc-b.md',
            'retrieval_score': 0.85,
        },
        {
            'chunk_id': 'distinct',
            'content': (
                'Warehouse alarms cold storage excursion investigation covers label '
                'reconciliation and shipping lane quarantine.'
            ),
            'file_path': 'doc-c.md',
            'retrieval_score': 0.7,
        },
    ]


def _run_process_chunks_for_mmr(chunks: list[dict[str, Any]]) -> list[dict]:
    return asyncio.run(
        process_chunks_unified(
            query='Describe manufacturing controls for release stability details',
            unique_chunks=chunks,
            query_param=QueryParam(chunk_top_k=3, enable_rerank=False),
            global_config={},
            source_type='mixed',
        )
    )


def test_mmr_reorder_preserves_order_with_full_relevance() -> None:
    chunks = [
        {
            'chunk_id': 'first',
            'content': 'Alpha batch release controls',
            'merge_score': 0.9,
        },
        {
            'chunk_id': 'second',
            'content': 'Beta warehouse alarm controls',
            'merge_score': 0.8,
        },
        {
            'chunk_id': 'third',
            'content': 'Gamma stability review controls',
            'merge_score': 0.7,
        },
    ]

    reordered = _mmr_reorder(chunks, lambda_=1.0)

    assert [chunk['chunk_id'] for chunk in reordered] == ['first', 'second', 'third']


def test_mmr_reorder_breaks_up_near_duplicates() -> None:
    reordered = _mmr_reorder(_make_mmr_chunks(), lambda_=0.7)
    reordered_ids = [chunk['chunk_id'] for chunk in reordered]

    assert reordered_ids.index('distinct') < reordered_ids.index('duplicate-b')


def test_mmr_reorder_handles_missing_content() -> None:
    chunks = [
        {'chunk_id': 'missing', 'merge_score': 0.99},
        {
            'chunk_id': 'present',
            'content': 'Distinct release evidence',
            'merge_score': 0.5,
        },
    ]

    reordered = _mmr_reorder(chunks, lambda_=0.7)

    assert [chunk['chunk_id'] for chunk in reordered] == ['present', 'missing']


def test_process_chunks_unified_applies_mmr_by_default(monkeypatch) -> None:
    monkeypatch.delenv('YAR_MMR_LAMBDA', raising=False)
    monkeypatch.setenv('ENABLE_LEXICAL_BOOST', 'false')

    processed = _run_process_chunks_for_mmr(_make_mmr_chunks())
    processed_ids = [chunk['chunk_id'] for chunk in processed]

    assert processed_ids.index('distinct') < processed_ids.index('duplicate-b')


def test_process_chunks_unified_disables_mmr_when_lambda_one(monkeypatch) -> None:
    monkeypatch.setenv('YAR_MMR_LAMBDA', '1.0')
    monkeypatch.setenv('ENABLE_LEXICAL_BOOST', 'false')

    processed = _run_process_chunks_for_mmr(_make_mmr_chunks())

    assert [chunk['chunk_id'] for chunk in processed] == [
        'duplicate-a',
        'duplicate-b',
        'distinct',
    ]


class TestValidateAndStripUnsupportedQuotes:
    """Tests for validate_and_strip_unsupported_quotes.

    Defends the post-LLM safety net that catches fabricated quoted strings
    in generator responses. The function is a no-op on clean answers and only
    intervenes when a double-quoted span cannot be found verbatim
    (case/whitespace/punctuation-insensitive) in the retrieved context.
    """

    @staticmethod
    def _call(response: str, context: str, **kwargs):
        return utils_module.validate_and_strip_unsupported_quotes(response, context, **kwargs)

    def test_noop_when_no_quotes(self):
        response = 'The compound was tested at site A in 2023. The protocol was revised.'
        context = 'Some retrieved context here.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_noop_when_quote_present_in_context(self):
        response = 'The document states "the assay was qualified for release".'
        context = 'Section 32S43: the assay was qualified for release in May 2024.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_strips_fabricated_quote(self):
        response = (
            'The Fitusiran document states: '
            '"Post-execution control. Document the change vs CM only after real '
            'implementation (describe the minor deviations if any)." '
            'This indicates documentation occurs after the fact.'
        )
        context = (
            'The Fitusiran presentation discusses PPQ batch 50mg PFS, mock audits, '
            'and joint quality reviews with CMO partners.'
        )
        modified, stripped = self._call(response, context)
        assert '"' not in modified.replace('Fitusiran', '_')  # no quote chars left in answer body
        assert len(stripped) == 1
        assert 'Post-execution control' in stripped[0]
        # The paraphrased content should still be present (we strip marks, not text).
        assert 'Post-execution control' in modified

    def test_strips_only_unsupported_when_mixed(self):
        response = (
            'Section A says "the assay was qualified for release". '
            'Section B says "this fabricated phrase was never written".'
        )
        context = 'The assay was qualified for release in 2024.'
        modified, stripped = self._call(response, context)
        assert len(stripped) == 1
        assert 'fabricated phrase' in stripped[0]
        # The supported quote keeps its quotation marks.
        assert '"the assay was qualified for release"' in modified
        # The fabricated quote loses them.
        assert '"this fabricated phrase was never written"' not in modified
        assert 'this fabricated phrase was never written' in modified

    def test_normalization_handles_case_and_whitespace(self):
        response = 'The note reads "Quality   Review accepted   by GQA".'
        context = 'CAPA outcome: quality review accepted by gqa on 2024-03-15.'
        modified, stripped = self._call(response, context)
        assert modified == response  # quote was supported, just differently formatted
        assert stripped == []

    def test_normalization_handles_punctuation(self):
        response = 'It states "FDA requested HPV L1 testing for every DS lot".'
        context = 'Per IR#1, the FDA requested HPV L1 testing for every DS lot.'
        modified, stripped = self._call(response, context)
        # Normalization drops all punctuation; same word order = verbatim match.
        assert modified == response
        assert stripped == []

    def test_word_order_change_is_fabrication(self):
        # Same words as context, but reordered. A "quote" with rearranged
        # word order is not verbatim — the model is claiming exactness that
        # is not there. The validator correctly strips it.
        response = 'It states "FDA per IR#1 requested HPV L1 testing".'
        context = 'Per IR#1, the FDA requested HPV L1 testing for every DS lot.'
        modified, stripped = self._call(response, context)
        assert len(stripped) == 1
        assert '"' not in modified.split('It states ')[1].split('.')[0]

    def test_curly_quotes_supported(self):
        # Some LLMs emit smart quotes \u201c \u201d instead of straight ".
        response = 'The doc reads \u201cthis is a fabricated curly-quoted line that should be stripped\u201d.'
        context = 'Completely unrelated content here.'
        modified, stripped = self._call(response, context)
        assert len(stripped) == 1
        assert '\u201c' not in modified
        assert '\u201d' not in modified

    def test_short_quotes_ignored(self):
        # Below min_quote_chars (12) — should not flag, even if absent from context.
        response = 'The team called it "weird" but moved on.'
        context = 'Nothing related here.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_empty_inputs_safe(self):
        assert self._call('', 'ctx') == ('', [])
        assert self._call('resp', '') == ('resp', [])
        assert self._call('', '') == ('', [])

    def test_single_quotes_not_flagged(self):
        # Single quotes are intentionally excluded to avoid contraction false positives.
        response = "The team's plan didn't include 'this fabricated phrase that is long enough'."
        context = 'Completely unrelated content here.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_idempotent(self):
        response = 'Fabricated: "this string is not in the context at all anywhere".'
        context = 'Completely unrelated.'
        once, stripped_once = self._call(response, context)
        twice, stripped_twice = self._call(once, context)
        assert twice == once
        assert stripped_once and not stripped_twice  # second pass is a no-op

    def test_preserves_text_outside_quotes(self):
        response = 'Before quote. "Fabricated quote not in any chunk text". After quote.'
        context = 'Unrelated.'
        modified, _ = self._call(response, context)
        assert modified.startswith('Before quote. ')
        assert modified.endswith(' After quote.')


class TestValidateAndStripUnsupportedAcronyms:
    """Tests for validate_and_strip_unsupported_acronyms.

    Targets the "invented acronym" failure mode where the generator coins
    short uppercase tokens (2-5 chars) that do not appear in any retrieved
    chunk. Validator is conservative — flags only ALL-CAPS standalone
    tokens, accepts case-insensitive context matches so "FDA" answer with
    "fda" context is fine.
    """

    @staticmethod
    def _call(response: str, context: str, **kwargs):
        return utils_module.validate_and_strip_unsupported_acronyms(response, context, **kwargs)

    def test_noop_when_no_acronyms(self):
        response = 'The compound was tested at site A in two thousand twenty three.'
        context = 'Unrelated retrieved content here.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_noop_when_acronym_present_in_context(self):
        response = 'The FDA approved the assay qualification.'
        context = 'FDA non-hold comments referenced in section 32S43.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_strips_invented_acronym_parenthetical(self):
        # Mirror of the original "NBO" failure: model coined an acronym
        # alongside a real entity, in parentheses.
        response = 'For Insulin Campus Frankfurt (NBO) the open CAPAs are tracked separately.'
        context = 'For Insulin Campus Frankfurt the open CAPAs are tracked alongside the ICF audit results.'
        modified, stripped = self._call(response, context)
        assert stripped == ['NBO']
        assert '(NBO)' not in modified
        assert 'NBO' not in modified
        # The legitimate entity name is preserved.
        assert 'Insulin Campus Frankfurt' in modified

    def test_strips_bare_acronym(self):
        # Bare standalone occurrence (not parenthetical).
        response = 'The findings showed NBO was responsible for the delay.'
        context = 'The findings discussed Insulin Campus Frankfurt and ICF specifically.'
        modified, stripped = self._call(response, context)
        assert stripped == ['NBO']
        assert 'NBO' not in modified
        # Whitespace tidied so we don't leave double spaces.
        assert '  ' not in modified

    def test_case_insensitive_context_match(self):
        # Response uses "FDA" (uppercase), context uses "fda" (lowercase).
        # Both forms should count as supported.
        response = 'The FDA reviewed the IR.'
        context = 'The fda has multiple ir submissions on file.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_dedupes_repeated_invented_acronym(self):
        # Same fake acronym appearing multiple times → reported once.
        response = 'XYZ said this, XYZ said that, and XYZ confirmed it.'
        context = 'Some unrelated content.'
        modified, stripped = self._call(response, context)
        assert stripped == ['XYZ']  # de-duplicated
        assert 'XYZ' not in modified

    def test_strips_only_unsupported_when_mixed(self):
        response = 'Both the FDA and NBO weighed in on the matter.'
        context = 'The FDA published its decision in March 2024.'
        modified, stripped = self._call(response, context)
        assert stripped == ['NBO']
        assert 'FDA' in modified
        assert 'NBO' not in modified

    def test_size_bounds_respected(self):
        # Single-letter tokens never match \b[A-Z]{2,5}\b — verified safely.
        # 6+ letter "acronyms" (e.g. NIOSHA) also not flagged (out of bounds).
        response = 'I said NIOSHA is responsible. The CDC also agreed.'
        context = 'NIOSH and the CDC published the guidance.'
        # NIOSHA (6 chars) skipped; CDC (3) found in context → both safe.
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_empty_inputs_safe(self):
        assert self._call('', 'ctx') == ('', [])
        assert self._call('resp', '') == ('resp', [])
        assert self._call('', '') == ('', [])

    def test_idempotent(self):
        response = 'The NBO is mentioned here.'
        context = 'Unrelated.'
        once, stripped_once = self._call(response, context)
        twice, stripped_twice = self._call(once, context)
        assert twice == once
        assert stripped_once == ['NBO']
        assert stripped_twice == []

    def test_punctuation_normalized_after_strip(self):
        # After stripping bare acronym at end-of-sentence, no orphan
        # whitespace should remain before the period.
        response = 'The site responsible was NBO.'
        context = 'Unrelated.'
        modified, _ = self._call(response, context)
        assert modified == 'The site responsible was.'

    def test_lowercase_words_never_flagged(self):
        # Validator only matches uppercase tokens; lowercase words like
        # "the" or "and" are not acronyms regardless of context match.
        response = 'the alphabet has many sequences like xyz embedded in it.'
        context = 'unrelated content with no overlap'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_preserves_text_outside_acronym(self):
        # Mirror of the quote-validator parity test: text on either side of
        # the stripped token must remain intact, including punctuation joints.
        response = 'Before the call, XYZ confirmed the result. After the call, all clear.'
        context = 'No acronyms here at all.'
        modified, stripped = self._call(response, context)
        assert stripped == ['XYZ']
        assert modified.startswith('Before the call,')
        assert modified.endswith('After the call, all clear.')
        assert 'XYZ' not in modified

    def test_substring_match_in_context_treated_as_support(self):
        # DESIGN NOTE: the validator uses a plain substring check
        # (`token.lower() in context.lower()`), not word-boundary matching.
        # That means an acronym like "API" is considered "supported" when the
        # letters "api" appear inside any word in the context — e.g. "rapid".
        # This is intentional (avoids false positives on plurals like "PCRs"
        # and inflected forms) but surprising. This test pins the behavior so
        # a future refactor to word-boundary matching is a deliberate choice,
        # not an accident.
        response = 'The API was reviewed.'
        context = 'A rapid review of the submission was performed.'  # contains 'api' inside 'rapid'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_mixed_case_tokens_never_flagged(self):
        # Regex is \b[A-Z]{2,5}\b — strictly all-caps. Title-case "Nbo" and
        # mixed-case "Fda" must not be matched even when absent from context.
        # Distinct from test_lowercase_words_never_flagged which covers fully
        # lowercase input.
        response = 'The Nbo and Fda were both mentioned, plus AbCd somewhere.'
        context = 'Unrelated content with zero overlap.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []

    def test_respects_custom_size_bounds_kwargs(self):
        # Tightening min to 3 should let a 2-char fabricated acronym slip
        # through; loosening max to 6 should catch a 6-char fabricated one.
        # Pins the kwargs as a real public knob rather than dead parameters.
        response_with_2char = 'The XY committee deliberated.'
        response_with_6char = 'The XYZABC committee deliberated.'
        context = 'Unrelated.'
        # Default bounds (2-5): 2-char XY is flagged.
        _, default_2 = self._call(response_with_2char, context)
        assert default_2 == ['XY']
        # min_acronym_chars=3 raises the floor; 2-char XY is now ignored.
        _, raised_min = self._call(response_with_2char, context, min_acronym_chars=3)
        assert raised_min == []
        # Default bounds (2-5): 6-char XYZABC is NOT flagged (above max — regex won't match either way).
        _, default_6 = self._call(response_with_6char, context)
        assert default_6 == []
        # max_acronym_chars=6 raises the ceiling; XYZABC is now flagged.
        # NOTE: regex \b[A-Z]{2,5}\b is hard-coded so even with max=6 the
        # token won't match — pins the regex/kwarg coupling as a known limit.
        _, raised_max = self._call(response_with_6char, context, max_acronym_chars=6)
        assert raised_max == []  # documents the regex hard-cap

    def test_acronym_with_internal_dots_not_matched(self):
        # "F.D.A." has no run of 2+ consecutive uppercase letters; each letter
        # is bordered by periods. Regex \b[A-Z]{2,5}\b cannot match. Pins the
        # behavior so a future regex change must explicitly decide to cover
        # dotted-acronym form.
        response = 'The F.D.A. and the U.S. agencies coordinated.'
        context = 'Coordination between agencies was documented.'
        modified, stripped = self._call(response, context)
        assert modified == response
        assert stripped == []
