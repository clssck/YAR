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

import json

import numpy as np
import pytest

from yar.constants import (
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    GRAPH_FIELD_SEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
)
from yar.utils import (
    _normalize_math_alphanumerics,
    _sanitize_string_for_json,
    apply_source_ids_limit,
    compute_args_hash,
    compute_incremental_chunk_ids,
    compute_mdhash_id,
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
    reciprocal_rank_fusion,
    remove_think_tags,
    sanitize_and_normalize_extracted_text,
    sanitize_text_for_encoding,
    split_string_by_multi_markers,
    subtract_source_ids,
    write_json,
)


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
        dirty_text = 'Test\uD800string'
        result = _sanitize_string_for_json(dirty_text)
        assert '\uD800' not in result
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
        text = 'Micro\u200Bsoft'
        result = normalize_unicode_for_entity_matching(text)
        assert '\u200B' not in result
        assert result == 'Microsoft'

    def test_normalize_unicode_for_entity_matching_bidirectional(self):
        """Test removing bidirectional control characters."""
        text = 'Test\u202Atext\u202C'
        result = normalize_unicode_for_entity_matching(text)
        assert '\u202A' not in result
        assert '\u202C' not in result

    def test_normalize_unicode_for_entity_matching_empty(self):
        """Test normalizing empty string."""
        assert normalize_unicode_for_entity_matching('') == ''
        assert normalize_unicode_for_entity_matching(None) is None

    def test_normalize_math_alphanumerics(self):
        """Test normalizing mathematical alphanumeric symbols."""
        # Mathematical bold A (U+1D400)
        text = '\U0001D400pple'
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
        text = 'Test\uD800data'
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
        results = [
            [{'id': 'doc1'}, {'id': 'doc2'}, {'id': 'doc3'}]
        ]
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
