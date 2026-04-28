from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
PAGE_MARKER_RE = re.compile(r'<!--\s*PAGE\s+(\d+)\s*-->')
SENTENCE_RE = re.compile(
    r'<!--\s*PAGE\s+\d+\s*-->|[^.!?\n]+[.!?]+(?:\s+|$)|[^.!?\n]+(?:\n|$)|[^.!?]+$',
    re.MULTILINE | re.DOTALL,
)
LIST_ITEM_RE = re.compile(r'^\s*(?:[-*+]\s+|\d+[.)]\s+)')
TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')
TABLE_SEPARATOR_RE = re.compile(r'^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$')
PAGE_MARKER_SEPARATOR_RE = re.compile(r'[\s\x00-\x1f\x7f]')
TRIM_RE = re.compile(r'^[ \t\r\n\f\v]+|[ \t\r\n\f\v]+$')
TRIM_END_RE = re.compile(r'[ \t\r\n\f\v]+$')
WHITESPACE_SPLIT_RE = re.compile(r'[ \t\r\n\f\v]+')
BLANK_LINE_SPLIT_RE = re.compile(r'\n[ \t\r\f\v]*\n+')
VISIBLE_WORD_SEPARATOR_RE = re.compile(r'[ \t\r\n\f\v]')


@dataclass(slots=True)
class HeadingNode:
    level: int
    text: str


@dataclass(slots=True)
class Section:
    level: int
    heading: str
    body: str
    heading_hierarchy: list[HeadingNode] | None = None


@dataclass(slots=True)
class ChunkData:
    content: str
    page_number: int | None
    chunk_index: int
    heading_context: str | None
    heading_hierarchy: list[HeadingNode] | None


def _trim(text: str) -> str:
    return TRIM_RE.sub('', text)


def _trim_end(text: str) -> str:
    return TRIM_END_RE.sub('', text)


def _split_whitespace(text: str) -> list[str]:
    return [part for part in WHITESPACE_SPLIT_RE.split(text) if part]


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _parse_into_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    lines = markdown.split('\n')
    current: Section | None = None
    body_lines: list[str] = []

    def flush_body() -> None:
        nonlocal body_lines
        if current is not None:
            current.body = '\n'.join(body_lines)
            body_lines = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush_body()
            if current is not None:
                sections.append(current)
            elif body_lines:
                sections.append(Section(level=0, heading='', body='\n'.join(body_lines)))
                body_lines = []
            current = Section(level=len(match.group(1)), heading=match.group(2), body='')
        else:
            body_lines.append(line)

    if current is not None:
        flush_body()
        sections.append(current)
    elif body_lines:
        sections.append(Section(level=0, heading='', body='\n'.join(body_lines)))

    return sections


def _build_heading_hierarchies(sections: list[Section]) -> None:
    stack: list[HeadingNode] = []
    for section in sections:
        if section.level == 0:
            section.heading_hierarchy = None
            continue
        while stack and stack[-1].level >= section.level:
            stack.pop()
        stack.append(HeadingNode(level=section.level, text=section.heading))
        section.heading_hierarchy = list(stack)


def _build_heading_prefix(section: Section) -> str:
    """Render the section's heading line(s) for inclusion at the top of the chunk text.

    When the section has a full heading hierarchy (H1 > H2 > H3 ...), every ancestor is
    emitted as its own markdown heading line. This gives BM25 and embeddings the parent
    context that's otherwise lost when a chunk represents a deep H3 detached from its H1.
    Without the breadcrumb, queries about the parent topic miss matching child chunks.
    """
    if section.level == 0:
        return ''
    hierarchy = section.heading_hierarchy
    if not hierarchy:
        return f'{"#" * section.level} {section.heading}'
    return '\n'.join(f'{"#" * node.level} {node.text}' for node in hierarchy)


def _section_full_text(section: Section) -> str:
    prefix = _build_heading_prefix(section)
    if not prefix:
        return section.body
    if not section.body:
        return prefix
    return f'{prefix}\n\n{section.body}'


def _same_heading_node(left: HeadingNode, right: HeadingNode) -> bool:
    return left.level == right.level and left.text == right.text


def _has_same_parent_hierarchy(left: list[HeadingNode], right: list[HeadingNode]) -> bool:
    if len(left) != len(right):
        return False
    return all(_same_heading_node(node, right[index]) for index, node in enumerate(left[:-1]))


def _has_compatible_branch(pred: Section, section: Section) -> bool:
    if pred.level == 0:
        return section.level in {0, 1}
    if pred.heading_hierarchy is None or section.heading_hierarchy is None:
        return False
    if pred.level != section.level:
        return False
    return _has_same_parent_hierarchy(pred.heading_hierarchy, section.heading_hierarchy)


def _can_merge(pred: Section, section: Section, join_threshold: int) -> bool:
    if not _has_compatible_branch(pred, section):
        return False
    combined_tokens = _estimate_tokens(_section_full_text(pred)) + _estimate_tokens(_section_full_text(section))
    return combined_tokens < join_threshold


def _merge_into(pred: Section, section: Section) -> None:
    append_text = f'{"#" * section.level} {section.heading}\n\n{section.body}'
    pred.body = f'{pred.body}\n\n{append_text}' if pred.body else append_text


def _merge_at_level(sections: list[Section], level: int, join_threshold: int) -> list[Section]:
    merged: list[Section] = []
    for section in sections:
        if section.level != level or not merged:
            merged.append(section)
            continue
        pred = merged[-1]
        if _can_merge(pred, section, join_threshold):
            _merge_into(pred, section)
        else:
            merged.append(section)
    return merged


_TINY_SECTION_THRESHOLD = 100


def _absorb_tiny_sections(sections: list[Section], join_threshold: int) -> list[Section]:
    """Merge sections smaller than *_TINY_SECTION_THRESHOLD* tokens into their predecessor.

    The level-based merge only merges siblings at the same heading depth.
    This pass catches short child sections (e.g. a 3-line ``### IMPACTS``
    under a ``## Topic 5``) that would otherwise become isolated chunks
    with no semantic link to the surrounding context.

    Only absorbs into a predecessor that already carries meaningful content,
    preventing cascading absorption of uniformly tiny sections.
    """
    if not sections:
        return sections
    result: list[Section] = [sections[0]]
    for section in sections[1:]:
        tokens = _estimate_tokens(_section_full_text(section))
        pred_tokens = _estimate_tokens(_section_full_text(result[-1]))
        if tokens < _TINY_SECTION_THRESHOLD and pred_tokens >= _TINY_SECTION_THRESHOLD:
            combined = pred_tokens + tokens
            if combined < join_threshold:
                _merge_into(result[-1], section)
                continue
        result.append(section)
    return result


def _merge_sections(sections: list[Section], join_threshold: int) -> list[Section]:
    max_level = 0
    for section in sections:
        if section.level > max_level:
            max_level = section.level
    merged = sections
    for level in range(max_level, 0, -1):
        merged = _merge_at_level(merged, level, join_threshold)
    # Absorb tiny child sections into their predecessor regardless of level.
    # Without this, a short "### IMPACTS:" section (level 3) remains isolated
    # from its parent "## Topic 5" section (level 2) and lacks the semantic
    # context that embeddings need for retrieval.
    merged = _absorb_tiny_sections(merged, join_threshold)
    return merged


def _extract_first_page_number(text: str) -> int | None:
    match = PAGE_MARKER_RE.search(text)
    return int(match.group(1)) if match else None


def _format_heading_context(hierarchy: list[HeadingNode] | None) -> str | None:
    if not hierarchy:
        return None
    return ' > '.join(node.text for node in hierarchy)


def _clone_section(section: Section, body: str) -> Section:
    return Section(
        level=section.level,
        heading=section.heading,
        body=body,
        heading_hierarchy=list(section.heading_hierarchy) if section.heading_hierarchy else None,
    )


def _attach_standalone_page_markers(units: list[str], separator: str) -> list[str]:
    combined: list[str] = []
    pending_markers: list[str] = []
    for unit in units:
        if PAGE_MARKER_RE.search(unit):
            pending_markers.append(_trim(unit))
            continue
        if pending_markers:
            combined.append(_trim(f'{separator.join(pending_markers)}{separator}{unit}'))
            pending_markers = []
            continue
        combined.append(unit)
    if pending_markers:
        if not combined:
            combined.append(separator.join(pending_markers))
        else:
            combined[-1] = _trim(f'{combined[-1]}{separator}{separator.join(pending_markers)}')
    return combined


def _split_paragraphs(text: str) -> list[str]:
    parts: list[str] = []
    for part in BLANK_LINE_SPLIT_RE.split(_trim(text)):
        trimmed = _trim(part)
        if trimmed:
            parts.append(trimmed)
    return _attach_standalone_page_markers(parts, '\n\n')


def _is_table_block(block: str) -> bool:
    lines = [_trim(line) for line in block.split('\n') if _trim(line)]
    if len(lines) < 2:
        return False
    return all(TABLE_ROW_RE.match(line) or TABLE_SEPARATOR_RE.match(line) for line in lines)


def _is_list_block(block: str) -> bool:
    lines = [_trim_end(line) for line in block.split('\n')]
    return any(LIST_ITEM_RE.match(line) for line in lines)


def _split_list_block(block: str) -> list[str]:
    lines = block.split('\n')
    groups: list[str] = []
    current: list[str] = []
    for line in lines:
        if LIST_ITEM_RE.match(line) and current:
            groups.append(_trim('\n'.join(current)))
            current = [line]
            continue
        current.append(line)
    if current:
        groups.append(_trim('\n'.join(current)))
    return [group for group in groups if group]


def _split_list_item_block(block: str, max_tokens: int) -> list[str]:
    if _estimate_tokens(block) <= max_tokens:
        return [_trim(block)]
    first_line, *rest_lines = block.split('\n')
    match = re.match(r'^(\s*(?:[-*+]\s+|\d+[.)]\s+))(.*)$', first_line)
    if match is None:
        return _split_sentence_block(block, max_tokens)
    marker = match.group(1) or ''
    remainder = _trim('\n'.join([match.group(2) or '', *rest_lines]))
    if not remainder:
        return [_trim(block)]
    content_budget = max(1, max_tokens - _estimate_tokens(marker))
    return [_trim_end(f'{marker}{part}') for part in _split_sentence_block(remainder, content_budget)]


def _split_table_block(block: str) -> list[str]:
    return [_trim(line) for line in block.split('\n') if _trim(line)]


def _is_inline_page_marker_transition(text: str, index: int, marker_length: int) -> bool:
    previous_char = text[index - 1] if index > 0 else None
    next_index = index + marker_length
    next_char = text[next_index] if next_index < len(text) else None
    has_inline_prefix = previous_char is not None and PAGE_MARKER_SEPARATOR_RE.search(previous_char) is None
    has_inline_suffix = next_char is not None and PAGE_MARKER_SEPARATOR_RE.search(next_char) is None
    return has_inline_prefix or has_inline_suffix


def _split_inline_page_marker_segments(text: str) -> list[str]:
    segments: list[str] = []
    cursor = 0
    for match in PAGE_MARKER_RE.finditer(text):
        index = match.start()
        marker = match.group(0)
        before_marker = _trim(text[cursor:index])
        if before_marker:
            segments.append(f'{before_marker}{marker}')
        elif segments:
            segments[-1] = f'{segments[-1]}{marker}'
        else:
            segments.append(marker)
        cursor = index + len(marker)
    after_last_marker = _trim(text[cursor:])
    if after_last_marker:
        segments.append(after_last_marker)
    return segments


def _append_sentence_segments(target: list[str], text: str) -> None:
    trimmed = _trim(text)
    if not trimmed:
        return
    if _has_inline_page_marker_transition(trimmed):
        target.extend(_split_inline_page_marker_segments(trimmed))
        return
    target.extend(_split_sentences_without_inline_markers(trimmed))


def _split_sentences_without_inline_markers(text: str) -> list[str]:
    sentences = [_trim(match.group(0)) for match in SENTENCE_RE.finditer(text)]
    sentences = [sentence for sentence in sentences if sentence]
    if sentences:
        separator = ' ' if _has_visible_word_separator(text) else ''
        return _attach_standalone_page_markers(sentences, separator)
    return _split_whitespace(_trim(text))


def _has_inline_page_marker_transition(text: str) -> bool:
    for match in PAGE_MARKER_RE.finditer(text):
        marker = match.group(0)
        if _is_inline_page_marker_transition(text, match.start(), len(marker)):
            return True
    return False


def _split_sentences(text: str) -> list[str]:
    segments: list[str] = []
    cursor = 0
    saw_standalone_marker = False
    for match in PAGE_MARKER_RE.finditer(text):
        index = match.start()
        marker = match.group(0)
        if _is_inline_page_marker_transition(text, index, len(marker)):
            continue
        saw_standalone_marker = True
        _append_sentence_segments(segments, text[cursor:index])
        segments.append(_trim(marker))
        cursor = index + len(marker)
    if saw_standalone_marker:
        _append_sentence_segments(segments, text[cursor:])
        separator = ' ' if _has_visible_word_separator(text) else ''
        return _attach_standalone_page_markers(segments, separator)
    if _has_inline_page_marker_transition(text):
        inline_segments = _split_inline_page_marker_segments(text)
        if len(inline_segments) > 1:
            return inline_segments
    return _split_sentences_without_inline_markers(text)


def _has_visible_word_separator(text: str) -> bool:
    text_without_markers = PAGE_MARKER_RE.sub('', text)
    return VISIBLE_WORD_SEPARATOR_RE.search(text_without_markers) is not None


def _split_words(text: str) -> list[str]:
    words: list[str] = []
    cursor = 0
    for match in PAGE_MARKER_RE.finditer(text):
        index = match.start()
        before_marker = _trim(text[cursor:index])
        if before_marker:
            words.extend(_split_whitespace(before_marker))
        words.append(match.group(0))
        cursor = index + len(match.group(0))
    after_last_marker = _trim(text[cursor:])
    if after_last_marker:
        words.extend(_split_whitespace(after_last_marker))
    return words


def _split_oversized_word(word: str, max_tokens: int) -> list[str]:
    trimmed = _trim(word)
    if not trimmed:
        return []
    max_chars = max(1, max_tokens * 4)
    return [trimmed[start : start + max_chars] for start in range(0, len(trimmed), max_chars)]


def _split_oversized_word_preserving_page_markers(word: str, max_tokens: int) -> list[str]:
    marker_matches = list(PAGE_MARKER_RE.finditer(word))
    if not marker_matches:
        return _split_oversized_word(word, max_tokens)
    units: list[str] = []
    cursor = 0
    for match in marker_matches:
        index = match.start()
        marker = match.group(0)
        before_marker = word[cursor:index]
        if before_marker:
            units.extend(_split_oversized_word(before_marker, max_tokens))
        units.append(marker)
        cursor = index + len(marker)
    after_last_marker = word[cursor:]
    if after_last_marker:
        units.extend(_split_oversized_word(after_last_marker, max_tokens))

    def split_oversized_unit(unit: str, unit_max_tokens: int) -> list[str]:
        if PAGE_MARKER_RE.search(unit):
            return [unit]
        return _split_oversized_word(unit, unit_max_tokens)

    return _pack_units(units, max_tokens, '', split_oversized_unit)


def _join_units(units: list[str], separator: str) -> str:
    return _trim(separator.join(units))


def _rebalance_tiny_tail(chunks: list[list[str]], separator: str, max_tokens: int) -> None:
    if len(chunks) < 2:
        return
    min_tail_tokens = max(1, math.floor(max_tokens * 0.35))
    tail = chunks[-1]
    previous = chunks[-2]
    tail_tokens = _estimate_tokens(_join_units(tail, separator))
    if tail_tokens >= min_tail_tokens or len(previous) <= 1:
        return
    previous_tokens = _estimate_tokens(_join_units(previous, separator))
    imbalance = abs(previous_tokens - tail_tokens)
    while len(previous) > 1:
        moved = previous[-1]
        candidate_previous = previous[:-1]
        candidate_tail = [moved, *tail]
        candidate_previous_tokens = _estimate_tokens(_join_units(candidate_previous, separator))
        candidate_tail_tokens = _estimate_tokens(_join_units(candidate_tail, separator))
        candidate_imbalance = abs(candidate_previous_tokens - candidate_tail_tokens)
        if (
            candidate_previous_tokens > max_tokens
            or candidate_tail_tokens > max_tokens
            or candidate_imbalance > imbalance
        ):
            break
        previous.pop()
        tail.insert(0, moved)
        previous_tokens = candidate_previous_tokens
        tail_tokens = candidate_tail_tokens
        imbalance = candidate_imbalance
        if tail_tokens >= min_tail_tokens:
            break


def _pack_units(
    units: list[str],
    max_tokens: int,
    separator: str,
    split_oversized_fn: Callable[[str, int], list[str]],
) -> list[str]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if _estimate_tokens(unit) > max_tokens:
            if current:
                chunks.append(current)
                current = []
            for split_unit in split_oversized_fn(unit, max_tokens):
                chunks.append([split_unit])
            continue
        candidate = [*current, unit]
        if current and _estimate_tokens(_join_units(candidate, separator)) > max_tokens:
            chunks.append(current)
            current = [unit]
            continue
        current = candidate
    if current:
        chunks.append(current)
    _rebalance_tiny_tail(chunks, separator, max_tokens)
    return [_join_units(chunk, separator) for chunk in chunks]


def _split_paragraph_block(block: str, max_tokens: int) -> list[str]:
    if _estimate_tokens(block) <= max_tokens:
        return [_trim(block)]
    if _is_table_block(block):
        rows = _split_table_block(block)
        if len(rows) > 1:
            return _pack_units(rows, max_tokens, '\n', _split_sentence_block)
    if _is_list_block(block):
        items = _split_list_block(block)
        if len(items) > 1:
            return _pack_units(items, max_tokens, '\n', _split_list_item_block)
        return _split_list_item_block(block, max_tokens)
    return _split_sentence_block(block, max_tokens)


def _split_sentence_block(block: str, max_tokens: int) -> list[str]:
    if _estimate_tokens(block) <= max_tokens:
        return [_trim(block)]
    sentences = _split_sentences(block)
    if len(sentences) > 1:
        separator = ' ' if _has_visible_word_separator(block) else ''
        return _pack_units(sentences, max_tokens, separator, _split_word_block)
    return _split_word_block(block, max_tokens)


def _split_word_block(block: str, max_tokens: int) -> list[str]:
    words = _split_words(block)
    if not words:
        return []
    separator = ' ' if _has_visible_word_separator(block) else ''
    if len(words) == 1 and _estimate_tokens(words[0]) > max_tokens:
        return _split_oversized_word_preserving_page_markers(words[0], max_tokens)

    def split_oversized_unit(unit: str, unit_max_tokens: int) -> list[str]:
        if _estimate_tokens(unit) > unit_max_tokens:
            return _split_oversized_word_preserving_page_markers(unit, unit_max_tokens)
        return [unit]

    return _pack_units(words, max_tokens, separator, split_oversized_unit)


def _split_section(section: Section, max_chunk_tokens: int) -> list[Section]:
    if _estimate_tokens(_section_full_text(section)) <= max_chunk_tokens:
        return [section]
    if not _trim(section.body):
        return [section]
    prefix = _build_heading_prefix(section)
    prefix_tokens = _estimate_tokens(f'{prefix}\n\n') if prefix else 0
    body_token_budget = max(1, max_chunk_tokens - prefix_tokens)
    paragraphs = _split_paragraphs(section.body)
    if len(paragraphs) > 1:
        body_chunks = _pack_units(paragraphs, body_token_budget, '\n\n', _split_paragraph_block)
    else:
        body_chunks = _split_paragraph_block(section.body, body_token_budget)
    return [_clone_section(section, chunk) for chunk in body_chunks if _trim(chunk)]


def _split_oversized_sections(sections: list[Section], max_chunk_tokens: int) -> list[Section]:
    split_sections: list[Section] = []
    for section in sections:
        split_sections.extend(_split_section(section, max_chunk_tokens))
    return split_sections


def chunk_markdown(markdown: str, *, join_threshold: int = 500) -> list[ChunkData]:
    if not markdown or not _trim(markdown):
        return []
    max_chunk_tokens = join_threshold * 2
    sections = _parse_into_sections(markdown)
    _build_heading_hierarchies(sections)
    merged = _merge_sections(sections, join_threshold)
    split_sections = _split_oversized_sections(merged, max_chunk_tokens)
    chunks: list[ChunkData] = []
    for index, section in enumerate(split_sections):
        content = _section_full_text(section)
        chunks.append(
            ChunkData(
                content=content,
                page_number=_extract_first_page_number(content),
                chunk_index=index,
                heading_context=_format_heading_context(section.heading_hierarchy),
                heading_hierarchy=section.heading_hierarchy,
            )
        )
    return chunks


__all__ = [
    'ChunkData',
    'HeadingNode',
    'chunk_markdown',
]
