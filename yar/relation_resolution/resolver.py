"""LLM review for relation predicate canonicalization."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import json_repair

from yar.graph_model import RelationPredicate
from yar.utils import logger

from .config import DEFAULT_CONFIG, RelationResolutionConfig


@dataclass(frozen=True)
class RelationReviewResult:
    src: str
    tgt: str
    original_keywords: tuple[str, ...]
    canonical_keywords: tuple[str, ...]
    primary: str
    reasoning: str
    confidence: float


@dataclass(frozen=True)
class _PreparedRelationItem:
    src: str
    tgt: str
    original_keywords: tuple[str, ...]
    allowed_keywords: frozenset[str]
    evidence_spans: tuple[str, ...]


async def llm_review_relation_predicates_batch(
    items: Iterable[Mapping[str, Any]],
    *,
    llm_func: Callable[[str, str | None], Awaitable[str]],
    config: RelationResolutionConfig = DEFAULT_CONFIG,
) -> list[RelationReviewResult]:
    """Review relation predicate keywords with an LLM and return safe canonical choices."""
    prepared_items = [_prepare_item(item, config) for item in items]
    unchanged_results = [_unchanged_result(item, reasoning='Not reviewed by LLM') for item in prepared_items]

    if not prepared_items or not config.enabled:
        return unchanged_results

    review_positions = [
        index for index, item in enumerate(prepared_items) if len(item.original_keywords) >= config.min_keywords_for_review
    ]
    if not review_positions:
        return unchanged_results

    review_items = [prepared_items[index] for index in review_positions]

    try:
        from yar.prompt import PROMPTS

        prompt_template = PROMPTS['relation_predicate_review']
        prompt = prompt_template.format(
            relation_items=json.dumps(_prompt_relation_items(review_items), ensure_ascii=False, indent=2),
            allowed_predicates=json.dumps(_prompt_allowed_predicates(review_items), ensure_ascii=False, indent=2),
        )
        response = await llm_func(prompt, None)
    except Exception as exc:
        logger.error(f'Relation predicate LLM review failed: {exc}')
        return unchanged_results

    parsed_results = _parse_llm_json_response(response)
    if not parsed_results:
        return unchanged_results

    results = list(unchanged_results)
    for position, parsed_item in zip(review_positions, parsed_results, strict=False):
        prepared = prepared_items[position]
        reviewed = _validated_review_result(parsed_item, prepared, config)
        if reviewed is not None:
            results[position] = reviewed
        else:
            results[position] = _unchanged_result(prepared, reasoning='LLM review rejected')

    return results


def _prepare_item(item: Mapping[str, Any], config: RelationResolutionConfig) -> _PreparedRelationItem:
    src = _string_value(item.get('src'))
    tgt = _string_value(item.get('tgt'))
    predicate = RelationPredicate.from_raw(item.get('candidate_keywords', ()), max_keywords=config.max_predicates_per_pair)
    original_keywords = predicate.keywords
    return _PreparedRelationItem(
        src=src,
        tgt=tgt,
        original_keywords=original_keywords,
        allowed_keywords=frozenset(original_keywords),
        evidence_spans=_string_tuple(item.get('evidence_spans', ())),
    )


def _unchanged_result(
    item: _PreparedRelationItem,
    *,
    reasoning: str,
    confidence: float = 0.0,
) -> RelationReviewResult:
    predicate = RelationPredicate(item.original_keywords)
    return RelationReviewResult(
        src=item.src,
        tgt=item.tgt,
        original_keywords=item.original_keywords,
        canonical_keywords=item.original_keywords,
        primary=predicate.primary,
        reasoning=reasoning,
        confidence=_clamp_confidence(confidence),
    )


def _validated_review_result(
    parsed_item: Any,
    prepared: _PreparedRelationItem,
    config: RelationResolutionConfig,
) -> RelationReviewResult | None:
    if not isinstance(parsed_item, Mapping):
        return None

    if _string_value(parsed_item.get('src')) != prepared.src or _string_value(parsed_item.get('tgt')) != prepared.tgt:
        return None

    try:
        confidence = _clamp_confidence(float(parsed_item.get('confidence', 0.0)))
    except (TypeError, ValueError):
        return None

    reasoning = _string_value(parsed_item.get('reasoning'))
    if confidence < config.confidence_threshold:
        return _unchanged_result(prepared, reasoning=reasoning or 'Low confidence LLM review', confidence=confidence)

    primary_keywords = _normalize_returned_keywords(parsed_item.get('primary'), config)
    if len(primary_keywords) != 1:
        return None
    primary = primary_keywords[0]
    if primary not in prepared.allowed_keywords:
        return None

    canonical_keywords = _normalize_returned_keywords(
        parsed_item.get('canonical_keywords', (primary,)),
        config,
    )
    if not canonical_keywords:
        return None
    if any(keyword not in prepared.allowed_keywords for keyword in canonical_keywords):
        return None

    ordered_keywords = tuple(dict.fromkeys((primary, *canonical_keywords)))[: config.max_predicates_per_pair]
    if not ordered_keywords or ordered_keywords[0] != primary:
        return None

    return RelationReviewResult(
        src=prepared.src,
        tgt=prepared.tgt,
        original_keywords=prepared.original_keywords,
        canonical_keywords=ordered_keywords,
        primary=primary,
        reasoning=reasoning,
        confidence=confidence,
    )


def _normalize_returned_keywords(value: Any, config: RelationResolutionConfig) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_keywords: Any
    if isinstance(value, str):
        raw_keywords = value
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        raw_keywords = [_string_value(keyword) for keyword in value]
    else:
        raw_keywords = _string_value(value)
    return RelationPredicate.from_raw(raw_keywords, max_keywords=config.max_predicates_per_pair).keywords


def _prompt_relation_items(items: Iterable[_PreparedRelationItem]) -> list[dict[str, Any]]:
    return [
        {
            'src': item.src,
            'tgt': item.tgt,
            'candidate_keywords': list(item.original_keywords),
            'evidence_spans': list(item.evidence_spans),
        }
        for item in items
    ]


def _prompt_allowed_predicates(items: Iterable[_PreparedRelationItem]) -> list[dict[str, Any]]:
    return [
        {
            'src': item.src,
            'tgt': item.tgt,
            'allowed_keywords': list(item.original_keywords),
        }
        for item in items
    ]


def _parse_llm_json_response(response: str) -> list[dict[str, Any]]:
    def _normalize_parsed(parsed: Any) -> list[dict[str, Any]]:
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    def _try_parse(candidate: str) -> list[dict[str, Any]] | None:
        if not candidate.strip():
            return None
        try:
            parsed = json_repair.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, (dict, list)):
            return None
        return _normalize_parsed(parsed)

    text = str(response).strip()
    if not text:
        return []

    try:
        return _normalize_parsed(json.loads(text))
    except json.JSONDecodeError as exc:
        last_error: Exception = exc

    candidates: list[str] = []
    seen_candidates: set[str] = set()

    def _add_candidate(candidate: str) -> None:
        normalized = candidate.strip()
        if normalized and normalized not in seen_candidates:
            seen_candidates.add(normalized)
            candidates.append(normalized)

    _add_candidate(text)
    for match in re.finditer(r'```(?:json)?\s*([\s\S]*?)\s*```', text):
        _add_candidate(match.group(1))
    for pattern in (r'\[[\s\S]*\]', r'\{[\s\S]*\}'):
        match = re.search(pattern, text)
        if match:
            _add_candidate(match.group())

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    logger.warning(f'Failed to parse relation predicate LLM JSON response after repair attempts: {last_error}')
    return []


def _string_value(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    return (str(value),)


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
