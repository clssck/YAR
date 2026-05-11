from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from yar.constants import DEFAULT_ENTITY_TYPES
from yar.utils import compute_mdhash_id, is_float_regex, logger, sanitize_and_normalize_extracted_text

_ENTITY_TYPE_ALIASES = {
    'activity': 'event',
    'animal': 'concept',
    'batch': 'artifact',
    'date': 'data',
    'dose': 'data',
    'group': 'organization',
    'other': 'concept',
    'people': 'person',
    'process': 'method',
    'project': 'event',
    'role': 'person',
    'stage': 'event',
    'study': 'document',
    'task': 'method',
    'team': 'organization',
    'time': 'data',
    'unknown': 'concept',
    'workstream': 'organization',
}


def _normalized_entity_type_lookup(entity_types: list[str] | None) -> dict[str, str]:
    configured_types = entity_types or DEFAULT_ENTITY_TYPES
    lookup: dict[str, str] = {}
    for entity_type in configured_types:
        normalized = str(entity_type).replace(' ', '').lower()
        if normalized:
            lookup[normalized] = normalized
    return lookup


def _fallback_entity_type(allowed_types: Mapping[str, str]) -> str:
    for preferred_type in ('concept', 'data', 'document'):
        if preferred_type in allowed_types:
            return allowed_types[preferred_type]
    return next(iter(allowed_types.values()), 'concept')


def normalize_extracted_entity_type(raw_entity_type: str, entity_types: list[str] | None) -> str:
    allowed_types = _normalized_entity_type_lookup(entity_types)
    normalized_type = raw_entity_type.replace(' ', '').lower()
    if normalized_type in allowed_types:
        return allowed_types[normalized_type]

    alias_type = _ENTITY_TYPE_ALIASES.get(normalized_type)
    if alias_type in allowed_types:
        return allowed_types[alias_type]

    fallback_type = _fallback_entity_type(allowed_types)
    logger.debug('Normalizing unsupported entity type %r to %r', raw_entity_type, fallback_type)
    return fallback_type


_RELATION_KEYWORD_LIMIT = 3
_RELATION_KEYWORD_CANONICAL_MAP = {
    'attend': 'attends',
    'attended': 'attends',
    'attending': 'attends',
    'attends': 'attends',
    'approve': 'approves',
    'approved': 'approves',
    'approves': 'approves',
    'approving': 'approves',
    'assess': 'assesses',
    'assessed': 'assesses',
    'assesses': 'assesses',
    'assessing': 'assesses',
    'collaborate': 'collaborates',
    'collaborated': 'collaborates',
    'collaborates': 'collaborates',
    'collaborating': 'collaborates',
    'collaborate with': 'collaborates with',
    'collaborated with': 'collaborates with',
    'collaborates with': 'collaborates with',
    'collaborating with': 'collaborates with',
    'partner with': 'partnered with',
    'partnered with': 'partnered with',
    'partnering with': 'partnered with',
    'partners with': 'partnered with',
    'develop': 'develops',
    'developed': 'develops',
    'developing': 'develops',
    'develops': 'develops',
    'evaluate': 'evaluates',
    'evaluated': 'evaluates',
    'evaluates': 'evaluates',
    'evaluating': 'evaluates',
    'include': 'includes',
    'included': 'includes',
    'includes': 'includes',
    'including': 'includes',
    'involve': 'involves',
    'involved': 'involves',
    'involves': 'involves',
    'involving': 'involves',
    'lead': 'leads',
    'leading': 'leads',
    'leads': 'leads',
    'led': 'leads',
    'manufacture': 'manufactures',
    'manufactured': 'manufactures',
    'manufactures': 'manufactures',
    'manufacturing': 'manufactures',
    'participate': 'participates in',
    'participated': 'participates in',
    'participated in': 'participates in',
    'participates': 'participates in',
    'participates in': 'participates in',
    'participating': 'participates in',
    'participating in': 'participates in',
    'produce': 'produces',
    'produced': 'produces',
    'produces': 'produces',
    'producing': 'produces',
    'pose risk of': 'poses risk of',
    'pose risk to': 'poses risk to',
    'posed risk of': 'poses risk of',
    'posed risk to': 'poses risk to',
    'poses risk of': 'poses risk of',
    'poses risk to': 'poses risk to',
    'poses risks of': 'poses risk of',
    'poses risks to': 'poses risk to',
    'require': 'requires',
    'required': 'requires',
    'requires': 'requires',
    'requiring': 'requires',
    'represent': 'represents',
    'represented': 'represents',
    'representing': 'represents',
    'represents': 'represents',
    'review': 'reviews',
    'reviewed': 'reviews',
    'reviews': 'reviews',
    'reviewing': 'reviews',
    'sent to': 'sends to',
    'send to': 'sends to',
    'sends to': 'sends to',
    'sending to': 'sends to',
    'specifies': 'specifies',
    'specified': 'specifies',
    'specify': 'specifies',
    'specifying': 'specifies',
    'submit': 'submits',
    'submitted': 'submits',
    'submitting': 'submits',
    'submits': 'submits',
    'submit to': 'submits to',
    'submitted to': 'submits to',
    'submitting to': 'submits to',
    'submits to': 'submits to',
    'target': 'targets',
    'targeted': 'targets',
    'targeting': 'targets',
    'targets': 'targets',
    'send': 'sends',
    'sends': 'sends',
    'sent': 'sends',
    'support': 'supports',
    'supported': 'supports',
    'supporting': 'supports',
    'supports': 'supports',
    'use': 'uses',
    'used': 'uses',
    'uses': 'uses',
    'using': 'uses',
    'utilize': 'uses',
    'utilized': 'uses',
    'utilizes': 'uses',
    'utilizing': 'uses',
}

_RELATION_SEARCH_HINT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ('represented by', 'represents', 'representative', 'on behalf'),
        'contributor contributes representative represented on behalf working group member',
    ),
    (
        ('collaborates with', 'collaborates', 'collaboration', 'alliance', 'partnership'),
        'collaboration partnership alliance joint transition relationship',
    ),
    (
        ('includes', 'included in', 'specifies', 'adds to', 'recommendations for'),
        'document section recommendation manual dossier specification included',
    ),
    (
        ('requires', 'must align', 'must be contacted', 'contacted for'),
        'requirement required contact approval workflow',
    ),
    (
        ('member of', 'is a member of', 'belongs to', 'participates in'),
        'team membership working group participant member of role',
    ),
    (
        ('manufactured by', 'manufactures', 'produced by', 'produces'),
        'vendor supplier manufacturer manufactured produces production',
    ),
    (
        ('treats', 'treatment for', 'indicated for', 'targets'),
        'treatment indication therapy targets disease patient',
    ),
    (
        ('develops', 'developed', 'developed in collaboration with'),
        'development program drug pipeline develops candidate',
    ),
    (
        ('applicable to', 'applies to', 'scope of', 'covers'),
        'scope applicable applies coverage applicability target population',
    ),
    (
        ('supports', 'supported by', 'supported_by', 'enables', 'supports use of'),
        'support supports enables backing rationale',
    ),
    (
        ('tested', 'tested in', 'verified', 'evaluated'),
        'testing tested verification evaluation study qualified',
    ),
    (
        (
            'approves',
            'approved',
            'approved by',
            'approval from',
            'cleared by',
            'cleared under',
            'regulated as',
            'must comply with',
            'consulted for',
        ),
        'regulatory approval clearance compliance authority consultation approved',
    ),
    (
        ('classified as', 'categorized as', 'category', 'type of', 'part of', 'among'),
        'classification category taxonomy type class part of grouping',
    ),
    (
        ('focuses on', 'focused on', 'addresses', 'addressed by', 'describes', 'defines', 'references'),
        'topic focus scope describes defines addresses references subject',
    ),
    (
        (
            'negotiated',
            'agreement',
            'contract',
            'license',
            'licensed to',
            'rights to',
            'royalties',
            'restructured alliance',
        ),
        'business agreement contract license rights commercial negotiation alliance',
    ),
    (
        ('tests', 'tested', 'released batch from', 'release', 'validation', 'qualified', 'non-gmp studies'),
        'testing batch release validation qualification study evidence verified',
    ),
    (
        ('communicates with', 'communication', 'sends memo to', 'reports within', 'coordinates', 'managed by'),
        'communication coordination reporting memo project management workflow',
    ),
    (
        ('authored', 'prepared by', 'published by', 'drafted', 'presentation', 'handbook'),
        'document author prepared published drafted presentation handbook source',
    ),
    (
        ('based in', 'located in', 'site', 'facility', 'country', 'market'),
        'location site facility geography country market region',
    ),
    (
        ('reviews', 'reviewed by', 'under review', 'audit', 'audited'),
        'review audit examination oversight peer review',
    ),
    (
        ('assesses', 'assessed by', 'assessment of', 'appraisal'),
        'assessment evaluation audit review appraisal',
    ),
    (
        ('involves', 'involved in', 'interaction with', 'component of'),
        'involvement coordination interaction component relationship',
    ),
    (
        ('depends_on', 'depends on', 'dependent on', 'prerequisite for'),
        'dependency prerequisite required-by depends upon',
    ),
)
_PREDICATE_ALIASES: dict[str, str] = {
    'applies': 'uses',
    'apply': 'uses',
    'mitigates': 'mitigates',
    'supportedby': 'supported_by',
    'dependson': 'depends_on',
}

_CANONICAL_PREDICATE_VALUES: frozenset[str] = frozenset(_RELATION_KEYWORD_CANONICAL_MAP.values()) | frozenset(
    _PREDICATE_ALIASES.values()
)

_DIRECTIONAL_RELATION_KEYWORD_SWAP_MAP = {
    'represented by': 'represents',
    'is represented by': 'represents',
    'was represented by': 'represents',
    'were represented by': 'represents',
}


def _predicate_alias_key(keyword: str) -> str:
    return re.sub(r'[-\s]+', '', keyword.casefold())


_RELATION_IMPACT_PHRASES = (
    'affect',
    'affected',
    'affects',
    'can pose risk',
    'can pose risks',
    'cause',
    'caused',
    'causes',
    'consequence',
    'consequences',
    'due to',
    'impact',
    'impacted',
    'impacts',
    'influence',
    'influenced',
    'influences',
    'led to',
    'leads to',
    'outcome',
    'outcomes',
    'pose risk',
    'pose risks',
    'poses risk',
    'poses risks',
    'risk to',
    'risks to',
    'resulted in',
    'resulting in',
    'results in',
)
_RELATION_NEGATIVE_EVIDENCE_PHRASES = (
    'does not support',
    'insufficient evidence',
    'lack of evidence',
    'lacked evidence',
    'lacking evidence',
    'lacks evidence',
    'limited evidence',
    'no association',
    'no evidence',
    'no stringent evidence',
    'not associated',
    'not support',
    'without evidence',
)


def _canonicalize_relation_keyword(keyword: str) -> str:
    # Exact-phrase mapping only. Do not collapse inverse forms such as
    # "manufactured by", "used in", or "evaluated in" unless direction is also changed.
    canonical = _RELATION_KEYWORD_CANONICAL_MAP.get(keyword, keyword)
    return _PREDICATE_ALIASES.get(_predicate_alias_key(canonical), canonical)


_COMPOUND_RELATION_CONNECTORS = (' and ', ' then ')


def _clean_relation_keyword_surface(keyword: str) -> str:
    return re.sub(r'\s+', ' ', keyword).strip(' \t\r\n,.;').replace('\\,', ',').lower()


def _split_unescaped_commas(keyword: str) -> list[str]:
    return list(re.split(r'(?<!\\),', keyword))


def _canonicalize_relation_keyword_part(keyword: str) -> str:
    return _canonicalize_relation_keyword(_clean_relation_keyword_surface(keyword))


def _expand_canonical_relation_keyword(keyword: str) -> tuple[str, ...]:
    canonical = _canonicalize_relation_keyword_part(keyword)
    raw_parts = _split_unescaped_commas(canonical) if ',' in canonical else [canonical]
    expanded: list[str] = []

    for raw_part in raw_parts:
        part = _canonicalize_relation_keyword_part(raw_part)
        if not part:
            continue

        connector_match = next((connector for connector in _COMPOUND_RELATION_CONNECTORS if connector in part), None)
        if connector_match is None:
            expanded.append(part)
            continue

        first_part, second_part = part.split(connector_match, 1)
        first_canonical = _canonicalize_relation_keyword_part(first_part)
        if first_canonical:
            expanded.append(first_canonical)
        second_canonical = _canonicalize_relation_keyword_part(second_part)
        if second_canonical in _CANONICAL_PREDICATE_VALUES:
            expanded.append(second_canonical)

    return tuple(expanded)


def _raw_relation_keyword_terms(raw_keywords: str | Iterable[str]) -> tuple[str, ...]:
    raw_values = [raw_keywords] if isinstance(raw_keywords, str) else raw_keywords
    normalized_keywords: list[str] = []
    seen_keywords: set[str] = set()

    for raw_value in raw_values:
        keyword = _clean_relation_keyword_surface(str(raw_value).replace('，', ','))
        if not keyword or keyword in seen_keywords:
            continue
        normalized_keywords.append(keyword)
        seen_keywords.add(keyword)

    return tuple(normalized_keywords)


def normalize_relation_keyword_terms(
    raw_keywords: str | Iterable[str],
    *,
    max_keywords: int = _RELATION_KEYWORD_LIMIT,
) -> tuple[str, ...]:
    normalized_keywords: list[str] = []
    seen_keywords: set[str] = set()

    for keyword in _raw_relation_keyword_terms(raw_keywords):
        for canonical_keyword in _expand_canonical_relation_keyword(keyword):
            if not canonical_keyword or canonical_keyword in seen_keywords:
                continue
            normalized_keywords.append(canonical_keyword)
            seen_keywords.add(canonical_keyword)
            if max_keywords > 0 and len(normalized_keywords) >= max_keywords:
                return tuple(normalized_keywords)

    return tuple(normalized_keywords)


def normalize_relation_keywords(
    raw_keywords: str | Iterable[str],
    *,
    max_keywords: int = _RELATION_KEYWORD_LIMIT,
) -> str:
    return ', '.join(normalize_relation_keyword_terms(raw_keywords, max_keywords=max_keywords))


def normalize_relation_direction(
    source: str,
    target: str,
    raw_keywords: str | Iterable[str],
) -> tuple[str, str, tuple[str, ...]]:
    raw_terms = _raw_relation_keyword_terms(raw_keywords)
    if raw_terms and all(term in _DIRECTIONAL_RELATION_KEYWORD_SWAP_MAP for term in raw_terms):
        swapped_keywords = [_DIRECTIONAL_RELATION_KEYWORD_SWAP_MAP[term] for term in raw_terms]
        return target, source, normalize_relation_keyword_terms(swapped_keywords)
    return source, target, normalize_relation_keyword_terms(raw_terms)


def _contains_phrase(value: str, phrase: str) -> bool:
    if ' ' in phrase:
        return phrase in value
    return bool(re.search(rf'\b{re.escape(phrase)}\b', value))


def relation_semantic_search_hints(keywords: str, description: str) -> str:
    normalized = f'{keywords} {description}'.casefold()
    hints: list[str] = []

    def add_hint(hint: str) -> None:
        if hint and hint not in hints:
            hints.append(hint)

    if any(_contains_phrase(normalized, phrase) for phrase in _RELATION_IMPACT_PHRASES):
        add_hint('impact consequence effect result outcome')
    if any(_contains_phrase(normalized, phrase) for phrase in _RELATION_NEGATIVE_EVIDENCE_PHRASES):
        add_hint('negative evidence no evidence insufficient evidence not supported')
    for triggers, hint in _RELATION_SEARCH_HINT_RULES:
        if any(_contains_phrase(normalized, trigger) for trigger in triggers):
            add_hint(hint)
    return '; '.join(hints)


_RELATION_HINT_STOPWORDS = frozenset(
    {
        'and',
        'for',
        'from',
        'into',
        'onto',
        'the',
        'to',
        'with',
    }
)


def relation_generic_search_hints(source: str, target: str, keywords: str, description: str) -> str:
    raw_terms = [keywords, source, target, description]
    hints: list[str] = []
    for raw_term in raw_terms:
        normalized = re.sub(r'[^a-z0-9]+', ' ', str(raw_term).casefold())
        for token in normalized.split():
            if len(token) <= 2 or token in _RELATION_HINT_STOPWORDS or token in hints:
                continue
            hints.append(token)
            if len(hints) >= 24:
                return ' '.join(hints)
    return ' '.join(hints)


class RelationPolarity(str, Enum):
    AFFIRMATIVE = 'affirmative'
    NEGATIVE_EVIDENCE = 'negative_evidence'
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'


class RelationFacet(str, Enum):
    IMPACT = 'impact'
    CONSEQUENCE = 'consequence'
    EVIDENCE = 'evidence'


@dataclass(frozen=True)
class RelationSemantics:
    polarity: RelationPolarity
    facets: frozenset[RelationFacet]

    @classmethod
    def from_text(cls, keywords: str, evidence_text: str) -> RelationSemantics:
        normalized = f'{keywords} {evidence_text}'.casefold()
        facets: set[RelationFacet] = set()
        if any(_contains_phrase(normalized, phrase) for phrase in _RELATION_IMPACT_PHRASES):
            facets.add(RelationFacet.IMPACT)
            facets.add(RelationFacet.CONSEQUENCE)

        if any(_contains_phrase(normalized, phrase) for phrase in _RELATION_NEGATIVE_EVIDENCE_PHRASES):
            facets.add(RelationFacet.EVIDENCE)
            if 'insufficient evidence' in normalized or 'limited evidence' in normalized:
                polarity = RelationPolarity.INSUFFICIENT_EVIDENCE
            else:
                polarity = RelationPolarity.NEGATIVE_EVIDENCE
        else:
            polarity = RelationPolarity.AFFIRMATIVE

        return cls(polarity=polarity, facets=frozenset(facets))


@dataclass(frozen=True)
class EntityFact:
    name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str
    timestamp: int

    @classmethod
    def from_record(
        cls,
        record_attributes: list[str],
        chunk_key: str,
        timestamp: int,
        file_path: str = 'unknown_source',
        entity_types: list[str] | None = None,
    ) -> EntityFact | None:
        if len(record_attributes) != 4 or 'entity' not in record_attributes[0]:
            return None

        entity_name = sanitize_and_normalize_extracted_text(record_attributes[1], remove_inner_quotes=True)
        if not entity_name or not entity_name.strip():
            logger.info("Empty entity name found after sanitization. Original: '%s'", record_attributes[1])
            return None

        entity_type = sanitize_and_normalize_extracted_text(record_attributes[2], remove_inner_quotes=True)
        if not entity_type.strip() or any(char in entity_type for char in ["'", '(', ')', '<', '>', '|', '/', '\\']):
            logger.warning('Entity extraction error: invalid entity type in: %s', record_attributes)
            return None
        entity_type = normalize_extracted_entity_type(entity_type, entity_types)

        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])
        if not entity_description.strip():
            logger.warning(
                "Entity extraction error: empty description for entity '%s' of type '%s'",
                entity_name,
                entity_type,
            )
            return None

        return cls(
            name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    def with_name(self, name: str) -> EntityFact:
        return replace(self, name=name)


@dataclass(frozen=True, order=True)
class RelationKey:
    src: str
    tgt: str

    @property
    def storage_pair(self) -> tuple[str, str]:
        first, second = sorted((self.src, self.tgt))
        return first, second

    @property
    def is_self_loop(self) -> bool:
        return self.src == self.tgt


@dataclass(frozen=True)
class RelationPredicate:
    keywords: tuple[str, ...]

    @classmethod
    def from_raw(
        cls,
        raw_keywords: str | Iterable[str],
        *,
        max_keywords: int = _RELATION_KEYWORD_LIMIT,
    ) -> RelationPredicate:
        return cls(normalize_relation_keyword_terms(raw_keywords, max_keywords=max_keywords))

    @property
    def text(self) -> str:
        return ', '.join(self.keywords)

    @property
    def primary(self) -> str:
        if not self.keywords:
            return ''

        first_keyword = self.keywords[0]
        if first_keyword in _CANONICAL_PREDICATE_VALUES:
            return first_keyword

        prefix_predicates = sorted(_CANONICAL_PREDICATE_VALUES, key=len, reverse=True)
        for predicate in prefix_predicates:
            if first_keyword.startswith(f'{predicate} '):
                return predicate

        return _clean_relation_keyword_surface(_split_unescaped_commas(first_keyword)[0])


@dataclass(frozen=True)
class RelationFact:
    key: RelationKey
    predicate: RelationPredicate
    evidence_text: str
    weight: float
    source_id: str
    file_path: str
    timestamp: int
    semantics: RelationSemantics
    evidence_spans: tuple[str, ...] = ()

    @classmethod
    def from_record(
        cls,
        record_attributes: list[str],
        chunk_key: str,
        timestamp: int,
        file_path: str = 'unknown_source',
        evidence_spans: Iterable[str] = (),
    ) -> RelationFact | None:
        if len(record_attributes) != 5 or 'relation' not in record_attributes[0]:
            return None

        source = sanitize_and_normalize_extracted_text(record_attributes[1], remove_inner_quotes=True)
        target = sanitize_and_normalize_extracted_text(record_attributes[2], remove_inner_quotes=True)
        if not source or not target or source == target:
            return None

        source, target, normalized_keywords = normalize_relation_direction(source, target, record_attributes[3])
        predicate = RelationPredicate(normalized_keywords)
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])
        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )
        return cls(
            key=RelationKey(source, target),
            predicate=predicate,
            evidence_text=edge_description,
            weight=weight,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
            semantics=RelationSemantics.from_text(predicate.text, edge_description),
            evidence_spans=tuple(dict.fromkeys(str(span).strip() for span in evidence_spans if str(span).strip())),
        )

    @property
    def description(self) -> str:
        return self.evidence_text

    @property
    def keywords(self) -> str:
        return self.predicate.text

    def with_key(self, key: RelationKey) -> RelationFact:
        return replace(self, key=key, semantics=RelationSemantics.from_text(self.predicate.text, self.evidence_text))

    def with_evidence_spans(self, evidence_spans: Iterable[str]) -> RelationFact:
        return replace(
            self, evidence_spans=tuple(dict.fromkeys(str(span).strip() for span in evidence_spans if str(span).strip()))
        )


@dataclass(frozen=True)
class ChunkExtractionResult:
    nodes: dict[str, list[EntityFact]]
    edges: dict[RelationKey, list[RelationFact]]


@dataclass(frozen=True)
class RelationSummary:
    key: RelationKey
    predicate: RelationPredicate
    description: str
    weight: float
    source_id: str
    file_path: str
    created_at: int
    truncate: str
    semantics: RelationSemantics
    evidence_spans: tuple[str, ...] = ()

    @property
    def keywords(self) -> str:
        return self.predicate.text


@dataclass(frozen=True)
class RelationStorageProjection:
    graph_edge_data: dict[str, Any]
    relation_vdb_id: str
    relation_vdb_delete_ids: list[str]
    relation_vdb_payload: dict[str, Any]


def build_relation_vector_content(summary: RelationSummary) -> str:
    base_content = '\n'.join(
        [
            f'{summary.keywords}\t{summary.key.src}',
            summary.key.tgt,
            summary.description,
            f'relation: {summary.key.src} --{summary.predicate.primary}--> {summary.key.tgt}',
            f'source_entity: {summary.key.src}',
            f'target_entity: {summary.key.tgt}',
            f'predicate: {summary.predicate.primary}',
        ]
    )
    semantic_lines: list[str] = []
    if summary.evidence_spans:
        semantic_lines.append(
            'evidence_spans: ' + ' | '.join(dict.fromkeys(span for span in summary.evidence_spans if span))
        )
    if summary.semantics.polarity != RelationPolarity.AFFIRMATIVE:
        semantic_lines.append(f'polarity: {summary.semantics.polarity.value}')
    if summary.semantics.facets:
        facets = ', '.join(facet.value for facet in sorted(summary.semantics.facets, key=lambda item: item.value))
        semantic_lines.append(f'facets: {facets}')
    semantic_hints = relation_semantic_search_hints(summary.keywords, summary.description)
    if not semantic_hints:
        semantic_hints = relation_generic_search_hints(
            summary.key.src,
            summary.key.tgt,
            summary.keywords,
            summary.description,
        )
    if semantic_hints:
        semantic_lines.append(f'search_hints: {semantic_hints}')
    if semantic_lines:
        return f'{base_content}\n' + '\n'.join(semantic_lines)
    return base_content


def build_relation_storage_projection(summary: RelationSummary) -> RelationStorageProjection:
    canonical_src, canonical_tgt = summary.key.storage_pair
    rel_vdb_id = compute_mdhash_id(canonical_src + canonical_tgt, prefix='rel-')
    rel_vdb_id_reverse = compute_mdhash_id(canonical_tgt + canonical_src, prefix='rel-')
    graph_edge_data = {
        'weight': summary.weight,
        'description': summary.description,
        'keywords': summary.keywords,
        'source_id': summary.source_id,
        'file_path': summary.file_path,
        'created_at': summary.created_at,
        'truncate': summary.truncate,
    }
    rel_content = build_relation_vector_content(summary)
    return RelationStorageProjection(
        graph_edge_data=graph_edge_data,
        relation_vdb_id=rel_vdb_id,
        relation_vdb_delete_ids=[rel_vdb_id, rel_vdb_id_reverse],
        relation_vdb_payload={
            rel_vdb_id: {
                'src_id': summary.key.src,
                'tgt_id': summary.key.tgt,
                'source_id': summary.source_id,
                'content': rel_content,
                'keywords': summary.keywords,
                'description': summary.description,
                'weight': summary.weight,
                'file_path': summary.file_path,
            }
        },
    )
