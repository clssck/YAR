from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from yar.utils import logger

_ALIAS_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'entity_aliases.yaml'
_ALIAS_BOUNDARY_TEMPLATE = r'(?<!\w){alias}(?!\w)'


@dataclass(frozen=True)
class _AliasRule:
    canonical: str
    aliases: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]


def resolve_entity_filter(query: str) -> str | None:
    if os.getenv('ENABLE_AUTO_ENTITY_FILTER', 'true').lower() != 'true':
        return None

    normalized_query = _normalize_text(query)
    if not normalized_query:
        return None

    rules = _load_alias_rules()
    if not rules:
        return None

    matches = _find_matching_canonicals(normalized_query, rules)
    if not matches:
        return None

    if len(matches) > 1:
        logger.info(
            f'Entity alias resolver matched multiple canonicals {matches}; using {matches[0]} by config order'
        )

    return matches[0]


@lru_cache(maxsize=1)
def _load_alias_rules() -> tuple[_AliasRule, ...]:
    try:
        raw_config = _ALIAS_CONFIG_PATH.read_text(encoding='utf-8')
    except FileNotFoundError:
        logger.warning(f'Entity alias config not found at {_ALIAS_CONFIG_PATH}')
        return ()
    except OSError as exc:
        logger.warning(f'Failed to read entity alias config at {_ALIAS_CONFIG_PATH}: {exc}')
        return ()

    if not raw_config.strip():
        logger.warning(f'Entity alias config at {_ALIAS_CONFIG_PATH} is empty')
        return ()

    try:
        loaded_config = yaml.safe_load(raw_config)
    except yaml.YAMLError as exc:
        logger.warning(f'Failed to parse entity alias config at {_ALIAS_CONFIG_PATH}: {exc}')
        return ()

    raw_entries = _extract_raw_entries(loaded_config)
    if not raw_entries:
        logger.warning(f'Entity alias config at {_ALIAS_CONFIG_PATH} does not define any aliases')
        return ()

    rules = tuple(
        rule
        for index, entry in enumerate(raw_entries, start=1)
        if (rule := _build_alias_rule(entry, index)) is not None
    )
    if not rules:
        logger.warning(f'Entity alias config at {_ALIAS_CONFIG_PATH} did not produce any valid alias rules')

    return rules


def _extract_raw_entries(loaded_config: Any) -> list[Any]:
    if isinstance(loaded_config, list):
        return loaded_config

    if not isinstance(loaded_config, dict):
        return []

    entity_entries = loaded_config.get('entities', loaded_config)
    if isinstance(entity_entries, list):
        return entity_entries

    if isinstance(entity_entries, dict):
        return [
            {'canonical': canonical, 'aliases': aliases}
            for canonical, aliases in entity_entries.items()
        ]

    return []


def _build_alias_rule(entry: Any, index: int) -> _AliasRule | None:
    if not isinstance(entry, dict):
        logger.warning(f'Ignoring non-mapping alias entry at position {index}')
        return None

    canonical = _normalize_canonical(entry.get('canonical'))
    if not canonical:
        logger.warning(f'Ignoring alias entry at position {index} without a canonical value')
        return None

    aliases = _normalize_aliases(canonical, entry.get('aliases'))
    if not aliases:
        logger.warning(f'Ignoring alias entry for {canonical} without any valid aliases')
        return None

    return _AliasRule(
        canonical=canonical,
        aliases=aliases,
        patterns=tuple(_compile_alias_pattern(alias) for alias in aliases),
    )


def _normalize_canonical(value: Any) -> str:
    if not isinstance(value, str):
        return ''

    return ' '.join(value.split())


def _normalize_aliases(canonical: str, raw_aliases: Any) -> tuple[str, ...]:
    candidates: list[str] = [canonical]
    if isinstance(raw_aliases, str):
        candidates.append(raw_aliases)
    elif isinstance(raw_aliases, list):
        candidates.extend(alias for alias in raw_aliases if isinstance(alias, str))

    normalized_aliases: list[str] = []
    seen_aliases: set[str] = set()
    for alias in candidates:
        normalized_alias = _normalize_text(alias)
        if not normalized_alias or normalized_alias in seen_aliases:
            continue
        seen_aliases.add(normalized_alias)
        normalized_aliases.append(normalized_alias)

    return tuple(normalized_aliases)


def _normalize_text(value: str) -> str:
    return ' '.join(value.lower().split())


def _compile_alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(_ALIAS_BOUNDARY_TEMPLATE.format(alias=re.escape(alias)))


def _find_matching_canonicals(query: str, rules: tuple[_AliasRule, ...]) -> list[str]:
    matches: list[str] = []
    seen_matches: set[str] = set()

    for rule in rules:
        if any(pattern.search(query) for pattern in rule.patterns):
            if rule.canonical not in seen_matches:
                seen_matches.add(rule.canonical)
                matches.append(rule.canonical)

    return matches
