from __future__ import annotations

from collections.abc import Iterator
from textwrap import dedent
from unittest.mock import Mock

import pytest

from yar.retrieval import resolve_entity_filter
from yar.retrieval import aliases as aliases_module


@pytest.fixture(autouse=True)
def clear_alias_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv('ENABLE_AUTO_ENTITY_FILTER', 'true')
    aliases_module._load_alias_rules.cache_clear()
    yield
    aliases_module._load_alias_rules.cache_clear()


def test_resolve_entity_filter_matches_aliases_with_whole_words(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: isatuximab
            aliases:
              - sarclisa
        """,
    )

    assert resolve_entity_filter('Latest SARCLISA response data') == 'isatuximab'
    assert resolve_entity_filter('Discuss isatuximab dosing guidance') == 'isatuximab'


def test_resolve_entity_filter_avoids_substring_false_positives(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: isatuximab
            aliases:
              - sarclisa
        """,
    )

    assert resolve_entity_filter('Compare sarclisab safety updates') is None


def test_resolve_entity_filter_uses_first_matching_canonical_and_logs_tie(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: First Canonical
            aliases:
              - shared alias
          - canonical: Second Canonical
            aliases:
              - shared alias
        """,
    )
    info_mock = Mock()
    monkeypatch.setattr(aliases_module.logger, 'info', info_mock)

    assert resolve_entity_filter('Need the shared alias benchmark') == 'First Canonical'
    info_mock.assert_called_once()
    assert 'using First Canonical by config order' in info_mock.call_args.args[0]


def test_resolve_entity_filter_returns_none_for_missing_or_empty_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    warning_mock = Mock()
    monkeypatch.setattr(aliases_module.logger, 'warning', warning_mock)
    monkeypatch.setattr(aliases_module, '_ALIAS_CONFIG_PATH', tmp_path / 'missing.yaml')

    assert resolve_entity_filter('sarclisa update') is None
    warning_mock.assert_called_once()

    warning_mock.reset_mock()
    empty_config_path = tmp_path / 'entity_aliases.yaml'
    empty_config_path.write_text('', encoding='utf-8')
    monkeypatch.setattr(aliases_module, '_ALIAS_CONFIG_PATH', empty_config_path)
    aliases_module._load_alias_rules.cache_clear()

    assert resolve_entity_filter('sarclisa update') is None
    warning_mock.assert_called_once()


def test_resolve_entity_filter_respects_disable_switch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: isatuximab
            aliases:
              - sarclisa
        """,
    )
    monkeypatch.setenv('ENABLE_AUTO_ENTITY_FILTER', 'false')

    assert resolve_entity_filter('sarclisa update') is None


def test_shipped_config_resolves_key_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify config/entity_aliases.yaml resolves the miss-case queries we target."""
    monkeypatch.delenv('ENABLE_AUTO_ENTITY_FILTER', raising=False)
    aliases_module._load_alias_rules.cache_clear()

    # Miss case #30 — brand name resolves to generic
    assert resolve_entity_filter('presentation of sarclisa') == 'isatuximab'

    # Miss case #11 — powder-in-bottle phrase resolves to program
    assert resolve_entity_filter('Do we already use Powder in a bottle directly for phase 1 study?') == 'myokardia'

    # Miss case #27 — typo variant resolves
    assert resolve_entity_filter('CMC team freezd project timeline') == 'project freeze'

    # Miss case #28 — topic phrase resolves
    assert resolve_entity_filter('CMC outsourcing licensing check points') == 'licencing'

    # Miss case #34 — alternate phrasing resolves
    assert resolve_entity_filter('What are the japan-specific activities?') == 'japanese icmc handbook'

    # Negative — an unrelated query does not trigger any alias
    assert resolve_entity_filter('What is the standard duration of shipment to depot?') is None


def _set_alias_config(monkeypatch: pytest.MonkeyPatch, tmp_path, content: str) -> None:
    config_path = tmp_path / 'entity_aliases.yaml'
    config_path.write_text(dedent(content).strip() + '\n', encoding='utf-8')
    monkeypatch.setattr(aliases_module, '_ALIAS_CONFIG_PATH', config_path)
    aliases_module._load_alias_rules.cache_clear()
