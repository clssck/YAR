from __future__ import annotations

from collections.abc import Iterator
from textwrap import dedent
from unittest.mock import Mock

import pytest

from yar.retrieval import aliases as aliases_module
from yar.retrieval import expand_query_aliases, resolve_entity_filter


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
          - canonical: product alpha
            aliases:
              - alpha brand
        """,
    )

    assert resolve_entity_filter('Latest ALPHA BRAND response data') == 'product alpha'
    assert resolve_entity_filter('Discuss product alpha dosing guidance') == 'product alpha'


def test_resolve_entity_filter_ignores_contextual_relationship_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: compound alpha
            aliases:
              - alpha collaboration
              - alpha alliance
              - alpha
        """,
    )

    query = 'What lessons were learned from the Alpha collaboration?'

    assert resolve_entity_filter(query) is None
    assert expand_query_aliases(query) == ['compound alpha', 'alpha collaboration', 'alpha alliance', 'alpha']
    assert resolve_entity_filter('What is the Alpha dosing plan?') == 'compound alpha'


def test_resolve_entity_filter_avoids_substring_false_positives(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: product alpha
            aliases:
              - alpha brand
        """,
    )

    assert resolve_entity_filter('Compare alphabrand safety updates') is None


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

    assert resolve_entity_filter('Need the shared alias comparison') == 'First Canonical'
    info_mock.assert_called_once()
    assert 'using First Canonical by config order' in info_mock.call_args.args[0]


def test_resolve_entity_filter_returns_none_for_missing_or_empty_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    warning_mock = Mock()
    monkeypatch.setattr(aliases_module.logger, 'warning', warning_mock)
    monkeypatch.setattr(aliases_module, '_ALIAS_CONFIG_PATH', tmp_path / 'missing.yaml')

    assert resolve_entity_filter('alpha brand update') is None
    warning_mock.assert_called_once()

    warning_mock.reset_mock()
    empty_config_path = tmp_path / 'entity_aliases.yaml'
    empty_config_path.write_text('', encoding='utf-8')
    monkeypatch.setattr(aliases_module, '_ALIAS_CONFIG_PATH', empty_config_path)
    aliases_module._load_alias_rules.cache_clear()

    assert resolve_entity_filter('alpha brand update') is None
    warning_mock.assert_called_once()


def test_resolve_entity_filter_respects_disable_switch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_alias_config(
        monkeypatch,
        tmp_path,
        """
        entities:
          - canonical: product alpha
            aliases:
              - alpha brand
        """,
    )
    monkeypatch.setenv('ENABLE_AUTO_ENTITY_FILTER', 'false')

    assert resolve_entity_filter('alpha brand update') is None


def test_shipped_config_resolves_generic_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify config/entity_aliases.yaml contains only generic aliases."""
    monkeypatch.delenv('ENABLE_AUTO_ENTITY_FILTER', raising=False)
    aliases_module._load_alias_rules.cache_clear()

    assert resolve_entity_filter('FDA approval timeline') == 'us food and drug administration'
    assert resolve_entity_filter('QMS audit findings') == 'quality management system'
    assert resolve_entity_filter('What is the standard duration before the launch milestone?') is None


def _set_alias_config(monkeypatch: pytest.MonkeyPatch, tmp_path, content: str) -> None:
    config_path = tmp_path / 'entity_aliases.yaml'
    config_path.write_text(dedent(content).strip() + '\n', encoding='utf-8')
    monkeypatch.setattr(aliases_module, '_ALIAS_CONFIG_PATH', config_path)
    aliases_module._load_alias_rules.cache_clear()
