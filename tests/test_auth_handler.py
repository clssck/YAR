"""Tests for JWT algorithm hardening in yar.api.auth."""

import importlib

import pytest
from fastapi import HTTPException


@pytest.mark.offline
class TestAuthHandler:
    """Verify insecure JWT algorithms are rejected."""

    def test_init_rejects_none_algorithm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AuthHandler refuses insecure JWT configuration at startup."""
        monkeypatch.setenv('TOKEN_SECRET', 'test-secret')
        monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
        auth_module = importlib.import_module('yar.api.auth')

        monkeypatch.setenv('JWT_ALGORITHM', 'none')

        with pytest.raises(ValueError, match="JWT_ALGORITHM must be set to a secure algorithm"):
            auth_module.AuthHandler()

    def test_validate_token_rejects_none_algorithm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_token fails closed if the algorithm is changed after init."""
        monkeypatch.setenv('TOKEN_SECRET', 'test-secret')
        monkeypatch.setenv('JWT_ALGORITHM', 'HS256')
        auth_module = importlib.import_module('yar.api.auth')
        handler = auth_module.AuthHandler()
        handler.algorithm = 'none'

        with pytest.raises(HTTPException) as exc_info:
            handler.validate_token('ignored-token')

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == 'Insecure JWT algorithm configuration'
