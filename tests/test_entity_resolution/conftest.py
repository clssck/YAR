"""Pytest configuration for entity resolution tests.

This conftest sets up module mocks BEFORE test files are collected,
preventing the argument parsing conflict between pytest and lightrag.api.config.
"""

import sys
from unittest.mock import MagicMock

# Only mock API modules if they haven't been imported yet
# This prevents lightrag.api.config from calling parse_args() during test collection
if 'lightrag.api.config' not in sys.modules:
    # Create mock config module
    mock_config = MagicMock()
    mock_config.get_env_value = lambda key, default=None: default
    mock_config.global_args = MagicMock()
    mock_config.global_args.key = None
    mock_config.global_args.working_dir = './test_rag_storage'
    mock_config.global_args.workspace = 'test_workspace'
    mock_config.ollama_server_infos = {}
    sys.modules['lightrag.api.config'] = mock_config

# Note: We no longer mock utils_api here. The real get_combined_auth_dependency
# needs to be used for auth tests to work correctly. Individual tests that don't
# need auth should create their own isolated mocks.
