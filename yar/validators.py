"""
Centralized validation and sanitization functions for YAR.

This module provides reusable validation functions for:
- SQL identifiers (table names, graph names, workspaces)
- Numeric configuration parameters
- Input sanitization for security

These functions help prevent SQL injection and ensure data integrity.
"""

from __future__ import annotations

import re
from typing import Any

# PostgreSQL maximum identifier length (63 bytes)
PG_MAX_IDENTIFIER_LENGTH = 63

# Workspace name validation pattern: alphanumeric + underscore only
# This prevents collision issues where 'foo-bar' and 'foo_bar' map to the same graph name
_VALID_WORKSPACE_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')

# Document ID validation: used as storage path segment and DB identifier.
_VALID_DOC_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
_MAX_DOC_ID_LENGTH = 128

# SQL identifier validation pattern: PostgreSQL identifiers must start with
# letter or underscore, contain only alphanumeric + underscore
_VALID_SQL_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# S3 path hardening: reject path traversal segments and dangerous control chars.
_S3_CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x1F\x7F]')
_S3_SEGMENT_FORBIDDEN = {'.', '..'}


def validate_workspace_name(workspace: str | None) -> str:
    """Validate workspace name to prevent collision issues in PostgreSQL graph names.

    Workspace names are used to generate Apache AGE graph names. Special characters
    get replaced with underscores, which can cause collisions:
    - 'my-workspace' and 'my_workspace' both become 'my_workspace'
    - This creates data corruption when two "different" workspaces share a graph

    Strict validation prevents this by rejecting problematic names at creation time.

    Args:
        workspace: The workspace name to validate

    Returns:
        The validated workspace name (unchanged if valid)

    Raises:
        ValueError: If workspace name contains invalid characters

    Valid names:
        - 'my_workspace' ✓
        - 'workspace123' ✓
        - 'MyWorkSpace' ✓
        - 'default' ✓

    Invalid names:
        - 'my-workspace' ✗ (hyphen not allowed)
        - 'my workspace' ✗ (space not allowed)
        - 'my.workspace' ✗ (dot not allowed)
        - '' → 'default' (empty converts to default)
    """
    # Empty or None workspace defaults to 'default' (legacy compatibility)
    if not workspace or not workspace.strip():
        return 'default'

    workspace = workspace.strip()

    # Allow 'default' as a special case (legacy compatibility)
    if workspace.lower() == 'default':
        return workspace

    if not _VALID_WORKSPACE_PATTERN.match(workspace):
        invalid_chars = {c for c in workspace if not c.isalnum() and c != '_'}
        raise ValueError(
            f"Invalid workspace name '{workspace}'. "
            f"Workspace names can only contain letters, numbers, and underscores. "
            f"Invalid characters found: {invalid_chars}. "
            f"Suggestion: Use '{re.sub(r'[^a-zA-Z0-9_]', '_', workspace)}' instead."
        )

    return workspace


def validate_sql_identifier(identifier: str, identifier_type: str = 'identifier') -> str:
    """Validate and return a safe SQL identifier (table/index/graph names).

    PostgreSQL identifiers must:
    - Start with letter or underscore
    - Contain only alphanumeric characters and underscores
    - Be max 63 bytes (PG_MAX_IDENTIFIER_LENGTH)

    This function prevents SQL injection by rejecting any identifier containing
    characters that could break out of SQL statements (quotes, semicolons, etc.).

    Args:
        identifier: The identifier to validate
        identifier_type: Description for error messages (e.g., 'graph_name', 'table_name')

    Returns:
        The validated identifier (unchanged if valid)

    Raises:
        ValueError: If identifier is empty, too long, or contains invalid characters
    """
    if not identifier:
        raise ValueError(f'Empty {identifier_type} is not allowed')

    if len(identifier.encode('utf-8')) > PG_MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{identifier_type} '{identifier}' exceeds PostgreSQL's 63-byte limit"
        )

    if not _VALID_SQL_IDENTIFIER_PATTERN.match(identifier):
        invalid_chars = {c for c in identifier if not c.isalnum() and c != '_'}
        raise ValueError(
            f"Invalid {identifier_type} '{identifier}'. "
            f'SQL identifiers can only contain letters, numbers, and underscores, '
            f'and must start with a letter or underscore. '
            f'Invalid characters found: {invalid_chars}'
        )

    return identifier


def validate_doc_id(doc_id: str | None) -> str:
    """Validate document ID used in S3 paths and API operations.

    This validator is intentionally strict because doc_id may be used as an
    S3 path segment. Restricting characters avoids traversal-like keys and
    keeps IDs stable across storage backends.
    """
    if doc_id is None:
        raise ValueError('Document ID cannot be empty')

    normalized = doc_id.strip()
    if not normalized:
        raise ValueError('Document ID cannot be empty')

    if len(normalized) > _MAX_DOC_ID_LENGTH:
        raise ValueError(f'Document ID exceeds maximum length of {_MAX_DOC_ID_LENGTH} characters')

    if normalized in _S3_SEGMENT_FORBIDDEN:
        raise ValueError('Document ID cannot be a relative path segment')

    if not _VALID_DOC_ID_PATTERN.match(normalized):
        invalid_chars = {c for c in normalized if not (c.isascii() and (c.isalnum() or c in {'_', '-'}))}
        raise ValueError(
            f"Invalid document ID '{normalized}'. "
            f'Document IDs can only contain ASCII letters, numbers, underscores, and hyphens. '
            f'Invalid characters found: {invalid_chars}'
        )

    return normalized


def validate_numeric_config(
    value: Any,
    param_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Validate numeric configuration parameters to prevent SQL injection.

    PostgreSQL SET commands cannot use parameterized queries ($1 placeholders).
    This function ensures values are strictly numeric before interpolation,
    eliminating injection risk via configuration parameters.

    Args:
        value: The value to validate (will be converted to float)
        param_name: Parameter name for error messages
        min_val: Optional minimum value (inclusive)
        max_val: Optional maximum value (inclusive)

    Returns:
        The validated numeric value as float

    Raises:
        ValueError: If value is not numeric or out of range
    """
    try:
        num = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f'Invalid {param_name}: must be numeric, got {type(value).__name__}'
        ) from e

    if min_val is not None and num < min_val:
        raise ValueError(f'Invalid {param_name}: {num} is below minimum {min_val}')
    if max_val is not None and num > max_val:
        raise ValueError(f'Invalid {param_name}: {num} exceeds maximum {max_val}')

    return num


def _validate_s3_path(path: str, path_type: str, allow_empty: bool, allow_trailing_slash: bool) -> str:
    """Validate an S3 key/prefix with conservative safety checks.

    This validator keeps compatibility with common S3 naming patterns while
    blocking path traversal-like constructs and control characters.
    """
    normalized = (path or '').strip()
    if not normalized:
        if allow_empty:
            return ''
        raise ValueError(f'{path_type} cannot be empty')

    if _S3_CONTROL_CHARS_PATTERN.search(normalized):
        raise ValueError(f'{path_type} contains control characters')
    if '\\' in normalized:
        raise ValueError(f"{path_type} cannot contain '\\\\'")
    if normalized.startswith('/'):
        raise ValueError(f'{path_type} cannot start with "/"')
    if '//' in normalized:
        raise ValueError(f'{path_type} cannot contain "//"')

    segments = normalized.split('/')
    if allow_trailing_slash and normalized.endswith('/'):
        segments = segments[:-1]

    for segment in segments:
        if not segment:
            raise ValueError(f'{path_type} contains an empty path segment')
        if segment in _S3_SEGMENT_FORBIDDEN:
            raise ValueError(f'{path_type} cannot contain relative path segments')

    if allow_trailing_slash and normalized and not normalized.endswith('/'):
        normalized = f'{normalized}/'
    if not allow_trailing_slash and normalized.endswith('/'):
        raise ValueError(f'{path_type} cannot end with "/"')

    return normalized


def validate_s3_key(key: str) -> str:
    """Validate a full S3 object key."""
    return _validate_s3_path(key, 'S3 key', allow_empty=False, allow_trailing_slash=False)


def validate_s3_prefix(prefix: str, *, allow_empty: bool = True) -> str:
    """Validate an S3 prefix path and normalize to trailing slash if non-empty."""
    return _validate_s3_path(prefix, 'S3 prefix', allow_empty=allow_empty, allow_trailing_slash=True)
