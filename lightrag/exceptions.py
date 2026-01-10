from __future__ import annotations

from typing import Literal

import httpx


class APIStatusError(Exception):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(self, message: str, *, response: httpx.Response, body: object | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get('x-request-id')


class APIConnectionError(Exception):
    def __init__(self, *, message: str = 'Connection error.', request: httpx.Request) -> None:
        super().__init__(message)


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400  # pyright: ignore[reportIncompatibleVariableOverride]


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401  # pyright: ignore[reportIncompatibleVariableOverride]


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403  # pyright: ignore[reportIncompatibleVariableOverride]


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404  # pyright: ignore[reportIncompatibleVariableOverride]


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409  # pyright: ignore[reportIncompatibleVariableOverride]


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422  # pyright: ignore[reportIncompatibleVariableOverride]


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message='Request timed out.', request=request)


class StorageNotInitializedError(RuntimeError):
    """Raised when storage operations are attempted before initialization."""

    def __init__(self, storage_type: str = 'Storage'):
        super().__init__(
            f'{storage_type} not initialized. Please ensure proper initialization:\n'
            f'\n'
            f'  rag = LightRAG(...)\n'
            f'  await rag.initialize_storages()  # Required - auto-initializes pipeline_status\n'
            f'\n'
            f'See: https://github.com/HKUDS/LightRAG#important-initialization-requirements'
        )


class PipelineNotInitializedError(KeyError):
    """Raised when pipeline status is accessed before initialization."""

    def __init__(self, namespace: str = ''):
        msg = (
            f"Pipeline namespace '{namespace}' not found.\n"
            f'\n'
            f'Pipeline status should be auto-initialized by initialize_storages().\n'
            f'If you see this error, please ensure:\n'
            f'\n'
            f'  1. You called await rag.initialize_storages()\n'
            f'  2. For multi-workspace setups, each LightRAG instance was properly initialized\n'
            f'\n'
            f'Standard initialization:\n'
            f"  rag = LightRAG(workspace='your_workspace')\n"
            f'  await rag.initialize_storages()  # Auto-initializes pipeline_status\n'
            f'\n'
            f'If you need manual control (advanced):\n'
            f'  from lightrag.kg.shared_storage import initialize_pipeline_status\n'
            f"  await initialize_pipeline_status(workspace='your_workspace')"
        )
        super().__init__(msg)


class PipelineCancelledException(Exception):
    """Raised when pipeline processing is cancelled by user request."""

    def __init__(self, message: str = 'User cancelled'):
        super().__init__(message)
        self.message = message


class LockTimeoutError(TimeoutError):
    """Raised when lock acquisition times out."""

    def __init__(self, key: str, timeout: float):
        super().__init__(f'Failed to acquire lock for "{key}" within {timeout}s')
        self.key = key
        self.timeout = timeout


class LightRAGError(Exception):
    """Base exception for all LightRAG errors."""


class StorageError(LightRAGError):
    """Raised when a storage operation fails."""

    def __init__(self, message: str, storage_type: str | None = None, operation: str | None = None):
        self.storage_type = storage_type
        self.operation = operation
        detail = f'[{storage_type}]' if storage_type else ''
        detail += f' {operation}:' if operation else ''
        super().__init__(f'{detail} {message}'.strip())


class ExtractionError(LightRAGError):
    """Raised when entity/relationship extraction fails."""

    def __init__(self, message: str, chunk_key: str | None = None):
        self.chunk_key = chunk_key
        prefix = f'[chunk={chunk_key}]' if chunk_key else ''
        super().__init__(f'{prefix} {message}'.strip())


class QueryError(LightRAGError):
    """Raised when a query operation fails."""

    def __init__(self, message: str, query_mode: str | None = None):
        self.query_mode = query_mode
        prefix = f'[mode={query_mode}]' if query_mode else ''
        super().__init__(f'{prefix} {message}'.strip())


class ConfigurationError(LightRAGError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str | None = None):
        self.config_key = config_key
        prefix = f'[{config_key}]' if config_key else ''
        super().__init__(f'{prefix} {message}'.strip())


class EmbeddingError(LightRAGError):
    """Raised when embedding computation fails."""

    def __init__(self, message: str, model: str | None = None):
        self.model = model
        prefix = f'[model={model}]' if model else ''
        super().__init__(f'{prefix} {message}'.strip())


class LLMError(LightRAGError):
    """Raised when LLM call fails."""

    def __init__(self, message: str, model: str | None = None, operation: str | None = None):
        self.model = model
        self.operation = operation
        parts = []
        if model:
            parts.append(f'model={model}')
        if operation:
            parts.append(f'op={operation}')
        prefix = f'[{", ".join(parts)}]' if parts else ''
        super().__init__(f'{prefix} {message}'.strip())
