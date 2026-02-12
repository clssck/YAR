"""Lazy exports for API routers.

Importing this package should not pull in optional storage dependencies
unless the caller actually needs those specific routes.
"""

from typing import Any

__all__ = (
    'create_s3_routes',
    'create_search_routes',
    'create_upload_routes',
    'document_router',
    'graph_router',
    'query_router',
)


def __getattr__(name: str) -> Any:
    """Lazily resolve router exports to avoid import-time side effects."""
    if name == 'create_s3_routes':
        from .s3_routes import create_s3_routes as exported

        return exported

    if name == 'create_search_routes':
        from .search_routes import create_search_routes as exported

        return exported

    if name == 'create_upload_routes':
        from .upload_routes import create_upload_routes as exported

        return exported

    if name == 'document_router':
        from .document_routes import router as exported

        return exported

    if name == 'graph_router':
        from .graph_routes import router as exported

        return exported

    if name == 'query_router':
        from .query_routes import router as exported

        return exported

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
