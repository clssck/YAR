"""Document processing module for LightRAG.

Powered by Kreuzberg - a polyglot document intelligence framework with Rust core.
Supports 56+ document formats with built-in semantic chunking for RAG.

See: https://github.com/Goldziher/kreuzberg
"""

from lightrag.document.kreuzberg_adapter import (
    KREUZBERG_AVAILABLE,
    ChunkingOptions,
    ExtractionOptions,
    ExtractionResult,
    HierarchyOptions,
    LanguageDetectionOptions,
    OcrOptions,
    PageOptions,
    PdfOptions,
    TextChunk,
    TokenReductionOptions,
    chunks_to_lightrag_format,
    create_chunking_options,
    extract_and_chunk,
    extract_and_chunk_sync,
    extract_with_kreuzberg,
    extract_with_kreuzberg_sync,
    is_kreuzberg_available,
    tokens_to_chars,
)

__all__ = [
    'KREUZBERG_AVAILABLE',
    'ChunkingOptions',
    'ExtractionOptions',
    'ExtractionResult',
    'HierarchyOptions',
    'LanguageDetectionOptions',
    'OcrOptions',
    'PageOptions',
    'PdfOptions',
    'TextChunk',
    'TokenReductionOptions',
    'chunks_to_lightrag_format',
    'create_chunking_options',
    'extract_and_chunk',
    'extract_and_chunk_sync',
    'extract_with_kreuzberg',
    'extract_with_kreuzberg_sync',
    'is_kreuzberg_available',
    'tokens_to_chars',
]
