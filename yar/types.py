from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypedDict

from pydantic import BaseModel

# Type alias for LLM model functions used throughout the codebase
# These are async callables that take a prompt and return a string response
LLMFunc = Callable[..., Coroutine[Any, Any, str]]


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: str | None
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False



class GlobalConfig(TypedDict, total=False):
    """Type definition for the global configuration dict passed through the RAG pipeline.

    Created via ``dataclasses.asdict(yar_instance)`` in ``YAR.__post_init__``.
    Uses ``total=False`` because the dict contains additional runtime fields
    beyond those typed here.
    """

    # Core LLM/embedding functions
    llm_model_func: Callable[..., Coroutine[Any, Any, str]]
    embedding_func: Any  # EmbeddingFunc dataclass or raw callable
    rerank_model_func: Callable[..., Any] | None
    tokenizer: Any  # tiktoken.Encoding

    # Pipeline configuration
    workspace: str
    addon_params: dict[str, Any]
    summary_context_size: int
    summary_max_tokens: int
    summary_length_recommended: int
    force_llm_summary_on_merge: int
    llm_model_max_async: int
    max_source_ids_per_entity: int
    max_source_ids_per_relation: int
    source_ids_limit_method: str
    file_path_more_placeholder: str
    max_file_paths: int
    max_extract_input_tokens: int | str
    embedding_token_limit: int | None
    MAX_TOTAL_TOKENS: int
    min_rerank_score: float | None

    # Cache control
    enable_llm_cache: bool
    enable_llm_cache_for_entity_extract: bool