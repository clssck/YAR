"""
Tests for yar/yar.py - Main YAR orchestrator class.

This module tests:
- YAR class initialization and __post_init__ logic
- Configuration validation and warnings
- Storage backend selection and validation
- Workspace handling
- Tokenizer initialization
- Embedding and LLM function setup
- Key public methods (with mocked dependencies)
- Storage lifecycle (initialize/finalize)
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from yar.base import (
    DeletionResult,
    DocProcessingStatus,
    DocStatus,
    StoragesStatus,
)
from yar.constants import DEFAULT_MAX_EXTRACT_INPUT_TOKENS
from yar.entity_resolution import EntityResolutionConfig
from yar.yar import YAR

if TYPE_CHECKING:
    pass


@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_embedding_func():
    """Create a mock embedding function."""

    async def embedding_inner_func(texts: list[str]) -> list[list[float]]:
        """Mock embedding inner function."""
        return [[0.1] * 1536 for _ in texts]

    @dataclass
    class MockEmbeddingFunc:
        max_token_size: int = 8191
        embedding_dim: int = 1536
        model_name: str = 'text-embedding-3-small'
        func: object = embedding_inner_func

    return MockEmbeddingFunc()


@pytest.fixture
def mock_llm_func():
    """Create a mock LLM function."""

    async def llm_func(prompt: str, **kwargs) -> str:
        """Mock LLM function."""
        return f'Mock response to: {prompt[:50]}'

    return llm_func


class TestYARInitialization:
    """Tests for YAR initialization and __post_init__."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_default_initialization(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test YAR initializes with default values."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Verify defaults
        assert rag.working_dir == temp_working_dir
        assert rag.kv_storage == 'PGKVStorage'
        assert rag.vector_storage == 'PGVectorStorage'
        assert rag.graph_storage == 'PGGraphStorage'
        assert rag.doc_status_storage == 'PGDocStatusStorage'
        assert rag._storages_status == StoragesStatus.CREATED
        assert os.path.exists(temp_working_dir)

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_working_dir_creation(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test working directory is created if it doesn't exist."""
        non_existent = os.path.join(temp_working_dir, 'new_dir')
        assert not os.path.exists(non_existent)

        rag = YAR(
            working_dir=non_existent,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert os.path.exists(non_existent)
        assert rag.working_dir == non_existent

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    def test_storage_verification_called(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test storage implementation verification is called for all storage types."""
        YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Verify all storage types are checked
        assert mock_verify_storage.call_count == 4
        storage_calls = [call[0] for call in mock_verify_storage.call_args_list]
        assert ('KV_STORAGE', 'PGKVStorage') in storage_calls
        assert ('VECTOR_STORAGE', 'PGVectorStorage') in storage_calls
        assert ('GRAPH_STORAGE', 'PGGraphStorage') in storage_calls
        assert ('DOC_STATUS_STORAGE', 'PGDocStatusStorage') in storage_calls

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    def test_env_vars_check_called(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test environment variable checking is called for all storage backends."""
        YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Verify all storage backends have env vars checked
        assert mock_check_env.call_count == 4
        env_calls = [call[0][0] for call in mock_check_env.call_args_list]
        assert 'PGKVStorage' in env_calls
        assert 'PGVectorStorage' in env_calls
        assert 'PGGraphStorage' in env_calls
        assert 'PGDocStatusStorage' in env_calls

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_embedding_func_required(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_llm_func,
    ):
        """Test ValueError is raised when embedding_func is None."""
        with pytest.raises(ValueError) as exc_info:
            YAR(
                working_dir=temp_working_dir,
                embedding_func=None,
                llm_model_func=mock_llm_func,
            )

        assert 'embedding_func must be provided' in str(exc_info.value)

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_tokenizer_default_initialization(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test default tokenizer is TiktokenTokenizer."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            tokenizer=None,
            tiktoken_model_name='gpt-4o-mini',
        )

        assert rag.tokenizer is not None
        # TiktokenTokenizer should be initialized with default model
        assert hasattr(rag.tokenizer, 'encode')

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_embedding_token_limit_captured(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test embedding_token_limit is captured from embedding_func.max_token_size."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert rag.embedding_token_limit == 8191
        assert rag.embedding_token_limit == mock_embedding_func.max_token_size


class TestYARConfigValidation:
    """Tests for configuration validation and warnings."""

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    @patch('yar.yar.logger')
    def test_force_llm_summary_warning(
        self,
        mock_logger,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test warning when force_llm_summary_on_merge < 3."""
        YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            force_llm_summary_on_merge=2,
        )

        # Check warning was logged
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any('force_llm_summary_on_merge should be at least 3' in call for call in warning_calls)

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    @patch('yar.yar.logger')
    def test_summary_context_size_warning(
        self,
        mock_logger,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test warning when summary_context_size > max_total_tokens."""
        YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            summary_context_size=50000,
            max_total_tokens=32000,
        )

        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any('summary_context_size' in call and 'max_total_tokens' in call for call in warning_calls)

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    @patch('yar.yar.logger')
    def test_summary_length_warning(
        self,
        mock_logger,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test warning when summary_length_recommended > summary_max_tokens."""
        YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            summary_length_recommended=600,
            summary_max_tokens=500,
        )

        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        # The warning message uses 'max_total_tokens' but means summary_max_tokens (likely a bug in the message)
        assert any('max_total_tokens' in call and 'summary_length_recommended' in call for call in warning_calls)


class TestYARDeprecatedParameters:
    """Tests for deprecated parameter handling."""

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    def test_log_level_deprecated_warning(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test deprecation warning for log_level parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            YAR(
                working_dir=temp_working_dir,
                embedding_func=mock_embedding_func,
                llm_model_func=mock_llm_func,
                log_level=20,
            )

            # Check deprecation warning
            warning_messages = [str(warning.message) for warning in w]
            assert any('log_level parameter is deprecated' in msg for msg in warning_messages)

        # Check attribute was removed after __post_init__
        # Note: The attribute is stored as None in dataclass, then deleted in __post_init__
        # So by the time we check, it should not exist OR should be the deleted attribute
        # Actually, dataclasses don't fully delete, they may leave field descriptors
        # Let's just verify the warning was issued

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    def test_log_file_path_deprecated_warning(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test deprecation warning for log_file_path parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            YAR(
                working_dir=temp_working_dir,
                embedding_func=mock_embedding_func,
                llm_model_func=mock_llm_func,
                log_file_path='/tmp/test.log',
            )

            # Check deprecation warning
            warning_messages = [str(warning.message) for warning in w]
            assert any('log_file_path parameter is deprecated' in msg for msg in warning_messages)

        # Just verify the warning was issued


class TestYARWorkspace:
    """Tests for workspace handling."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_default_workspace_from_env(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test workspace defaults to WORKSPACE environment variable."""
        with patch.dict(os.environ, {'WORKSPACE': 'test_workspace'}):
            rag = YAR(
                working_dir=temp_working_dir,
                embedding_func=mock_embedding_func,
                llm_model_func=mock_llm_func,
            )
            assert rag.workspace == 'test_workspace'

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_workspace_explicit_override(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test explicit workspace parameter overrides environment."""
        with patch.dict(os.environ, {'WORKSPACE': 'env_workspace'}):
            rag = YAR(
                working_dir=temp_working_dir,
                embedding_func=mock_embedding_func,
                llm_model_func=mock_llm_func,
                workspace='explicit_workspace',
            )
            assert rag.workspace == 'explicit_workspace'

    @patch('yar.yar.verify_storage_implementation')
    @patch('yar.yar.check_storage_env_vars')
    def test_workspace_empty_default(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test workspace defaults to empty string when not set."""
        # Clear WORKSPACE env var
        with patch.dict(os.environ, {'WORKSPACE': ''}, clear=False):
            rag = YAR(
                working_dir=temp_working_dir,
                embedding_func=mock_embedding_func,
                llm_model_func=mock_llm_func,
            )
            # Should be empty string or 'default' depending on workspace validation
            assert rag.workspace in ('', 'default')


class TestYAREntityResolution:
    """Tests for entity resolution configuration."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_entity_resolution_default(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test entity resolution config is initialized by default."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert isinstance(rag.entity_resolution_config, EntityResolutionConfig)

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_entity_resolution_custom(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test custom entity resolution config."""
        custom_config = EntityResolutionConfig(
            enabled=False,
        )

        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            entity_resolution_config=custom_config,
        )

        assert rag.entity_resolution_config == custom_config
        assert rag.entity_resolution_config.enabled is False


class TestYARQueryParameters:
    """Tests for query parameter defaults."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_query_param_defaults(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test query parameter fields have correct defaults."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Check query-related defaults
        assert rag.top_k > 0
        assert rag.chunk_top_k > 0
        assert rag.max_entity_tokens > 0
        assert rag.max_relation_tokens > 0
        assert rag.max_total_tokens > 0
        assert rag.related_chunk_number > 0

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_query_param_custom_values(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test custom query parameter values."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            top_k=100,
            chunk_top_k=50,
            max_entity_tokens=10000,
        )

        assert rag.top_k == 100
        assert rag.chunk_top_k == 50
        assert rag.max_entity_tokens == 10000

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_max_extract_input_tokens_custom_value(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test custom max_extract_input_tokens value is stored in YAR config."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            max_extract_input_tokens=4096,
        )

        assert rag.max_extract_input_tokens == 4096
        assert rag.max_extract_input_tokens != DEFAULT_MAX_EXTRACT_INPUT_TOKENS


class TestYARChunkingConfiguration:
    """Tests for text chunking configuration."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_chunk_size_defaults(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test chunk size defaults."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert rag.chunk_token_size > 0
        assert rag.chunk_overlap_token_size > 0
        assert rag.chunk_overlap_token_size < rag.chunk_token_size

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_chunk_size_custom(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test custom chunk sizes."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            chunk_token_size=2000,
            chunk_overlap_token_size=200,
        )

        assert rag.chunk_token_size == 2000
        assert rag.chunk_overlap_token_size == 200

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_chunking_func_default(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test default chunking function is set."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert rag.chunking_func is not None
        assert callable(rag.chunking_func)


class TestYAROrphanConnection:
    """Tests for orphan connection configuration."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_orphan_connection_defaults(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test orphan connection defaults."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert isinstance(rag.auto_connect_orphans, bool)
        assert 0.0 <= rag.orphan_connection_threshold <= 1.0
        assert 0.0 <= rag.orphan_confidence_threshold <= 1.0
        assert isinstance(rag.orphan_cross_connect, bool)
        assert rag.orphan_connection_max_degree >= 0

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_orphan_connection_custom(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test custom orphan connection config."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            auto_connect_orphans=True,
            orphan_connection_threshold=0.5,
            orphan_confidence_threshold=0.8,
            orphan_cross_connect=False,
            orphan_connection_max_degree=2,
        )

        assert rag.auto_connect_orphans is True
        assert rag.orphan_connection_threshold == 0.5
        assert rag.orphan_confidence_threshold == 0.8
        assert rag.orphan_cross_connect is False
        assert rag.orphan_connection_max_degree == 2


@pytest.mark.offline
class TestYARStorageLifecycle:
    """Tests for storage initialization and finalization lifecycle."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_initialize_storages(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test storage initialization."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Initially CREATED
        assert rag._storages_status == StoragesStatus.CREATED

        # Mock all storage initialize methods
        rag.full_docs.initialize = AsyncMock()
        rag.text_chunks.initialize = AsyncMock()
        rag.full_entities.initialize = AsyncMock()
        rag.full_relations.initialize = AsyncMock()
        rag.entity_chunks.initialize = AsyncMock()
        rag.relation_chunks.initialize = AsyncMock()
        rag.entities_vdb.initialize = AsyncMock()
        rag.relationships_vdb.initialize = AsyncMock()
        rag.chunks_vdb.initialize = AsyncMock()
        rag.chunk_entity_relation_graph.initialize = AsyncMock()
        rag.llm_response_cache.initialize = AsyncMock()
        rag.doc_status.initialize = AsyncMock()

        await rag.initialize_storages()

        # Should be INITIALIZED
        assert rag._storages_status == StoragesStatus.INITIALIZED

        # Verify all initialize methods were called
        rag.full_docs.initialize.assert_called_once()
        rag.text_chunks.initialize.assert_called_once()
        rag.entities_vdb.initialize.assert_called_once()
        rag.doc_status.initialize.assert_called_once()

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_finalize_storages(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test storage finalization."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Set to INITIALIZED
        rag._storages_status = StoragesStatus.INITIALIZED

        # Mock all storage finalize methods
        rag.full_docs.finalize = AsyncMock()
        rag.text_chunks.finalize = AsyncMock()
        rag.full_entities.finalize = AsyncMock()
        rag.full_relations.finalize = AsyncMock()
        rag.entity_chunks.finalize = AsyncMock()
        rag.relation_chunks.finalize = AsyncMock()
        rag.entities_vdb.finalize = AsyncMock()
        rag.relationships_vdb.finalize = AsyncMock()
        rag.chunks_vdb.finalize = AsyncMock()
        rag.chunk_entity_relation_graph.finalize = AsyncMock()
        rag.llm_response_cache.finalize = AsyncMock()
        rag.doc_status.finalize = AsyncMock()

        await rag.finalize_storages()

        # Should be FINALIZED
        assert rag._storages_status == StoragesStatus.FINALIZED

        # Verify all finalize methods were called
        rag.full_docs.finalize.assert_called_once()
        rag.text_chunks.finalize.assert_called_once()
        rag.entities_vdb.finalize.assert_called_once()
        rag.doc_status.finalize.assert_called_once()

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_finalize_handles_errors(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test finalization continues even if one storage fails."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Set to INITIALIZED
        rag._storages_status = StoragesStatus.INITIALIZED

        # Make one storage fail
        rag.full_docs.finalize = AsyncMock(side_effect=Exception('Finalize failed'))
        rag.text_chunks.finalize = AsyncMock()
        rag.full_entities.finalize = AsyncMock()
        rag.full_relations.finalize = AsyncMock()
        rag.entity_chunks.finalize = AsyncMock()
        rag.relation_chunks.finalize = AsyncMock()
        rag.entities_vdb.finalize = AsyncMock()
        rag.relationships_vdb.finalize = AsyncMock()
        rag.chunks_vdb.finalize = AsyncMock()
        rag.chunk_entity_relation_graph.finalize = AsyncMock()
        rag.llm_response_cache.finalize = AsyncMock()
        rag.doc_status.finalize = AsyncMock()

        # Should not raise
        await rag.finalize_storages()

        # Other storages should still be finalized
        rag.text_chunks.finalize.assert_called_once()
        rag.entities_vdb.finalize.assert_called_once()


@pytest.mark.offline
class TestYARPublicMethods:
    """Tests for key public methods with mocked dependencies."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_adelete_by_doc_id(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test adelete_by_doc_id returns DeletionResult."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Mock storage state
        rag._storages_status = StoragesStatus.INITIALIZED

        # Mock doc_status storage
        mock_doc_status = DocProcessingStatus(
            content_summary='Test document',
            content_length=100,
            file_path='/test/doc.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        rag.doc_status.get_by_id = AsyncMock(return_value=mock_doc_status)
        rag.doc_status.delete = AsyncMock()
        rag.full_docs.get_by_id = AsyncMock(return_value=None)
        rag.text_chunks.get_by_ids = AsyncMock(return_value=[])
        rag.chunks_vdb.delete = AsyncMock()
        rag.entities_vdb.delete = AsyncMock()
        rag.relationships_vdb.delete = AsyncMock()
        rag.chunk_entity_relation_graph.delete_node = AsyncMock()

        result = await rag.adelete_by_doc_id('test_doc_id')

        assert isinstance(result, DeletionResult)
        assert result.doc_id == 'test_doc_id'
        assert result.status in ['success', 'not_found', 'fail']

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_adelete_not_found(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test adelete_by_doc_id when document not found."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Mock storage state
        rag._storages_status = StoragesStatus.INITIALIZED

        # Document doesn't exist
        rag.doc_status.get_by_id = AsyncMock(return_value=None)

        result = await rag.adelete_by_doc_id('nonexistent_doc')

        assert isinstance(result, DeletionResult)
        assert result.doc_id == 'nonexistent_doc'
        assert result.status == 'not_found'
        assert result.status_code == 404

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_apipeline_enqueue_documents_creates_duplicate_records(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Duplicate enqueue attempts should create trackable failed records."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        rag.doc_status.filter_keys = AsyncMock(return_value=set())
        rag.doc_status.get_by_id = AsyncMock(
            return_value={
                'status': DocStatus.PROCESSED,
                'track_id': 'orig_track_123',
            }
        )
        rag.doc_status.upsert = AsyncMock()
        rag.full_docs.upsert = AsyncMock()
        rag.full_docs.index_done_callback = AsyncMock()

        track_id = await rag.apipeline_enqueue_documents(
            input=['duplicate-content'],
            file_paths=['dup.txt'],
            track_id='enqueue_track_abc',
        )

        assert track_id == 'enqueue_track_abc'
        rag.doc_status.upsert.assert_called_once()
        rag.full_docs.upsert.assert_not_called()
        rag.full_docs.index_done_callback.assert_not_called()

        upsert_payload = rag.doc_status.upsert.call_args[0][0]
        assert len(upsert_payload) == 1
        duplicate_record = next(iter(upsert_payload.values()))
        assert duplicate_record['status'] == DocStatus.FAILED
        assert duplicate_record['track_id'] == 'enqueue_track_abc'
        assert duplicate_record['metadata']['is_duplicate'] is True
        assert duplicate_record['metadata']['original_track_id'] == 'orig_track_123'
        assert duplicate_record['metadata']['original_doc_id'].startswith('doc-')

    @patch('yar.yar.get_storage_keyed_lock')
    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    @pytest.mark.asyncio
    async def test_apipeline_enqueue_documents_uses_keyed_lock_for_doc_ids(
        self,
        mock_check_env,
        mock_verify_storage,
        mock_get_storage_keyed_lock,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Enqueue path should use a keyed lock around duplicate filtering/upsert."""

        class _DummyAsyncLock:
            def __init__(self):
                self.entered = False
                self.exited = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self.exited = True
                return False

        dummy_lock = _DummyAsyncLock()
        mock_get_storage_keyed_lock.return_value = dummy_lock

        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        rag.doc_status.filter_keys = AsyncMock(return_value=set())
        rag.doc_status.get_by_id = AsyncMock(return_value={'status': DocStatus.PROCESSED, 'track_id': 'orig_track'})
        rag.doc_status.upsert = AsyncMock()
        rag.full_docs.upsert = AsyncMock()
        rag.full_docs.index_done_callback = AsyncMock()

        await rag.apipeline_enqueue_documents(
            input=['duplicate-content'],
            file_paths=['dup.txt'],
            track_id='enqueue_track_abc',
        )

        mock_get_storage_keyed_lock.assert_called_once()
        lock_args, lock_kwargs = mock_get_storage_keyed_lock.call_args
        assert len(lock_args[0]) == 1
        assert lock_args[0][0].startswith('doc-')
        assert lock_kwargs['namespace'] == f'doc_enqueue:{rag.workspace}'
        assert lock_kwargs['enable_logging'] is False
        assert dummy_lock.entered is True
        assert dummy_lock.exited is True


@pytest.mark.offline
class TestYARStorageClasses:
    """Tests for storage class initialization."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_storage_classes_set(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test storage classes are properly initialized."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Verify storage class attributes exist
        assert hasattr(rag, 'key_string_value_json_storage_cls')
        assert hasattr(rag, 'vector_db_storage_cls')
        assert hasattr(rag, 'graph_storage_cls')
        assert hasattr(rag, 'doc_status_storage_cls')

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_kv_storages_created(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test KV storage instances are created."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Verify all KV storage instances exist
        assert rag.llm_response_cache is not None
        assert rag.text_chunks is not None
        assert rag.full_docs is not None
        assert rag.full_entities is not None
        assert rag.full_relations is not None
        assert rag.entity_chunks is not None
        assert rag.relation_chunks is not None

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_vector_storages_created(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test vector storage instances are created."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        # Verify all vector storage instances exist
        assert rag.entities_vdb is not None
        assert rag.relationships_vdb is not None
        assert rag.chunks_vdb is not None

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_graph_storage_created(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test graph storage instance is created."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert rag.chunk_entity_relation_graph is not None

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_doc_status_storage_created(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test doc status storage instance is created."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert rag.doc_status is not None


@pytest.mark.offline
class TestYAREmbeddingConfiguration:
    """Tests for embedding function configuration."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_embedding_cache_config_default(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test default embedding cache configuration."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
        )

        assert 'enabled' in rag.embedding_cache_config
        assert 'similarity_threshold' in rag.embedding_cache_config
        assert 'use_llm_check' in rag.embedding_cache_config
        assert isinstance(rag.embedding_cache_config['enabled'], bool)

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_embedding_batch_num(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test embedding batch number configuration."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            embedding_batch_num=20,
        )

        assert rag.embedding_batch_num == 20

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_embedding_func_max_async(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test embedding function max async configuration."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            embedding_func_max_async=16,
        )

        assert rag.embedding_func_max_async == 16


@pytest.mark.offline
class TestYARLLMConfiguration:
    """Tests for LLM configuration."""

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_llm_model_name(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test LLM model name configuration."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            llm_model_name='gpt-4o',
        )

        assert rag.llm_model_name == 'gpt-4o'

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_llm_model_kwargs(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test LLM model kwargs configuration."""
        custom_kwargs = {'temperature': 0.7, 'max_tokens': 1000}

        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            llm_model_kwargs=custom_kwargs,
        )

        assert rag.llm_model_kwargs == custom_kwargs

    @patch('yar.kg.verify_storage_implementation')
    @patch('yar.utils.check_storage_env_vars')
    def test_llm_model_max_async(
        self,
        mock_check_env,
        mock_verify_storage,
        temp_working_dir,
        mock_embedding_func,
        mock_llm_func,
    ):
        """Test LLM model max async configuration."""
        rag = YAR(
            working_dir=temp_working_dir,
            embedding_func=mock_embedding_func,
            llm_model_func=mock_llm_func,
            llm_model_max_async=32,
        )

        assert rag.llm_model_max_async == 32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
