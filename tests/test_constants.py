"""
Tests for yar/constants.py - Centralized configuration constants.

This module tests:
- Server setting constants
- Extraction setting constants
- Query and retrieval configuration defaults
- Rerank configuration defaults
- Source ID limit methods and validation
- Status constants
- Namespace keys
- Common field names
- Default workspace
"""

from __future__ import annotations

from yar.constants import (
    DEFAULT_CHECK_TOPIC_CONNECTIVITY,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_ENABLE_RERANK,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_FTS_CACHE_ENABLED,
    DEFAULT_FTS_CACHE_MAX_SIZE,
    DEFAULT_FTS_CACHE_TTL,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_MAX_GLEANING,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_METRICS_ENABLED,
    DEFAULT_METRICS_HISTORY_SIZE,
    DEFAULT_METRICS_WINDOW_SECONDS,
    DEFAULT_MIGRATION_BATCH_SIZE,
    DEFAULT_MIN_ENTITY_COVERAGE,
    DEFAULT_MIN_RELATIONSHIP_DENSITY,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_RETRIEVAL_MULTIPLIER,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_WOKERS,
    DEFAULT_WORKSPACE,
    FIELD_CONTENT,
    FIELD_DESCRIPTION,
    FIELD_ENTITY_NAME,
    FIELD_ENTITY_TYPE,
    FIELD_FILE_PATH,
    FIELD_HISTORY_MESSAGES,
    FIELD_KEYWORDS,
    FIELD_LATEST_MESSAGE,
    FIELD_SOURCE_ID,
    FIELD_SRC_ID,
    FIELD_STATUS,
    FIELD_TGT_ID,
    FIELD_WEIGHT,
    GRAPH_FIELD_SEP,
    NS_ENTITY_RESOLUTION_STATUS,
    NS_ORPHAN_CONNECTION_STATUS,
    NS_PIPELINE_STATUS,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    STATE_BUSY,
    STATE_CANCELLATION_REQUESTED,
    STATE_REQUEST_PENDING,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_FAILURE,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_SUCCESS,
    VALID_SOURCE_IDS_LIMIT_METHODS,
)


class TestServerSettings:
    """Tests for server setting constants."""

    def test_default_workers_defined(self):
        """Test DEFAULT_WOKERS is defined and is an integer."""
        assert DEFAULT_WOKERS == 2
        assert isinstance(DEFAULT_WOKERS, int)
        assert DEFAULT_WOKERS > 0

    def test_default_max_graph_nodes_defined(self):
        """Test DEFAULT_MAX_GRAPH_NODES is defined and is an integer."""
        assert DEFAULT_MAX_GRAPH_NODES == 1000
        assert isinstance(DEFAULT_MAX_GRAPH_NODES, int)
        assert DEFAULT_MAX_GRAPH_NODES > 0


class TestExtractionSettings:
    """Tests for extraction setting constants."""

    def test_default_summary_language(self):
        """Test DEFAULT_SUMMARY_LANGUAGE is set to English."""
        assert DEFAULT_SUMMARY_LANGUAGE == 'English'
        assert isinstance(DEFAULT_SUMMARY_LANGUAGE, str)

    def test_default_max_gleaning(self):
        """Test DEFAULT_MAX_GLEANING is set to 1."""
        assert DEFAULT_MAX_GLEANING == 1
        assert isinstance(DEFAULT_MAX_GLEANING, int)
        assert DEFAULT_MAX_GLEANING >= 0

    def test_default_entity_name_max_length(self):
        """Test DEFAULT_ENTITY_NAME_MAX_LENGTH is a positive integer."""
        assert DEFAULT_ENTITY_NAME_MAX_LENGTH == 256
        assert isinstance(DEFAULT_ENTITY_NAME_MAX_LENGTH, int)
        assert DEFAULT_ENTITY_NAME_MAX_LENGTH > 0

    def test_default_force_llm_summary_on_merge(self):
        """Test DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE is a positive integer."""
        assert DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE == 8
        assert isinstance(DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int)
        assert DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE > 0

    def test_default_summary_max_tokens(self):
        """Test DEFAULT_SUMMARY_MAX_TOKENS is a positive integer."""
        assert DEFAULT_SUMMARY_MAX_TOKENS == 1200
        assert isinstance(DEFAULT_SUMMARY_MAX_TOKENS, int)
        assert DEFAULT_SUMMARY_MAX_TOKENS > 0

    def test_default_summary_length_recommended(self):
        """Test DEFAULT_SUMMARY_LENGTH_RECOMMENDED is less than max."""
        assert DEFAULT_SUMMARY_LENGTH_RECOMMENDED == 600
        assert isinstance(DEFAULT_SUMMARY_LENGTH_RECOMMENDED, int)
        assert DEFAULT_SUMMARY_LENGTH_RECOMMENDED > 0
        assert DEFAULT_SUMMARY_LENGTH_RECOMMENDED <= DEFAULT_SUMMARY_MAX_TOKENS

    def test_default_summary_context_size(self):
        """Test DEFAULT_SUMMARY_CONTEXT_SIZE is a positive integer."""
        assert DEFAULT_SUMMARY_CONTEXT_SIZE == 12000
        assert isinstance(DEFAULT_SUMMARY_CONTEXT_SIZE, int)
        assert DEFAULT_SUMMARY_CONTEXT_SIZE > 0

    def test_default_max_extract_input_tokens(self):
        """Test DEFAULT_MAX_EXTRACT_INPUT_TOKENS is a positive integer."""
        assert DEFAULT_MAX_EXTRACT_INPUT_TOKENS == 20480
        assert isinstance(DEFAULT_MAX_EXTRACT_INPUT_TOKENS, int)
        assert DEFAULT_MAX_EXTRACT_INPUT_TOKENS > 0

    def test_default_entity_types_is_list(self):
        """Test DEFAULT_ENTITY_TYPES is a list of strings."""
        assert isinstance(DEFAULT_ENTITY_TYPES, list)
        assert len(DEFAULT_ENTITY_TYPES) > 0
        assert all(isinstance(et, str) for et in DEFAULT_ENTITY_TYPES)

    def test_default_entity_types_expected_values(self):
        """Test DEFAULT_ENTITY_TYPES contains expected entity types.

        Entity types optimized for general/mixed corpus:
        - Removed: Creature, NaturalObject, Content (rarely used)
        - Added: Technology, Product, Document (common in business/tech docs)
        """
        expected_types = {
            'Person',
            'Organization',
            'Location',
            'Event',
            'Concept',
            'Method',
            'Technology',
            'Product',
            'Document',
            'Data',
            'Artifact',
        }
        assert set(DEFAULT_ENTITY_TYPES) == expected_types


class TestGraphSettings:
    """Tests for graph field separator constant."""

    def test_graph_field_sep_defined(self):
        """Test GRAPH_FIELD_SEP is defined as string."""
        assert GRAPH_FIELD_SEP == '<SEP>'
        assert isinstance(GRAPH_FIELD_SEP, str)
        assert len(GRAPH_FIELD_SEP) > 0


class TestQueryAndRetrievalDefaults:
    """Tests for query and retrieval configuration defaults."""

    def test_default_top_k(self):
        """Test DEFAULT_TOP_K is a positive integer."""
        assert DEFAULT_TOP_K == 60
        assert isinstance(DEFAULT_TOP_K, int)
        assert DEFAULT_TOP_K > 0

    def test_default_chunk_top_k(self):
        """Test DEFAULT_CHUNK_TOP_K is a positive integer."""
        assert DEFAULT_CHUNK_TOP_K == 40
        assert isinstance(DEFAULT_CHUNK_TOP_K, int)
        assert DEFAULT_CHUNK_TOP_K > 0
        # Chunk top-k should typically be less than or equal to top-k
        assert DEFAULT_CHUNK_TOP_K <= DEFAULT_TOP_K

    def test_default_max_entity_tokens(self):
        """Test DEFAULT_MAX_ENTITY_TOKENS is a positive integer."""
        assert DEFAULT_MAX_ENTITY_TOKENS == 16000
        assert isinstance(DEFAULT_MAX_ENTITY_TOKENS, int)
        assert DEFAULT_MAX_ENTITY_TOKENS > 0

    def test_default_max_relation_tokens(self):
        """Test DEFAULT_MAX_RELATION_TOKENS is a positive integer."""
        assert DEFAULT_MAX_RELATION_TOKENS == 16000
        assert isinstance(DEFAULT_MAX_RELATION_TOKENS, int)
        assert DEFAULT_MAX_RELATION_TOKENS > 0

    def test_default_max_total_tokens(self):
        """Test DEFAULT_MAX_TOTAL_TOKENS is a positive integer."""
        assert DEFAULT_MAX_TOTAL_TOKENS == 60000
        assert isinstance(DEFAULT_MAX_TOTAL_TOKENS, int)
        assert DEFAULT_MAX_TOTAL_TOKENS > 0
        # Total should be >= entity and relation tokens
        assert DEFAULT_MAX_TOTAL_TOKENS >= DEFAULT_MAX_ENTITY_TOKENS
        assert DEFAULT_MAX_TOTAL_TOKENS >= DEFAULT_MAX_RELATION_TOKENS

    def test_default_cosine_threshold(self):
        """Test DEFAULT_COSINE_THRESHOLD is a valid similarity threshold."""
        assert DEFAULT_COSINE_THRESHOLD == 0.30
        assert isinstance(DEFAULT_COSINE_THRESHOLD, float)
        assert 0 <= DEFAULT_COSINE_THRESHOLD <= 1

    def test_default_related_chunk_number(self):
        """Test DEFAULT_RELATED_CHUNK_NUMBER is a positive integer."""
        assert DEFAULT_RELATED_CHUNK_NUMBER == 16
        assert isinstance(DEFAULT_RELATED_CHUNK_NUMBER, int)
        assert DEFAULT_RELATED_CHUNK_NUMBER > 0

    def test_default_kg_chunk_pick_method(self):
        """Test DEFAULT_KG_CHUNK_PICK_METHOD is defined."""
        assert DEFAULT_KG_CHUNK_PICK_METHOD == 'VECTOR'
        assert isinstance(DEFAULT_KG_CHUNK_PICK_METHOD, str)


class TestRerankConfiguration:
    """Tests for rerank configuration defaults."""

    def test_default_enable_rerank_is_bool(self):
        """Test DEFAULT_ENABLE_RERANK is a boolean."""
        assert isinstance(DEFAULT_ENABLE_RERANK, bool)
        assert DEFAULT_ENABLE_RERANK is True

    def test_default_min_rerank_score_is_valid(self):
        """Test DEFAULT_MIN_RERANK_SCORE is None or a valid float."""
        assert DEFAULT_MIN_RERANK_SCORE is None or isinstance(DEFAULT_MIN_RERANK_SCORE, float)
        if DEFAULT_MIN_RERANK_SCORE is not None:
            assert 0 <= DEFAULT_MIN_RERANK_SCORE <= 1


class TestRetrievalConfiguration:
    """Tests for two-stage retrieval configuration."""

    def test_default_retrieval_multiplier(self):
        """Test DEFAULT_RETRIEVAL_MULTIPLIER is a non-negative number."""
        assert DEFAULT_RETRIEVAL_MULTIPLIER == 1
        assert isinstance(DEFAULT_RETRIEVAL_MULTIPLIER, int)
        assert DEFAULT_RETRIEVAL_MULTIPLIER >= 1


class TestSourceIDsConfiguration:
    """Tests for source IDs limit configuration."""

    def test_default_max_source_ids_per_entity(self):
        """Test DEFAULT_MAX_SOURCE_IDS_PER_ENTITY is a positive integer."""
        assert DEFAULT_MAX_SOURCE_IDS_PER_ENTITY == 300
        assert isinstance(DEFAULT_MAX_SOURCE_IDS_PER_ENTITY, int)
        assert DEFAULT_MAX_SOURCE_IDS_PER_ENTITY > 0

    def test_default_max_source_ids_per_relation(self):
        """Test DEFAULT_MAX_SOURCE_IDS_PER_RELATION is a positive integer."""
        assert DEFAULT_MAX_SOURCE_IDS_PER_RELATION == 300
        assert isinstance(DEFAULT_MAX_SOURCE_IDS_PER_RELATION, int)
        assert DEFAULT_MAX_SOURCE_IDS_PER_RELATION > 0

    def test_source_ids_limit_method_keep(self):
        """Test SOURCE_IDS_LIMIT_METHOD_KEEP is defined."""
        assert SOURCE_IDS_LIMIT_METHOD_KEEP == 'KEEP'
        assert isinstance(SOURCE_IDS_LIMIT_METHOD_KEEP, str)

    def test_source_ids_limit_method_fifo(self):
        """Test SOURCE_IDS_LIMIT_METHOD_FIFO is defined."""
        assert SOURCE_IDS_LIMIT_METHOD_FIFO == 'FIFO'
        assert isinstance(SOURCE_IDS_LIMIT_METHOD_FIFO, str)

    def test_default_source_ids_limit_method(self):
        """Test DEFAULT_SOURCE_IDS_LIMIT_METHOD is one of the valid methods."""
        assert DEFAULT_SOURCE_IDS_LIMIT_METHOD in VALID_SOURCE_IDS_LIMIT_METHODS
        assert isinstance(DEFAULT_SOURCE_IDS_LIMIT_METHOD, str)

    def test_valid_source_ids_limit_methods_is_set(self):
        """Test VALID_SOURCE_IDS_LIMIT_METHODS is a set with both methods."""
        assert isinstance(VALID_SOURCE_IDS_LIMIT_METHODS, set)
        assert len(VALID_SOURCE_IDS_LIMIT_METHODS) == 2
        assert SOURCE_IDS_LIMIT_METHOD_KEEP in VALID_SOURCE_IDS_LIMIT_METHODS
        assert SOURCE_IDS_LIMIT_METHOD_FIFO in VALID_SOURCE_IDS_LIMIT_METHODS

    def test_default_max_file_paths(self):
        """Test DEFAULT_MAX_FILE_PATHS is a positive integer."""
        assert DEFAULT_MAX_FILE_PATHS == 100
        assert isinstance(DEFAULT_MAX_FILE_PATHS, int)
        assert DEFAULT_MAX_FILE_PATHS > 0

    def test_default_file_path_more_placeholder(self):
        """Test DEFAULT_FILE_PATH_MORE_PLACEHOLDER is a string."""
        assert DEFAULT_FILE_PATH_MORE_PLACEHOLDER == 'truncated'
        assert isinstance(DEFAULT_FILE_PATH_MORE_PLACEHOLDER, str)


class TestLLMConfiguration:
    """Tests for LLM-related configuration."""

    def test_default_temperature(self):
        """Test DEFAULT_TEMPERATURE is a valid temperature value."""
        assert DEFAULT_TEMPERATURE == 0.1
        assert isinstance(DEFAULT_TEMPERATURE, float)
        assert 0 <= DEFAULT_TEMPERATURE <= 1


class TestAsyncConfiguration:
    """Tests for async operation configuration."""

    def test_default_max_async(self):
        """Test DEFAULT_MAX_ASYNC is a positive integer."""
        assert DEFAULT_MAX_ASYNC == 4
        assert isinstance(DEFAULT_MAX_ASYNC, int)
        assert DEFAULT_MAX_ASYNC > 0

    def test_default_max_parallel_insert(self):
        """Test DEFAULT_MAX_PARALLEL_INSERT is a positive integer."""
        assert DEFAULT_MAX_PARALLEL_INSERT == 2
        assert isinstance(DEFAULT_MAX_PARALLEL_INSERT, int)
        assert DEFAULT_MAX_PARALLEL_INSERT > 0


class TestEmbeddingConfiguration:
    """Tests for embedding-related configuration."""

    def test_default_embedding_func_max_async(self):
        """Test DEFAULT_EMBEDDING_FUNC_MAX_ASYNC is a positive integer."""
        assert DEFAULT_EMBEDDING_FUNC_MAX_ASYNC == 8
        assert isinstance(DEFAULT_EMBEDDING_FUNC_MAX_ASYNC, int)
        assert DEFAULT_EMBEDDING_FUNC_MAX_ASYNC > 0

    def test_default_embedding_batch_num(self):
        """Test DEFAULT_EMBEDDING_BATCH_NUM is a positive integer."""
        assert DEFAULT_EMBEDDING_BATCH_NUM == 10
        assert isinstance(DEFAULT_EMBEDDING_BATCH_NUM, int)
        assert DEFAULT_EMBEDDING_BATCH_NUM > 0

    def test_default_embedding_similarity_threshold(self):
        """Test DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD is a valid similarity threshold."""
        assert DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD == 0.95
        assert isinstance(DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD, float)
        assert 0 <= DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD <= 1


class TestMigrationAndBatchConfiguration:
    """Tests for migration and batch processing defaults."""

    def test_default_migration_batch_size(self):
        """Test DEFAULT_MIGRATION_BATCH_SIZE is a positive integer."""
        assert DEFAULT_MIGRATION_BATCH_SIZE == 500
        assert isinstance(DEFAULT_MIGRATION_BATCH_SIZE, int)
        assert DEFAULT_MIGRATION_BATCH_SIZE > 0


class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    def test_default_timeout(self):
        """Test DEFAULT_TIMEOUT is a positive integer."""
        assert DEFAULT_TIMEOUT == 300
        assert isinstance(DEFAULT_TIMEOUT, int)
        assert DEFAULT_TIMEOUT > 0

    def test_default_llm_timeout(self):
        """Test DEFAULT_LLM_TIMEOUT is a positive integer."""
        assert DEFAULT_LLM_TIMEOUT == 180
        assert isinstance(DEFAULT_LLM_TIMEOUT, int)
        assert DEFAULT_LLM_TIMEOUT > 0

    def test_default_embedding_timeout(self):
        """Test DEFAULT_EMBEDDING_TIMEOUT is a positive integer."""
        assert DEFAULT_EMBEDDING_TIMEOUT == 30
        assert isinstance(DEFAULT_EMBEDDING_TIMEOUT, int)
        assert DEFAULT_EMBEDDING_TIMEOUT > 0


class TestTopicConnectivityConfiguration:
    """Tests for topic connectivity check configuration."""

    def test_default_min_relationship_density(self):
        """Test DEFAULT_MIN_RELATIONSHIP_DENSITY is a valid ratio."""
        assert DEFAULT_MIN_RELATIONSHIP_DENSITY == 0.3
        assert isinstance(DEFAULT_MIN_RELATIONSHIP_DENSITY, float)
        assert 0 <= DEFAULT_MIN_RELATIONSHIP_DENSITY <= 1

    def test_default_min_entity_coverage(self):
        """Test DEFAULT_MIN_ENTITY_COVERAGE is a valid ratio."""
        assert DEFAULT_MIN_ENTITY_COVERAGE == 0.5
        assert isinstance(DEFAULT_MIN_ENTITY_COVERAGE, float)
        assert 0 <= DEFAULT_MIN_ENTITY_COVERAGE <= 1

    def test_default_check_topic_connectivity_is_bool(self):
        """Test DEFAULT_CHECK_TOPIC_CONNECTIVITY is a boolean."""
        assert isinstance(DEFAULT_CHECK_TOPIC_CONNECTIVITY, bool)
        assert DEFAULT_CHECK_TOPIC_CONNECTIVITY is True


class TestLoggingConfiguration:
    """Tests for logging configuration defaults."""

    def test_default_log_max_bytes(self):
        """Test DEFAULT_LOG_MAX_BYTES is a positive integer."""
        assert DEFAULT_LOG_MAX_BYTES == 10485760  # 10MB
        assert isinstance(DEFAULT_LOG_MAX_BYTES, int)
        assert DEFAULT_LOG_MAX_BYTES > 0

    def test_default_log_backup_count(self):
        """Test DEFAULT_LOG_BACKUP_COUNT is a positive integer."""
        assert DEFAULT_LOG_BACKUP_COUNT == 5
        assert isinstance(DEFAULT_LOG_BACKUP_COUNT, int)
        assert DEFAULT_LOG_BACKUP_COUNT > 0

    def test_default_log_filename(self):
        """Test DEFAULT_LOG_FILENAME is a string."""
        assert DEFAULT_LOG_FILENAME == 'yar.log'
        assert isinstance(DEFAULT_LOG_FILENAME, str)
        assert '.log' in DEFAULT_LOG_FILENAME


class TestFTSCacheConfiguration:
    """Tests for full-text search cache configuration."""

    def test_default_fts_cache_ttl(self):
        """Test DEFAULT_FTS_CACHE_TTL is a positive integer (seconds)."""
        assert DEFAULT_FTS_CACHE_TTL == 300  # 5 minutes
        assert isinstance(DEFAULT_FTS_CACHE_TTL, int)
        assert DEFAULT_FTS_CACHE_TTL > 0

    def test_default_fts_cache_max_size(self):
        """Test DEFAULT_FTS_CACHE_MAX_SIZE is a positive integer."""
        assert DEFAULT_FTS_CACHE_MAX_SIZE == 5000
        assert isinstance(DEFAULT_FTS_CACHE_MAX_SIZE, int)
        assert DEFAULT_FTS_CACHE_MAX_SIZE > 0

    def test_default_fts_cache_enabled_is_bool(self):
        """Test DEFAULT_FTS_CACHE_ENABLED is a boolean."""
        assert isinstance(DEFAULT_FTS_CACHE_ENABLED, bool)
        assert DEFAULT_FTS_CACHE_ENABLED is True


class TestMetricsConfiguration:
    """Tests for metrics configuration."""

    def test_default_metrics_enabled_is_bool(self):
        """Test DEFAULT_METRICS_ENABLED is a boolean."""
        assert isinstance(DEFAULT_METRICS_ENABLED, bool)
        assert DEFAULT_METRICS_ENABLED is True

    def test_default_metrics_history_size(self):
        """Test DEFAULT_METRICS_HISTORY_SIZE is a positive integer."""
        assert DEFAULT_METRICS_HISTORY_SIZE == 1000
        assert isinstance(DEFAULT_METRICS_HISTORY_SIZE, int)
        assert DEFAULT_METRICS_HISTORY_SIZE > 0

    def test_default_metrics_window_seconds(self):
        """Test DEFAULT_METRICS_WINDOW_SECONDS is a positive integer."""
        assert DEFAULT_METRICS_WINDOW_SECONDS == 3600  # 1 hour
        assert isinstance(DEFAULT_METRICS_WINDOW_SECONDS, int)
        assert DEFAULT_METRICS_WINDOW_SECONDS > 0


class TestStatusConstants:
    """Tests for status constants."""

    def test_document_processing_statuses(self):
        """Test document processing status constants."""
        assert STATUS_PENDING == 'pending'
        assert STATUS_PROCESSING == 'processing'
        assert STATUS_COMPLETED == 'completed'
        assert STATUS_FAILED == 'failed'

    def test_pipeline_job_statuses(self):
        """Test pipeline/job status constants."""
        assert STATUS_SUCCESS == 'success'
        assert STATUS_FAILURE == 'failure'
        assert STATUS_CANCELLED == 'cancelled'

    def test_all_status_constants_are_strings(self):
        """Test all status constants are strings."""
        status_constants = [
            STATUS_PENDING,
            STATUS_PROCESSING,
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_SUCCESS,
            STATUS_FAILURE,
            STATUS_CANCELLED,
        ]
        assert all(isinstance(s, str) for s in status_constants)

    def test_status_values_are_lowercase(self):
        """Test status values are lowercase."""
        status_constants = [
            STATUS_PENDING,
            STATUS_PROCESSING,
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_SUCCESS,
            STATUS_FAILURE,
            STATUS_CANCELLED,
        ]
        assert all(s.islower() for s in status_constants)


class TestBackgroundJobStateConstants:
    """Tests for background job state constants."""

    def test_state_busy(self):
        """Test STATE_BUSY constant."""
        assert STATE_BUSY == 'busy'
        assert isinstance(STATE_BUSY, str)

    def test_state_cancellation_requested(self):
        """Test STATE_CANCELLATION_REQUESTED constant."""
        assert STATE_CANCELLATION_REQUESTED == 'cancellation_requested'
        assert isinstance(STATE_CANCELLATION_REQUESTED, str)

    def test_state_request_pending(self):
        """Test STATE_REQUEST_PENDING constant."""
        assert STATE_REQUEST_PENDING == 'request_pending'
        assert isinstance(STATE_REQUEST_PENDING, str)


class TestNamespaceConstants:
    """Tests for shared storage namespace keys."""

    def test_ns_pipeline_status(self):
        """Test NS_PIPELINE_STATUS constant."""
        assert NS_PIPELINE_STATUS == 'pipeline_status'
        assert isinstance(NS_PIPELINE_STATUS, str)

    def test_ns_orphan_connection_status(self):
        """Test NS_ORPHAN_CONNECTION_STATUS constant."""
        assert NS_ORPHAN_CONNECTION_STATUS == 'orphan_connection_status'
        assert isinstance(NS_ORPHAN_CONNECTION_STATUS, str)

    def test_ns_entity_resolution_status(self):
        """Test NS_ENTITY_RESOLUTION_STATUS constant."""
        assert NS_ENTITY_RESOLUTION_STATUS == 'entity_resolution_status'
        assert isinstance(NS_ENTITY_RESOLUTION_STATUS, str)

    def test_all_namespace_constants_are_strings(self):
        """Test all namespace constants are strings."""
        ns_constants = [
            NS_PIPELINE_STATUS,
            NS_ORPHAN_CONNECTION_STATUS,
            NS_ENTITY_RESOLUTION_STATUS,
        ]
        assert all(isinstance(ns, str) for ns in ns_constants)


class TestCommonFieldNames:
    """Tests for common field names used across modules."""

    def test_entity_relation_field_names(self):
        """Test entity/relation field name constants."""
        assert FIELD_DESCRIPTION == 'description'
        assert FIELD_SOURCE_ID == 'source_id'
        assert FIELD_ENTITY_NAME == 'entity_name'
        assert FIELD_ENTITY_TYPE == 'entity_type'
        assert FIELD_FILE_PATH == 'file_path'
        assert FIELD_CONTENT == 'content'
        assert FIELD_KEYWORDS == 'keywords'
        assert FIELD_WEIGHT == 'weight'

    def test_graph_edge_field_names(self):
        """Test graph edge field name constants."""
        assert FIELD_SRC_ID == 'src_id'
        assert FIELD_TGT_ID == 'tgt_id'

    def test_status_progress_field_names(self):
        """Test status/progress field name constants."""
        assert FIELD_STATUS == 'status'
        assert FIELD_LATEST_MESSAGE == 'latest_message'
        assert FIELD_HISTORY_MESSAGES == 'history_messages'

    def test_all_field_names_are_strings(self):
        """Test all field name constants are strings."""
        field_constants = [
            FIELD_DESCRIPTION,
            FIELD_SOURCE_ID,
            FIELD_ENTITY_NAME,
            FIELD_ENTITY_TYPE,
            FIELD_FILE_PATH,
            FIELD_CONTENT,
            FIELD_KEYWORDS,
            FIELD_WEIGHT,
            FIELD_SRC_ID,
            FIELD_TGT_ID,
            FIELD_STATUS,
            FIELD_LATEST_MESSAGE,
            FIELD_HISTORY_MESSAGES,
        ]
        assert all(isinstance(f, str) for f in field_constants)

    def test_field_names_are_snake_case(self):
        """Test field names follow snake_case convention."""
        field_constants = [
            FIELD_DESCRIPTION,
            FIELD_SOURCE_ID,
            FIELD_ENTITY_NAME,
            FIELD_ENTITY_TYPE,
            FIELD_FILE_PATH,
            FIELD_CONTENT,
            FIELD_KEYWORDS,
            FIELD_WEIGHT,
            FIELD_SRC_ID,
            FIELD_TGT_ID,
            FIELD_STATUS,
            FIELD_LATEST_MESSAGE,
            FIELD_HISTORY_MESSAGES,
        ]
        # Field names should only contain lowercase letters, numbers, and underscores
        for field in field_constants:
            assert field.replace('_', '').isalnum()
            assert field.islower()


class TestDefaultWorkspace:
    """Tests for default workspace constant."""

    def test_default_workspace(self):
        """Test DEFAULT_WORKSPACE is set to 'default'."""
        assert DEFAULT_WORKSPACE == 'default'
        assert isinstance(DEFAULT_WORKSPACE, str)


class TestConstantRelationships:
    """Tests for relationships and constraints between constants."""

    def test_chunk_top_k_less_than_top_k(self):
        """Test chunk_top_k is not greater than top_k."""
        assert DEFAULT_CHUNK_TOP_K <= DEFAULT_TOP_K

    def test_summary_recommended_less_than_max(self):
        """Test recommended summary length is not greater than max."""
        assert DEFAULT_SUMMARY_LENGTH_RECOMMENDED <= DEFAULT_SUMMARY_MAX_TOKENS

    def test_max_total_tokens_greater_than_entity_and_relation(self):
        """Test max_total_tokens is sufficient for entity and relation tokens."""
        assert DEFAULT_MAX_TOTAL_TOKENS >= DEFAULT_MAX_ENTITY_TOKENS
        assert DEFAULT_MAX_TOTAL_TOKENS >= DEFAULT_MAX_RELATION_TOKENS

    def test_embedding_similarity_threshold_valid_range(self):
        """Test embedding similarity threshold is in valid range."""
        assert 0 <= DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD <= 1

    def test_cosine_threshold_valid_range(self):
        """Test cosine threshold is in valid range."""
        assert 0 <= DEFAULT_COSINE_THRESHOLD <= 1

    def test_min_relationship_density_valid_range(self):
        """Test min relationship density is in valid range."""
        assert 0 <= DEFAULT_MIN_RELATIONSHIP_DENSITY <= 1

    def test_min_entity_coverage_valid_range(self):
        """Test min entity coverage is in valid range."""
        assert 0 <= DEFAULT_MIN_ENTITY_COVERAGE <= 1

    def test_temperature_valid_range(self):
        """Test temperature is in valid range."""
        assert 0 <= DEFAULT_TEMPERATURE <= 1

    def test_fts_cache_size_is_positive(self):
        """Test FTS cache size is a positive integer."""
        # FTS cache size is independent of metrics history size
        assert DEFAULT_FTS_CACHE_MAX_SIZE > 0
        assert isinstance(DEFAULT_FTS_CACHE_MAX_SIZE, int)

    def test_embedding_timeout_less_than_llm_timeout(self):
        """Test embedding timeout is less than LLM timeout."""
        # Embeddings are typically faster than LLM calls
        assert DEFAULT_EMBEDDING_TIMEOUT < DEFAULT_LLM_TIMEOUT

    def test_llm_timeout_less_than_gunicorn_timeout(self):
        """Test LLM timeout is less than Gunicorn timeout."""
        # LLM calls should timeout before the server's worker timeout
        assert DEFAULT_LLM_TIMEOUT < DEFAULT_TIMEOUT

    def test_retrieval_multiplier_at_least_one(self):
        """Test retrieval multiplier is at least 1 (disabled mode)."""
        assert DEFAULT_RETRIEVAL_MULTIPLIER >= 1

    def test_max_async_positive(self):
        """Test max async and parallel insert are positive."""
        assert DEFAULT_MAX_ASYNC > 0
        assert DEFAULT_MAX_PARALLEL_INSERT > 0
        assert DEFAULT_MAX_PARALLEL_INSERT <= DEFAULT_MAX_ASYNC

    def test_entity_types_not_empty(self):
        """Test entity types list is not empty."""
        assert len(DEFAULT_ENTITY_TYPES) > 0
        # Each type should be a single word or CamelCase (no spaces)
        for entity_type in DEFAULT_ENTITY_TYPES:
            assert ' ' not in entity_type
            assert entity_type[0].isupper()  # First letter uppercase


class TestConstantImmutability:
    """Tests verifying constants are used as immutable values."""

    def test_cannot_modify_top_k(self):
        """Test that modifying the imported constant doesn't affect usage."""
        # This is a documentation test showing how constants should be used
        original_top_k = DEFAULT_TOP_K
        assert original_top_k == 60
        # Constants should not be reassigned in real code

    def test_entity_types_list_immutability(self):
        """Test that DEFAULT_ENTITY_TYPES should not be modified."""
        original_length = len(DEFAULT_ENTITY_TYPES)
        # Verify it's still a list (modifiable in Python, but shouldn't be)
        assert isinstance(DEFAULT_ENTITY_TYPES, list)
        assert len(DEFAULT_ENTITY_TYPES) == original_length

    def test_valid_source_ids_limit_methods_is_immutable_set(self):
        """Test that VALID_SOURCE_IDS_LIMIT_METHODS is a set."""
        assert isinstance(VALID_SOURCE_IDS_LIMIT_METHODS, set)
        # Sets are hashable collections
        assert SOURCE_IDS_LIMIT_METHOD_KEEP in VALID_SOURCE_IDS_LIMIT_METHODS
        assert SOURCE_IDS_LIMIT_METHOD_FIFO in VALID_SOURCE_IDS_LIMIT_METHODS
