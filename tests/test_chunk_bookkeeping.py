"""Regression tests for document-scoped chunk bookkeeping."""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from yar.base import DocProcessingStatus, DocStatus, StoragesStatus
from yar.utils import compute_mdhash_id
from yar.yar import YAR


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


def _make_status_doc(file_path: str, track_id: str) -> DocProcessingStatus:
	return DocProcessingStatus(
		content_summary='doc',
		content_length=12,
		file_path=file_path,
		status=DocStatus.PENDING,
		created_at='2024-01-01T00:00:00Z',
		updated_at='2024-01-01T00:00:00Z',
		track_id=track_id,
	)


class _DummyStorage:
	def __init__(self, *args, **kwargs):
		self.db = None

	async def get_by_id(self, *args, **kwargs):
		return None

	async def get_by_ids(self, *args, **kwargs):
		return []

	async def get_docs_by_status(self, *args, **kwargs):
		return {}

	async def upsert(self, *args, **kwargs):
		return None

	async def delete(self, *args, **kwargs):
		return None

	async def delete_by_doc_id(self, *args, **kwargs):
		return []

	async def get_chunk_ids_by_doc_id(self, *args, **kwargs):
		return []

	async def initialize(self, *args, **kwargs):
		return None

	async def finalize(self, *args, **kwargs):
		return None

	async def index_done_callback(self, *args, **kwargs):
		return None

	async def get_nodes_batch(self, *args, **kwargs):
		return {}

	async def get_edges_batch(self, *args, **kwargs):
		return {}

	async def remove_nodes(self, *args, **kwargs):
		return None

def _build_rag(temp_working_dir: str, mock_embedding_func, mock_llm_func, **kwargs) -> YAR:
	with patch.object(YAR, '_get_storage_class', return_value=_DummyStorage):
		rag = YAR(
			working_dir=temp_working_dir,
			embedding_func=mock_embedding_func,
			llm_model_func=mock_llm_func,
			workspace='workspace-test',
			**kwargs,
		)
	rag._storages_status = StoragesStatus.INITIALIZED
	rag._insert_done = AsyncMock()
	rag._invalidate_fts_cache = AsyncMock()
	return rag


@pytest.mark.offline
class TestChunkBookkeeping:
	@patch('yar.yar.merge_nodes_and_edges', new_callable=AsyncMock)
	@patch('yar.yar.get_namespace_lock')
	@patch('yar.yar.get_namespace_data')
	@patch('yar.yar.verify_storage_implementation')
	@patch('yar.yar.check_storage_env_vars')
	@pytest.mark.asyncio
	async def test_pipeline_chunk_ids_are_doc_scoped_and_preserve_full_doc_id(
		self,
		mock_check_env,
		mock_verify_storage,
		mock_get_namespace_data,
		mock_get_namespace_lock,
		mock_merge_nodes_and_edges,
		temp_working_dir,
		mock_embedding_func,
		mock_llm_func,
	):
		pipeline_status = {'busy': False, 'history_messages': []}
		pipeline_status_lock = asyncio.Lock()
		mock_get_namespace_data.return_value = pipeline_status
		mock_get_namespace_lock.return_value = pipeline_status_lock

		rag = _build_rag(
			temp_working_dir,
			mock_embedding_func,
			mock_llm_func,
			max_parallel_insert=1,
		)
		rag._process_extract_entities = AsyncMock(return_value=[])
		rag.chunking_func = lambda *_args, **_kwargs: [{'content': 'shared chunk'}]

		docs_to_process = {
			'doc-1': _make_status_doc('/tmp/doc-1.txt', 'track-1'),
			'doc-2': _make_status_doc('/tmp/doc-2.txt', 'track-2'),
		}
		rag.doc_status.get_docs_by_status = AsyncMock(side_effect=[{}, {}, docs_to_process])
		rag._validate_and_fix_document_consistency = AsyncMock(return_value=docs_to_process)
		rag.full_docs.get_by_id = AsyncMock(side_effect=[{'content': 'same content'}, {'content': 'same content'}])
		rag.doc_status.upsert = AsyncMock()
		rag.text_chunks.delete_by_doc_id = AsyncMock(return_value=[])
		rag.chunks_vdb.delete_by_doc_id = AsyncMock(return_value=0)

		text_chunk_payloads: list[dict[str, dict[str, object]]] = []

		async def record_text_chunk_upsert(chunks: dict[str, dict[str, object]]) -> None:
			text_chunk_payloads.append(chunks)

		rag.text_chunks.upsert = AsyncMock(side_effect=record_text_chunk_upsert)
		rag.chunks_vdb.upsert = AsyncMock()

		await rag.apipeline_process_enqueue_documents()

		assert len(text_chunk_payloads) == 2

		recorded_chunks = {}
		for chunk_payload in text_chunk_payloads:
			assert len(chunk_payload) == 1
			chunk_id, chunk_data = next(iter(chunk_payload.items()))
			recorded_chunks[chunk_data['full_doc_id']] = (chunk_id, chunk_data)

		assert set(recorded_chunks) == {'doc-1', 'doc-2'}
		assert recorded_chunks['doc-1'][0] == compute_mdhash_id('doc-1shared chunk', prefix='chunk-')
		assert recorded_chunks['doc-2'][0] == compute_mdhash_id('doc-2shared chunk', prefix='chunk-')
		assert recorded_chunks['doc-1'][0] != recorded_chunks['doc-2'][0]
		assert recorded_chunks['doc-1'][1]['full_doc_id'] == 'doc-1'
		assert recorded_chunks['doc-2'][1]['full_doc_id'] == 'doc-2'
		mock_merge_nodes_and_edges.assert_awaited()

	@patch('yar.yar.merge_nodes_and_edges', new_callable=AsyncMock)
	@patch('yar.yar.get_namespace_lock')
	@patch('yar.yar.get_namespace_data')
	@patch('yar.yar.verify_storage_implementation')
	@patch('yar.yar.check_storage_env_vars')
	@pytest.mark.asyncio
	async def test_pipeline_reprocessing_deletes_doc_scoped_chunks_before_reupsert(
		self,
		mock_check_env,
		mock_verify_storage,
		mock_get_namespace_data,
		mock_get_namespace_lock,
		mock_merge_nodes_and_edges,
		temp_working_dir,
		mock_embedding_func,
		mock_llm_func,
	):
		pipeline_status = {'busy': False, 'history_messages': []}
		pipeline_status_lock = asyncio.Lock()
		mock_get_namespace_data.return_value = pipeline_status
		mock_get_namespace_lock.return_value = pipeline_status_lock

		rag = _build_rag(
			temp_working_dir,
			mock_embedding_func,
			mock_llm_func,
			max_parallel_insert=1,
		)
		rag._process_extract_entities = AsyncMock(return_value=[])
		rag.chunking_func = lambda *_args, **_kwargs: [{'content': 'refreshed chunk'}]

		status_doc = _make_status_doc('/tmp/doc-1.txt', 'track-1')
		docs_to_process = {'doc-1': status_doc}
		rag.doc_status.get_docs_by_status = AsyncMock(side_effect=[{}, {}, docs_to_process])
		rag._validate_and_fix_document_consistency = AsyncMock(return_value=docs_to_process)
		rag.full_docs.get_by_id = AsyncMock(return_value={'content': 'same content'})
		rag.doc_status.upsert = AsyncMock()

		events: list[str] = []

		async def delete_text_chunks(doc_id: str) -> list[str]:
			events.append(f'text-delete:{doc_id}')
			return ['old-chunk']

		async def delete_chunk_vectors(doc_id: str) -> int:
			events.append(f'vector-delete:{doc_id}')
			return 1

		async def upsert_text_chunks(chunks: dict[str, dict[str, object]]) -> None:
			events.append(f"text-upsert:{next(iter(chunks))}")

		async def upsert_chunk_vectors(chunks: dict[str, dict[str, object]]) -> None:
			events.append(f"vector-upsert:{next(iter(chunks))}")

		rag.text_chunks.delete_by_doc_id = AsyncMock(side_effect=delete_text_chunks)
		rag.chunks_vdb.delete_by_doc_id = AsyncMock(side_effect=delete_chunk_vectors)
		rag.text_chunks.upsert = AsyncMock(side_effect=upsert_text_chunks)
		rag.chunks_vdb.upsert = AsyncMock(side_effect=upsert_chunk_vectors)

		await rag.apipeline_process_enqueue_documents()

		first_upsert_index = min(i for i, event in enumerate(events) if event.startswith('text-upsert') or event.startswith('vector-upsert'))
		assert events.index('text-delete:doc-1') < first_upsert_index
		assert events.index('vector-delete:doc-1') < first_upsert_index
		rag.text_chunks.delete_by_doc_id.assert_awaited_once_with('doc-1')
		rag.chunks_vdb.delete_by_doc_id.assert_awaited_once_with('doc-1')
		mock_merge_nodes_and_edges.assert_awaited_once()

	@patch('yar.yar.verify_storage_implementation')
	@patch('yar.yar.check_storage_env_vars')
	@pytest.mark.asyncio
	async def test_validate_and_fix_document_consistency_deletes_orphan_chunks_before_doc_status(
		self,
		mock_check_env,
		mock_verify_storage,
		temp_working_dir,
		mock_embedding_func,
		mock_llm_func,
	):
		rag = _build_rag(temp_working_dir, mock_embedding_func, mock_llm_func)
		rag.full_docs.get_by_id = AsyncMock(return_value=None)

		events: list[str] = []

		async def delete_text_chunks(doc_id: str) -> list[str]:
			events.append(f'text-delete:{doc_id}')
			return ['chunk-1']

		async def delete_chunk_vectors(doc_id: str) -> int:
			events.append(f'vector-delete:{doc_id}')
			return 1

		async def delete_doc_status(doc_ids: list[str]) -> None:
			events.append(f"doc-status-delete:{','.join(doc_ids)}")

		rag.text_chunks.delete_by_doc_id = AsyncMock(side_effect=delete_text_chunks)
		rag.chunks_vdb.delete_by_doc_id = AsyncMock(side_effect=delete_chunk_vectors)
		rag.doc_status.delete = AsyncMock(side_effect=delete_doc_status)

		to_process_docs = {'doc-1': _make_status_doc('/tmp/doc-1.txt', 'track-1')}
		pipeline_status = {'history_messages': []}

		remaining_docs = await rag._validate_and_fix_document_consistency(
			to_process_docs,
			pipeline_status,
			asyncio.Lock(),
		)

		assert remaining_docs == {}
		assert events.index('text-delete:doc-1') < events.index('doc-status-delete:doc-1')
		assert events.index('vector-delete:doc-1') < events.index('doc-status-delete:doc-1')
		rag.doc_status.delete.assert_awaited_once_with(['doc-1'])

	@patch('yar.yar.get_namespace_lock')
	@patch('yar.yar.get_namespace_data')
	@patch('yar.yar.verify_storage_implementation')
	@patch('yar.yar.check_storage_env_vars')
	@pytest.mark.asyncio
	async def test_adelete_by_doc_id_merges_stale_chunk_ids_from_doc_scoped_lookup(
		self,
		mock_check_env,
		mock_verify_storage,
		mock_get_namespace_data,
		mock_get_namespace_lock,
		temp_working_dir,
		mock_embedding_func,
		mock_llm_func,
	):
		pipeline_status = {'busy': False, 'history_messages': []}
		pipeline_status_lock = asyncio.Lock()
		mock_get_namespace_data.return_value = pipeline_status
		mock_get_namespace_lock.return_value = pipeline_status_lock

		rag = _build_rag(temp_working_dir, mock_embedding_func, mock_llm_func)
		rag.doc_status.get_by_id = AsyncMock(
			return_value={
				'status': DocStatus.PROCESSED,
				'file_path': '/tmp/doc-1.txt',
				'chunks_list': ['stored-chunk'],
			}
		)
		rag.text_chunks.get_chunk_ids_by_doc_id = AsyncMock(return_value=['stored-chunk', 'stale-chunk'])
		rag.chunks_vdb.delete = AsyncMock()
		rag.text_chunks.delete = AsyncMock()
		rag.full_entities.get_by_id = AsyncMock(return_value=None)
		rag.full_relations.get_by_id = AsyncMock(return_value=None)
		rag.entities_vdb.delete = AsyncMock()
		rag.relationships_vdb.delete = AsyncMock()
		rag.full_entities.delete = AsyncMock()
		rag.full_relations.delete = AsyncMock()
		rag.full_docs.delete = AsyncMock()
		rag.doc_status.delete = AsyncMock()

		result = await rag.adelete_by_doc_id('doc-1')

		assert result.status == 'success'
		rag.text_chunks.get_chunk_ids_by_doc_id.assert_awaited_once_with('doc-1')
		assert set(rag.chunks_vdb.delete.await_args.args[0]) == {'stored-chunk', 'stale-chunk'}
		assert set(rag.text_chunks.delete.await_args.args[0]) == {'stored-chunk', 'stale-chunk'}
		assert len(rag.chunks_vdb.delete.await_args.args[0]) == 2
		assert len(rag.text_chunks.delete.await_args.args[0]) == 2
