from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import numpy as np
import pytest

from yar.base import BaseVectorStorage
from yar.kg.postgres_impl import PGVectorStorage, PostgreSQLDB
from yar.namespace import NameSpace
from yar.yar import YAR


def _make_storage(namespace: str, db_query: AsyncMock) -> PGVectorStorage:
    return PGVectorStorage(
        namespace=namespace,
        workspace='workspace-test',
        global_config={
            'embedding_batch_num': 1,
            'vector_db_storage_cls_kwargs': {'cosine_better_than_threshold': 0.25},
        },
        embedding_func=AsyncMock(),
        meta_fields=set(),
        db=cast(PostgreSQLDB, cast(object, SimpleNamespace(query=db_query))),
    )


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('namespace', 'expected_column'),
    [
        (NameSpace.VECTOR_STORE_ENTITIES, 'e.content_vector'),
        (NameSpace.VECTOR_STORE_RELATIONSHIPS, 'r.content_vector'),
        (NameSpace.VECTOR_STORE_CHUNKS, 'c.content_vector'),
    ],
)
async def test_pgvector_query_binds_embedding_parameter(namespace: str, expected_column: str) -> None:
    db_query = AsyncMock(return_value=[])
    storage = _make_storage(namespace, db_query)

    _ = await storage.query('ignored', top_k=5, query_embedding=[0.1, 0.2])

    call = db_query.await_args_list[0]
    sql = cast(str, call.args[0])
    params = cast(list[object], call.kwargs['params'])

    assert f'ORDER BY {expected_column}' in sql
    assert '$2' in sql
    assert '::vector' not in sql
    assert '[0.1,0.2]' not in sql
    assert params == ['workspace-test', [0.1, 0.2], 0.75, 5]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pgvector_query_rejects_nan_vectors_before_db_call() -> None:
    db_query = AsyncMock(return_value=[])
    storage = _make_storage(NameSpace.VECTOR_STORE_ENTITIES, db_query)

    with pytest.raises(ValueError, match='Embedding vector contains NaN or Inf values'):
        _ = await storage.query('ignored', top_k=5, query_embedding=[0.1, float('nan')])

    db_query.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('orphan_vector', 'expected_vector'),
    [
        ([0.2, 0.4], [0.2, 0.4]),
        (np.array([0.2, 0.4], dtype=float), [0.2, 0.4]),
        ('[0.2, 0.4]', [0.2, 0.4]),
    ],
)
async def test_orphan_candidate_query_binds_normalized_vector(orphan_vector: object, expected_vector: list[float]) -> None:
    db_query = AsyncMock(
        side_effect=[
            [
                {
                    'entity_name': 'orphan-1',
                    'content': 'person: orphaned node',
                    'content_vector': orphan_vector,
                }
            ],
            [],
        ]
    )

    rag = object.__new__(YAR)
    rag.workspace = 'workspace-test'
    rag.entities_vdb = cast(
        BaseVectorStorage,
        cast(
            object,
            SimpleNamespace(
                workspace='workspace-test',
                db=SimpleNamespace(query=db_query),
            ),
        ),
    )
    rag.orphan_connection_threshold = 0.25
    rag.orphan_confidence_threshold = 0.75
    rag.orphan_cross_connect = True
    rag.orphan_connection_max_degree = 0
    rag.llm_model_func = None

    result = await YAR.aconnect_orphan_entities(rag)

    candidate_call = db_query.await_args_list[1]
    sql = cast(str, candidate_call.args[0])
    params = cast(list[object], candidate_call.args[1])

    assert result['orphans_found'] == 1
    assert '$3' in sql
    assert '::vector' not in sql
    assert '[0.2, 0.4]' not in sql
    assert params == ['workspace-test', 'orphan-1', expected_vector, 0.25, 3]
