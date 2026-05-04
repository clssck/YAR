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
async def test_pgvector_relationship_query_fuses_vector_and_bm25_primary_results() -> None:
    db_query = AsyncMock(
        side_effect=[
            [
                {
                    'id': 'rel-vector',
                    'src_id': 'CSTD',
                    'tgt_id': 'Product Quality',
                    'created_at': 1,
                    'distance': 0.2,
                    'vector_score': 0.8,
                    'source_type': 'vector',
                }
            ],
            [
                {
                    'id': 'rel-vector',
                    'src_id': 'CSTD',
                    'tgt_id': 'Product Quality',
                    'created_at': 1,
                    'bm25_score': 0.7,
                    'source_type': 'bm25',
                },
                {
                    'id': 'rel-bm25',
                    'src_id': 'CSTD',
                    'tgt_id': 'Accurate Dosing',
                    'created_at': 2,
                    'bm25_score': 0.6,
                    'source_type': 'bm25',
                },
            ],
        ]
    )
    storage = _make_storage(NameSpace.VECTOR_STORE_RELATIONSHIPS, db_query)

    results = await storage.query(
        'product quality risks, dosing risks',
        top_k=5,
        query_embedding=[0.1, 0.2],
    )

    vector_call, bm25_call = db_query.await_args_list
    vector_sql = cast(str, vector_call.args[0])
    bm25_sql = cast(str, bm25_call.args[0])

    assert 'FROM YAR_VDB_RELATION r' in vector_sql
    assert 'ORDER BY r.content_vector' in vector_sql
    assert vector_call.kwargs['params'] == ['workspace-test', [0.1, 0.2], 0.75, 10]
    assert 'to_tsvector' in bm25_sql
    assert 'plainto_tsquery' in bm25_sql
    assert bm25_call.kwargs['params'][0] == 'workspace-test'
    lexical_terms = bm25_call.kwargs['params'][1]
    assert lexical_terms[:3] == ['product quality risks, dosing risks', 'product quality risks', 'dosing risks']
    assert {'poses risk to', 'poses risk of', 'mitigates risk'} <= set(lexical_terms)
    lexical_weights = dict(zip(lexical_terms, bm25_call.kwargs['params'][2], strict=False))
    assert lexical_weights['poses risk to'] == 2.0
    assert bm25_call.kwargs['params'][3] == 10
    assert [(result['src_id'], result['tgt_id']) for result in results] == [
        ('CSTD', 'Product Quality'),
        ('CSTD', 'Accurate Dosing'),
    ]
    assert results[0]['source_type'] == 'vector+bm25'
    assert results[0]['score'] == results[0]['rrf_score']


@pytest.mark.offline
def test_relationship_lexical_terms_expand_query_intent_without_fallback() -> None:
    contributor_terms = PGVectorStorage._relationship_lexical_terms(
        'Who contributes to the CSTD Strategy Working Group?'
    )
    collaboration_terms = PGVectorStorage._relationship_lexical_terms(
        'What did Sanofi learn from collaboration with Alnylam on Fitusiran?'
    )

    assert 'represented by' in contributor_terms
    assert 'on behalf' in contributor_terms
    contributor_weights = dict(PGVectorStorage._relationship_lexical_query_terms('contributors'))
    assert contributor_weights['represented by'] == 3.0
    assert 'collaborates with' in collaboration_terms
    assert 'restructured alliance' in collaboration_terms


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
async def test_orphan_candidate_query_binds_normalized_vector(
    orphan_vector: object, expected_vector: list[float]
) -> None:
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
