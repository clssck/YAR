from __future__ import annotations

import contextlib
import datetime as dt
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
import pytest_asyncio

os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'yar')
os.environ.setdefault('POSTGRES_PASSWORD', 'yar_pass')
os.environ.setdefault('POSTGRES_DATABASE', 'yar')

from yar.base import DocStatus
from yar.kg.postgres_impl import PGDocStatusStorage, PostgreSQLDB


async def _insert_doc_status_rows(db: PostgreSQLDB, rows: list[tuple[object, ...]]) -> None:
    await db.executemany(
        """
        INSERT INTO YAR_DOC_STATUS (
            workspace,
            id,
            content_summary,
            content_length,
            chunks_count,
            status,
            file_path,
            chunks_list,
            track_id,
            metadata,
            error_msg,
            s3_key,
            created_at,
            updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8::jsonb, $9, $10::jsonb, $11, $12, $13, $14
        )
        """,
        rows,
    )


@pytest_asyncio.fixture
async def doc_status_storage() -> AsyncIterator[PGDocStatusStorage]:
    workspace = f'test_doc_status_page_{uuid.uuid4().hex[:8]}'
    db = PostgreSQLDB(
        {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'user': os.getenv('POSTGRES_USER', 'yar'),
            'password': os.getenv('POSTGRES_PASSWORD', 'yar_pass'),
            'database': os.getenv('POSTGRES_DATABASE', 'yar'),
            'workspace': workspace,
            'max_connections': int(os.getenv('POSTGRES_MAX_CONNECTIONS', '10')),
            'min_connections': 1,
            'connection_retry_attempts': min(10, int(os.getenv('POSTGRES_CONNECTION_RETRIES', '3'))),
            'connection_retry_backoff': min(5.0, float(os.getenv('POSTGRES_CONNECTION_RETRY_BACKOFF', '0.5'))),
            'connection_retry_backoff_max': min(
                60.0,
                float(os.getenv('POSTGRES_CONNECTION_RETRY_BACKOFF_MAX', '5.0')),
            ),
            'pool_close_timeout': min(30.0, float(os.getenv('POSTGRES_POOL_CLOSE_TIMEOUT', '5.0'))),
        }
    )

    await db.initdb()
    storage = PGDocStatusStorage(
        namespace='doc_status',
        workspace=workspace,
        global_config={},
        embedding_func=cast(Any, None),
        db=db,
    )

    with contextlib.suppress(Exception):
        await db.execute('DELETE FROM YAR_DOC_STATUS WHERE workspace = $1', data={'workspace': workspace})

    try:
        yield storage
    finally:
        with contextlib.suppress(Exception):
            await db.execute('DELETE FROM YAR_DOC_STATUS WHERE workspace = $1', data={'workspace': workspace})
        if db.pool is not None:
            with contextlib.suppress(Exception):
                await db.pool.close()


@pytest.mark.integration
@pytest.mark.requires_db
class TestDocStatusPaginationIntegration:
    @pytest.mark.asyncio
    async def test_filtered_out_of_range_page_keeps_filtered_total(self, doc_status_storage: PGDocStatusStorage):
        storage = doc_status_storage
        workspace = storage.workspace
        now = dt.datetime(2024, 1, 5, 12, 0, 0)

        await _insert_doc_status_rows(
            storage._db_required(),
            [
                (
                    workspace,
                    'processed-1',
                    'processed one',
                    100,
                    2,
                    DocStatus.PROCESSED.value,
                    '/docs/processed-1.txt',
                    '["chunk-a", "chunk-b"]',
                    'track-1',
                    '{"kind": "processed"}',
                    None,
                    's3://processed-1',
                    now,
                    now,
                ),
                (
                    workspace,
                    'processed-2',
                    'processed two',
                    120,
                    1,
                    DocStatus.PROCESSED.value,
                    '/docs/processed-2.txt',
                    '[]',
                    'track-2',
                    '{"kind": "processed"}',
                    None,
                    's3://processed-2',
                    now,
                    now,
                ),
                (
                    workspace,
                    'pending-1',
                    'pending one',
                    80,
                    0,
                    DocStatus.PENDING.value,
                    '/docs/pending-1.txt',
                    '[]',
                    'track-3',
                    '{"kind": "pending"}',
                    'still processing',
                    's3://pending-1',
                    now,
                    now,
                ),
            ],
        )

        documents, total_count = await storage.get_docs_paginated(
            status_filter=DocStatus.PROCESSED,
            page=2,
            page_size=10,
        )

        assert documents == []
        assert total_count == 2

    @pytest.mark.asyncio
    async def test_updated_at_desc_uses_nulls_last(self, doc_status_storage: PGDocStatusStorage):
        storage = doc_status_storage
        workspace = storage.workspace
        recent = dt.datetime(2024, 1, 3, 12, 0, 0)
        older = dt.datetime(2024, 1, 1, 12, 0, 0)

        await _insert_doc_status_rows(
            storage._db_required(),
            [
                (
                    workspace,
                    'null-updated-at',
                    'null updated timestamp',
                    50,
                    0,
                    DocStatus.PROCESSED.value,
                    '/docs/null.txt',
                    '["chunk-null"]',
                    'track-null',
                    '{"order": "null"}',
                    'kept for assertion',
                    's3://null',
                    older,
                    None,
                ),
                (
                    workspace,
                    'older-updated-at',
                    'older updated timestamp',
                    60,
                    1,
                    DocStatus.PROCESSED.value,
                    '/docs/older.txt',
                    '["chunk-older"]',
                    'track-older',
                    '{"order": "older"}',
                    None,
                    's3://older',
                    older,
                    older,
                ),
                (
                    workspace,
                    'recent-updated-at',
                    'recent updated timestamp',
                    70,
                    2,
                    DocStatus.PROCESSED.value,
                    '/docs/recent.txt',
                    '["chunk-recent-1", "chunk-recent-2"]',
                    'track-recent',
                    '{"order": "recent"}',
                    None,
                    's3://recent',
                    recent,
                    recent,
                ),
            ],
        )

        documents, total_count = await storage.get_docs_paginated(
            status_filter=DocStatus.PROCESSED,
            page=1,
            page_size=10,
            sort_field='updated_at',
            sort_direction='desc',
        )

        assert total_count == 3
        assert [doc_id for doc_id, _ in documents[:3]] == [
            'recent-updated-at',
            'older-updated-at',
            'null-updated-at',
        ]

        null_doc = dict(documents)['null-updated-at']
        assert null_doc.updated_at == ''
        assert null_doc.chunks_list == ['chunk-null']
        assert null_doc.metadata == {'order': 'null'}
        assert null_doc.error_msg == 'kept for assertion'
        assert null_doc.s3_key == 's3://null'
