#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the working directory aligned with the server's config/.env lookup behavior.
os.chdir(REPO_ROOT)

from yar.base import DocStatus
from yar.kg.postgres_impl import ClientManager, PostgreSQLDB

ORPHAN_DOCS_SQL = """
SELECT
    orphan.full_doc_id,
    orphan.doc_chunk_rows,
    COALESCE(vdb.vdb_chunk_rows, 0)::int AS vdb_chunk_rows
FROM (
    SELECT
        chunks.full_doc_id,
        COUNT(*)::int AS doc_chunk_rows
    FROM YAR_DOC_CHUNKS AS chunks
    LEFT JOIN YAR_DOC_STATUS AS status
        ON status.workspace = chunks.workspace
       AND status.id = chunks.full_doc_id
    LEFT JOIN YAR_DOC_FULL AS doc_full
        ON doc_full.workspace = chunks.workspace
       AND doc_full.id = chunks.full_doc_id
    WHERE chunks.workspace = $1
      AND status.id IS NULL
      AND doc_full.id IS NULL
    GROUP BY chunks.full_doc_id
) AS orphan
LEFT JOIN (
    SELECT full_doc_id, COUNT(*)::int AS vdb_chunk_rows
    FROM YAR_VDB_CHUNKS
    WHERE workspace = $1
    GROUP BY full_doc_id
) AS vdb
    ON vdb.full_doc_id = orphan.full_doc_id
ORDER BY orphan.full_doc_id
"""

MISMATCHED_DOCS_SQL = """
SELECT
    status.id AS doc_id,
    status.chunks_count AS stored_chunks_count,
    COALESCE(doc_chunks.doc_chunk_rows, 0)::int AS doc_chunk_rows,
    COALESCE(vdb.vdb_chunk_rows, 0)::int AS vdb_chunk_rows
FROM YAR_DOC_STATUS AS status
LEFT JOIN (
    SELECT full_doc_id, COUNT(*)::int AS doc_chunk_rows
    FROM YAR_DOC_CHUNKS
    WHERE workspace = $1
    GROUP BY full_doc_id
) AS doc_chunks
    ON doc_chunks.full_doc_id = status.id
LEFT JOIN (
    SELECT full_doc_id, COUNT(*)::int AS vdb_chunk_rows
    FROM YAR_VDB_CHUNKS
    WHERE workspace = $1
    GROUP BY full_doc_id
) AS vdb
    ON vdb.full_doc_id = status.id
WHERE status.workspace = $1
  AND status.chunks_count IS DISTINCT FROM COALESCE(doc_chunks.doc_chunk_rows, 0)
ORDER BY status.id
"""

DELETE_DOC_CHUNKS_SQL = """
DELETE FROM YAR_DOC_CHUNKS
WHERE workspace = $1
  AND full_doc_id = ANY($2::varchar[])
"""

DELETE_VDB_CHUNKS_SQL = """
DELETE FROM YAR_VDB_CHUNKS
WHERE workspace = $1
  AND full_doc_id = ANY($2::varchar[])
"""

RESET_DOC_STATUS_SQL = """
UPDATE YAR_DOC_STATUS
SET status = $2,
    chunks_count = 0,
    chunks_list = '[]'::jsonb,
    error_msg = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE workspace = $1
  AND id = ANY($3::varchar[])
"""


@dataclass(frozen=True, slots=True)
class OrphanDoc:
    doc_id: str
    doc_chunk_rows: int
    vdb_chunk_rows: int


@dataclass(frozen=True, slots=True)
class MismatchedDoc:
    doc_id: str
    stored_chunks_count: int | None
    doc_chunk_rows: int
    vdb_chunk_rows: int


@dataclass(slots=True)
class RepairSummary:
    orphan_docs: int = 0
    orphan_doc_chunk_rows: int = 0
    orphan_vdb_chunk_rows: int = 0
    mismatched_docs: int = 0
    mismatched_doc_chunk_rows: int = 0
    mismatched_vdb_chunk_rows: int = 0
    status_rows_reset: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Repair orphaned and mismatched chunk bookkeeping for one workspace.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Discover required repairs and print the plan without changing PostgreSQL.',
    )
    return parser.parse_args()


def _parse_execute_count(result: str | None) -> int:
    if not result:
        return 0

    try:
        return int(result.rsplit(' ', 1)[-1])
    except (ValueError, IndexError):
        return 0


async def _fetch_orphan_docs(conn, workspace: str) -> list[OrphanDoc]:
    rows = await conn.fetch(ORPHAN_DOCS_SQL, workspace)
    return [
        OrphanDoc(
            doc_id=str(row['full_doc_id']),
            doc_chunk_rows=int(row['doc_chunk_rows']),
            vdb_chunk_rows=int(row['vdb_chunk_rows']),
        )
        for row in rows
    ]


async def _fetch_mismatched_docs(conn, workspace: str) -> list[MismatchedDoc]:
    rows = await conn.fetch(MISMATCHED_DOCS_SQL, workspace)
    return [
        MismatchedDoc(
            doc_id=str(row['doc_id']),
            stored_chunks_count=row['stored_chunks_count'],
            doc_chunk_rows=int(row['doc_chunk_rows']),
            vdb_chunk_rows=int(row['vdb_chunk_rows']),
        )
        for row in rows
    ]


def _planned_summary(orphan_docs: list[OrphanDoc], mismatched_docs: list[MismatchedDoc]) -> RepairSummary:
    return RepairSummary(
        orphan_docs=len(orphan_docs),
        orphan_doc_chunk_rows=sum(doc.doc_chunk_rows for doc in orphan_docs),
        orphan_vdb_chunk_rows=sum(doc.vdb_chunk_rows for doc in orphan_docs),
        mismatched_docs=len(mismatched_docs),
        mismatched_doc_chunk_rows=sum(doc.doc_chunk_rows for doc in mismatched_docs),
        mismatched_vdb_chunk_rows=sum(doc.vdb_chunk_rows for doc in mismatched_docs),
        status_rows_reset=len(mismatched_docs),
    )


def _print_plan(workspace: str, dry_run: bool, orphan_docs: list[OrphanDoc], mismatched_docs: list[MismatchedDoc]) -> None:
    mode = 'dry-run' if dry_run else 'repair'
    print(f'Workspace: {workspace}')
    print(f'Mode: {mode}')

    if orphan_docs:
        print('\nOrphan doc IDs to clean from chunk tables:')
        for doc in orphan_docs:
            print(
                f'  - {doc.doc_id}: '
                f'doc_chunks={doc.doc_chunk_rows}, vdb_chunks={doc.vdb_chunk_rows}'
            )
    else:
        print('\nOrphan doc IDs to clean from chunk tables: none')

    if mismatched_docs:
        print('\nDocs with mismatched chunks_count to reset:')
        for doc in mismatched_docs:
            print(
                f'  - {doc.doc_id}: '
                f'stored_chunks_count={doc.stored_chunks_count}, '
                f'doc_chunks={doc.doc_chunk_rows}, '
                f'vdb_chunks={doc.vdb_chunk_rows}'
            )
    else:
        print('\nDocs with mismatched chunks_count to reset: none')


def _print_summary(summary: RepairSummary, dry_run: bool) -> None:
    if dry_run:
        orphan_doc_label = 'orphan docs found'
        orphan_doc_chunk_label = 'orphan YAR_DOC_CHUNKS rows planned for deletion'
        orphan_vdb_chunk_label = 'orphan YAR_VDB_CHUNKS rows planned for deletion'
        mismatched_doc_label = 'mismatched docs to reset'
        mismatched_doc_chunk_label = 'mismatched YAR_DOC_CHUNKS rows planned for deletion'
        mismatched_vdb_chunk_label = 'mismatched YAR_VDB_CHUNKS rows planned for deletion'
        status_reset_label = 'YAR_DOC_STATUS rows planned for reset'
    else:
        orphan_doc_label = 'orphan docs deleted'
        orphan_doc_chunk_label = 'orphan YAR_DOC_CHUNKS rows deleted'
        orphan_vdb_chunk_label = 'orphan YAR_VDB_CHUNKS rows deleted'
        mismatched_doc_label = 'mismatched docs reset'
        mismatched_doc_chunk_label = 'mismatched YAR_DOC_CHUNKS rows deleted'
        mismatched_vdb_chunk_label = 'mismatched YAR_VDB_CHUNKS rows deleted'
        status_reset_label = 'YAR_DOC_STATUS rows reset'

    print('\nSummary:')
    print(f'  {orphan_doc_label}: {summary.orphan_docs}')
    print(f'  {orphan_doc_chunk_label}: {summary.orphan_doc_chunk_rows}')
    print(f'  {orphan_vdb_chunk_label}: {summary.orphan_vdb_chunk_rows}')
    print(f'  {mismatched_doc_label}: {summary.mismatched_docs}')
    print(f'  {mismatched_doc_chunk_label}: {summary.mismatched_doc_chunk_rows}')
    print(f'  {mismatched_vdb_chunk_label}: {summary.mismatched_vdb_chunk_rows}')
    print(f'  {status_reset_label}: {summary.status_rows_reset}')


async def _apply_repairs(conn, workspace: str, orphan_docs: list[OrphanDoc], mismatched_docs: list[MismatchedDoc]) -> RepairSummary:
    summary = RepairSummary(
        orphan_docs=len(orphan_docs),
        mismatched_docs=len(mismatched_docs),
    )

    orphan_doc_ids = [doc.doc_id for doc in orphan_docs]
    if orphan_doc_ids:
        summary.orphan_doc_chunk_rows = _parse_execute_count(
            await conn.execute(DELETE_DOC_CHUNKS_SQL, workspace, orphan_doc_ids)
        )
        summary.orphan_vdb_chunk_rows = _parse_execute_count(
            await conn.execute(DELETE_VDB_CHUNKS_SQL, workspace, orphan_doc_ids)
        )

    mismatched_doc_ids = [doc.doc_id for doc in mismatched_docs]
    if mismatched_doc_ids:
        summary.mismatched_doc_chunk_rows = _parse_execute_count(
            await conn.execute(DELETE_DOC_CHUNKS_SQL, workspace, mismatched_doc_ids)
        )
        summary.mismatched_vdb_chunk_rows = _parse_execute_count(
            await conn.execute(DELETE_VDB_CHUNKS_SQL, workspace, mismatched_doc_ids)
        )
        summary.status_rows_reset = _parse_execute_count(
            await conn.execute(
                RESET_DOC_STATUS_SQL,
                workspace,
                DocStatus.PENDING.value,
                mismatched_doc_ids,
            )
        )

    return summary


async def _run(args: argparse.Namespace) -> int:
    db: PostgreSQLDB | None = None
    try:
        db = await ClientManager.get_client()
        workspace = db.workspace

        if db.pool is None:
            raise RuntimeError('PostgreSQL pool was not initialized')

        async with db.pool.acquire() as conn:
            async with conn.transaction():
                orphan_docs = await _fetch_orphan_docs(conn, workspace)
                mismatched_docs = await _fetch_mismatched_docs(conn, workspace)

                _print_plan(workspace, args.dry_run, orphan_docs, mismatched_docs)

                if not orphan_docs and not mismatched_docs:
                    print('\nNo repairs needed.')
                    return 0

                if args.dry_run:
                    print('\nDry run only. No changes applied.')
                    _print_summary(_planned_summary(orphan_docs, mismatched_docs), dry_run=True)
                    return 0

                summary = await _apply_repairs(conn, workspace, orphan_docs, mismatched_docs)

        print('\nRepairs applied.')
        _print_summary(summary, dry_run=False)
        return 0
    finally:
        if db is not None:
            await ClientManager.release_client(db)


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    raise SystemExit(main())
