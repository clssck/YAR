#!/usr/bin/env python3
"""
Backfill S3 keys for documents uploaded via /documents/upload.

This script:
1. Finds documents without s3_key that have file_path pointing to local files
2. Uploads them to S3 archive
3. Updates s3_key in doc_status and doc_chunks tables
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg

# Configuration from environment
POSTGRES_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.environ.get('POSTGRES_PORT', '5433'))
POSTGRES_USER = os.environ.get('POSTGRES_USER', 'lightrag')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'lightrag_pass')
POSTGRES_DATABASE = os.environ.get('POSTGRES_DATABASE', 'lightrag')

S3_ENDPOINT_URL = os.environ.get('S3_ENDPOINT_URL', 'http://localhost:9000')
S3_ACCESS_KEY_ID = os.environ.get('S3_ACCESS_KEY_ID', 'rustfsadmin')
S3_SECRET_ACCESS_KEY = os.environ.get('S3_SECRET_ACCESS_KEY', 'rustfsadmin')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'lightrag')
S3_REGION = os.environ.get('S3_REGION', 'us-east-1')

INPUT_DIR = Path(os.environ.get('INPUT_DIR', '/Users/clssck/Projects/LightRAG/inputs'))


async def get_s3_client():
    """Create S3 client."""
    import aioboto3

    session = aioboto3.Session()
    return session.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )


async def upload_to_s3(s3_client, local_path: Path, s3_key: str) -> bool:
    """Upload file to S3."""
    try:
        async with s3_client as s3:
            with open(local_path, 'rb') as f:
                await s3.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=s3_key,
                    Body=f.read(),
                )
        print(f'  ✓ Uploaded to S3: {s3_key}')
        return True
    except Exception as e:
        print(f'  ✗ Failed to upload {local_path}: {e}')
        return False


async def main():
    print('=' * 60)
    print('Backfilling S3 keys for existing documents')
    print('=' * 60)

    # Connect to database
    conn = await asyncpg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DATABASE,
    )

    # Find documents without s3_key
    docs = await conn.fetch("""
        SELECT id, file_path, status, workspace
        FROM lightrag_doc_status
        WHERE (s3_key IS NULL OR s3_key = '')
          AND file_path IS NOT NULL
          AND file_path != ''
          AND LOWER(status) = 'processed'
    """)

    print(f'\nFound {len(docs)} processed documents without S3 keys')

    if not docs:
        print('Nothing to do!')
        await conn.close()
        return

    # Process each document
    updated = 0
    for doc in docs:
        doc_id = doc['id']
        file_path = doc['file_path']
        workspace = doc['workspace']

        print(f'\n[{doc_id[:12]}...] {file_path} (workspace: {workspace})')

        # Find local file
        local_path = INPUT_DIR / file_path
        if not local_path.exists():
            # Try __enqueued__ subfolder
            local_path = INPUT_DIR / '__enqueued__' / file_path

        if not local_path.exists():
            print('  ✗ Local file not found, skipping')
            continue

        # Generate S3 key
        s3_key = f'archive/default/{doc_id}/{file_path}'

        # Upload to S3
        s3_client = await get_s3_client()
        if not await upload_to_s3(s3_client, local_path, s3_key):
            continue

        # Update doc_status
        await conn.execute("""
            UPDATE lightrag_doc_status
            SET s3_key = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2 AND workspace = $3
        """, s3_key, doc_id, workspace)

        # Update doc_chunks
        result = await conn.execute("""
            UPDATE lightrag_doc_chunks
            SET s3_key = $1, update_time = CURRENT_TIMESTAMP
            WHERE full_doc_id = $2 AND workspace = $3
        """, s3_key, doc_id, workspace)
        # Extract count from 'UPDATE N' result
        chunk_count = int(result.split()[-1]) if result else 0

        print(f'  ✓ Updated {chunk_count or 0} chunks with s3_key')
        updated += 1

    await conn.close()

    print('\n' + '=' * 60)
    print(f'Backfill complete: {updated}/{len(docs)} documents updated')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(main())
