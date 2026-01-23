"""Integration tests verifying Unicode normalization in the full extraction pipeline.

These tests verify that entity names stored in the database are properly normalized,
not just that the normalization function works in isolation.

CRITICAL: These tests would FAIL if normalization stops working, even if
semantic matching catches duplicates. This is the "canary in the coal mine"
for Unicode security hardening.

Run with:
    pytest tests/test_unicode_integration.py -v --run-integration --forked

Or run individually (avoids asyncpg connection pool event loop issues):
    pytest "tests/test_unicode_integration.py::TestUnicodeNormalizationIntegration::test_stored_entity_names_have_no_zero_width_chars" -v --run-integration
"""

import contextlib
import os
import uuid

import pytest

os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5433')
os.environ.setdefault('POSTGRES_USER', 'yar')
os.environ.setdefault('POSTGRES_PASSWORD', 'yar_pass')
os.environ.setdefault('POSTGRES_DATABASE', 'yar')

from yar.utils import UNICODE_SECURITY_STRIP


@pytest.fixture
async def setup_rag(tmp_path):
    """Create a YAR instance for testing."""
    import os

    from yar import YAR
    from yar.kg.postgres_impl import ClientManager
    from yar.llm.openai import gpt_4o_mini_complete, openai_embed
    from yar.utils import EmbeddingFunc

    # Reset connection pool state before each test
    if ClientManager._db_instance is not None:
        with contextlib.suppress(Exception):
            if ClientManager._db_instance.pool is not None:
                await ClientManager._db_instance.pool.close()
        ClientManager._db_instance = None
        ClientManager._ref_count = 0

    embedding_dim = int(os.getenv('EMBEDDING_DIM', '1536'))
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=openai_embed,
    )
    embedding_func.send_dimensions = os.getenv('EMBEDDING_SEND_DIM', 'false').lower() == 'true'

    workspace = f'test_unicode_int_{uuid.uuid4().hex[:8]}'
    rag = YAR(
        working_dir=str(tmp_path / workspace),
        workspace=workspace,
        embedding_func=embedding_func,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage='PGGraphStorage',
        kv_storage='PGKVStorage',
        vector_storage='PGVectorStorage',
        doc_status_storage='PGDocStatusStorage',
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()

    db = rag.entities_vdb._db_required()
    for table in [
        'YAR_DOC_FULL',
        'YAR_DOC_CHUNKS',
        'YAR_DOC_STATUS',
        'YAR_VDB_ENTITY',
        'YAR_VDB_RELATION',
        'YAR_VDB_CHUNKS',
        'YAR_ENTITY_ALIASES',
    ]:
        with contextlib.suppress(Exception):
            await db.execute(f'DELETE FROM {table} WHERE workspace = $1', data={'workspace': workspace})

    yield rag

    # Proper cleanup
    await rag.finalize_storages()
    if ClientManager._db_instance is not None:
        with contextlib.suppress(Exception):
            if ClientManager._db_instance.pool is not None:
                await ClientManager._db_instance.pool.close()
        ClientManager._db_instance = None
        ClientManager._ref_count = 0


@pytest.mark.integration
@pytest.mark.requires_db
class TestUnicodeNormalizationIntegration:
    """End-to-end tests for Unicode normalization in extraction pipeline.

    These tests verify that entity names stored in the database are properly
    normalized after going through the full pipeline:
    LLM extraction → JSON parsing → normalization → DB insertion → retrieval
    """

    async def test_stored_entity_names_have_no_zero_width_chars(self, setup_rag):
        """CANARY TEST: Verify entity names in DB contain no zero-width characters.

        This test would FAIL if normalize_unicode_for_entity_matching() stops being
        called or stops working, even if LLM semantic matching catches duplicates.

        This is the most critical test - if this fails, the security hardening is broken.
        """
        rag = setup_rag

        doc_with_zwc = (
            f'[{rag.workspace}] '
            'Micro\u200bsoft announced new Azure\u200d features. '
            'Apple\ufeff Inc released iOS updates. '
            'Google\u034f Cloud expanded its services.'
        )
        await rag.ainsert(doc_with_zwc)

        # Get all entity names from database
        db = rag.entities_vdb._db_required()
        entities = await db.query(
            'SELECT entity_name FROM YAR_VDB_ENTITY WHERE workspace = $1', params=[rag.workspace], multirows=True
        )

        entity_names = [e['entity_name'] for e in (entities or [])]

        # Ensure we actually got some entities (sanity check)
        assert len(entity_names) > 0, 'No entities extracted - test inconclusive'

        # CRITICAL ASSERTION: No entity name should contain zero-width characters
        for entity_name in entity_names:
            for zwc in UNICODE_SECURITY_STRIP:
                assert zwc not in entity_name, (
                    f"NORMALIZATION FAILURE: Entity '{entity_name!r}' "
                    f'contains zero-width character {zwc!r} (U+{ord(zwc):04X})'
                )

        print(f'✅ Verified {len(entity_names)} entities contain no zero-width characters')

    async def test_nfd_accents_normalized_to_nfc_in_db(self, setup_rag):
        """Verify that NFD accented entity names are stored as NFC in DB.

        NFD: "Café" = 'C' + 'a' + 'f' + 'e' + COMBINING_ACUTE_ACCENT (5 chars)
        NFC: "Café" = 'C' + 'a' + 'f' + 'é' (4 chars, precomposed)

        Without normalization, these would be stored as different byte sequences.
        """
        import unicodedata

        rag = setup_rag

        nfd_cafe = 'Cafe\u0301'
        doc = f'[{rag.workspace}] The {nfd_cafe} company opened new locations in Paris.'
        await rag.ainsert(doc)

        # Get entity names from database
        db = rag.entities_vdb._db_required()
        entities = await db.query(
            'SELECT entity_name FROM YAR_VDB_ENTITY WHERE workspace = $1', params=[rag.workspace], multirows=True
        )

        entity_names = [e['entity_name'] for e in (entities or [])]

        # Ensure we got some entities
        assert len(entity_names) > 0, 'No entities extracted - test inconclusive'

        # Verify all names are NFC normalized (no combining characters left decomposed)
        for entity_name in entity_names:
            nfc_form = unicodedata.normalize('NFC', entity_name)
            assert entity_name == nfc_form, (
                f"NORMALIZATION FAILURE: Entity '{entity_name!r}' is not NFC normalized. Expected: '{nfc_form!r}'"
            )

        print(f'✅ Verified {len(entity_names)} entities are NFC normalized')

    async def test_math_alphanumerics_normalized_in_db(self, setup_rag):
        """CANARY TEST: Verify mathematical alphanumerics are normalized to ASCII in DB.

        Mathematical alphanumeric symbols (U+1D400-U+1D7FF) look identical to regular
        letters but have different codepoints. Without normalization, "𝐀𝐩𝐩𝐥𝐞" and
        "Apple" would be stored as different entities.

        This test would FAIL if _normalize_math_alphanumerics() stops being called.
        """
        rag = setup_rag

        math_apple = '\U0001d400\U0001d429\U0001d429\U0001d425\U0001d41e'
        doc = f'[{rag.workspace}] The company {math_apple} announced new products today.'
        await rag.ainsert(doc)

        # Get entity names from database
        db = rag.entities_vdb._db_required()
        entities = await db.query(
            'SELECT entity_name FROM YAR_VDB_ENTITY WHERE workspace = $1', params=[rag.workspace], multirows=True
        )

        entity_names = [e['entity_name'] for e in (entities or [])]

        # Ensure we got some entities
        assert len(entity_names) > 0, 'No entities extracted - test inconclusive'

        # CRITICAL ASSERTION: No entity name should contain mathematical alphanumerics
        MATH_ALPHA_RANGE = (0x1D400, 0x1D7FF)
        for entity_name in entity_names:
            for char in entity_name:
                cp = ord(char)
                assert not (MATH_ALPHA_RANGE[0] <= cp <= MATH_ALPHA_RANGE[1]), (
                    f"NORMALIZATION FAILURE: Entity '{entity_name!r}' "
                    f'contains mathematical alphanumeric U+{cp:04X}. '
                    f'_normalize_math_alphanumerics() may not be called!'
                )

        print(f'✅ Verified {len(entity_names)} entities contain no math alphanumerics')


@pytest.mark.integration
@pytest.mark.requires_db
class TestAliasResolutionNormalization:
    """Tests for normalized alias resolution in resolver.py."""

    async def test_alias_cache_normalized_lookup(self, setup_rag):
        """Verify alias cache lookups work with Unicode variants.

        If normalize_unicode_for_entity_matching isn't called in resolver.py,
        cache lookups would fail for variant entity names.
        """
        from yar.entity_resolution.resolver import get_cached_alias, store_alias

        rag = setup_rag
        db = rag.entities_vdb._db_required()

        # Store an alias with clean name
        await store_alias(
            alias='microsoft',
            canonical='Microsoft Corporation',
            method='manual',
            confidence=1.0,
            db=db,
            workspace=rag.workspace,
        )

        # Lookup with zero-width character variant - should still find it
        # because resolver.py normalizes before lookup
        cached = await get_cached_alias('micro\u200bsoft', db, rag.workspace)

        assert cached is not None, (
            'RESOLVER FAILURE: Alias lookup failed for zero-width variant. '
            'normalize_unicode_for_entity_matching() may not be called in resolver.py'
        )
        assert cached[0] == 'Microsoft Corporation', (
            f"RESOLVER FAILURE: Expected 'Microsoft Corporation', got '{cached[0]}'"
        )

        print('✅ Verified alias cache normalizes lookups correctly')

    async def test_alias_cache_nfd_lookup(self, setup_rag):
        """Verify alias cache lookups work with NFD accent variants."""
        from yar.entity_resolution.resolver import get_cached_alias, store_alias

        rag = setup_rag
        db = rag.entities_vdb._db_required()

        # Store an alias with NFC name
        await store_alias(
            alias='café',  # NFC form
            canonical='Café Restaurant Group',
            method='manual',
            confidence=1.0,
            db=db,
            workspace=rag.workspace,
        )

        # Lookup with NFD variant - should still find it
        nfd_cafe = 'cafe\u0301'  # e + combining acute
        cached = await get_cached_alias(nfd_cafe, db, rag.workspace)

        assert cached is not None, (
            'RESOLVER FAILURE: Alias lookup failed for NFD variant. NFC normalization may not be applied in resolver.py'
        )

        print('✅ Verified alias cache normalizes NFD lookups correctly')


@pytest.mark.integration
@pytest.mark.requires_db
class TestPipelineIntegrationSmoke:
    """Smoke tests to verify the full pipeline is wired correctly."""

    async def test_normalization_function_is_called_during_extraction(self, setup_rag):
        """Verify normalize_unicode_for_entity_matching is actually called.

        This test uses a marker character that would only be removed by our
        normalization function, proving it's being called in the pipeline.
        """
        rag = setup_rag

        # Use Combining Grapheme Joiner (CGJ) - only stripped by our function
        # This character is specifically in UNICODE_SECURITY_STRIP Phase 2
        entity_with_cgj = 'Test\u034fCompany'  # CGJ between Test and Company

        doc = f'The {entity_with_cgj} released new products today.'
        await rag.ainsert(doc)

        # Get entity names from database
        db = rag.entities_vdb._db_required()
        entities = await db.query(
            'SELECT entity_name FROM YAR_VDB_ENTITY WHERE workspace = $1', params=[rag.workspace], multirows=True
        )

        entity_names = [e['entity_name'] for e in (entities or [])]

        # If any entity contains CGJ, normalization is NOT being called
        for entity_name in entity_names:
            assert '\u034f' not in entity_name, (
                f"PIPELINE FAILURE: Entity '{entity_name!r}' contains CGJ (U+034F). "
                f'This proves normalize_unicode_for_entity_matching() is NOT being called!'
            )

        print('✅ Verified normalization function is called during extraction')


if __name__ == '__main__':
    import asyncio

    async def run_quick_test():
        """Quick manual test without pytest."""
        from yar import YAR
        from yar.llm.openai import gpt_4o_mini_complete, openai_embed

        workspace = f'manual_test_{uuid.uuid4().hex[:8]}'
        rag = YAR(
            working_dir=f'/tmp/{workspace}',
            workspace=workspace,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
            graph_storage='PGGraphStorage',
            kv_storage='PGKVStorage',
            vector_storage='PGVectorStorage',
            doc_status_storage='PGDocStatusStorage',
            enable_llm_cache_for_entity_extract=False,
        )
        await rag.initialize_storages()

        # Test with zero-width characters
        doc = 'Micro\u200bsoft announced Azure\u200d updates. Apple\ufeff Inc released iOS.'
        print(f'Inserting document with zero-width chars: {doc[:60]!r}')
        await rag.ainsert(doc)

        # Check entities
        db = rag.entities_vdb._db_required()
        entities = await db.query(
            'SELECT entity_name FROM YAR_VDB_ENTITY WHERE workspace = $1', params=[workspace], multirows=True
        )

        print(f'\nExtracted entities ({len(entities or [])}):')
        for e in entities or []:
            name = e['entity_name']
            has_zwc = any(zwc in name for zwc in UNICODE_SECURITY_STRIP)
            status = '❌ CONTAINS ZWC' if has_zwc else '✅ CLEAN'
            print(f'  {status}: {name!r}')

        await rag.finalize_storages()

    asyncio.run(run_quick_test())
