from __future__ import annotations

import pytest

from scripts.graph_evidence_coverage import collect_graph_evidence_coverage
from yar.constants import GRAPH_FIELD_SEP
from yar.utils import compute_mdhash_id, make_relation_chunk_key


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_evidence_coverage_reports_json_shape_from_fake_storages():
    supported_src = 'CSTD strategy'
    supported_tgt = 'Overdose risk'
    unsupported_src = 'Generic strategy'
    unsupported_tgt = 'Outcome'
    chunk_id = 'chunk-evidence'
    supported_key = make_relation_chunk_key(supported_src, supported_tgt)
    evidence_span = 'CSTD strategy mitigates overdose risk in the source table.'

    class FakeGraphStorage:
        async def get_all_edges(self):
            return [
                {
                    'source': supported_src,
                    'target': supported_tgt,
                    'keywords': 'mitigates',
                    'source_id': chunk_id,
                    'weight': 3.0,
                },
                {
                    'source': unsupported_src,
                    'target': unsupported_tgt,
                    'keywords': 'related_to',
                    'source_id': 'chunk-unsupported',
                    'weight': 5.0,
                },
            ]

    class FakeRelationChunksStorage:
        def __init__(self):
            self.batch_calls = []

        async def get_by_ids(self, keys):
            self.batch_calls.append(keys)
            records = {
                supported_key: {
                    'chunk_ids': [chunk_id],
                    'count': 1,
                    'evidence_by_chunk': {chunk_id: [evidence_span]},
                }
            }
            return [records.get(key) for key in keys]

    class FakeRelationshipsVDB:
        def __init__(self):
            self.batch_calls = []
            self.records = {
                compute_mdhash_id(supported_src + supported_tgt, prefix='rel-'): {
                    'src_id': supported_src,
                    'tgt_id': supported_tgt,
                    'source_id': chunk_id,
                    'content': 'mitigates',
                },
                compute_mdhash_id(unsupported_src + unsupported_tgt, prefix='rel-'): {
                    'src_id': unsupported_src,
                    'tgt_id': unsupported_tgt,
                    'source_id': GRAPH_FIELD_SEP.join(['chunk-unsupported']),
                    'content': 'related_to',
                },
            }

        async def get_by_ids(self, ids):
            self.batch_calls.append(ids)
            return [self.records.get(relation_id) for relation_id in ids]

    relation_chunks = FakeRelationChunksStorage()
    relationships_vdb = FakeRelationshipsVDB()

    summary = await collect_graph_evidence_coverage(
        FakeGraphStorage(),
        relation_chunks,
        relationships_vdb,
        limit=1,
        min_weight=2.0,
    )

    assert summary == {
        'relations_total': 2,
        'relations_with_evidence': 1,
        'evidence_coverage_pct': 50.0,
        'high_weight_unsupported': [{'src': unsupported_src, 'tgt': unsupported_tgt, 'weight': 5.0}],
        'predicate_histogram': {'mitigates': 1, 'related_to': 1},
    }
    assert relation_chunks.batch_calls == [[supported_key, make_relation_chunk_key(unsupported_src, unsupported_tgt)]]
    assert relationships_vdb.batch_calls
