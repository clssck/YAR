from __future__ import annotations

import json
from pathlib import Path

from yar.evaluation.generate_ragas_questions import (
    REPO_ROOT,
    SourceChunk,
    _anchor_question_to_sources,
    build_eval_cases_from_testset_rows,
    collect_source_documents,
    infer_source_documents,
    select_source_documents,
    write_generated_dataset,
)


def test_collect_source_documents_from_dataset_and_cli_dedupes(tmp_path: Path):
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(
        json.dumps(
            {
                'test_cases': [
                    {'source_documents': ['inputs/doc-a.pdf', 'inputs/doc-b.pdf']},
                    {'source_documents': 'inputs/doc-a.pdf'},
                ]
            }
        ),
        encoding='utf-8',
    )

    paths = collect_source_documents(
        source_dataset=dataset,
        source_documents=['inputs/doc-b.pdf', 'inputs/doc-c.pdf'],
    )

    repo_root = REPO_ROOT.resolve(strict=False)
    assert [path.resolve(strict=False).relative_to(repo_root).as_posix() for path in paths] == [
        'inputs/doc-a.pdf',
        'inputs/doc-b.pdf',
        'inputs/doc-c.pdf',
    ]


def test_select_source_documents_is_seeded_and_bounded():
    paths = [Path(f'doc-{index}.pdf') for index in range(5)]

    first = select_source_documents(paths, max_source_docs=2, seed=7)
    second = select_source_documents(paths, max_source_docs=2, seed=7)

    assert first == second
    assert len(first) == 2
    assert set(first) <= set(paths)


def test_infer_source_documents_matches_reference_contexts_to_chunks():
    chunks = [
        SourceChunk(
            content='Alpha source content about manufacturing site and batch yield.',
            source_document='a.pdf',
            chunk_order_index=0,
        ),
        SourceChunk(
            content='Beta source content about SARP pre-IND strategy.', source_document='b.pdf', chunk_order_index=0
        ),
    ]

    assert infer_source_documents(['SARP pre-IND strategy'], chunks) == ['b.pdf']


def test_anchor_question_to_sources_adds_document_title_when_missing():
    anchored = _anchor_question_to_sources(
        'Who is Danielle Combessis in the context of the clinical study?',
        [
            '/tmp/yar_ragas_ingested_sources/'
            'doc_94a146ae87c53b9b59fdd61b9239bf4a_'
            '18-lessons-learned-session-02-development-supply-outcome.md'
        ],
    )

    assert anchored.startswith('In the 18-lessons-learned-session-02-development-supply-outcome document, ')
    assert 'Who is Danielle Combessis' not in anchored
    assert 'who is Danielle Combessis' in anchored


def test_anchor_question_to_sources_preserves_existing_anchor():
    question = 'In the adapter strategy workgroup document, what is the definition of transfer adapters?'

    assert (
        _anchor_question_to_sources(
            question,
            ['doc_9ece4c702bc5e6e24cf45f3654730866_adapter_strategy_workgroup_final.md'],
        )
        == question
    )


def test_build_eval_cases_from_testset_rows_maps_generated_schema():
    chunks = [
        SourceChunk(
            content='Best Practice recommends an ad hoc meeting with the iCMC team and Subject Matter Expert contributors.',
            source_document='technology.pdf',
            chunk_order_index=3,
        )
    ]
    rows = [
        {
            'user_input': 'What is the first recommended step?',
            'reference': 'Organize an ad hoc meeting with the iCMC team.',
            'reference_contexts': ['ad hoc meeting with the iCMC team'],
            'synthesizer_name': 'single_hop',
            'persona_name': 'auditor',
            'query_style': 'formal',
            'query_length': 'short',
        }
    ]

    cases = build_eval_cases_from_testset_rows(
        rows,
        source_chunks=chunks,
        project='generated-project',
        retrieval_mode='naive',
    )

    assert cases == [
        {
            'id': 'ragas-generated-001',
            'question': 'In the technology document, what is the first recommended step?',
            'ground_truth': 'Organize an ad hoc meeting with the iCMC team.',
            'context_reference': 'Organize an ad hoc meeting with the iCMC team.',
            'retrieval_query': 'In the technology document, what is the first recommended step?',
            'retrieval_mode': 'naive',
            'source_documents': ['technology.pdf'],
            'project': 'generated-project',
            'generated_by': 'ragas.testset.TestsetGenerator',
            'synthesizer_name': 'single_hop',
            'original_question': 'What is the first recommended step?',
            'reference_contexts': ['ad hoc meeting with the iCMC team'],
            'persona_name': 'auditor',
            'query_style': 'formal',
            'query_length': 'short',
        }
    ]


def test_write_generated_dataset_marks_supplemental_track(tmp_path: Path):
    output = tmp_path / 'generated.json'

    write_generated_dataset(
        output_path=output,
        test_cases=[{'question': 'Q?', 'ground_truth': 'A'}],
        source_documents=[Path('inputs/source.pdf')],
        source_chunks=[SourceChunk(content='chunk', source_document='inputs/source.pdf', chunk_order_index=0)],
        testset_size=1,
        llm_model='judge',
        embedding_model='embedder',
    )

    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['metadata']['generated'] is True
    assert 'do not mix silently' in payload['metadata']['warning']
    assert payload['test_cases'][0]['question'] == 'Q?'
