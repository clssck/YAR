from __future__ import annotations

import json
from unittest.mock import patch

from yar.evaluation.phoenix_query_generation import (
    GenerationConfig,
    SourceDocument,
    _comparison_axes_supported_by_both,
    _comparison_uses_supported_shared_labels,
    _gen_single_passage_intent,
    _query_mentions_source_anchor,
)


def test_query_mentions_source_anchor_requires_distinctive_title_term() -> None:
    title = '16-LLsession 12-TechTransfer implementation-final'

    assert not _query_mentions_source_anchor(
        'Which internal document number is referenced as the TT guideline in the Best Practice section?',
        title,
    )
    assert _query_mentions_source_anchor(
        'Which internal document number is referenced in 16-LLsession-12 E2E Tech Transfer?',
        title,
    )
    assert _query_mentions_source_anchor(
        'Which studies were delayed or impacted by the project freeze?',
        '190917 Lessons Learned - Project freeze',
    )


def test_answer_expected_generation_rejects_context_dependent_query() -> None:
    config = GenerationConfig(judge_model='dummy')
    doc = SourceDocument(
        doc_id='doc-1',
        file_path='default/doc-1/file.canonical.md',
        title='16-LLsession 12-TechTransfer implementation-final',
        content='',
    )

    with patch(
        'yar.evaluation.phoenix_query_generation._llm_call',
        return_value=json.dumps(
            {
                'query': 'List the facilitators mentioned in the passage.',
                'expected_items': ['Eric Lacoste', 'Claire Poyer'],
            }
        ),
    ):
        example = _gen_single_passage_intent(
            config,
            intent='enumeration',
            prompt_template='DOCUMENT TITLE: {document_title}\nPASSAGE: {passage}',
            doc=doc,
            passage='Facilitators: Eric Lacoste, Claire Poyer',
            extra_meta_keys=('expected_items',),
        )

    assert example is None


def test_answer_expected_generation_accepts_source_anchored_query() -> None:
    config = GenerationConfig(judge_model='dummy')
    doc = SourceDocument(
        doc_id='doc-1',
        file_path='default/doc-1/file.canonical.md',
        title='16-LLsession 12-TechTransfer implementation-final',
        content='',
    )

    with patch(
        'yar.evaluation.phoenix_query_generation._llm_call',
        return_value=json.dumps(
            {
                'query': 'List the facilitators in 16-LLsession-12 E2E Tech Transfer implementation.',
                'expected_items': ['Eric Lacoste', 'Claire Poyer'],
            }
        ),
    ):
        example = _gen_single_passage_intent(
            config,
            intent='enumeration',
            prompt_template='DOCUMENT TITLE: {document_title}\nPASSAGE: {passage}',
            doc=doc,
            passage='Facilitators: Eric Lacoste, Claire Poyer',
            extra_meta_keys=('expected_items',),
        )

    assert example is not None
    assert example['query'] == 'List the facilitators in 16-LLsession-12 E2E Tech Transfer implementation.'
    assert example['expected_items'] == ['Eric Lacoste', 'Claire Poyer']


def test_comparison_validation_rejects_one_sided_axes_and_labels() -> None:
    pku_passage = 'AAV-PCL platform FDA feedback required impurity assays and HPV L1 protein testing.'
    freeze_passage = 'Efpeglenatide project freeze delayed clinical development timelines and study starts.'

    assert not _comparison_axes_supported_by_both(
        ['impurity/protein testing requirements', 'clinical development timeline impact'],
        pku_passage,
        freeze_passage,
    )
    assert not _comparison_uses_supported_shared_labels(
        'How did the respective peptide-based drug products differ?',
        pku_passage,
        freeze_passage,
    )
    assert _comparison_axes_supported_by_both(
        ['timeline risk'],
        'The timeline risk affected submission planning.',
        'The project timeline risk affected clinical planning.',
    )
