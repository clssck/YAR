from __future__ import annotations

import json
from unittest.mock import patch

from yar.evaluation.phoenix_query_generation import (
    GenerationConfig,
    SourceDocument,
    _comparison_axes_supported_by_both,
    _comparison_pair_has_shared_topic,
    _comparison_uses_supported_shared_labels,
    _gen_comparison,
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


def test_mechanism_bait_generation_rejects_answerable_admin_targets() -> None:
    config = GenerationConfig(judge_model='dummy')
    doc = SourceDocument(
        doc_id='doc-1',
        file_path='default/doc-1/file.canonical.md',
        title='2024-02-21 Fitusiran PMG Green Light Presentation',
        content='',
    )

    with patch(
        'yar.evaluation.phoenix_query_generation._llm_call',
        return_value=json.dumps(
            {
                'query': 'what is the detection principle behind the PPQ assay',
                'target_entity': 'PPQ assay',
                'why_should_refuse': 'The passage only mentions PPQ administratively.',
            }
        ),
    ):
        example = _gen_single_passage_intent(
            config,
            intent='mechanism_bait',
            prompt_template='DOCUMENT TITLE: {document_title}\nPASSAGE: {passage}',
            doc=doc,
            passage='PPQ campaign and release testing are described.',
            extra_meta_keys=('target_entity', 'why_should_refuse'),
        )

    assert example is None


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
    assert not _comparison_axes_supported_by_both(
        ['meeting governance'],
        'The first passage discusses meeting cadence only.',
        'The second passage discusses governance structure only.',
    )


def test_comparison_pair_prefilter_requires_concrete_shared_topic() -> None:
    assert _comparison_pair_has_shared_topic(
        'Comparator sourcing session',
        'The iCMC team planned comparator sourcing, blinding, and PDP coordination.',
        'Comparator stability session',
        'The comparator plan covered stability data, blinding strategy, and sourcing risk.',
    )
    assert not _comparison_pair_has_shared_topic(
        'Fitusiran PMG Green Light Presentation',
        'Fitusiran site readiness covered GMP systems, CAPA closure, and quality agreements.',
        'Blinded comparator session',
        'The iCMC team planned comparator sourcing, stakeholder timelines, and clinical supply.',
    )


def test_comparison_generation_rejects_missing_source_anchor() -> None:
    config = GenerationConfig(judge_model='dummy')
    doc_a = SourceDocument('doc-a', 'default/doc-a/a.canonical.md', '16-LLsession-09- outcome Jan 18 2017', '')
    doc_b = SourceDocument(
        'doc-b',
        'default/doc-b/b.canonical.md',
        '18-LLsession-02-Devpt and supply of blinded comparator- outcome Oct 18 VF',
        '',
    )

    with patch(
        'yar.evaluation.phoenix_query_generation._llm_call',
        return_value=json.dumps(
            {
                'query': 'How do the Sanofi team recommendations compare with blinded comparator management?',
                'expected_axes': ['risk alignment'],
            }
        ),
    ):
        example = _gen_comparison(
            config,
            doc_a=doc_a,
            doc_b=doc_b,
            passage_a='The risk alignment approach used SMEs and partner trust.',
            passage_b='The blinded comparator management approach used risk alignment and sourcing governance.',
        )

    assert example is None
