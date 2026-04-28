from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

litellm_stub: Any = cast(Any, types.ModuleType('litellm'))
litellm_stub.completion = lambda **kwargs: None
litellm_stub.suppress_debug_info = False
litellm_stub.set_verbose = False
sys.modules.setdefault('litellm', litellm_stub)

from yar.evaluation import grade_qa_answers


@pytest.mark.offline
@pytest.mark.parametrize(
	('text', 'expected'),
	[
		('Plain answer without citations.', False),
		('Answer with inline citation [1].', True),
		('Answer with multi-digit inline citation [12].', True),
		('Bracketed word [index] should not count.', False),
		('Adjacent token section[1] should not count.', False),
	],
)
def test_has_inline_citation_markers(text: str, expected: bool) -> None:
	assert grade_qa_answers.has_inline_citation_markers(text) is expected


@pytest.mark.offline
def test_parse_args_supports_allow_inline_citations(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr('sys.argv', ['grade_qa_answers.py', '--allow-inline-citations'])

	args = grade_qa_answers.parse_args()

	assert args.allow_inline_citations is True


@pytest.mark.offline
def test_grade_row_fails_inline_citations_without_calling_judge(monkeypatch: pytest.MonkeyPatch) -> None:
	judge_called = False

	def fake_completion(**kwargs: object) -> None:
		nonlocal judge_called
		judge_called = True
		raise AssertionError('judge should not be called when inline citations are disallowed')

	monkeypatch.setattr(grade_qa_answers, 'completion', fake_completion)

	result = grade_qa_answers.grade_row(
		row={
			'question': 'What storage backend does YAR use?',
			'expectedResponse': 'YAR uses PostgreSQL.',
			'actualResponse': 'YAR uses PostgreSQL [1].',
		},
		judge_model='gpt-4.1-mini',
		judge_api_base='https://judge.example',
		judge_api_key='test-key',
		temperature=0.0,
		max_tokens=300,
	)

	assert judge_called is False
	assert result == {
		'generalQuality': 'Fail',
		'seemsRelevant': 'Fail',
		'seemsComplete': 'Skipped',
		'basedOnKnowledgeSources': 'Skipped',
		'reason': 'Formatting violation: default answers must not include inline citation markers like [1].',
	}


@pytest.mark.offline
def test_grade_row_allows_inline_citations_when_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
	judge_calls: list[dict[str, object]] = []

	def fake_completion(**kwargs: object) -> SimpleNamespace:
		judge_calls.append(kwargs)
		return SimpleNamespace(
			choices=[
				SimpleNamespace(
					message=SimpleNamespace(
						content=(
							'{"generalQuality":"Pass","seemsRelevant":"Pass",'
							'"seemsComplete":"Pass","basedOnKnowledgeSources":"Pass",'
							'"reason":"Grounded and complete."}'
						),
					),
				),
			],
		)

	monkeypatch.setattr(grade_qa_answers, 'completion', fake_completion)

	result = grade_qa_answers.grade_row(
		row={
			'question': 'What storage backend does YAR use?',
			'expectedResponse': 'YAR uses PostgreSQL.',
			'actualResponse': 'YAR uses PostgreSQL [1].',
		},
		judge_model='gpt-4.1-mini',
		judge_api_base='https://judge.example',
		judge_api_key='test-key',
		temperature=0.0,
		max_tokens=300,
		allow_inline_citations=True,
	)

	assert len(judge_calls) == 1
	assert judge_calls[0]['model'] == 'openai/gpt-4.1-mini'
	assert result == {
		'generalQuality': 'Pass',
		'seemsRelevant': 'Pass',
		'seemsComplete': 'Pass',
		'basedOnKnowledgeSources': 'Pass',
		'reason': 'Grounded and complete.',
	}
