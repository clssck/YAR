#!/usr/bin/env python3
from __future__ import annotations

"""Grade a QA answers CSV with LiteLLM.

Happy path:
    uv run python yar/evaluation/grade_qa_answers.py

This defaults to:
- the latest yar/evaluation/results/qa_answers_*.csv
- judge endpoint/key from repo .env
- judge model name discovered from YAR /health
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any

import litellm
from litellm import completion

from yar.evaluation.qa_eval_common import (
	DEFAULT_JUDGE_API_BASE,
	DEFAULT_JUDGE_API_KEY,
	DEFAULT_YAR_API_KEY,
	DEFAULT_YAR_API_URL,
	fetch_yar_health,
	latest_results_csv,
	parse_json_object,
	timestamped_results_path,
)

ALLOWED_PASS_FAIL = {'Pass', 'Fail'}
ALLOWED_OPTIONAL = {'Pass', 'Fail', 'Skipped'}

litellm.suppress_debug_info = True
litellm.set_verbose = False

KNOWN_PROVIDER_PREFIXES = (
	'openrouter/',
	'openai/',
	'custom_openai/',
	'anthropic/',
	'azure/',
	'bedrock/',
	'gemini/',
	'ollama/',
)

INLINE_CITATION_PATTERN = re.compile(r'(?<![\w`])\[\d+\](?!\w)')

SYSTEM_PROMPT = """You are grading a single QA evaluation row.

Inputs:
- QUESTION: the original user question
- EXPECTED RESPONSE: the source-grounded expected answer
- ACTUAL RESPONSE: the answer produced by the agent

Return JSON with exactly these fields:
{
  "generalQuality": "Pass" | "Fail",
  "seemsRelevant": "Pass" | "Fail",
  "seemsComplete": "Pass" | "Fail" | "Skipped",
  "basedOnKnowledgeSources": "Pass" | "Fail" | "Skipped",
  "reason": "short explanation"
}

Grading rules:
- Treat EXPECTED RESPONSE as the reference answer grounded in the knowledge source.
- generalQuality is Pass only if all non-skipped checks pass.
- If ACTUAL RESPONSE does not answer the question or is not relevant, set:
  - generalQuality = "Fail"
  - seemsRelevant = "Fail"
  - seemsComplete = "Skipped"
  - basedOnKnowledgeSources = "Skipped"
  - reason = "Question not answered. Further checks skipped because relevance failed."
- seemsComplete should fail when major expected information is missing.
- basedOnKnowledgeSources should fail when the actual response contradicts expected facts or adds unsupported claims.
- Keep the reason concise and concrete.
- Return JSON only.
"""


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Grade a QA answers CSV with LiteLLM.',
		epilog='Typical usage: uv run python yar/evaluation/grade_qa_answers.py',
	)
	parser.add_argument('--input-csv', type=Path, default=None, help='Answers CSV path (default: latest qa_answers_*.csv)')
	parser.add_argument('--output-csv', type=Path, default=None, help='Output graded CSV path (default: yar/evaluation/results/qa_grades_<timestamp>.csv)')
	parser.add_argument('--judge-api-base', type=str, default=DEFAULT_JUDGE_API_BASE, help='LiteLLM API base URL')
	parser.add_argument('--judge-api-key', type=str, default=DEFAULT_JUDGE_API_KEY, help='LiteLLM API key')
	parser.add_argument('--judge-model', type=str, default='', help='Judge model name (default: YAR llm_model from /health)')
	parser.add_argument('--yar-api-url', type=str, default=DEFAULT_YAR_API_URL, help='YAR API URL used to discover the default judge model')
	parser.add_argument('--yar-api-key', type=str, default=DEFAULT_YAR_API_KEY, help='Optional YAR API key for /health lookup')
	parser.add_argument('--temperature', type=float, default=0.0, help='Judge sampling temperature')
	parser.add_argument('--max-tokens', type=int, default=300, help='Judge max tokens')
	parser.add_argument(
		'--allow-inline-citations',
		action='store_true',
		help='Allow inline citation markers like [1] in actualResponse instead of failing before judge evaluation.',
	)
	return parser.parse_args()


def load_answers_csv(path: Path) -> list[dict[str, str]]:
	with path.open(newline='', encoding='utf-8') as handle:
		reader = csv.DictReader(handle)
		rows = list(reader)
	if reader.fieldnames != ['question', 'actualResponse', 'expectedResponse']:
		raise ValueError(f'Unexpected CSV headers in {path}: {reader.fieldnames}')
	if not rows:
		raise ValueError(f'No answer rows found in {path}')
	return rows


def build_user_prompt(row: dict[str, str]) -> str:
	return (
		f"QUESTION:\n{row['question']}\n\n"
		f"EXPECTED RESPONSE:\n{row['expectedResponse']}\n\n"
		f"ACTUAL RESPONSE:\n{row['actualResponse']}"
	)

def has_inline_citation_markers(text: str) -> bool:
	return bool(INLINE_CITATION_PATTERN.search(text))


def format_inline_citation_failure() -> dict[str, str]:
	return {
		'generalQuality': 'Fail',
		'seemsRelevant': 'Fail',
		'seemsComplete': 'Skipped',
		'basedOnKnowledgeSources': 'Skipped',
		'reason': 'Formatting violation: default answers must not include inline citation markers like [1].',
	}

def normalize_grade(payload: dict[str, Any]) -> dict[str, str]:
	general_quality = str(payload.get('generalQuality', 'Fail'))
	seems_relevant = str(payload.get('seemsRelevant', 'Fail'))
	seems_complete = str(payload.get('seemsComplete', 'Skipped'))
	based_on_sources = str(payload.get('basedOnKnowledgeSources', 'Skipped'))
	reason = str(payload.get('reason', '')).strip()

	if general_quality not in ALLOWED_PASS_FAIL:
		general_quality = 'Fail'
	if seems_relevant not in ALLOWED_PASS_FAIL:
		seems_relevant = 'Fail'
	if seems_complete not in ALLOWED_OPTIONAL:
		seems_complete = 'Skipped'
	if based_on_sources not in ALLOWED_OPTIONAL:
		based_on_sources = 'Skipped'
	if not reason:
		reason = 'No reason returned by judge.'

	if seems_relevant == 'Fail':
		general_quality = 'Fail'
		seems_complete = 'Skipped'
		based_on_sources = 'Skipped'
		if reason == 'No reason returned by judge.':
			reason = 'Question not answered. Further checks skipped because relevance failed.'
	elif general_quality == 'Pass' and ('Fail' in {seems_complete, based_on_sources}):
		general_quality = 'Fail'

	return {
		'generalQuality': general_quality,
		'seemsRelevant': seems_relevant,
		'seemsComplete': seems_complete,
		'basedOnKnowledgeSources': based_on_sources,
		'reason': reason,
	}


def normalize_litellm_model_name(judge_model: str, judge_api_base: str) -> str:
	if judge_model.startswith(KNOWN_PROVIDER_PREFIXES):
		return judge_model
	if judge_api_base:
		return f'openai/{judge_model}'
	return judge_model


def grade_row(
	row: dict[str, str],
	judge_model: str,
	judge_api_base: str,
	judge_api_key: str,
	temperature: float,
	max_tokens: int,
	allow_inline_citations: bool = False,
) -> dict[str, str]:
	if not allow_inline_citations and has_inline_citation_markers(row.get('actualResponse', '')):
		return format_inline_citation_failure()

	model_name = normalize_litellm_model_name(judge_model, judge_api_base)
	response = completion(
		model=model_name,
		messages=[
			{'role': 'system', 'content': SYSTEM_PROMPT},
			{'role': 'user', 'content': build_user_prompt(row)},
		],
		api_base=judge_api_base,
		api_key=judge_api_key,
		temperature=temperature,
		max_tokens=max_tokens,
	)
	raw_content = response.choices[0].message.content
	payload = parse_json_object(str(raw_content))
	return normalize_grade(payload)


def export_grades(output_csv: Path, rows: list[dict[str, str]]) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		'question',
		'actualResponse',
		'expectedResponse',
		'generalQuality',
		'seemsRelevant',
		'seemsComplete',
		'basedOnKnowledgeSources',
		'reason',
	]
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	input_csv = args.input_csv or latest_results_csv('qa_answers')
	output_csv = args.output_csv or timestamped_results_path('qa_grades')

	judge_model = args.judge_model
	if not judge_model:
		try:
			health = fetch_yar_health(args.yar_api_url, args.yar_api_key)
			judge_model = str(health['configuration']['llm_model'])
		except Exception:
			judge_model = os.getenv('LLM_MODEL', 'gpt-4.1-mini')
			print(f'Could not reach YAR /health; using LLM_MODEL={judge_model}')

	if not args.judge_api_key:
		raise ValueError('No judge API key. Set LLM_BINDING_API_KEY or LITELLM_MASTER_KEY in .env, or pass --judge-api-key.')

	rows = load_answers_csv(input_csv)
	graded_rows: list[dict[str, str]] = []
	for index, row in enumerate(rows, start=1):
		print(f"[{index}/{len(rows)}] Grading: {row['question'][:90]}")
		try:
			grades = grade_row(
				row=row,
				judge_model=judge_model,
				judge_api_base=args.judge_api_base,
				judge_api_key=args.judge_api_key,
				temperature=args.temperature,
				max_tokens=args.max_tokens,
				allow_inline_citations=args.allow_inline_citations,
			)
		except Exception as exc:
			grades = {
				'generalQuality': 'Fail',
				'seemsRelevant': 'Fail',
				'seemsComplete': 'Skipped',
				'basedOnKnowledgeSources': 'Skipped',
				'reason': f'Judge error: {exc}',
			}
		graded_rows.append({**row, **grades})

	export_grades(output_csv, graded_rows)
	passes = sum(1 for row in graded_rows if row['generalQuality'] == 'Pass')
	print(f'Wrote graded CSV: {output_csv}')
	print(f'General quality pass rate: {passes}/{len(graded_rows)}')


if __name__ == '__main__':
	main()
