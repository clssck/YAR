#!/usr/bin/env python3
from __future__ import annotations

"""Export YAR answers for the QA CSV.

Happy path:
    uv run python yar/evaluation/export_qa_answers.py

This defaults to:
- EvaluationTemplate_filled.csv at repo root
- YAR at http://localhost:9621
- output CSV under yar/evaluation/results/
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import httpx

from yar.evaluation.qa_eval_common import (
	DEFAULT_QA_CSV,
	DEFAULT_YAR_API_KEY,
	DEFAULT_YAR_API_URL,
	build_api_headers,
	fetch_yar_health,
	load_qa_csv,
	timestamped_results_path,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Run the QA CSV through YAR and export answers to CSV.',
		epilog='Typical usage: uv run python yar/evaluation/export_qa_answers.py',
	)
	parser.add_argument('--input-csv', type=Path, default=DEFAULT_QA_CSV, help='QA CSV with question and expectedResponse columns')
	parser.add_argument('--output-csv', type=Path, default=None, help='Output CSV path (default: yar/evaluation/results/qa_answers_<timestamp>.csv)')
	parser.add_argument('--api-url', type=str, default=DEFAULT_YAR_API_URL, help='YAR API base URL')
	parser.add_argument('--api-key', type=str, default=DEFAULT_YAR_API_KEY, help='Optional YAR API key')
	parser.add_argument('--mode', type=str, default='mix', help='YAR query mode to use')
	parser.add_argument('--response-type', type=str, default='Multiple Paragraphs', help='Response type (default: Multiple Paragraphs)')
	parser.add_argument('--timeout', type=float, default=300.0, help='Per-question timeout in seconds')
	return parser.parse_args()


def query_yar(client: httpx.Client, api_url: str, api_key: str, question: str, mode: str, response_type: str = 'Multiple Paragraphs') -> str:
	response = client.post(
		f'{api_url.rstrip("/")}/query',
		json={
			'query': question,
			'mode': mode,
			'stream': False,
			'include_references': False,
			'response_type': response_type,
			'user_prompt': 'Answer concisely using only facts from the provided context. Do not add information beyond what the context contains.',
		},
		headers=build_api_headers(api_key, {'Content-Type': 'application/json'}),
	)
	response.raise_for_status()
	payload: dict[str, Any] = response.json()
	return str(payload.get('response', '')).strip()


def export_answers(output_csv: Path, rows: list[dict[str, str]]) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=['question', 'actualResponse', 'expectedResponse'])
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	test_cases = load_qa_csv(args.input_csv)
	output_csv = args.output_csv or timestamped_results_path('qa_answers')

	health = fetch_yar_health(args.api_url, args.api_key)
	print(f"Connected to YAR: {args.api_url.rstrip('/')} ({health['status']})")
	print(f"Model: {health['configuration']['llm_model']}")
	print(f'Questions: {len(test_cases)}')

	results: list[dict[str, str]] = []
	with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=30.0)) as client:
		for index, test_case in enumerate(test_cases, start=1):
			print(f'[{index}/{len(test_cases)}] {test_case.question[:90]}')
			try:
				actual_response = query_yar(client, args.api_url, args.api_key, test_case.question, args.mode, args.response_type)
			except Exception as exc:
				actual_response = f'ERROR: {exc}'
			results.append(
				{
					'question': test_case.question,
					'actualResponse': actual_response,
					'expectedResponse': test_case.expected_response,
				}
			)

	export_answers(output_csv, results)
	print(f'Wrote answers CSV: {output_csv}')


if __name__ == '__main__':
	main()
