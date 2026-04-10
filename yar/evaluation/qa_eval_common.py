from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / 'results'
DEFAULT_QA_CSV = REPO_ROOT / 'EvaluationTemplate_filled.csv'

# Load repository .env once so host-side scripts can run with minimal flags.
load_dotenv(REPO_ROOT / '.env', override=False)

DEFAULT_YAR_API_URL = os.getenv('YAR_API_URL', 'http://localhost:9621')
DEFAULT_YAR_API_KEY = os.getenv('YAR_API_KEY', '')
DEFAULT_JUDGE_API_BASE = os.getenv('EVAL_LLM_BINDING_HOST') or os.getenv('LLM_BINDING_HOST') or 'https://openrouter.ai/api/v1'
DEFAULT_JUDGE_API_KEY = os.getenv('EVAL_LLM_BINDING_API_KEY') or os.getenv('LLM_BINDING_API_KEY') or ''


@dataclass(frozen=True)
class QATestCase:
	question: str
	expected_response: str


def build_api_headers(api_key: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
	headers = dict(extra or {})
	if api_key:
		headers['X-API-Key'] = api_key
	return headers


def load_qa_csv(path: Path) -> list[QATestCase]:
	with path.open(newline='', encoding='utf-8') as handle:
		rows = list(csv.reader(handle))

	try:
		header_index = next(i for i, row in enumerate(rows) if row == ['question', 'expectedResponse'])
	except StopIteration as exc:
		raise ValueError(f'Could not find question/expectedResponse header in {path}') from exc

	test_cases: list[QATestCase] = []
	for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
		if not row:
			continue
		if len(row) < 2:
			raise ValueError(f'Row {row_number} in {path} does not have two columns')
		question = row[0].strip()
		expected = row[1].strip()
		if not question or question.startswith('#'):
			continue
		test_cases.append(QATestCase(question=question, expected_response=expected))

	if not test_cases:
		raise ValueError(f'No test cases found in {path}')

	return test_cases


def timestamped_results_path(prefix: str, suffix: str = '.csv') -> Path:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	return RESULTS_DIR / f'{prefix}_{timestamp}{suffix}'


def fetch_yar_health(api_url: str, api_key: str | None) -> dict[str, Any]:
	response = httpx.get(
		f'{api_url.rstrip("/")}/health',
		headers=build_api_headers(api_key),
		timeout=httpx.Timeout(20.0, connect=10.0),
	)
	response.raise_for_status()
	return response.json()


def latest_results_csv(prefix: str) -> Path:
	candidates = list(RESULTS_DIR.glob(f'{prefix}_*.csv'))
	if not candidates:
		raise FileNotFoundError(f'No CSV results found for prefix {prefix!r} in {RESULTS_DIR}')
	return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_json_object(raw_text: str) -> dict[str, Any]:
	start = raw_text.find('{')
	end = raw_text.rfind('}')
	if start == -1 or end == -1 or end <= start:
		raise ValueError(f'No JSON object found in response: {raw_text[:200]}')
	return json.loads(raw_text[start : end + 1])
