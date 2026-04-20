#!/usr/bin/env python3
"""Grade the concierge QA pair JSON with a LiteLLM judge.

The concierge export (eval_docs/qa_pair_concierge.json) only carries
question/expected_answer/actual_answer — there are no retrieval contexts.
We score four RAGAS-style dimensions that only need those three fields:

	correctness  — factual match of actual vs expected (0-1)
	relevance    — does the actual response address the question (0-1)
	completeness — coverage of key facts from expected (0-1)
	faithfulness — absence of contradictions/fabrications vs expected (0-1)

composite_score is the plain mean of the four metrics.

Output JSON mirrors yar/evaluation/results/results_*.json so the
generate_heatmaps.py script can treat both sources symmetrically.

Typical use:
	.venv/bin/python yar/evaluation/grade_concierge.py \
		--input eval_docs/qa_pair_concierge.json \
		--judge-model tuna
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from litellm import completion

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / 'results'
DEFAULT_INPUT = REPO_ROOT / 'eval_docs' / 'qa_pair_concierge.json'

load_dotenv(REPO_ROOT / '.env', override=False)

litellm.suppress_debug_info = True


METRIC_KEYS = ('correctness', 'relevance', 'completeness', 'faithfulness')

SYSTEM_PROMPT = """You are a strict RAG evaluation judge.

You score a single QA row against a reference (expected) answer. You do NOT
have the source documents; treat EXPECTED ANSWER as the ground truth.

Return ONE JSON object with exactly these float fields in [0.0, 1.0]:
{
  "correctness":  <0-1>,
  "relevance":    <0-1>,
  "completeness": <0-1>,
  "faithfulness": <0-1>,
  "reason": "<<=2 short sentences>"
}

Definitions:
- correctness:  Does the ACTUAL answer match the factual content of the
                EXPECTED answer? Trivial-yes/no questions where expected
                is "yes"/"no" score 1.0 iff actual gives the same verdict.
                Partial match -> partial credit.
- relevance:    Does the ACTUAL answer address the QUESTION's topic and
                produce a direct answer? Asking clarifying questions,
                refusing, or dumping off-topic content scores low.
- completeness: Do all key facts from EXPECTED appear in ACTUAL?
                Missing major facts drops the score proportionally.
                If EXPECTED is a single short verdict (yes/no/name),
                completeness tracks correctness.
- faithfulness: Is ACTUAL free of claims that contradict EXPECTED or are
                clearly fabricated (invented document IDs, wrong project
                names, hallucinated procedures)? Extra supported detail
                is fine. Direct contradictions drop the score sharply.

Rules:
- Output JSON only. No prose outside the JSON.
- Be harsh with scores. Use the full 0-1 range. 1.0 = perfect; 0.5 =
  partial; 0.0 = wrong or missing.
- If ACTUAL is an error, refusal, or asks for clarification instead of
  answering, score every metric <= 0.3.
"""


def build_user_prompt(question: str, expected: str, actual: str) -> str:
    return f'QUESTION:\n{question}\n\nEXPECTED ANSWER:\n{expected}\n\nACTUAL ANSWER:\n{actual}'


def parse_json_object(raw: str) -> dict[str, Any]:
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f'No JSON object found in judge output: {raw[:200]}')
    return json.loads(raw[start : end + 1])


def clamp(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def normalize_model(judge_model: str, api_base: str) -> str:
    known = ('openai/', 'openrouter/', 'anthropic/', 'gemini/', 'ollama/', 'azure/', 'bedrock/', 'custom_openai/')
    if judge_model.startswith(known):
        return judge_model
    if api_base:
        return f'openai/{judge_model}'
    return judge_model


def call_judge(
    *,
    question: str,
    expected: str,
    actual: str,
    judge_model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    model_name = normalize_model(judge_model, api_base)
    response = completion(
        model=model_name,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_prompt(question, expected, actual)},
        ],
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={'type': 'json_object'},
    )
    raw = str(response.choices[0].message.content)
    payload = parse_json_object(raw)
    metrics = {key: clamp(payload.get(key)) for key in METRIC_KEYS}
    reason = str(payload.get('reason', '')).strip() or 'No reason returned.'
    return {'metrics': metrics, 'reason': reason, 'raw': raw}


def grade_one(
    pair: dict[str, Any],
    judge_model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int = 2,
) -> dict[str, Any]:
    question = str(pair.get('question', '')).strip()
    expected = str(pair.get('expected_answer', '')).strip()
    actual = str(pair.get('actual_answer', '')).strip()
    test_number = int(pair.get('id') or 0)

    base_row = {
        'test_number': test_number,
        'question': question,
        'answer': actual,
        'ground_truth': expected,
        'timestamp': datetime.now().isoformat(),
    }

    if not actual:
        return {
            **base_row,
            'metrics': dict.fromkeys(METRIC_KEYS, 0.0),
            'composite_score': 0.0,
            'status': 'empty_actual',
            'reason': 'No actual_answer present in source.',
        }

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = call_judge(
                question=question,
                expected=expected,
                actual=actual,
                judge_model=judge_model,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            metrics = result['metrics']
            composite = round(sum(metrics.values()) / len(metrics), 4)
            metrics_rounded = {k: round(v, 4) for k, v in metrics.items()}
            return {
                **base_row,
                'metrics': metrics_rounded,
                'composite_score': composite,
                'status': 'success',
                'reason': result['reason'],
            }
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    return {
        **base_row,
        'metrics': dict.fromkeys(METRIC_KEYS, 0.0),
        'composite_score': 0.0,
        'status': 'error',
        'error': f'{type(last_err).__name__}: {last_err}',
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Grade concierge QA pairs with a LiteLLM judge.')
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Concierge qa_pair JSON file.')
    p.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output graded JSON path (default: results/concierge_graded_<ts>.json).',
    )
    p.add_argument('--judge-model', type=str, default=os.getenv('CONCIERGE_JUDGE_MODEL', 'tuna'))
    p.add_argument(
        '--judge-api-base',
        type=str,
        default=os.getenv('EVAL_LLM_BINDING_HOST') or os.getenv('LLM_BINDING_HOST') or 'http://localhost:4000/v1',
    )
    p.add_argument(
        '--judge-api-key',
        type=str,
        default=os.getenv('EVAL_LLM_BINDING_API_KEY')
        or os.getenv('LLM_BINDING_API_KEY')
        or os.getenv('LITELLM_MASTER_KEY', ''),
    )
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=400)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--concurrency', type=int, default=6)
    p.add_argument('--limit', type=int, default=0, help='Only grade first N items (0 = all).')
    p.add_argument('--ids', type=str, default='', help='Comma-separated list of QA ids to grade (overrides --limit).')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_api_key:
        print(
            'ERROR: no judge API key. Set LLM_BINDING_API_KEY/LITELLM_MASTER_KEY in .env or pass --judge-api-key.',
            file=sys.stderr,
        )
        sys.exit(2)

    source = json.loads(args.input.read_text(encoding='utf-8'))
    qa_pairs: list[dict[str, Any]] = source.get('qa_pairs', [])

    if args.ids:
        keep = {int(x) for x in args.ids.split(',') if x.strip()}
        qa_pairs = [p for p in qa_pairs if int(p.get('id') or 0) in keep]
    elif args.limit > 0:
        qa_pairs = qa_pairs[: args.limit]

    if not qa_pairs:
        print('No QA pairs selected.', file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = args.output or (RESULTS_DIR / f'concierge_graded_{ts}.json')

    print(f'Grading {len(qa_pairs)} concierge QA pairs with model={args.judge_model} via {args.judge_api_base}')
    start = time.time()

    results: list[dict[str, Any]] = [None] * len(qa_pairs)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        future_to_idx = {
            pool.submit(
                grade_one,
                pair,
                args.judge_model,
                args.judge_api_base,
                args.judge_api_key,
                args.temperature,
                args.max_tokens,
                args.timeout,
            ): idx
            for idx, pair in enumerate(qa_pairs)
        }
        for done, future in enumerate(as_completed(future_to_idx), start=1):
            idx = future_to_idx[future]
            try:
                row = future.result()
            except Exception as exc:
                pair = qa_pairs[idx]
                row = {
                    'test_number': int(pair.get('id') or 0),
                    'question': pair.get('question', ''),
                    'answer': pair.get('actual_answer', ''),
                    'ground_truth': pair.get('expected_answer', ''),
                    'metrics': dict.fromkeys(METRIC_KEYS, 0.0),
                    'composite_score': 0.0,
                    'status': 'error',
                    'error': f'{type(exc).__name__}: {exc}',
                    'timestamp': datetime.now().isoformat(),
                }
            results[idx] = row
            comp = row.get('composite_score', 0.0)
            status = row.get('status', '?')
            q_preview = (row.get('question') or '')[:70]
            print(f'[{done}/{len(qa_pairs)}] id={row.get("test_number"):>2} score={comp:.3f} [{status}] {q_preview}')

    results.sort(key=lambda r: r.get('test_number', 0))
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        averages = {
            metric: round(sum(r['metrics'][metric] for r in successful) / len(successful), 4) for metric in METRIC_KEYS
        }
        avg_composite = round(sum(r['composite_score'] for r in successful) / len(successful), 4)
    else:
        averages = dict.fromkeys(METRIC_KEYS, 0.0)
        avg_composite = 0.0

    payload = {
        'timestamp': datetime.now().isoformat(),
        'source': source.get('source', 'concierge'),
        'input_file': str(args.input.resolve()),
        'judge_model': args.judge_model,
        'judge_api_base': args.judge_api_base,
        'total_tests': len(results),
        'successful_tests': len(successful),
        'failed_tests': len(results) - len(successful),
        'elapsed_time_seconds': round(time.time() - start, 2),
        'average_metrics': {**averages, 'composite_score': avg_composite},
        'results': results,
    }

    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nWrote {output}')
    print(f'Average metrics: {payload["average_metrics"]}')


if __name__ == '__main__':
    main()
