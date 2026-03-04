#!/usr/bin/env python3
"""Benchmark Kreuzberg OCR backends on a sample PDF.

This script runs extraction with each OCR backend and reports:
- latency (ms)
- extracted text size (characters/words)
- optional similarity vs ground truth text

Use this to compare backend quality/speed on your own document set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from kreuzberg import ExtractionConfig, OcrConfig, TesseractConfig, extract_file_sync, get_valid_ocr_backends


@dataclass
class BackendResult:
    backend: str
    status: str
    runs: int
    latency_ms_avg: float | None
    latency_ms_min: float | None
    latency_ms_max: float | None
    content_chars: int | None
    content_words: int | None
    non_whitespace_chars: int | None
    similarity: float | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark Kreuzberg OCR backends on one PDF.')
    parser.add_argument('pdf', type=Path, help='Path to sample PDF')
    parser.add_argument(
        '--language',
        default='eng',
        help='OCR language (Kreuzberg format, e.g. eng, eng+fra)',
    )
    parser.add_argument(
        '--backends',
        default='all',
        help='Comma-separated backend list or "all" (default, tesseract, easyocr, paddleocr when available)',
    )
    parser.add_argument('--iterations', type=int, default=3, help='Runs per backend (default: 3)')
    parser.add_argument(
        '--ground-truth',
        type=Path,
        default=None,
        help='Optional text file for accuracy comparison (SequenceMatcher ratio)',
    )
    parser.add_argument(
        '--force-ocr',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Force OCR even if PDF has text layer (default: true)',
    )
    parser.add_argument(
        '--tesseract-oem',
        type=int,
        default=None,
        help='Optional Tesseract OCR Engine Mode (0-3) for tesseract backend only',
    )
    parser.add_argument(
        '--json-out',
        type=Path,
        default=None,
        help='Optional output path for JSON report',
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return ' '.join(value.split())


def text_similarity(extracted: str, ground_truth: str) -> float:
    return SequenceMatcher(None, normalize_text(extracted), normalize_text(ground_truth)).ratio()


def normalize_language_for_backend(language: str, backend: str) -> str:
    """Normalize language codes for backend-specific expectations."""

    iso3_to_iso2 = {
        'eng': 'en',
        'deu': 'de',
        'fra': 'fr',
        'spa': 'es',
        'ita': 'it',
        'por': 'pt',
        'rus': 'ru',
        'jpn': 'ja',
        'kor': 'ko',
        'ara': 'ar',
        'chi_sim': 'ch_sim',
        'chi_tra': 'ch_tra',
        'zho': 'ch_sim',
    }
    iso2_to_iso3 = {v: k for k, v in iso3_to_iso2.items() if len(v) == 2}

    parts = [part.strip() for part in language.split('+') if part.strip()]
    if not parts:
        return language

    if backend == 'tesseract':
        normalized = [iso2_to_iso3.get(part, part) for part in parts]
    else:
        normalized = [iso3_to_iso2.get(part, part) for part in parts]

    return '+'.join(normalized)


def run_backend(
    pdf_path: Path,
    backend: str,
    language: str,
    iterations: int,
    ground_truth: str | None,
    force_ocr: bool,
    tesseract_oem: int | None,
) -> BackendResult:
    latencies: list[float] = []
    last_content = ''

    try:
        ocr_kwargs: dict[str, Any]
        if backend == 'default':
            ocr_kwargs = {}
        else:
            backend_language = normalize_language_for_backend(language, backend)
            ocr_kwargs = {
                'backend': backend,
                'language': backend_language,
            }
            if backend == 'tesseract' and tesseract_oem is not None:
                ocr_kwargs['tesseract_config'] = TesseractConfig(oem=tesseract_oem)

        config = ExtractionConfig(
            ocr=OcrConfig(**ocr_kwargs),
            force_ocr=force_ocr,
            output_format='plain',
        )

        for _ in range(iterations):
            start = time.perf_counter()
            result = extract_file_sync(str(pdf_path), config=config)
            latencies.append((time.perf_counter() - start) * 1000)
            last_content = result.content or ''

        similarity = None
        if ground_truth is not None:
            similarity = text_similarity(last_content, ground_truth)

        words = len(last_content.split())
        non_ws = len(''.join(last_content.split()))

        return BackendResult(
            backend=backend,
            status='ok',
            runs=iterations,
            latency_ms_avg=statistics.mean(latencies),
            latency_ms_min=min(latencies),
            latency_ms_max=max(latencies),
            content_chars=len(last_content),
            content_words=words,
            non_whitespace_chars=non_ws,
            similarity=similarity,
            error=None,
        )
    except BaseException as exc:
        return BackendResult(
            backend=backend,
            status='error',
            runs=0,
            latency_ms_avg=None,
            latency_ms_min=None,
            latency_ms_max=None,
            content_chars=None,
            content_words=None,
            non_whitespace_chars=None,
            similarity=None,
            error=str(exc),
        )


def choose_backends(raw: str) -> list[str]:
    available = get_valid_ocr_backends()
    selectable = ['default', *available]
    if raw == 'all':
        return selectable

    requested = [item.strip() for item in raw.split(',') if item.strip()]
    valid = [item for item in requested if item in selectable]
    invalid = [item for item in requested if item not in selectable]

    # Preserve order while removing duplicates.
    valid = list(dict.fromkeys(valid))

    if invalid:
        print(f'Warning: skipping unavailable/unknown backends: {", ".join(invalid)}')

    return valid


def ensure_tesseract_traineddata(language: str) -> None:
    """Ensure required tesseract traineddata exists locally."""

    tessdata_dir_raw = os.environ.get('TESSDATA_PREFIX')
    tessdata_dir = Path(tessdata_dir_raw).expanduser() if tessdata_dir_raw else Path.home() / '.local/share/tessdata'
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TESSDATA_PREFIX'] = str(tessdata_dir)

    required = normalize_language_for_backend(language, 'tesseract').split('+')
    for code in required:
        traineddata = tessdata_dir / f'{code}.traineddata'
        if traineddata.exists():
            continue

        url = f'https://github.com/tesseract-ocr/tessdata_fast/raw/main/{code}.traineddata'
        print(f'Downloading tesseract language model: {code}')
        urllib.request.urlretrieve(url, traineddata)  # nosec B310
        print(f'  saved: {traineddata}')


def ensure_onnxruntime_linkable() -> None:
    """Ensure libonnxruntime.so is discoverable by Kreuzberg loaders."""

    import ctypes

    import onnxruntime

    capi_dir = Path(onnxruntime.__file__).resolve().parent / 'capi'
    candidates = sorted(capi_dir.glob('libonnxruntime.so.*'))
    if not candidates:
        return

    target = candidates[-1]
    for soname in ('libonnxruntime.so', 'libonnxruntime.so.1'):
        link = capi_dir / soname
        if link.exists():
            continue
        link.symlink_to(target.name)

    venv_lib = Path(sys.executable).resolve().parents[1] / 'lib'
    existing = os.environ.get('LD_LIBRARY_PATH', '')
    required_prefixes = [str(capi_dir), str(venv_lib)]

    if not all(prefix in existing.split(':') for prefix in required_prefixes):
        new_ld_path = ':'.join(required_prefixes + ([existing] if existing else []))
        if os.environ.get('KREUZBERG_OCR_BENCH_REEXEC') != '1':
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = new_ld_path
            env['KREUZBERG_OCR_BENCH_REEXEC'] = '1'
            os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
        os.environ['LD_LIBRARY_PATH'] = new_ld_path

    # Load with an absolute path as a final guard for symbol availability.
    ctypes.CDLL(str(capi_dir / 'libonnxruntime.so'), mode=ctypes.RTLD_GLOBAL)


def prepare_runtime_environment(backends: list[str], language: str) -> None:
    """Prepare backend runtime dependencies so benchmarks run out of the box."""

    if 'tesseract' in backends or 'default' in backends:
        ensure_tesseract_traineddata(language)
    if 'paddleocr' in backends or 'default' in backends:
        ensure_onnxruntime_linkable()


def print_table(results: list[BackendResult]) -> None:
    print('\n=== OCR Benchmark Results ===')
    print(
        'backend'.ljust(12),
        'status'.ljust(8),
        'avg_ms'.rjust(10),
        'chars'.rjust(8),
        'words'.rjust(8),
        'sim'.rjust(8),
    )
    print('-' * 60)

    for row in results:
        avg_ms = f'{row.latency_ms_avg:.1f}' if row.latency_ms_avg is not None else '-'
        chars = str(row.content_chars) if row.content_chars is not None else '-'
        words = str(row.content_words) if row.content_words is not None else '-'
        sim = f'{row.similarity:.4f}' if row.similarity is not None else '-'

        print(
            row.backend.ljust(12),
            row.status.ljust(8),
            avg_ms.rjust(10),
            chars.rjust(8),
            words.rjust(8),
            sim.rjust(8),
        )
        if row.error:
            print(f'  error: {row.error}')


def recommend(results: list[BackendResult], has_ground_truth: bool) -> str:
    ok_rows = [r for r in results if r.status == 'ok' and (r.content_chars or 0) > 0]
    if not ok_rows:
        return 'No backend produced usable text.'

    if has_ground_truth:
        ranked = sorted(
            ok_rows,
            key=lambda r: (
                -(r.similarity or 0.0),
                r.latency_ms_avg if r.latency_ms_avg is not None else float('inf'),
            ),
        )
        best = ranked[0]
        return f'Best by accuracy: {best.backend} (similarity={best.similarity:.4f}, avg={best.latency_ms_avg:.1f}ms)'

    ranked = sorted(
        ok_rows,
        key=lambda r: (
            r.latency_ms_avg if r.latency_ms_avg is not None else float('inf'),
            -(r.non_whitespace_chars or 0),
        ),
    )
    best = ranked[0]
    return (
        f'No ground truth provided; fastest non-empty backend is {best.backend} '
        f'({best.latency_ms_avg:.1f}ms avg). '
        'Use --ground-truth for a quality-based winner.'
    )


def main() -> int:
    args = parse_args()

    if not args.pdf.exists():
        print(f'Error: PDF not found: {args.pdf}')
        return 1
    if args.iterations < 1:
        print('Error: --iterations must be >= 1')
        return 1

    ground_truth = None
    if args.ground_truth is not None:
        if not args.ground_truth.exists():
            print(f'Error: ground truth file not found: {args.ground_truth}')
            return 1
        ground_truth = args.ground_truth.read_text(encoding='utf-8')

    backends = choose_backends(args.backends)
    if not backends:
        print('Error: no valid backends selected')
        return 1

    prepare_runtime_environment(backends, args.language)
    print(f'PDF: {args.pdf}')
    print(f'Backends: {", ".join(backends)}')
    print(f'Iterations per backend: {args.iterations}')
    print(f'Force OCR: {args.force_ocr}')
    print(f'Ground truth: {"yes" if ground_truth else "no"}')

    results = [
        run_backend(
            pdf_path=args.pdf,
            backend=backend,
            language=args.language,
            iterations=args.iterations,
            ground_truth=ground_truth,
            force_ocr=args.force_ocr,
            tesseract_oem=args.tesseract_oem,
        )
        for backend in backends
    ]

    print_table(results)
    print('\nRecommendation:')
    print(recommend(results, has_ground_truth=ground_truth is not None))

    if args.json_out is not None:
        payload = {
            'pdf': str(args.pdf),
            'iterations': args.iterations,
            'force_ocr': args.force_ocr,
            'language': args.language,
            'results': [asdict(item) for item in results],
            'recommendation': recommend(results, has_ground_truth=ground_truth is not None),
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nJSON report written to: {args.json_out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
