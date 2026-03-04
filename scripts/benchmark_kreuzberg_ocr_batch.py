#!/usr/bin/env python3
"""Benchmark Kreuzberg OCR backends across a folder of PDFs.

This wrapper reuses the single-file benchmark implementation and produces
an aggregate recommendation over multiple documents.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_kreuzberg_ocr import choose_backends, prepare_runtime_environment, run_backend


@dataclass
class FileBackendResult:
    pdf: str
    has_ground_truth: bool
    backend: str
    status: str
    latency_ms_avg: float | None
    content_chars: int | None
    similarity: float | None
    error: str | None


@dataclass
class BackendAggregate:
    backend: str
    files_total: int
    files_ok: int
    files_error: int
    files_with_similarity: int
    latency_ms_avg: float | None
    latency_ms_median: float | None
    similarity_avg: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark Kreuzberg OCR backends over a PDF folder.')
    parser.add_argument('pdf_dir', type=Path, help='Directory containing sample PDFs')
    parser.add_argument('--glob', default='*.pdf', help='Glob pattern inside pdf_dir (default: *.pdf)')
    parser.add_argument(
        '--ground-truth-dir',
        type=Path,
        default=None,
        help='Optional directory of .txt files matching PDF stem names for similarity scoring',
    )
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
    parser.add_argument('--iterations', type=int, default=1, help='Runs per backend per PDF (default: 1)')
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
    parser.add_argument('--json-out', type=Path, default=None, help='Optional output path for JSON report')
    return parser.parse_args()


def resolve_ground_truth(pdf_path: Path, ground_truth_dir: Path | None) -> str | None:
    if ground_truth_dir is None:
        return None

    txt_path = ground_truth_dir / f'{pdf_path.stem}.txt'
    if not txt_path.exists():
        return None
    return txt_path.read_text(encoding='utf-8')


def aggregate_backend_results(backend: str, rows: list[FileBackendResult]) -> BackendAggregate:
    ok_rows = [row for row in rows if row.status == 'ok' and (row.content_chars or 0) > 0]
    latencies = [row.latency_ms_avg for row in ok_rows if row.latency_ms_avg is not None]
    similarities = [row.similarity for row in ok_rows if row.similarity is not None]

    return BackendAggregate(
        backend=backend,
        files_total=len(rows),
        files_ok=len(ok_rows),
        files_error=len(rows) - len(ok_rows),
        files_with_similarity=len(similarities),
        latency_ms_avg=statistics.mean(latencies) if latencies else None,
        latency_ms_median=statistics.median(latencies) if latencies else None,
        similarity_avg=statistics.mean(similarities) if similarities else None,
    )


def recommend(aggregates: list[BackendAggregate]) -> str:
    usable = [item for item in aggregates if item.files_ok > 0]
    if not usable:
        return 'No backend produced usable text on this dataset.'

    scored = [item for item in usable if item.files_with_similarity > 0 and item.similarity_avg is not None]
    if scored:
        best = sorted(
            scored,
            key=lambda item: (
                -item.similarity_avg,
                item.latency_ms_avg if item.latency_ms_avg is not None else float('inf'),
                -item.files_ok,
            ),
        )[0]
        return (
            f'Best by aggregate accuracy: {best.backend} '
            f'(similarity_avg={best.similarity_avg:.4f}, latency_avg={best.latency_ms_avg:.1f}ms, files_ok={best.files_ok}/{best.files_total})'
        )

    fastest = sorted(
        usable,
        key=lambda item: (
            item.latency_ms_avg if item.latency_ms_avg is not None else float('inf'),
            -item.files_ok,
        ),
    )[0]
    return (
        f'No ground truth matched; fastest non-empty backend is {fastest.backend} '
        f'(latency_avg={fastest.latency_ms_avg:.1f}ms, files_ok={fastest.files_ok}/{fastest.files_total}).'
    )


def print_summary(aggregates: list[BackendAggregate]) -> None:
    print('\n=== Aggregate OCR Benchmark Results ===')
    print(
        'backend'.ljust(12),
        'ok/total'.ljust(10),
        'lat_avg_ms'.rjust(12),
        'lat_med_ms'.rjust(12),
        'sim_avg'.rjust(10),
    )
    print('-' * 64)

    for row in aggregates:
        ok_total = f'{row.files_ok}/{row.files_total}'
        lat_avg = f'{row.latency_ms_avg:.1f}' if row.latency_ms_avg is not None else '-'
        lat_median = f'{row.latency_ms_median:.1f}' if row.latency_ms_median is not None else '-'
        sim_avg = f'{row.similarity_avg:.4f}' if row.similarity_avg is not None else '-'

        print(
            row.backend.ljust(12),
            ok_total.ljust(10),
            lat_avg.rjust(12),
            lat_median.rjust(12),
            sim_avg.rjust(10),
        )


def main() -> int:
    args = parse_args()

    if not args.pdf_dir.exists() or not args.pdf_dir.is_dir():
        print(f'Error: PDF directory not found or not a directory: {args.pdf_dir}')
        return 1
    if args.iterations < 1:
        print('Error: --iterations must be >= 1')
        return 1
    if args.ground_truth_dir is not None and (not args.ground_truth_dir.exists() or not args.ground_truth_dir.is_dir()):
        print(f'Error: ground truth directory not found or not a directory: {args.ground_truth_dir}')
        return 1

    pdf_paths = sorted(path for path in args.pdf_dir.glob(args.glob) if path.is_file())
    if not pdf_paths:
        print(f'Error: no PDF files found under {args.pdf_dir} with pattern {args.glob}')
        return 1

    backends = choose_backends(args.backends)
    if not backends:
        print('Error: no valid backends selected')
        return 1

    prepare_runtime_environment(backends, args.language)

    print(f'PDF directory: {args.pdf_dir}')
    print(f'Pattern: {args.glob}')
    print(f'Files discovered: {len(pdf_paths)}')
    print(f'Backends: {", ".join(backends)}')
    print(f'Iterations per backend per PDF: {args.iterations}')

    file_results: list[FileBackendResult] = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        ground_truth = resolve_ground_truth(pdf_path, args.ground_truth_dir)
        print(
            f'[{index}/{len(pdf_paths)}] {pdf_path.name} (ground truth: {"yes" if ground_truth is not None else "no"})'
        )

        for backend in backends:
            result = run_backend(
                pdf_path=pdf_path,
                backend=backend,
                language=args.language,
                iterations=args.iterations,
                ground_truth=ground_truth,
                force_ocr=args.force_ocr,
                tesseract_oem=args.tesseract_oem,
            )
            file_results.append(
                FileBackendResult(
                    pdf=str(pdf_path),
                    has_ground_truth=ground_truth is not None,
                    backend=backend,
                    status=result.status,
                    latency_ms_avg=result.latency_ms_avg,
                    content_chars=result.content_chars,
                    similarity=result.similarity,
                    error=result.error,
                )
            )

            if result.status == 'ok':
                print(
                    f'  {backend}: ok '
                    f'(avg={result.latency_ms_avg:.1f}ms, chars={result.content_chars}, sim={result.similarity if result.similarity is not None else "-"})'
                )
            else:
                print(f'  {backend}: error ({result.error})')

    aggregates = [
        aggregate_backend_results(
            backend=backend,
            rows=[row for row in file_results if row.backend == backend],
        )
        for backend in backends
    ]

    print_summary(aggregates)
    recommendation = recommend(aggregates)
    print('\nRecommendation:')
    print(recommendation)

    if args.json_out is not None:
        payload = {
            'pdf_dir': str(args.pdf_dir),
            'glob': args.glob,
            'iterations': args.iterations,
            'force_ocr': args.force_ocr,
            'language': args.language,
            'backends': backends,
            'file_results': [asdict(item) for item in file_results],
            'aggregates': [asdict(item) for item in aggregates],
            'recommendation': recommendation,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nJSON report written to: {args.json_out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
