"""Extraction-quality diagnostics for processed/canonical document artifacts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from yar.evaluation.qa_eval_common import timestamped_results_path

PAGE_MARKER_RE = re.compile(r'<!--\s*PAGE\s+(\d+)\s*-->', flags=re.IGNORECASE)
SIMPLE_MARKDOWN_TABLE_RE = re.compile(r'^\s*\|.*\|\s*$', flags=re.MULTILINE)
HTML_TABLE_RE = re.compile(r'<table\b[\s\S]*?</table>', flags=re.IGNORECASE)


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def count_page_markers(content: str) -> int:
    return len(PAGE_MARKER_RE.findall(content))


def _brand_line_counts(content: str) -> dict[str, int]:
    exact = 0
    heading = 0
    for line in content.splitlines():
        stripped = line.strip()
        normalized = re.sub(r'^#+\s*', '', stripped).strip().lower()
        if stripped.lower() == 'sanofi':
            exact += 1
        if normalized == 'sanofi' and stripped.startswith('#'):
            heading += 1
    return {'exact_sanofi_lines': exact, 'heading_sanofi_lines': heading}


def _table_counts(content: str) -> dict[str, int]:
    return {
        'html_tables': len(HTML_TABLE_RE.findall(content)),
        'markdown_table_lines': len(SIMPLE_MARKDOWN_TABLE_RE.findall(content)),
    }


def _quality_value(report: dict[str, Any], key: str, default: Any) -> Any:
    value = report.get(key)
    return default if value is None else value


def summarize_artifacts(
    *,
    name: str,
    processed_text: str,
    canonical_text: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    quality_report_raw = manifest.get('quality_report')
    quality_report: dict[str, Any] = quality_report_raw if isinstance(quality_report_raw, dict) else {}
    page_records_raw = manifest.get('page_records')
    page_records: list[Any] = page_records_raw if isinstance(page_records_raw, list) else []
    canonical_marker_count = count_page_markers(canonical_text)
    retrieval_marker_count = count_page_markers(processed_text)
    expected_page_count = int(_quality_value(quality_report, 'page_count', len(page_records) or canonical_marker_count))
    table_counts = quality_report.get('table_counts') if isinstance(quality_report.get('table_counts'), dict) else {}
    processed_table_counts = _table_counts(processed_text)
    canonical_table_counts = _table_counts(canonical_text)

    return {
        'name': name,
        'expected_page_count': expected_page_count,
        'canonical_page_marker_count': canonical_marker_count,
        'retrieval_page_marker_count': retrieval_marker_count,
        'dropped_retrieval_pages': _quality_value(quality_report, 'dropped_retrieval_pages', []),
        'empty_pages': _quality_value(quality_report, 'empty_pages', []),
        'tiny_pages': _quality_value(quality_report, 'tiny_pages', []),
        'unexplained_tiny_pages': _quality_value(quality_report, 'unexplained_tiny_pages', []),
        'boilerplate_only_pages': _quality_value(quality_report, 'boilerplate_only_pages', []),
        'native_fallback_pages': _quality_value(quality_report, 'native_fallback_pages', []),
        'warning_counts': _quality_value(quality_report, 'warning_counts', {}),
        'manifest_table_counts': table_counts,
        'processed_table_counts': processed_table_counts,
        'canonical_table_counts': canonical_table_counts,
        'processed_boilerplate_counts': _brand_line_counts(processed_text),
        'canonical_boilerplate_counts': _brand_line_counts(canonical_text),
        'hashes': {
            'processed_sha256': sha256_text(processed_text),
            'canonical_sha256': sha256_text(canonical_text),
            'manifest_sha256': sha256_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
        },
        'extractor_version': manifest.get('extractor_version'),
        'prompt_version': manifest.get('prompt_version'),
        'model': manifest.get('model'),
    }


def format_markdown_summary(summaries: list[dict[str, Any]]) -> str:
    lines = [
        '# Extraction Quality Summary',
        '',
        '| Document | Expected pages | Canonical markers | Retrieval markers | Dropped retrieval | Empty | Tiny | Unexplained tiny | Native fallback |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for summary in summaries:
        lines.append(
            '| {name} | {expected} | {canonical} | {retrieval} | {dropped} | {empty} | {tiny} | {unexplained} | {native} |'.format(
                name=str(summary['name']).replace('|', '\\|'),
                expected=summary['expected_page_count'],
                canonical=summary['canonical_page_marker_count'],
                retrieval=summary['retrieval_page_marker_count'],
                dropped=len(summary['dropped_retrieval_pages']),
                empty=len(summary['empty_pages']),
                tiny=len(summary['tiny_pages']),
                unexplained=len(summary['unexplained_tiny_pages']),
                native=len(summary['native_fallback_pages']),
            )
        )
    lines.append('')
    lines.append('## Artifact hashes')
    for summary in summaries:
        hashes = summary['hashes']
        lines.extend(
            [
                '',
                f'### {summary["name"]}',
                f'- processed_sha256: `{hashes["processed_sha256"]}`',
                f'- canonical_sha256: `{hashes["canonical_sha256"]}`',
                f'- manifest_sha256: `{hashes["manifest_sha256"]}`',
            ]
        )
    return '\n'.join(lines) + '\n'


def _read_local_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _load_local_group(spec: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    processed = _read_local_text(Path(str(spec['processed'])))
    canonical = _read_local_text(Path(str(spec['canonical'])))
    manifest = json.loads(_read_local_text(Path(str(spec['manifest']))))
    if not isinstance(manifest, dict):
        raise ValueError(f'Manifest is not a JSON object: {spec["manifest"]}')
    return processed, canonical, manifest


async def _read_s3_text(client: Any, key: str) -> str:
    content, _metadata = await client.get_object(key)
    if isinstance(content, str):
        return content
    return bytes(content).decode('utf-8')


async def _load_s3_group(spec: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    from yar.storage.s3_client import S3Client, S3Config

    prefix = str(spec['s3_prefix']).removesuffix('.')
    client = S3Client(S3Config())
    await client.initialize()
    try:
        processed = await _read_s3_text(client, f'{prefix}.processed.md')
        canonical = await _read_s3_text(client, f'{prefix}.canonical.md')
        manifest_raw = await _read_s3_text(client, f'{prefix}.extraction.json')
    finally:
        await client.finalize()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise ValueError(f'Manifest is not a JSON object for S3 prefix: {prefix}')
    return processed, canonical, manifest


async def _load_group(spec: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if 's3_prefix' in spec:
        return await _load_s3_group(spec)
    return _load_local_group(spec)


def _specs_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if args.input_manifest:
        raw_specs = json.loads(args.input_manifest.read_text(encoding='utf-8'))
        if not isinstance(raw_specs, list):
            raise SystemExit('--input-manifest must contain a JSON array')
        specs.extend(spec for spec in raw_specs if isinstance(spec, dict))
    if args.s3_prefix:
        specs.append({'name': args.name or args.s3_prefix, 's3_prefix': args.s3_prefix})
    if args.processed or args.canonical or args.manifest:
        if not (args.processed and args.canonical and args.manifest):
            raise SystemExit('--processed, --canonical, and --manifest must be provided together')
        specs.append(
            {
                'name': args.name or args.processed.stem,
                'processed': str(args.processed),
                'canonical': str(args.canonical),
                'manifest': str(args.manifest),
            }
        )
    if not specs:
        raise SystemExit('Provide --input-manifest, --s3-prefix, or local --processed/--canonical/--manifest paths')
    return specs


async def run_diagnostics(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        processed_text, canonical_text, manifest = await _load_group(spec)
        summaries.append(
            summarize_artifacts(
                name=str(spec.get('name') or spec.get('s3_prefix') or spec.get('processed') or 'document'),
                processed_text=processed_text,
                canonical_text=canonical_text,
                manifest=manifest,
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize YAR extraction-quality artifacts.')
    parser.add_argument('--input-manifest', type=Path, help='JSON array of local artifact paths or S3 prefixes')
    parser.add_argument('--processed', type=Path, help='Local .processed.md path')
    parser.add_argument('--canonical', type=Path, help='Local .canonical.md path')
    parser.add_argument('--manifest', type=Path, help='Local .extraction.json path')
    parser.add_argument('--s3-prefix', help='S3 key prefix without .processed.md/.canonical.md/.extraction.json suffix')
    parser.add_argument('--name', help='Display name for a single local/S3 artifact group')
    parser.add_argument('--output-json', type=Path, help='Output JSON path')
    parser.add_argument('--output-markdown', type=Path, help='Output Markdown path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _specs_from_args(args)
    summaries = asyncio.run(run_diagnostics(specs))
    output_json = args.output_json or timestamped_results_path('extraction_quality', suffix='.json')
    output_markdown = args.output_markdown or output_json.with_suffix('.md')
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    output_markdown.write_text(format_markdown_summary(summaries), encoding='utf-8')
    print(f'Wrote extraction quality JSON: {output_json}')
    print(f'Wrote extraction quality Markdown: {output_markdown}')


if __name__ == '__main__':
    main()
