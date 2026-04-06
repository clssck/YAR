#!/usr/bin/env python3
"""
E2E RAGAS Test Harness for YAR.

Complete end-to-end testing pipeline:
1. Clear existing data (optional, used by A/B runs)
2. Ingest bundled sample documents into YAR
3. Wait for document processing via current track-status endpoint
4. Use an explicit or bundled evaluation dataset
5. Run RAGAS evaluation
6. Optional: A/B comparison

Usage:
    # Full E2E test using bundled sample documents + bundled dataset
    python yar/evaluation/e2e_test_harness.py

    # A/B comparison (with/without orphan connections)
    python yar/evaluation/e2e_test_harness.py --ab-test

    # Use an explicit dataset
    python yar/evaluation/e2e_test_harness.py --dataset existing_dataset.json

    # Skip ingest when documents are already loaded in YAR
    python yar/evaluation/e2e_test_harness.py --skip-ingest
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from yar.utils import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv(dotenv_path='.env', override=False)

# Configuration
DEFAULT_RAG_URL = 'http://localhost:9622'
POLL_INTERVAL_SECONDS = 5
MAX_WAIT_SECONDS = 600
BUNDLED_DATASET_FILENAMES = ('sample_dataset.json', 'wiki_test_dataset.json')
BUNDLED_DOCUMENT_DIRNAMES = ('sample_documents', 'wiki_documents')


def resolve_dataset_path(explicit_path: Path | None, bundled_candidates: Sequence[Path]) -> Path:
    """Resolve the evaluation dataset path or raise an actionable error."""
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f'Explicit dataset not found: {explicit_path}')

    for candidate in bundled_candidates:
        if candidate.exists():
            return candidate

    formatted_candidates = ', '.join(str(path) for path in bundled_candidates)
    raise FileNotFoundError(
        'No evaluation dataset is available. Provide --dataset /path/to/dataset.json '
        f'or add one of the bundled datasets: {formatted_candidates}'
    )


class E2ETestHarness:
    """End-to-end test harness for YAR RAGAS evaluation."""

    def __init__(
        self,
        rag_url: str | None = None,
        paper_ids: list[str] | None = None,
        questions_per_paper: int = 5,
        skip_download: bool = False,
        skip_ingest: bool = False,
        dataset_path: str | None = None,
        output_dir: str | None = None,
    ):
        self.rag_url = (rag_url or os.getenv('YAR_API_URL', DEFAULT_RAG_URL)).rstrip('/')
        self.document_filters = paper_ids or []
        self.questions_per_paper = questions_per_paper
        self.skip_download = skip_download
        self.skip_ingest = skip_ingest
        self.dataset_path = Path(dataset_path) if dataset_path else None

        self.eval_dir = Path(__file__).resolve().parent
        self.repo_root = self.eval_dir.parent.parent
        self.results_dir = Path(output_dir) if output_dir else self.eval_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.bundled_dataset_candidates = [self.eval_dir / name for name in BUNDLED_DATASET_FILENAMES]
        self.bundled_document_dirs = [self.repo_root / name for name in BUNDLED_DOCUMENT_DIRNAMES]

        self.api_key = os.getenv('YAR_API_KEY')

    def _request_headers(self) -> dict[str, str]:
        return {'X-API-Key': self.api_key} if self.api_key else {}

    async def check_yar_health(self) -> bool:
        """Check if YAR API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f'{self.rag_url}/health')
                response.raise_for_status()
            logger.info('YAR API accessible at %s', self.rag_url)
            return True
        except Exception as e:
            logger.error('Cannot connect to YAR API: %s', e)
            return False

    def resolve_source_documents(self) -> list[Path]:
        """Resolve bundled source documents that match the current harness configuration."""
        if self.skip_download:
            logger.info('Skipping deprecated download step; using local bundled documents')

        for candidate_dir in self.bundled_document_dirs:
            if not candidate_dir.exists():
                continue

            files = sorted(
                path
                for path in candidate_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {'.txt', '.md'}
            )
            if not files:
                continue

            if not self.document_filters:
                logger.info('Using bundled documents from %s', candidate_dir)
                return files

            filtered_files = [
                path
                for path in files
                if any(token in path.stem or token in path.name for token in self.document_filters)
            ]
            if filtered_files:
                logger.info('Using %d filtered bundled documents from %s', len(filtered_files), candidate_dir)
                return filtered_files

        filter_display = ', '.join(self.document_filters) if self.document_filters else '<all bundled documents>'
        candidate_display = ', '.join(str(path) for path in self.bundled_document_dirs)
        raise FileNotFoundError(
            'No bundled source documents are available for ingestion. '
            f'Filters: {filter_display}. Checked: {candidate_display}. '
            'Either add bundled sample documents or re-run with --skip-ingest and an already prepared YAR instance.'
        )

    async def clear_existing_data(self) -> bool:
        """Clear existing documents in YAR using the supported clear endpoint."""
        logger.info('Clearing existing data...')
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.delete(
                    f'{self.rag_url}/documents',
                    headers=self._request_headers(),
                )
                response.raise_for_status()
                payload = response.json()

            status = payload.get('status', 'unknown')
            message = payload.get('message', '')
            logger.info('Clear documents status=%s message=%s', status, message)
            return status in {'success', 'partial_success'}
        except Exception as e:
            logger.warning('Could not clear data: %s', e)
            return False

    async def ingest_documents(self, document_paths: list[Path]) -> str | None:
        """Ingest bundled text documents into YAR using the current texts endpoint."""
        if self.skip_ingest:
            logger.info('Document ingestion skipped (--skip-ingest)')
            return None

        if not document_paths:
            raise FileNotFoundError('No bundled source documents were resolved for ingestion')

        logger.info('STEP 1: Ingest Bundled Documents into YAR')

        texts: list[str] = []
        sources: list[str] = []
        for path in document_paths:
            content = path.read_text(encoding='utf-8').strip()
            if not content:
                logger.warning('Skipping empty bundled document: %s', path)
                continue
            texts.append(content)
            sources.append(path.name)

        if not texts:
            raise ValueError('Bundled source documents were found but all resolved documents were empty')

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f'{self.rag_url}/documents/texts',
                json={'texts': texts, 'file_sources': sources},
                headers=self._request_headers(),
            )
            response.raise_for_status()
            payload = response.json()

        status = payload.get('status', 'unknown')
        track_id = payload.get('track_id')
        logger.info('Ingest response status=%s track_id=%s', status, track_id)

        if status not in {'success', 'duplicated'}:
            raise RuntimeError(f'Unexpected ingest status: {status}')
        if not track_id:
            raise RuntimeError('Document ingest did not return a track_id for status polling')

        return track_id

    async def wait_for_processing(self, track_id: str) -> bool:
        """Wait for documents associated with the given track ID to finish processing."""
        logger.info('STEP 2: Wait for Document Processing')
        start_time = time.time()
        has_seen_documents = False

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() - start_time < MAX_WAIT_SECONDS:
                try:
                    response = await client.get(
                        f'{self.rag_url}/documents/track_status/{track_id}',
                        headers=self._request_headers(),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    total_count = int(payload.get('total_count', 0))
                    status_summary = payload.get('status_summary', {})

                    if total_count > 0:
                        has_seen_documents = True

                    pending = int(status_summary.get('pending', 0))
                    processing = int(status_summary.get('processing', 0))
                    processed = int(status_summary.get('processed', 0))
                    failed = int(status_summary.get('failed', 0))

                    elapsed = int(time.time() - start_time)
                    logger.info(
                        '[%ss] Track %s -> total=%s processed=%s failed=%s pending=%s processing=%s',
                        elapsed,
                        track_id,
                        total_count,
                        processed,
                        failed,
                        pending,
                        processing,
                    )

                    if has_seen_documents and pending == 0 and processing == 0:
                        if processed == 0 and failed > 0:
                            logger.error('All tracked documents failed processing for track_id=%s', track_id)
                            return False
                        if failed > 0:
                            logger.warning('Some tracked documents failed processing for track_id=%s', track_id)
                        logger.info('All tracked documents finished processing')
                        return True
                except Exception as e:
                    logger.warning('Track status check failed for %s: %s', track_id, e)

                await asyncio.sleep(POLL_INTERVAL_SECONDS)

        logger.error('Timeout waiting for document processing for track_id=%s', track_id)
        return False

    def generate_dataset(self) -> Path:
        """Resolve an explicit or bundled dataset without importing missing generators."""
        logger.info('STEP 3: Resolve Evaluation Dataset')
        dataset_path = resolve_dataset_path(self.dataset_path, self.bundled_dataset_candidates)
        logger.info('Using evaluation dataset: %s', dataset_path)
        return dataset_path

    async def run_ragas_evaluation(self, dataset_path: Path) -> dict:
        """Run RAGAS evaluation."""
        logger.info('STEP 4: Run RAGAS Evaluation')

        from yar.evaluation.eval_rag_quality import RAGEvaluator

        evaluator = RAGEvaluator(
            test_dataset_path=str(dataset_path),
            rag_api_url=self.rag_url,
        )
        return await evaluator.run()

    async def run_full_pipeline(self) -> dict:
        """Run the complete E2E test pipeline."""
        logger.info('E2E RAGAS TEST HARNESS FOR YAR')
        logger.info('RAG URL:    %s', self.rag_url)
        logger.info('Filters:    %s', ', '.join(self.document_filters) if self.document_filters else '<all bundled>')
        logger.info('Results:    %s', self.results_dir)

        start_time = time.time()

        if not await self.check_yar_health():
            return {'error': 'YAR API not accessible'}

        if not self.skip_ingest:
            try:
                document_paths = self.resolve_source_documents()
                track_id = await self.ingest_documents(document_paths)
            except Exception as e:
                return {'error': str(e)}

            if track_id and not await self.wait_for_processing(track_id):
                return {'error': f'Document processing did not complete successfully for track_id={track_id}'}

        try:
            dataset_path = self.generate_dataset()
        except Exception as e:
            return {'error': str(e)}

        results = await self.run_ragas_evaluation(dataset_path)
        elapsed_time = time.time() - start_time

        summary = {
            'pipeline_completed_at': datetime.now().isoformat(),
            'total_elapsed_seconds': round(elapsed_time, 2),
            'document_filters': self.document_filters,
            'dataset_path': str(dataset_path),
            'ragas_results': results.get('benchmark_stats', {}),
        }

        summary_path = self.results_dir / f'e2e_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info('E2E PIPELINE COMPLETE')
        logger.info('Total time: %.1f seconds', elapsed_time)
        logger.info('Summary saved: %s', summary_path)

        return summary


async def run_ab_test(
    harness_config: dict,
    clear_between_runs: bool = True,
) -> dict:
    """Run A/B test comparing with/without orphan connections."""
    logger.info('A/B TEST: WITH vs WITHOUT ORPHAN CONNECTIONS')

    results = {}

    logger.info('[A] Running WITHOUT orphan connections...')
    os.environ['AUTO_CONNECT_ORPHANS'] = 'false'
    harness_a = E2ETestHarness(**harness_config)
    results['without_orphans'] = await harness_a.run_full_pipeline()

    if clear_between_runs:
        await harness_a.clear_existing_data()

    logger.info('[B] Running WITH orphan connections...')
    os.environ['AUTO_CONNECT_ORPHANS'] = 'true'
    harness_b = E2ETestHarness(**harness_config)
    results['with_orphans'] = await harness_b.run_full_pipeline()

    logger.info('A/B COMPARISON')
    a_stats = results['without_orphans'].get('ragas_results', {}).get('average_metrics', {})
    b_stats = results['with_orphans'].get('ragas_results', {}).get('average_metrics', {})

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'without_orphans': a_stats,
        'with_orphans': b_stats,
        'improvement': {},
    }

    for metric in ['faithfulness', 'answer_relevance', 'context_recall', 'context_precision', 'ragas_score']:
        a_val = a_stats.get(metric, 0)
        b_val = b_stats.get(metric, 0)
        diff = b_val - a_val
        pct = (diff / a_val * 100) if a_val > 0 else 0
        comparison['improvement'][metric] = {
            'absolute': round(diff, 4),
            'percent': round(pct, 2),
        }
        status = 'UP' if diff > 0 else ('DOWN' if diff < 0 else '~')
        logger.info('%-20s A: %.4f  B: %.4f  [%s] %+0.1f%%', metric, a_val, b_val, status, pct)

    ragas_improvement = comparison['improvement'].get('ragas_score', {}).get('percent', 0)
    if ragas_improvement > 5:
        verdict = 'ORPHAN CONNECTIONS IMPROVE QUALITY'
    elif ragas_improvement < -5:
        verdict = 'ORPHAN CONNECTIONS DEGRADE QUALITY'
    else:
        verdict = 'NO SIGNIFICANT DIFFERENCE'

    comparison['verdict'] = verdict
    logger.info('VERDICT: %s', verdict)

    comp_path = harness_a.results_dir / f'ab_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(comp_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    logger.info('Comparison saved: %s', comp_path)

    return comparison


async def main():
    parser = argparse.ArgumentParser(
        description='E2E RAGAS Test Harness for YAR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full E2E test using bundled sample documents + bundled dataset
  python yar/evaluation/e2e_test_harness.py

  # A/B test (with/without orphan connections)
  python yar/evaluation/e2e_test_harness.py --ab-test

  # Restrict ingestion to bundled files matching these tokens
  python yar/evaluation/e2e_test_harness.py --papers yar,architecture

  # Use an explicit dataset
  python yar/evaluation/e2e_test_harness.py --dataset yar/evaluation/sample_dataset.json
        """,
    )

    parser.add_argument(
        '--rag-url',
        '-r',
        type=str,
        default=None,
        help=f'YAR API URL (default: {DEFAULT_RAG_URL})',
    )
    parser.add_argument(
        '--papers',
        '-p',
        type=str,
        default=None,
        help='Comma-separated filename/stem filters for bundled source documents',
    )
    parser.add_argument(
        '--questions',
        '-q',
        type=int,
        default=5,
        help='Retained for CLI compatibility; bundled datasets control actual question count',
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Retained for CLI compatibility; bundled local documents are used instead of downloads',
    )
    parser.add_argument(
        '--skip-ingest',
        action='store_true',
        help='Skip document ingestion (use existing data already loaded in YAR)',
    )
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        default=None,
        help='Path to existing Q&A dataset (defaults to bundled evaluation datasets)',
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default=None,
        help='Output directory for results',
    )
    parser.add_argument(
        '--ab-test',
        action='store_true',
        help='Run A/B test comparing with/without orphan connections',
    )

    args = parser.parse_args()
    paper_ids = [p.strip() for p in args.papers.split(',')] if args.papers else None

    harness_config = {
        'rag_url': args.rag_url,
        'paper_ids': paper_ids,
        'questions_per_paper': args.questions,
        'skip_download': args.skip_download,
        'skip_ingest': args.skip_ingest,
        'dataset_path': args.dataset,
        'output_dir': args.output_dir,
    }

    if args.ab_test:
        await run_ab_test(harness_config)
    else:
        harness = E2ETestHarness(**harness_config)
        await harness.run_full_pipeline()


if __name__ == '__main__':
    asyncio.run(main())
