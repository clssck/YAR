from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yar.evaluation.qa_eval_common import REPO_ROOT, RESULTS_DIR
from yar.operate import chunking_by_semantic
from yar.utils import logger

load_dotenv(REPO_ROOT / '.env', override=False)

DEFAULT_PROJECT = 'ragas_generated_real_source_docs'
DEFAULT_CONTEXT = (
    'Generate realistic, answerable RAG evaluation questions from the provided source chunks. '
    'Questions should be independent of the original hand-written benchmark and must be answerable '
    'from the reference contexts.'
)


@dataclass(frozen=True)
class SourceChunk:
    """A generated-testset source chunk with enough metadata for provenance."""

    content: str
    source_document: str
    chunk_order_index: int


def _repo_relative(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve(strict=False)))
    except ValueError:
        return str(resolved)


def _resolve_input_path(raw_path: str) -> Path:
    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return expanded
    return REPO_ROOT / expanded


def _unique_preserving_order(values: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for value in values:
        key = str(value.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _coerce_source_documents(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str) and item.strip()]
    return []


def _dataset_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get('test_cases'), list):
        return [case for case in payload['test_cases'] if isinstance(case, dict)]
    if isinstance(payload.get('qa_pairs'), list):
        return [case for case in payload['qa_pairs'] if isinstance(case, dict)]
    raise ValueError('Expected dataset with test_cases or qa_pairs root key')


def collect_source_documents(
    *,
    source_dataset: Path | None,
    source_documents: Sequence[str],
) -> list[Path]:
    """Collect real source document paths from explicit CLI values and/or a dataset."""
    collected: list[Path] = []
    if source_dataset is not None:
        with source_dataset.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        for case in _dataset_cases(payload):
            collected.extend(
                _resolve_input_path(path) for path in _coerce_source_documents(case.get('source_documents'))
            )

    for raw_path in source_documents:
        path = _resolve_input_path(raw_path)
        if any(char in raw_path for char in '*?['):
            parent = path.parent if str(path.parent) else REPO_ROOT
            collected.extend(parent.glob(path.name))
        elif path.is_dir():
            collected.extend(child for child in path.rglob('*') if child.is_file())
        else:
            collected.append(path)

    return _unique_preserving_order(collected)


def select_source_documents(paths: Sequence[Path], *, max_source_docs: int | None, seed: int | None) -> list[Path]:
    if max_source_docs is None or max_source_docs >= len(paths):
        return list(paths)
    if max_source_docs <= 0:
        raise ValueError('--max-source-docs must be greater than zero when provided')
    sampled = list(paths)
    random.Random(seed).shuffle(sampled)
    return sampled[:max_source_docs]


def _read_source_document_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError(
            f'Source document is not UTF-8 text after removing binary document extraction: {_repo_relative(path)}. '
            'Provide a processed Markdown/text artifact instead.'
        ) from exc
    if '\x00' in content:
        raise ValueError(
            f'Source document appears to be binary after removing binary document extraction: {_repo_relative(path)}. '
            'Provide a processed Markdown/text artifact instead.'
        )
    return content


def extract_source_chunks(
    paths: Sequence[Path],
    *,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
    max_chunks_per_doc: int | None,
    max_chunks: int | None,
    min_chunk_chars: int,
) -> list[SourceChunk]:
    """Read UTF-8 source artifacts into chunks for RAGAS testset generation."""
    _ = chunk_overlap_token_size
    source_chunks: list[SourceChunk] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'Source document not found: {_repo_relative(path)}')
        content = _read_source_document_text(path)
        chunk_dicts = chunking_by_semantic(
            content,
            chunk_token_size=chunk_token_size,
        )
        if max_chunks_per_doc is not None:
            chunk_dicts = chunk_dicts[:max_chunks_per_doc]
        for chunk in chunk_dicts:
            content = str(chunk.get('content') or '').strip()
            if len(content) < min_chunk_chars:
                continue
            source_chunks.append(
                SourceChunk(
                    content=content,
                    source_document=_repo_relative(path),
                    chunk_order_index=int(chunk.get('chunk_order_index') or 0),
                )
            )
            if max_chunks is not None and len(source_chunks) >= max_chunks:
                return source_chunks
    return source_chunks


def _normalize_for_match(value: Any) -> str:
    return ' '.join(str(value or '').casefold().split())


def infer_source_documents(reference_contexts: Sequence[str], source_chunks: Sequence[SourceChunk]) -> list[str]:
    """Infer source documents for generated samples by matching contexts to input chunks."""
    matched: list[str] = []
    normalized_chunks = [(_normalize_for_match(chunk.content), chunk.source_document) for chunk in source_chunks]
    for context in reference_contexts:
        normalized_context = _normalize_for_match(context)
        if not normalized_context:
            continue
        for normalized_chunk, source_document in normalized_chunks:
            if normalized_context in normalized_chunk or normalized_chunk in normalized_context:
                matched.append(source_document)
                break
    return sorted(set(matched))


def build_eval_cases_from_testset_rows(
    rows: Sequence[dict[str, Any]],
    *,
    source_chunks: Sequence[SourceChunk],
    project: str,
    retrieval_mode: str,
) -> list[dict[str, Any]]:
    """Convert RAGAS testset rows to eval_rag_quality.py's JSON schema."""
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        question = str(row.get('user_input') or '').strip()
        reference = str(row.get('reference') or '').strip()
        reference_contexts = [
            str(context).strip() for context in row.get('reference_contexts') or [] if str(context).strip()
        ]
        if not question or not reference:
            logger.warning('Skipping generated row %s without question/reference', index)
            continue

        source_documents = infer_source_documents(reference_contexts, source_chunks)
        case: dict[str, Any] = {
            'id': f'ragas-generated-{index:03d}',
            'question': question,
            'ground_truth': reference,
            'context_reference': reference,
            'retrieval_query': question,
            'retrieval_mode': retrieval_mode,
            'source_documents': source_documents,
            'project': project,
            'generated_by': 'ragas.testset.TestsetGenerator',
            'synthesizer_name': row.get('synthesizer_name') or '',
        }
        if reference_contexts:
            case['reference_contexts'] = reference_contexts
        if row.get('persona_name'):
            case['persona_name'] = row['persona_name']
        if row.get('query_style'):
            case['query_style'] = row['query_style']
        if row.get('query_length'):
            case['query_length'] = row['query_length']
        if not source_documents:
            case['comments'] = 'Generated source document could not be inferred from reference contexts.'
        cases.append(case)
    return cases


def _default_output_path(prefix: str, suffix: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f'{prefix}_{timestamp}{suffix}'


def write_generated_dataset(
    *,
    output_path: Path,
    test_cases: Sequence[dict[str, Any]],
    source_documents: Sequence[Path],
    source_chunks: Sequence[SourceChunk],
    testset_size: int,
    llm_model: str,
    embedding_model: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata': {
            'generated': True,
            'generator': 'ragas.testset.TestsetGenerator.generate_with_chunks',
            'created_at': datetime.now().isoformat(),
            'warning': 'Generated questions are for an impartial supplemental track; do not mix silently with the canonical real QA benchmark.',
            'requested_testset_size': testset_size,
            'emitted_test_cases': len(test_cases),
            'llm_model': llm_model,
            'embedding_model': embedding_model,
            'source_documents': [_repo_relative(path) for path in source_documents],
            'source_chunk_count': len(source_chunks),
        },
        'test_cases': list(test_cases),
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def _build_ragas_generator(llm_context: str):
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.testset.synthesizers.generate import TestsetGenerator
    except ImportError as exc:
        raise ImportError(
            'RAGAS testset generation dependencies are missing. Install with `pip install -e .[evaluation]`.'
        ) from exc

    from yar.evaluation.eval_rag_quality import OpenAICompatibleEmbeddings

    eval_llm_api_key = os.getenv('EVAL_LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not eval_llm_api_key:
        raise OSError('EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY is required for RAGAS question generation.')
    eval_embedding_api_key = (
        os.getenv('EVAL_EMBEDDING_BINDING_API_KEY')
        or os.getenv('EVAL_LLM_BINDING_API_KEY')
        or os.getenv('OPENAI_API_KEY')
    )
    eval_model = os.getenv('EVAL_LLM_MODEL', 'gpt-4o-mini')
    eval_embedding_model = os.getenv('EVAL_EMBEDDING_MODEL', 'text-embedding-3-large')
    eval_llm_base_url = os.getenv('EVAL_LLM_BINDING_HOST')
    eval_embedding_base_url = os.getenv('EVAL_EMBEDDING_BINDING_HOST')
    max_retries = int(os.getenv('EVAL_LLM_MAX_RETRIES', '5'))
    timeout = int(os.getenv('EVAL_LLM_TIMEOUT', '180'))

    llm_kwargs: dict[str, Any] = {
        'model': eval_model,
        'api_key': eval_llm_api_key,
        'max_retries': max_retries,
        'request_timeout': timeout,
        'temperature': float(os.getenv('EVAL_LLM_TEMPERATURE', '0')),
    }
    if eval_llm_base_url:
        llm_kwargs['base_url'] = eval_llm_base_url
    base_llm = ChatOpenAI(**llm_kwargs)

    if eval_embedding_base_url:
        base_embeddings = OpenAICompatibleEmbeddings(
            model=eval_embedding_model,
            api_key=eval_embedding_api_key,
            base_url=eval_embedding_base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
    else:
        base_embeddings = OpenAIEmbeddings(model=eval_embedding_model, api_key=eval_embedding_api_key)

    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True),
        embedding_model=LangchainEmbeddingsWrapper(base_embeddings),
        llm_context=llm_context,
    )
    return generator, eval_model, eval_embedding_model


def generate_testset_rows(
    *,
    source_chunks: Sequence[SourceChunk],
    testset_size: int,
    llm_context: str,
    with_debugging_logs: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError('langchain-core is required for RAGAS question generation.') from exc

    documents = [
        Document(
            page_content=chunk.content,
            metadata={
                'source': chunk.source_document,
                'chunk_order_index': chunk.chunk_order_index,
            },
        )
        for chunk in source_chunks
    ]
    generator, llm_model, embedding_model = _build_ragas_generator(llm_context)
    testset = generator.generate_with_chunks(
        chunks=documents,
        testset_size=testset_size,
        with_debugging_logs=with_debugging_logs,
    )
    return testset.to_list(), llm_model, embedding_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate a separate RAGAS question dataset from real source documents/chunks.',
    )
    parser.add_argument(
        '--source-dataset', type=Path, help='Existing eval JSON dataset whose source_documents should seed generation.'
    )
    parser.add_argument(
        '--source-document',
        action='append',
        default=[],
        help='Source file, directory, or glob. May be passed multiple times.',
    )
    parser.add_argument('--testset-size', type=int, default=10, help='Number of generated questions to request.')
    parser.add_argument(
        '--max-source-docs', type=int, default=None, help='Optional random cap on source documents before extraction.'
    )
    parser.add_argument('--max-chunks', type=int, default=40, help='Maximum total extracted chunks passed to RAGAS.')
    parser.add_argument(
        '--max-chunks-per-doc', type=int, default=4, help='Maximum extracted chunks per source document.'
    )
    parser.add_argument(
        '--min-chunk-chars', type=int, default=200, help='Drop extracted chunks shorter than this many characters.'
    )
    parser.add_argument('--chunk-token-size', type=int, default=1200, help='Local semantic chunk token size.')
    parser.add_argument(
        '--chunk-overlap-token-size', type=int, default=100, help='Reserved for pipeline compatibility.'
    )
    parser.add_argument('--retrieval-mode', default='naive', help='retrieval_mode written into generated eval cases.')
    parser.add_argument('--project', default=DEFAULT_PROJECT, help='project label written into generated eval cases.')
    parser.add_argument(
        '--llm-context', default=DEFAULT_CONTEXT, help='Context instruction passed to RAGAS TestsetGenerator.'
    )
    parser.add_argument(
        '--seed', type=int, default=13, help='Random seed used when --max-source-docs samples documents.'
    )
    parser.add_argument('--output-json', type=Path, default=None, help='Eval-compatible JSON output path.')
    parser.add_argument(
        '--raw-output-json', type=Path, default=None, help='Optional raw RAGAS testset row output path.'
    )
    parser.add_argument('--debug', action='store_true', help='Enable RAGAS debugging logs during generation.')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.testset_size <= 0:
        raise ValueError('--testset-size must be greater than zero')
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError('--max-chunks must be greater than zero when provided')
    if args.max_chunks_per_doc is not None and args.max_chunks_per_doc <= 0:
        raise ValueError('--max-chunks-per-doc must be greater than zero when provided')

    source_dataset = args.source_dataset.resolve(strict=False) if args.source_dataset else None
    source_documents = collect_source_documents(
        source_dataset=source_dataset,
        source_documents=args.source_document,
    )
    if not source_documents:
        raise ValueError('Provide --source-dataset and/or at least one --source-document.')
    selected_documents = select_source_documents(
        source_documents,
        max_source_docs=args.max_source_docs,
        seed=args.seed,
    )
    source_chunks = extract_source_chunks(
        selected_documents,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        max_chunks_per_doc=args.max_chunks_per_doc,
        max_chunks=args.max_chunks,
        min_chunk_chars=args.min_chunk_chars,
    )
    if not source_chunks:
        raise ValueError('No usable source chunks were extracted from the selected documents.')

    logger.info(
        'Generating %s RAGAS questions from %s chunks across %s source documents',
        args.testset_size,
        len(source_chunks),
        len(selected_documents),
    )
    raw_rows, llm_model, embedding_model = generate_testset_rows(
        source_chunks=source_chunks,
        testset_size=args.testset_size,
        llm_context=args.llm_context,
        with_debugging_logs=args.debug,
    )
    test_cases = build_eval_cases_from_testset_rows(
        raw_rows,
        source_chunks=source_chunks,
        project=args.project,
        retrieval_mode=args.retrieval_mode,
    )

    output_json = args.output_json or _default_output_path('ragas_generated_questions', '.json')
    write_generated_dataset(
        output_path=output_json,
        test_cases=test_cases,
        source_documents=selected_documents,
        source_chunks=source_chunks,
        testset_size=args.testset_size,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )
    logger.info('Wrote eval-compatible generated dataset: %s', output_json)

    if args.raw_output_json:
        args.raw_output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_output_json.open('w', encoding='utf-8') as handle:
            json.dump({'rows': raw_rows}, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        logger.info('Wrote raw RAGAS generated rows: %s', args.raw_output_json)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
