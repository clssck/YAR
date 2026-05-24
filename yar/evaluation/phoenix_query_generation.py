"""Generate intent-typed evaluation queries from the corpus.

The hand-rolled baseline (5 queries, one per intent) was too small to draw
real conclusions about prompt or pipeline changes. This module fixes that by
sampling source documents from S3, asking an LLM to produce queries grounded
in the actual content, and uploading the result as a Phoenix dataset.

Five intent types are generated, each with a `should_refuse` flag that the
refusal evaluator can compare against:

* ``factual_lookup`` — single-fact question whose answer is directly in one
  passage. ``should_refuse`` is False.
* ``enumeration`` — list/count question that requires gathering items from
  one passage. ``should_refuse`` is False.
* ``comparison`` — cross-document question that requires synthesizing two
  passages from different docs. ``should_refuse`` is False.
* ``out_of_scope`` — question on the same broad topic as the passage but
  whose specific answer requires domain knowledge outside the corpus.
  ``should_refuse`` is True. Catches mechanism/biology/history bait.
* ``mechanism_bait`` — explicitly asks for a mechanism/process the corpus
  does not describe. ``should_refuse`` is True. Sharper version of
  ``out_of_scope`` for the well-known training-data hallucination case.

CLI::

    python -m yar.evaluation.phoenix_query_generation \
        --server-url http://localhost:9621 \
        --judge-model tuna --judge-provider openai \
        --workspace default \
        --queries-per-intent 5 \
        --dataset-name yar-generated-baseline-2026-05-08 \
        --output /tmp/yar_generated_queries.json

Generation cost: ~1 LLM call per generated query plus 1-2 calls for
cross-document comparisons. ~30 queries ≈ 35 LLM calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation prompts
# ---------------------------------------------------------------------------


_FACTUAL_PROMPT = """You are an evaluation-set author for a RAG system over
the user's internal documents.

Read the PASSAGE. Generate ONE specific question whose precise answer is
stated in this passage and only this passage. Avoid questions that could be
answered from general knowledge — the question must require reading THIS
passage to answer.
The question must be self-contained for retrieval. The downstream RAG system
receives only the question, not the passage metadata, so include the document
title or a distinctive title/topic anchor from the passage. Do NOT ask about
"the passage", "the document", or a generic section such as "Best Practice"
unless the same question also names the source document/topic precisely.

Output strict JSON with these keys: ``query`` (string), ``expected_answer``
(string, the actual answer drawn verbatim or near-verbatim from the
passage), ``why_specific`` (string, one sentence on why this question
requires the passage). No prose, no markdown fences, no preamble.

DOCUMENT TITLE: {document_title}

PASSAGE:
{passage}
"""


_ENUMERATION_PROMPT = """You are an evaluation-set author for a RAG system.

Read the PASSAGE. Generate ONE list-style question (e.g. "What are the
risks discussed?", "List the audit findings", "Which sites are
mentioned?") whose answer is a set of items explicitly enumerated in the
passage. The passage must contain at least 2 items that satisfy the
question.


The question must be self-contained for retrieval. The downstream RAG system
receives only the question, not the passage metadata, so include the document
title or a distinctive title/topic anchor from the passage. Do NOT write
context-dependent questions such as "List the facilitators mentioned in the
passage"; instead name the session, document, project, or section that makes
the requested list retrievable.
Output strict JSON with these keys: ``query`` (string), ``expected_items``
(list of strings — the items the answer should enumerate). No prose, no
markdown fences.

DOCUMENT TITLE: {document_title}

PASSAGE:
{passage}
"""


_COMPARISON_PROMPT = """You are an evaluation-set author for a RAG system.

Read the TWO PASSAGES from different source documents. Generate ONE
comparison question that requires synthesizing material from BOTH passages
to answer.

CRITICAL constraints on the question:
* Use the actual document titles (provided below) or topical descriptors
  drawn from the passages. Do NOT use abstract labels like "Document A",
  "the first document", "Source 1" — the downstream RAG system has no idea
  what those refer to. Write the question as a real human reader of the
  corpus would phrase it.
* The question must be answerable from the union of the two passages but
  not from either alone (otherwise it isn't a comparison).
* Anchor on concrete entities (drug names, projects, sites, processes,
  authors) rather than meta-document labels.
* Do NOT invent product-class, modality, technology, or methodology
  labels for the documents that are not stated in the passages
  themselves. If the passage describes an AAV gene therapy, do not
  re-label it as "mRNA gene therapy"; if the passage describes a
  peptide drug, do not re-label it as "small-molecule peptide"; if a
  passage describes a partnership-conflict-management session, do not
  re-label it as "regulatory response strategy". Use only labels that
  appear verbatim or near-verbatim in the passage you are labeling.
* The comparison axis you ask about (the thing the two docs are
  compared on) MUST be a topic that BOTH passages explicitly cover.
  Do not ask "How do A and B differ in their X?" when only passage A
  covers X; that produces a malformed comparison the downstream RAG
  system cannot answer. Before writing the question, check that each
  axis you intend to ask about has at least one supporting sentence in
  BOTH passages.
* Do not compare presence versus absence. If passage A covers impurity
  testing, assay validation, or FDA feedback and passage B only covers
  project-freeze timelines, SKIP. A good comparison needs positive evidence
  about the same axis in both passages.
* Do not apply a shared product class or modality label to both passages
  unless both passages state that label. For example, do not call both items
  "peptide-based" unless both passages explicitly say peptide.
* If the two passages share no common dimension that supports a
  meaningful comparison (e.g. one is about partner-conflict
  governance, the other is about FDA CMC dossier interactions —
  topically disjoint), abandon this pair: output the literal token
  ``SKIP`` instead of JSON. The pipeline retries with a different
  pair.

Output strict JSON with these keys: ``query`` (string),
``expected_axes`` (list of strings — the comparison axes the answer should
cover, e.g. ``["scope", "timeline", "stakeholders"]``). No prose, no
markdown fences.

DOCUMENT A TITLE: {doc_a_title}
DOCUMENT A PASSAGE:
{passage_a}

DOCUMENT B TITLE: {doc_b_title}
DOCUMENT B PASSAGE:
{passage_b}
"""


_OUT_OF_SCOPE_PROMPT = """You are an evaluation-set author for a RAG system.

Read the PASSAGE. Generate ONE question that is on the same broad TOPIC as
this passage (so a retriever will surface it) but whose specific answer is
truly NOT in the passage and would require domain knowledge or sources
outside the document. The correct system behaviour for this question is to
REFUSE ("the documents do not contain that information").

CRITICAL constraints:
* The answer to the question must NOT be derivable from anything in the
  passage — not even partially. Before finalizing the question, mentally
  scan the passage one more time. If you can find ANY sentence that
  partially answers the question, REJECT this question and pick a
  different angle.
* Do NOT ask about regulatory milestones (FDA Type C, IND submission, lot
  release criteria, specifications) when the passage already mentions
  those events — even tangentially. Those become answerable.
* Prefer questions about: scientific theory the doc doesn't cover (e.g.
  pharmacology, statistical analysis methodology), historical context
  (when was X discovered, who funded the original research), or external
  comparisons (how does X compare to a competitor not mentioned).

Examples of good ``out_of_scope`` questions when the passage describes a
drug's manufacturing site: "What is the drug's pharmacokinetic half-life?",
"What is the active site geometry of X?", "Who were the original academic
discoverers of this drug class?".

Examples of BAD ``out_of_scope`` questions (these are answerable from the
passage even if the framing differs):
* "What is the FDA classification?" if any FDA classification is mentioned.
* "What are the Phase 3 specs?" if Phase 1/2 specs are listed and the
  passage says they apply going forward.
* "How does the risk review process work?" if the passage describes the
  process steps (even at a high level).

Output strict JSON with these keys: ``query`` (string), ``why_out_of_scope``
(string, one sentence on what is missing from the passage that the question
would require). No prose, no markdown fences.

DOCUMENT TITLE: {document_title}

PASSAGE:
{passage}
"""


_MECHANISM_BAIT_PROMPT = """You are an evaluation-set author for a RAG system.

Read the PASSAGE. Generate ONE question that explicitly asks about the
CHEMISTRY, BIOLOGY, or INSTRUMENTATION-LEVEL MECHANISM of an entity
mentioned in the passage — even though the passage only describes the
entity in regulatory, manufacturing, or programmatic context (i.e. NOT
its underlying mechanism).

The question MUST require knowledge that is genuinely absent from the
passage. Specifically:
* For an ASSAY mentioned in the passage: ask about its detection
  principle (NOT its release-criteria status, sample-prep workflow, or
  pass/fail result — those are usually in the passage).
* For a DRUG / PROTEIN: ask about molecular target, binding mechanism,
  enzymatic step, or pathway (NOT its development history, partner list,
  or trial milestones).
* For a PROCESS NAME mentioned in the passage: ask about the *physical
  chemistry* of the process step itself (e.g. "what chemistry occurs
  during heat virus inactivation"), NOT the overall workflow.

Phrasing must use: "what is the molecular target of X", "what binding
chemistry does X exploit", "describe the enzymatic mechanism of X",
"what is the detection principle behind <assay>". Avoid vague phrasing
like "how does X work in this context" — that often becomes answerable.

Before finalizing the question, scan the passage once more. If the
passage describes the process steps even at a high level (e.g. "we run
assay X to confirm criterion Y"), that is enough material for the
system to give a process answer — REJECT and ask about something
deeper.

Output strict JSON with these keys: ``query`` (string), ``target_entity``
(string — the entity from the passage whose mechanism is being asked
about), ``why_should_refuse`` (string, one sentence on what the passage
covers vs what the question asks). No prose, no markdown fences.

DOCUMENT TITLE: {document_title}

PASSAGE:
{passage}
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    server_url: str = 'http://localhost:9621'
    workspace: str = 'default'
    api_key: str | None = None  # Optional X-API-Key header value.
    judge_provider: str = 'openai'
    judge_model: str = 'tuna'
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    queries_per_intent: int = 5
    intents: list[str] = field(
        default_factory=lambda: [
            'factual_lookup',
            'enumeration',
            'comparison',
            'out_of_scope',
            'mechanism_bait',
        ]
    )
    passage_chars: int = 3000
    passage_min_chars: int = 800
    seed: int = 13
    dataset_name: str | None = None
    dataset_description: str | None = None
    output_path: str | None = None
    phoenix_base_url: str | None = None
    phoenix_api_key: str | None = None
    source_suffix: str = '.canonical.md'
    canonical_suffix: str | None = None  # Legacy alias; prefer source_suffix.


# ---------------------------------------------------------------------------
# Corpus loading via S3 API
# ---------------------------------------------------------------------------


def _configured_source_suffix(config: GenerationConfig) -> str:
    return config.canonical_suffix or config.source_suffix


def _http_get_json(url: str, *, headers: dict[str, str], timeout: float = 30.0) -> Any:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _http_get_text(url: str, *, headers: dict[str, str], timeout: float = 60.0) -> str:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


def _yar_headers(config: GenerationConfig) -> dict[str, str]:
    headers: dict[str, str] = {'Accept': 'application/json'}
    if config.api_key:
        headers['X-API-Key'] = config.api_key
    return headers


def _list_s3_prefix(config: GenerationConfig, prefix: str) -> dict[str, Any]:
    return _http_get_json(
        f'{config.server_url}/s3/list?prefix={quote(prefix, safe="")}',
        headers=_yar_headers(config),
    )


def _fetch_s3_text(config: GenerationConfig, key: str) -> str:
    encoded = quote(key, safe='/')
    return _http_get_text(
        f'{config.server_url}/s3/content/{encoded}',
        headers=_yar_headers(config),
    )


@dataclass
class SourceDocument:
    """A single canonical-markdown source document."""

    doc_id: str
    file_path: str
    title: str
    content: str


def _strip_md_artifacts(text: str) -> str:
    """Drop page markers / boilerplate so passages read like prose."""
    text = re.sub(r'<!--\s*PAGE\s*\d+\s*-->', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[.+?\]\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _document_title(file_path: str) -> str:
    leaf = file_path.rsplit('/', 1)[-1]
    leaf = re.sub(r'\.canonical\.md$|\.processed\.md$|\.pdf$|\.md$|\.json$', '', leaf, flags=re.IGNORECASE)
    return leaf.replace('_', ' ').strip()


def _load_corpus(config: GenerationConfig) -> list[SourceDocument]:
    """Walk uploaded S3 text artifacts under ``<workspace>/<doc>/`` and fetch their text."""
    docs: list[SourceDocument] = []
    source_suffix = _configured_source_suffix(config).lower()
    workspace_listing = _list_s3_prefix(config, f'{config.workspace}/')
    doc_prefixes = [folder for folder in workspace_listing.get('folders', []) if folder.endswith('/')]

    for doc_prefix in doc_prefixes:
        doc_id = doc_prefix.rstrip('/').rsplit('/', 1)[-1]
        doc_listing = _list_s3_prefix(config, doc_prefix)
        source_objects = [
            obj for obj in doc_listing.get('objects', []) if str(obj.get('key', '')).lower().endswith(source_suffix)
        ]
        if not source_objects:
            logger.warning('No %s artifact found under %s; skipping', source_suffix, doc_prefix)
            continue
        # Prefer the largest matching artifact (in case there are multiple).
        source_objects.sort(key=lambda obj: int(obj.get('size') or 0), reverse=True)
        key = str(source_objects[0]['key'])
        try:
            text = _fetch_s3_text(config, key)
        except Exception as exc:
            logger.warning('Failed to fetch %s: %s', key, exc)
            continue
        docs.append(
            SourceDocument(
                doc_id=doc_id,
                file_path=key,
                title=_document_title(key),
                content=_strip_md_artifacts(text),
            )
        )
    return docs


def _split_into_passages(doc: SourceDocument, *, max_chars: int, min_chars: int) -> list[str]:
    """Greedy paragraph-aware split into target-sized passages."""
    paragraphs = [chunk.strip() for chunk in re.split(r'\n{2,}', doc.content) if chunk.strip()]
    passages: list[str] = []
    buffer: list[str] = []
    buffer_chars = 0
    for paragraph in paragraphs:
        if buffer_chars + len(paragraph) + 2 > max_chars and buffer:
            passages.append('\n\n'.join(buffer))
            buffer = []
            buffer_chars = 0
        buffer.append(paragraph)
        buffer_chars += len(paragraph) + 2
    if buffer:
        passages.append('\n\n'.join(buffer))
    return [p for p in passages if len(p) >= min_chars]


# ---------------------------------------------------------------------------
# LLM call (lightweight client around OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------


def _llm_call(config: GenerationConfig, *, prompt: str, temperature: float = 0.7) -> str:
    model_lower = (config.judge_model or '').lower()
    # Route tuna (and other LiteLLM-only aliases) through the LiteLLM proxy by
    # default; route OpenAI-named models (gpt-*, o1, o3) through OpenAI unless
    # an explicit ``--judge-base-url`` overrides it.
    if config.judge_base_url:
        base_url = config.judge_base_url
    elif model_lower.startswith('gpt-') or model_lower.startswith('o1') or model_lower.startswith('o3'):
        base_url = os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    else:
        base_url = 'http://localhost:4000/v1'
    if config.judge_api_key:
        api_key = config.judge_api_key
    elif base_url.startswith('https://api.openai.com'):
        api_key = os.getenv('OPENAI_API_KEY') or ''
    else:
        api_key = 'sk-litellm-master-key'
    is_reasoning = model_lower.startswith('gpt-5') or model_lower.startswith('o1') or model_lower.startswith('o3')
    body: dict[str, Any] = {
        'model': config.judge_model,
        'messages': [
            {
                'role': 'system',
                'content': 'You write evaluation-set entries. Output strict JSON only. No markdown fences, no commentary.',
            },
            {'role': 'user', 'content': prompt},
        ],
    }
    if is_reasoning:
        # gpt-5 / o1 / o3 reject temperature != 1 and require max_completion_tokens.
        body['max_completion_tokens'] = 4000
    else:
        body['temperature'] = temperature
        body['max_tokens'] = 4000
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    with httpx.Client(timeout=120) as client:
        resp = client.post(f'{base_url.rstrip("/")}/chat/completions', json=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    return str(payload['choices'][0]['message']['content']).strip()


def _parse_json_lenient(text: str) -> Any:
    """Tolerate models that wrap JSON in markdown fences or add prose around it."""
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    # Try direct, then a fallback that pulls the first {...} object.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


_SELF_CONTAINED_QUERY_INTENTS = frozenset({'factual_lookup', 'enumeration'})
_TITLE_ANCHOR_STOPWORDS = frozenset(
    {
        'and',
        'best',
        'document',
        'documents',
        'final',
        'for',
        'implementation',
        'learned',
        'learning',
        'lesson',
        'lessons',
        'outcome',
        'practice',
        'project',
        'session',
        'the',
        'version',
    }
)


def _tokenize_title_anchor_text(text: str) -> set[str]:
    """Return normalized alphanumeric tokens, splitting simple CamelCase titles."""
    camel_spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return set(re.findall(r'[a-z0-9]+', camel_spaced.lower()))


def _source_anchor_terms(document_title: str) -> set[str]:
    """Extract source-title terms distinctive enough to make a query retrievable."""
    return {
        token
        for token in _tokenize_title_anchor_text(document_title)
        if token not in _TITLE_ANCHOR_STOPWORDS and (token.isdigit() or len(token) >= 4)
    }


def _query_mentions_source_anchor(query: str, document_title: str) -> bool:
    """Check that answer-expected generated queries name the source/topic anchor."""
    anchor_terms = _source_anchor_terms(document_title)
    if not anchor_terms:
        return True
    return bool(_tokenize_title_anchor_text(query) & anchor_terms)


_COMPARISON_AXIS_STOPWORDS = frozenset(
    {
        'approach',
        'approaches',
        'between',
        'compare',
        'compared',
        'comparison',
        'communication',
        'considerations',
        'controls',
        'described',
        'development',
        'data',
        'differ',
        'different',
        'expectations',
        'feedback',
        'impact',
        'impacts',
        'decision',
        'decisions',
        'management',
        'manufacturing',
        'plans',
        'planning',
        'practices',
        'options',
        'quality',
        'requirements',
        'respective',
        'roles',
        'scope',
        'stakeholder',
        'stakeholders',
        'strategy',
        'strategies',
        'systems',
        'team',
        'teams',
        'timeline',
        'topics',
        'timelines',
        'workload',
    }
)
_COMPARISON_TOPIC_STOPWORDS = frozenset(
    {
        *_COMPARISON_AXIS_STOPWORDS,
        'about',
        'after',
        'also',
        'around',
        'activity',
        'activities',
        'clinical',
        'closed',
        'description',
        'both',
        'document',
        'documents',
        'final',
        'lesson',
        'lessons',
        'does',
        'each',
        'llsession',
        'outcome',
        'passage',
        'passages',
        'information',
        'had',
        'has',
        'have',
        'phase',
        'process',
        'processes',
        'program',
        'programs',
        'project',
        'projects',
        'regulatory',
        'report',
        'review',
        'sanofi',
        'session',
        'source',
        'request',
        'requested',
        'sensitive',
        'should',
        'share',
        'shared',
        'sources',
        'study',
        'studies',
        'reported',
        'that',
        'their',
        'these',
        'those',
        'submission',
        'using',
        'were',
        'what',
        'when',
        'where',
        'which',
        'with',
        'would',
    }
)
_SHARED_LABEL_TERMS = frozenset({'gene therapy', 'peptide', 'protein', 'small molecule'})
_COMPARISON_FALLBACK_PREFERRED_TERMS = frozenset(
    {
        'agency',
        'agreement',
        'agreements',
        'alignment',
        'audit',
        'batch',
        'batches',
        'comparator',
        'conflict',
        'device',
        'devices',
        'dossier',
        'governance',
        'impurity',
        'license',
        'manufacturing',
        'partner',
        'partners',
        'pooling',
        'quality',
        'risk',
        'specifications',
        'stakeholders',
        'submission',
        'supply',
        'testing',
        'timeline',
        'transfer',
    }
)

_MECHANISM_BAIT_UNSAFE_TARGET_RE = re.compile(
    r'\b(?:technology\s+transfer|leader\s+role|process\s+performance\s+qualification|'
    r'ppq(?:\s+assay)?|governance|ways?\s+of\s+working)\b',
    re.IGNORECASE,
)


def _tokenize_comparison_text(text: str) -> set[str]:
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def _comparison_axis_terms(axis: str) -> set[str]:
    return {
        token
        for token in _tokenize_comparison_text(axis)
        if token not in _COMPARISON_AXIS_STOPWORDS and len(token) >= 4
    }


def _comparison_topic_terms(document_title: str, passage: str) -> set[str]:
    """Extract specific shared-topic terms for pre-filtering comparison pairs."""
    terms = _tokenize_comparison_text(f'{document_title} {passage}')
    return {
        token for token in terms if token not in _COMPARISON_TOPIC_STOPWORDS and len(token) >= 4 and not token.isdigit()
    }


def _comparison_pair_has_shared_topic(
    doc_a_title: str,
    passage_a: str,
    doc_b_title: str,
    passage_b: str,
) -> bool:
    """Reject passage pairs with no concrete shared topic before spending an LLM call."""
    terms_a = _comparison_topic_terms(doc_a_title, passage_a)
    terms_b = _comparison_topic_terms(doc_b_title, passage_b)
    return bool(terms_a & terms_b)


def _comparison_axes_supported_by_both(
    expected_axes: Any,
    passage_a: str,
    passage_b: str,
) -> bool:
    """Reject comparison outputs when any requested axis lacks support on either side."""
    if not isinstance(expected_axes, list) or not expected_axes:
        return False
    terms_a = _tokenize_comparison_text(passage_a)
    terms_b = _tokenize_comparison_text(passage_b)
    for raw_axis in expected_axes:
        axis_terms = _comparison_axis_terms(str(raw_axis))
        if not axis_terms or not axis_terms.intersection(terms_a, terms_b):
            return False
    return True


def _comparison_uses_supported_shared_labels(query: str, passage_a: str, passage_b: str) -> bool:
    normalized_a = passage_a.casefold()
    normalized_b = passage_b.casefold()
    for label in _SHARED_LABEL_TERMS:
        shared_label = re.compile(
            rf'\b(?:respective|both|their\s+respective)\b[^.?!]*\b{re.escape(label)}\b'
            rf'|\b{re.escape(label)}(?:-based)?\s+(?:drug\s+products|products|projects)\b',
            re.IGNORECASE,
        )
        if shared_label.search(query) and (label not in normalized_a or label not in normalized_b):
            return False
    return True


def _format_comparison_topic(term: str) -> str:
    if 2 <= len(term) <= 4:
        return term.upper()
    return term.replace('_', ' ')


def _fallback_side_anchor(document_title: str, passage: str, shared_terms: set[str]) -> str:
    side_terms = (
        _comparison_topic_terms(document_title, passage) & _COMPARISON_FALLBACK_PREFERRED_TERMS
    ) - shared_terms
    if side_terms:
        return _format_comparison_topic(max(side_terms, key=lambda term: (len(term), term)))
    title_terms = _source_anchor_terms(document_title) & _COMPARISON_FALLBACK_PREFERRED_TERMS
    if title_terms:
        return _format_comparison_topic(max(title_terms, key=lambda term: (len(term), term)))
    return ''


def _fallback_comparison_example(
    *,
    doc_a: SourceDocument,
    doc_b: SourceDocument,
    passage_a: str,
    passage_b: str,
) -> dict[str, Any] | None:
    """Build a conservative comparison from a concrete term present in both passages."""
    shared_terms = _comparison_topic_terms(doc_a.title, passage_a) & _comparison_topic_terms(doc_b.title, passage_b)
    preferred_terms = shared_terms & _COMPARISON_FALLBACK_PREFERRED_TERMS
    if not preferred_terms:
        return None
    topic = max(preferred_terms, key=lambda term: (len(term), term))
    topic_label = _format_comparison_topic(topic)
    anchor_a = _fallback_side_anchor(doc_a.title, passage_a, shared_terms)
    anchor_b = _fallback_side_anchor(doc_b.title, passage_b, shared_terms)
    source_hint = f' ({anchor_a})' if anchor_a else ''
    target_hint = f' ({anchor_b})' if anchor_b else ''
    return {
        'query': (
            f'How do "{doc_a.title}"{source_hint} and "{doc_b.title}"{target_hint} each discuss '
            f'{topic_label}, and what concrete actions, decisions, or risks does each source describe?'
        ),
        'intent': 'comparison',
        'should_refuse': False,
        'source_doc_ids': [doc_a.doc_id, doc_b.doc_id],
        'source_file_paths': [doc_a.file_path, doc_b.file_path],
        'document_titles': [doc_a.title, doc_b.title],
        'expected_axes': [topic_label],
        'passage_preview_a': passage_a[:300],
        'passage_preview_b': passage_b[:300],
    }


# ---------------------------------------------------------------------------
# Per-intent generators
# ---------------------------------------------------------------------------


def _gen_single_passage_intent(
    config: GenerationConfig,
    *,
    intent: str,
    prompt_template: str,
    doc: SourceDocument,
    passage: str,
    extra_meta_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    prompt = prompt_template.format(document_title=doc.title, passage=passage)
    try:
        response = _llm_call(config, prompt=prompt)
        parsed = _parse_json_lenient(response)
    except Exception as exc:
        logger.warning('%s generation failed: %s', intent, exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning('%s generation returned non-dict; skipping', intent)
        return None
    query = str(parsed.get('query') or '').strip()
    if not query:
        return None
    if intent in _SELF_CONTAINED_QUERY_INTENTS and not _query_mentions_source_anchor(query, doc.title):
        logger.info(
            '%s generation omitted source anchor for %s; skipping query: %s',
            intent,
            doc.title,
            query,
        )
        return None
    if intent == 'mechanism_bait':
        target_text = f'{query} {parsed.get("target_entity") or ""}'
        if _MECHANISM_BAIT_UNSAFE_TARGET_RE.search(target_text):
            logger.info(
                'mechanism-bait generation used an unsafe administrative or answerable target; skipping query: %s',
                query,
            )
            return None
    extra = {key: parsed.get(key) for key in extra_meta_keys if key in parsed}
    return {
        'query': query,
        'intent': intent,
        'should_refuse': intent in {'out_of_scope', 'mechanism_bait'},
        'source_doc_id': doc.doc_id,
        'source_file_path': doc.file_path,
        'document_title': doc.title,
        'passage_preview': passage[:400],
        **extra,
    }


def _gen_comparison(
    config: GenerationConfig,
    *,
    doc_a: SourceDocument,
    doc_b: SourceDocument,
    passage_a: str,
    passage_b: str,
) -> dict[str, Any] | None:
    fallback = _fallback_comparison_example(doc_a=doc_a, doc_b=doc_b, passage_a=passage_a, passage_b=passage_b)
    prompt = _COMPARISON_PROMPT.format(
        doc_a_title=doc_a.title,
        passage_a=passage_a,
        doc_b_title=doc_b.title,
        passage_b=passage_b,
    )
    try:
        response = _llm_call(config, prompt=prompt)
    except Exception as exc:
        logger.warning('comparison generation failed: %s', exc)
        return fallback
    if response.strip().upper().startswith('SKIP'):
        logger.info(
            'comparison generator skipped pair (topically disjoint): %s vs %s',
            doc_a.title[:40],
            doc_b.title[:40],
        )
        return fallback
    try:
        parsed = _parse_json_lenient(response)
    except Exception as exc:
        logger.warning('comparison JSON parse failed: %s', exc)
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    query = str(parsed.get('query') or '').strip()
    expected_axes = parsed.get('expected_axes') or []
    if not query:
        return fallback
    if not (_query_mentions_source_anchor(query, doc_a.title) and _query_mentions_source_anchor(query, doc_b.title)):
        logger.info(
            'comparison generation omitted one or both source anchors (%s vs %s); skipping query: %s',
            doc_a.title,
            doc_b.title,
            query,
        )
        return fallback
    if not _comparison_uses_supported_shared_labels(query, passage_a, passage_b):
        logger.info('comparison generator used unsupported shared label; skipping query: %s', query)
        return fallback
    if not _comparison_axes_supported_by_both(expected_axes, passage_a, passage_b):
        logger.info('comparison generator produced one-sided axes; skipping query: %s', query)
        return fallback
    return {
        'query': query,
        'intent': 'comparison',
        'should_refuse': False,
        'source_doc_ids': [doc_a.doc_id, doc_b.doc_id],
        'source_file_paths': [doc_a.file_path, doc_b.file_path],
        'document_titles': [doc_a.title, doc_b.title],
        'expected_axes': expected_axes,
        'passage_preview_a': passage_a[:300],
        'passage_preview_b': passage_b[:300],
    }


# ---------------------------------------------------------------------------
# Generator orchestration
# ---------------------------------------------------------------------------


_SINGLE_PASSAGE_INTENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    'factual_lookup': (_FACTUAL_PROMPT, ('expected_answer', 'why_specific')),
    'enumeration': (_ENUMERATION_PROMPT, ('expected_items',)),
    'out_of_scope': (_OUT_OF_SCOPE_PROMPT, ('why_out_of_scope',)),
    'mechanism_bait': (_MECHANISM_BAIT_PROMPT, ('target_entity', 'why_should_refuse')),
}


def generate_queries(config: GenerationConfig) -> list[dict[str, Any]]:
    """Drive query generation across all configured intents."""
    rng = random.Random(config.seed)
    docs = _load_corpus(config)
    if not docs:
        raise RuntimeError('No source documents found via the YAR S3 API.')

    logger.info('Loaded %d source documents from %s', len(docs), config.server_url)
    doc_passages: dict[str, list[str]] = {}
    for doc in docs:
        passages = _split_into_passages(doc, max_chars=config.passage_chars, min_chars=config.passage_min_chars)
        if not passages:
            # If a doc is short enough that no passage met min_chars, fall
            # back to the whole document so we still get coverage.
            passages = [doc.content[: config.passage_chars]]
        doc_passages[doc.doc_id] = passages
        logger.info('  %s -> %d passages', doc.title, len(passages))

    examples: list[dict[str, Any]] = []

    for intent in config.intents:
        if intent == 'comparison':
            examples.extend(
                _generate_comparison_set(
                    config=config,
                    docs=docs,
                    doc_passages=doc_passages,
                    rng=rng,
                )
            )
            continue
        if intent not in _SINGLE_PASSAGE_INTENTS:
            logger.warning('Unknown intent %r; skipping', intent)
            continue
        prompt_template, meta_keys = _SINGLE_PASSAGE_INTENTS[intent]
        examples.extend(
            _generate_single_passage_set(
                config=config,
                intent=intent,
                prompt_template=prompt_template,
                meta_keys=meta_keys,
                docs=docs,
                doc_passages=doc_passages,
                rng=rng,
            )
        )

    return examples


def _generate_single_passage_set(
    *,
    config: GenerationConfig,
    intent: str,
    prompt_template: str,
    meta_keys: tuple[str, ...],
    docs: list[SourceDocument],
    doc_passages: dict[str, list[str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    target = config.queries_per_intent
    out: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = target * 3
    # Round-robin across docs so every doc contributes; sample a passage per doc per round.
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        for doc in docs:
            if len(out) >= target:
                break
            passages = doc_passages.get(doc.doc_id) or []
            if not passages:
                continue
            passage = rng.choice(passages)
            example = _gen_single_passage_intent(
                config,
                intent=intent,
                prompt_template=prompt_template,
                doc=doc,
                passage=passage,
                extra_meta_keys=meta_keys,
            )
            if example:
                out.append(example)
    if len(out) < target:
        logger.warning('Only generated %d/%d %s queries', len(out), target, intent)
    return out


def _generate_comparison_set(
    *,
    config: GenerationConfig,
    docs: list[SourceDocument],
    doc_passages: dict[str, list[str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    target = config.queries_per_intent
    out: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    attempts = 0
    max_attempts = target * 16
    if len(docs) < 2:
        logger.warning('Need at least 2 source documents for comparison queries; have %d', len(docs))
        return out
    pair_indices: list[tuple[int, int]] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            pair_indices.append((i, j))
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(pair_indices)
        doc_a, doc_b = docs[i], docs[j]
        passages_a = doc_passages.get(doc_a.doc_id) or []
        passages_b = doc_passages.get(doc_b.doc_id) or []
        if not passages_a or not passages_b:
            continue
        passage_a = rng.choice(passages_a)
        passage_b = rng.choice(passages_b)
        if not _comparison_pair_has_shared_topic(doc_a.title, passage_a, doc_b.title, passage_b):
            logger.info(
                'comparison generator skipped pair with no concrete shared topic: %s vs %s',
                doc_a.title[:40],
                doc_b.title[:40],
            )
            continue
        example = _gen_comparison(
            config,
            doc_a=doc_a,
            doc_b=doc_b,
            passage_a=passage_a,
            passage_b=passage_b,
        )
        if example:
            query_key = str(example.get('query') or '').casefold()
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            out.append(example)
    if len(out) < target:
        logger.warning('Only generated %d/%d comparison queries', len(out), target)
    return out


# ---------------------------------------------------------------------------
# Output: Phoenix dataset + optional local JSON dump
# ---------------------------------------------------------------------------


def upload_to_phoenix(
    *,
    config: GenerationConfig,
    examples: list[dict[str, Any]],
) -> Any:
    """Push the generated set to a Phoenix dataset matching the export format."""
    if not config.dataset_name:
        return None
    try:
        from phoenix.client import Client
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('phoenix.client is required. Install via `pip install -e .[observability]`.') from exc

    client_kwargs: dict[str, Any] = {}
    if config.phoenix_base_url:
        client_kwargs['base_url'] = config.phoenix_base_url
    if config.phoenix_api_key:
        client_kwargs['api_key'] = config.phoenix_api_key
    client = Client(**client_kwargs)

    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for ex in examples:
        # ``input`` mirrors the manual baseline's ``{query, mode}`` shape so the
        # experiment runner can drive it without changes.
        inputs.append({'query': ex['query'], 'mode': 'mix'})
        outputs.append({'expected_answer': ex.get('expected_answer', '')})
        metadata.append(
            {
                'intent': ex['intent'],
                'should_refuse': ex.get('should_refuse', False),
                'source_doc_id': ex.get('source_doc_id') or '',
                'source_doc_ids': ex.get('source_doc_ids') or [],
                'document_title': ex.get('document_title') or '',
                'document_titles': ex.get('document_titles') or [],
                'expected_items': ex.get('expected_items') or [],
                'expected_axes': ex.get('expected_axes') or [],
                'target_entity': ex.get('target_entity') or '',
                'why_out_of_scope': ex.get('why_out_of_scope') or '',
                'why_should_refuse': ex.get('why_should_refuse') or '',
            }
        )

    return client.datasets.create_dataset(
        name=config.dataset_name,
        dataset_description=(
            config.dataset_description or f'Auto-generated YAR eval queries ({len(examples)} examples) from S3 corpus'
        ),
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate intent-typed eval queries from the YAR corpus.')
    parser.add_argument('--server-url', default=os.getenv('YAR_SERVER_URL', 'http://localhost:9621'))
    parser.add_argument('--api-key', default=os.getenv('YAR_API_KEY') or None)
    parser.add_argument('--workspace', default=os.getenv('WORKSPACE', 'default'))
    parser.add_argument('--judge-model', default='tuna')
    parser.add_argument('--judge-provider', default='openai')
    parser.add_argument('--judge-base-url', default=os.getenv('OPENAI_BASE_URL') or None)
    parser.add_argument('--judge-api-key', default=os.getenv('OPENAI_API_KEY') or None)
    parser.add_argument('--queries-per-intent', type=int, default=5)
    parser.add_argument(
        '--intents',
        default='factual_lookup,enumeration,comparison,out_of_scope,mechanism_bait',
    )
    parser.add_argument('--passage-chars', type=int, default=3000)
    parser.add_argument('--passage-min-chars', type=int, default=800)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--dataset-name', default=None, help='Phoenix dataset name; omit to skip upload.')
    parser.add_argument('--dataset-description', default=None)
    parser.add_argument('--output', default=None, help='Optional local JSON dump path.')
    parser.add_argument('--phoenix-base-url', default=os.getenv('PHOENIX_BASE_URL') or None)
    parser.add_argument('--phoenix-api-key', default=os.getenv('PHOENIX_API_KEY') or None)
    parser.add_argument(
        '--source-suffix',
        default=os.getenv('YAR_EVAL_SOURCE_SUFFIX', '.canonical.md'),
        help='Uploaded S3 text artifact suffix to sample, e.g. .canonical.md or .processed.md.',
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    config = GenerationConfig(
        server_url=args.server_url,
        api_key=args.api_key,
        workspace=args.workspace,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        queries_per_intent=args.queries_per_intent,
        intents=[name.strip() for name in args.intents.split(',') if name.strip()],
        passage_chars=args.passage_chars,
        passage_min_chars=args.passage_min_chars,
        seed=args.seed,
        dataset_name=args.dataset_name,
        dataset_description=args.dataset_description,
        output_path=args.output,
        phoenix_base_url=args.phoenix_base_url,
        phoenix_api_key=args.phoenix_api_key,
        source_suffix=args.source_suffix,
    )

    examples = generate_queries(config)
    logger.info('Generated %d queries across %d intents', len(examples), len(config.intents))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(examples, fh, ensure_ascii=False, indent=2, default=str)
        logger.info('Wrote %d examples to %s', len(examples), args.output)

    if config.dataset_name:
        dataset = upload_to_phoenix(config=config, examples=examples)
        logger.info('Uploaded to Phoenix dataset %s', getattr(dataset, 'id', config.dataset_name))

    return 0


if __name__ == '__main__':
    sys.exit(main())
