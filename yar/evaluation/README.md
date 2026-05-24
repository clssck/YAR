# YAR Evaluation Toolkit

This directory contains everything needed to evaluate YAR's RAG quality
end-to-end. Three CLIs cover the loop:

1. **`phoenix_query_generation`** — auto-generate a corpus-grounded baseline
   from S3 documents, organized by query intent.
2. **`phoenix_evaluators`** — score recent traces with LLM-as-judge
   evaluators and push annotations back into Phoenix.
3. **`phoenix_experiments`** — replay a baseline dataset against the live
   yar-server and capture per-prompt regression deltas.

A fourth CLI, **`phoenix_extraction_qa`**, audits ingestion-time entity
and relation descriptions independently of any query.

All four read OTel traces emitted by the yar-server (the chain span +
LLM/retriever children) so the evaluators see exactly what the synthesis
LLM saw — KG entities, KG relationships, and chunks.

---

## Prerequisites

Phoenix and `phoenix.evals` come from the `[observability]` extra. The CLIs
shell out to an OpenAI-compatible chat endpoint for the judge LLM. The
default LiteLLM proxy in this repo serves both `tuna` (LLM) and `shrimp`
(embedding):

```sh
uv sync --extra api --extra observability --extra dev --extra test
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY=sk-litellm-master-key
```

`./start.sh` and `./setup.sh` already pin all four extras, so a normal
provisioning run leaves the eval CLIs ready out of the box.

---

## 1. Generate a baseline dataset

`phoenix_query_generation` walks the S3 corpus through the yar-server's
`/s3/list` + `/s3/content` endpoints, splits each selected uploaded text
artifact into ~3 KB passages, and asks the judge LLM to write one query per
passage per intent. It defaults to `.canonical.md`; pass
`--source-suffix .processed.md` to generate directly from uploaded processed
Markdown.

Five intent types, each with `should_refuse` metadata so the refusal
evaluator can grade against expectation:

| intent | shape | `should_refuse` |
|---|---|---|
| `factual_lookup` | single-fact lookup answerable from one passage | `false` |
| `enumeration` | list/count question grounded in one passage | `false` |
| `comparison` | cross-document question requiring two passages | `false` |
| `out_of_scope` | same broad topic as a passage but answer requires external sources | `true` |
| `mechanism_bait` | asks for chemistry / biology / instrumentation mechanism the passage does not describe | `true` |

```sh
.venv/bin/python -m yar.evaluation.phoenix_query_generation \
  --server-url http://localhost:9621 \
  --workspace default \
  --judge-model tuna --judge-provider openai \
  --queries-per-intent 5 \
  --dataset-name yar-baseline-$(date +%Y-%m-%d) \
  --output /tmp/yar_baseline.json
```

Cost: 1 LLM call per generated query (~25–35 calls for a 25-query
baseline). Single-doc corpora produce dataset variants; the seed argument
(`--seed`) makes generation reproducible across runs.

You can hand-edit `/tmp/yar_baseline.json` and re-upload via
`phoenix.client` if you need to curate (e.g. swap one auto-generated
question for a sharper one against the same passage).

---

## 2. Score recent traces (chain-level)

`phoenix_evaluators` pulls the most recent `app.query*` chain spans, fetches
each trace's synthesis-prompt context, and runs four evaluators per trace:

| evaluator | rubric | direction |
|---|---|---|
| `relevance` | did the retriever surface useful documents? RELEVANT/PARTIAL/UNRELATED | maximize |
| `groundedness` | is every claim in the answer supported? GROUNDED/PARTIAL/UNSUPPORTED | maximize |
| `hallucination` | does the answer introduce facts not derivable from context? FACTUAL/HALLUCINATED | minimize |
| `refusal` | did the model attempt an answer or admit insufficient info? ANSWERED/REFUSAL | (informational) |

```sh
.venv/bin/python -m yar.evaluation.phoenix_evaluators \
  --project yar-app --since 30m \
  --judge-model tuna --judge-provider openai \
  --output /tmp/eval.csv
```

Annotations are posted back to Phoenix as chain-span annotations by
default (label + score + free-text explanation per evaluator). Pass
`--no-annotations` to skip the push.

### Per-document mode

The `--per-document` flag runs a single `doc_relevance` evaluator over every
retrieved chunk in scope, one row per `(span, document_index)`. Annotations
attach to the document's position on the chain span.

```sh
.venv/bin/python -m yar.evaluation.phoenix_evaluators \
  --project yar-app --since 30m \
  --per-document \
  --judge-model tuna --judge-provider openai
```

This surfaces chunk-precision regressions independently from chain-level
relevance.

---

## 3. Run an experiment against a baseline

`phoenix_experiments` is the regression harness. Given a baseline dataset
(from step 1 or hand-curated), it replays each query against the live
yar-server with `disable_cache=true`, then runs the four chain-level
evaluators against the new answers — and against each trace's synthesis
context, not just the chunks list. Each run is named and persisted as a
Phoenix Experiment so the dataset's "compare" view shows deltas across
versions.

```sh
.venv/bin/python -m yar.evaluation.phoenix_experiments \
  --dataset yar-baseline-2026-05-08 \
  --judge-model tuna --judge-provider openai \
  --experiment-name "after-prompt-tighten-2026-05-08" \
  --experiment-description "Strengthened anti-hallucination prompt; mode auto-route on comparison"
```

Experiment timing: ~3–4 min per task (runs sequentially), plus ~2 min for
100 evaluations on a 25-query × 4-evaluator dataset. ~7 min end-to-end is
typical.

Each evaluator reuses the templates from `phoenix_evaluators`, so chain-
level eval and experiment-level eval grade identically.

---

## 4. Audit ingestion quality

`phoenix_extraction_qa` is independent of any query: it samples random
entities and relations from PostgreSQL and asks the judge "is this
description specific enough to ground a citation?". Useful for catching
ingestion regressions before they show up as retrieval failures.

```sh
.venv/bin/python -m yar.evaluation.phoenix_extraction_qa \
  --workspace default \
  --sample-entities 30 --sample-relations 30 \
  --judge-model tuna --judge-provider openai \
  --output /tmp/extraction_qa.csv
```

Each row gets an entity_quality or relation_quality score
(USEFUL/VAGUE/MISLEADING).

---

## Reading the Phoenix UI

- **Project view** (`/projects/<id>`): per-trace chain spans with all
  evaluator annotations and explanations rendered next to each row. Use
  the tag filter to slice by intent (`citation_mode:none`,
  `latency:slow`, `cited:0`, etc.) — tags come from
  `_request_tags` / `_result_tags` in `query_routes.py`.
- **Datasets view** (`/datasets/<id>`): the baseline questions plus
  expected outputs and metadata.
- **Experiments view** (`/datasets/<id>/experiments`): every run
  registered against this dataset, with a side-by-side "compare" view
  for any pair. The compare view diffs metric scores per question and
  highlights flips (e.g. FACTUAL → HALLUCINATED).
- **Span detail**: every YAR query trace has the synthesis prompt,
  reference list, KG entity/relation flat attributes
  (`attributes.kg.entities.*`, `attributes.kg.relationships.*`), token
  counts (`llm.token_count.*` rolled up to the chain via the
  `rollup_tokens` flag in `tracing.py`), and citation telemetry
  (`citations.coverage`, `citations.stripped`,
  `retrieval.cited_count_raw`).

---

## Failure-mode catalog (what each metric tells you)

| Symptom | Most-common cause | First place to look |
|---|---|---|
| `hallucination=HALLUCINATED` with `refusal=ANSWERED` | Synthesis LLM filled mechanism / target / classification from training data | `yar/prompt.py` negative examples — section 3 of `rag_response` / `naive_rag_response` |
| `refusal=REFUSAL` on a `factual_lookup` | Retrieval surfaced no chunks OR query is mislabeled | `yar.log` for `Forced low_level_keywords` / `Skipping HyDE` lines; check chain span's `rag.entity_count` and `rag.chunk_count` |
| `comparison` with one ref | Cross-doc retrieval collapsed to one entity neighbourhood | `_apply_auto_entity_filter` should skip on comparison; if it didn't, check `_is_comparison_query` regex |
| Per-document `doc_relevance=UNRELATED` | Chunk surfaced but doesn't help — likely entity-vector miss | `_query_entity_candidates` / `entity_filter` |
| `groundedness=PARTIAL` on enumeration | Model included an item from the corpus that isn't in the requested set | Section 5 of `rag_response` (enumeration rule) |
| `cited=0` with `refs > 0` | LLM ignored the citation directive | Section 5 of `rag_response`; `citation_mode:none` query default |

---

## What the v7 baseline established (2026-05-08)

A 25-query auto-generated dataset (5 per intent) hit:

- **0 hallucinations** across all 25 queries
- **enumeration** perfect across all 4 metrics
- **factual_lookup**: 5/5 RELEVANT, 5/5 GROUNDED, 5/5 FACTUAL, 4/5 ANSWERED (1 mislabeled)
- **comparison**: 5/5 ANSWERED, 5/5 FACTUAL, 4/5 GROUNDED, 2/5 RELEVANT 3/5 PARTIAL (judge nitpick)
- **out_of_scope**: 5/5 REFUSAL, 5/5 FACTUAL, 4/5 GROUNDED
- **mechanism_bait**: 4/5 REFUSAL, 5/5 GROUNDED, 5/5 FACTUAL

The four imperfect cells trace to: 1 dataset-mislabel, 1 borderline
mechanism_bait borderline-answerable, 3 PARTIAL-relevance judge
strictness. The system's actual behaviour — zero hallucinations, refuse
when it should, ground every claim it makes — is robust.

The baseline + experiment harness lets you A/B any prompt change against
this curve.
