# Evaluation Framework

Two independent pipelines for measuring YAR retrieval and generation quality.

**QA Eval** (Pipeline A) runs 25 benchmark questions through YAR and grades answers with an LLM judge. Simple, fast, produces a Pass/Fail rubric per question. Use this for quick regression checks.

**RAGAS Eval** (Pipeline B) scores four quantitative metrics (faithfulness, answer relevance, context recall, context precision) using the RAGAS framework. More thorough, slower, requires additional dependencies. Use this for deep retrieval quality analysis.

Both pipelines share the same test corpus and require a running YAR server with documents ingested.

## Directory Layout

```
yar/evaluation/
  qa_eval_common.py           # Shared constants and helpers (Pipeline A)
  export_qa_answers.py        # Step 2: query YAR, record answers
  grade_qa_answers.py         # Step 3: LLM-judge grading
  ingest_test_docs.py         # Step 1: upload docs to YAR (both pipelines)
  eval_rag_quality.py         # RAGAS evaluator (Pipeline B)
  e2e_test_harness.py         # End-to-end RAGAS orchestrator
  questions.md                # Human-readable question list
  download_wikipedia.py       # Fetches wiki articles into wiki_documents/
  wiki_documents/             # 8 Wikipedia articles (test corpus)
  sample_documents/           # 5 YAR-about markdown files (RAGAS corpus)
  results/                    # All output lands here (gitignored)
EvaluationTemplate_filled.csv # 25 Q&A pairs (repo root, Pipeline A input)
```

---

## QA Eval (Pipeline A)

A three-step pipeline: ingest documents, export YAR's answers, grade them.

### Prerequisites

- YAR server running (`yar-server` or `uvicorn yar.api.yar_server:app`)
- LLM API key for the grading judge (OpenRouter, OpenAI, or any LiteLLM-compatible provider)
- Python environment with `pip install -e .` (no extra deps beyond base YAR)

### Environment Variables

Set these in `.env` at the repo root or export them in your shell:

| Variable | Default | Used By |
|----------|---------|---------|
| `YAR_API_URL` | `http://localhost:9621` | All scripts |
| `YAR_API_KEY` | (empty) | All scripts |
| `EVAL_LLM_BINDING_HOST` | `https://openrouter.ai/api/v1` | grade_qa_answers |
| `EVAL_LLM_BINDING_API_KEY` | (empty, **required** for grading) | grade_qa_answers |

Fallbacks: `EVAL_LLM_BINDING_HOST` falls back to `LLM_BINDING_HOST`. `EVAL_LLM_BINDING_API_KEY` falls back to `LLM_BINDING_API_KEY`.

### Step 1: Ingest Test Documents

Upload the bundled Wikipedia articles into YAR:

```bash
python yar/evaluation/ingest_test_docs.py
```

This reads all `.txt` and `.md` files from `yar/evaluation/wiki_documents/`, uploads them via `POST /documents/texts`, and polls `/documents/track_status` until processing completes (up to 30 minutes).

Options:

```
--input, -i DIR     Source directory (default: yar/evaluation/wiki_documents)
--rag-url, -r URL   YAR API URL (default: $YAR_API_URL or http://localhost:9621)
```

To ingest a different document set:

```bash
python yar/evaluation/ingest_test_docs.py --input /path/to/your/docs
```

### Step 2: Export Answers

Query YAR with each of the 25 benchmark questions and record the responses:

```bash
python yar/evaluation/export_qa_answers.py
```

This reads `EvaluationTemplate_filled.csv`, sends each question to `POST /query` (mode=mix, response_type=Single Paragraph), and writes a timestamped CSV to `yar/evaluation/results/`.

Options:

```
--input-csv PATH    Q&A source file (default: EvaluationTemplate_filled.csv)
--output-csv PATH   Output path (default: results/qa_answers_<timestamp>.csv)
--api-url URL       YAR API URL (default: $YAR_API_URL)
--api-key KEY       YAR API key (default: $YAR_API_KEY)
--mode MODE         Query mode (default: mix)
--timeout SECS      Per-question timeout (default: 120)
```

Output columns: `question`, `actualResponse`, `expectedResponse`.

### Step 3: Grade Answers

Run an LLM judge over the exported answers:

```bash
python yar/evaluation/grade_qa_answers.py
```

By default this picks up the most recent `qa_answers_*.csv` from `results/`. Each row is graded on four dimensions:

| Dimension | Meaning |
|-----------|---------|
| `generalQuality` | Overall answer quality (Pass/Fail) |
| `seemsRelevant` | Does the answer address the question? (Pass/Fail) |
| `seemsComplete` | Does the answer cover the expected content? (Pass/Fail/Skipped) |
| `basedOnKnowledgeSources` | Is the answer grounded in retrieved docs? (Pass/Fail/Skipped) |

If `seemsRelevant` fails, the remaining dimensions are skipped.

Options:

```
--input-csv PATH              Answers CSV (default: latest qa_answers_*.csv)
--output-csv PATH             Grades output (default: results/qa_grades_<timestamp>.csv)
--judge-api-base URL          Judge LLM endpoint (default: $EVAL_LLM_BINDING_HOST)
--judge-api-key KEY           Judge LLM API key (default: $EVAL_LLM_BINDING_API_KEY)
--judge-model MODEL           Judge model (default: auto-detected from YAR /health)
--temperature FLOAT           Sampling temperature (default: 0.0)
--max-tokens INT              Max response tokens (default: 300)
--allow-inline-citations      Skip the [N] citation pre-check (see below)
```

**Citation guard:** Answers containing inline citation markers like `[1]`, `[2]` are auto-failed by default. This catches responses that leak retrieval metadata into prose. Pass `--allow-inline-citations` to disable this check.

### Full Run Example

```bash
# 1. Start YAR server (separate terminal)
yar-server

# 2. Ingest the test corpus
python yar/evaluation/ingest_test_docs.py

# 3. Export answers
python yar/evaluation/export_qa_answers.py

# 4. Grade with an OpenRouter judge
export EVAL_LLM_BINDING_API_KEY=sk-or-...
python yar/evaluation/grade_qa_answers.py

# 5. View results
cat yar/evaluation/results/qa_grades_*.csv | column -t -s,
```

### Output Files

All outputs are written to `yar/evaluation/results/` (gitignored):

```
results/
  qa_answers_20260413_103000.csv   # Step 2 output
  qa_grades_20260413_103200.csv    # Step 3 output
```

The grades CSV has columns: `question`, `actualResponse`, `expectedResponse`, `generalQuality`, `seemsRelevant`, `seemsComplete`, `basedOnKnowledgeSources`, `reason`.

---

## RAGAS Eval (Pipeline B)

Quantitative scoring using four RAGAS metrics. Requires additional Python packages.

### Prerequisites

```bash
pip install -e ".[evaluation]"
# or: pip install ragas datasets langchain_openai
```

Plus an OpenAI-compatible API key for the RAGAS judge LLM and embedding model.

### Quick Start

```bash
# Set your API key
export EVAL_LLM_BINDING_API_KEY=sk-...

# Run against the bundled sample dataset
python yar/evaluation/eval_rag_quality.py
```

### RAGAS Metrics

| Metric | Measures | Target |
|--------|----------|--------|
| Faithfulness | Is the answer factually grounded in retrieved context? | > 0.80 |
| Answer Relevance | Does the answer address the question? | > 0.80 |
| Context Recall | Was all relevant information retrieved? | > 0.80 |
| Context Precision | Is retrieved context free of irrelevant noise? | > 0.80 |
| RAGAS Score | Average of the four metrics | > 0.80 |

### Options

```
--dataset, -d PATH        Test dataset JSON (default: sample_dataset.json)
--ragendpoint, -r URL     YAR API URL (default: $YAR_API_URL or http://localhost:9621)
--mode, -m MODE           Query mode: local|global|hybrid|mix|naive (default: mix)
--debug, -v               Log retrieved contexts per question
--compare-modes           Run all 5 modes and produce a comparison CSV
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_LLM_MODEL` | `gpt-4o-mini` | Judge LLM model |
| `EVAL_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `EVAL_LLM_BINDING_API_KEY` | falls back to `OPENAI_API_KEY` | **Required** |
| `EVAL_LLM_BINDING_HOST` | OpenAI official API | Custom LLM endpoint |
| `EVAL_EMBEDDING_BINDING_API_KEY` | falls back to LLM key | Embedding API key |
| `EVAL_EMBEDDING_BINDING_HOST` | falls back to LLM host | Custom embedding endpoint |
| `EVAL_MAX_CONCURRENT` | `2` | Concurrent evaluations |
| `EVAL_QUERY_TOP_K` | `15` | Documents retrieved per query |
| `EVAL_CHUNK_TOP_K` | `15` | Chunks per query |
| `EVAL_LLM_MAX_RETRIES` | `5` | LLM request retries |
| `EVAL_LLM_TIMEOUT` | `180` | LLM timeout (seconds) |

Both LLM and embedding endpoints must be OpenAI-compatible.

### Custom Test Dataset

```json
{
  "test_cases": [
    {
      "question": "Your question here",
      "ground_truth": "Expected answer based on your documents"
    }
  ]
}
```

### End-to-End Harness

`e2e_test_harness.py` wraps the full RAGAS pipeline: ingest, wait, evaluate. It also supports A/B testing with `AUTO_CONNECT_ORPHANS`:

```bash
# Full pipeline
python yar/evaluation/e2e_test_harness.py

# Skip ingestion (docs already loaded)
python yar/evaluation/e2e_test_harness.py --skip-ingest

# A/B test orphan connections
python yar/evaluation/e2e_test_harness.py --ab-test

# Filter documents by name
python yar/evaluation/e2e_test_harness.py --papers covid,diabetes
```

Note: `e2e_test_harness.py` defaults to port **9622** (not 9621). Override with `--rag-url` or `$YAR_API_URL`.

### RAGAS Troubleshooting

**"LM returned 1 generations instead of 3"** or NaN metrics: reduce `EVAL_MAX_CONCURRENT` to 1 and/or lower `EVAL_QUERY_TOP_K`. This is usually API rate limiting.

**ModuleNotFoundError for ragas**: run `pip install -e ".[evaluation]"`.

**Context Precision returns NaN**: lower `EVAL_QUERY_TOP_K` to reduce per-test-case LLM calls.

---

## Test Corpus

The bundled corpus covers five domains:

| Domain | Documents |
|--------|-----------|
| Medical | `medical_covid-19.txt`, `medical_diabetes.txt` |
| Finance | `finance_stock_market.txt`, `finance_cryptocurrency.txt` |
| Climate | `climate_climate_change.txt`, `climate_renewable_energy.txt` |
| Sports | `sports_olympic_games.txt`, `sports_fifa_world_cup.txt` |

The 25 benchmark questions (in `EvaluationTemplate_filled.csv` and `questions.md`) span all four domains plus cross-domain multi-hop reasoning.

To refresh the Wikipedia articles:

```bash
python yar/evaluation/download_wikipedia.py
```

## Adding Your Own Questions

1. Edit `EvaluationTemplate_filled.csv` (columns: `question`, `expectedResponse`)
2. Ingest documents that contain the answers
3. Run the QA eval pipeline (Steps 2-3 above)

For RAGAS eval, create a JSON dataset file with `test_cases` containing `question` and `ground_truth` fields, then pass it with `--dataset`.
