# Development commands (npm-style)
# Usage: make <command>

.PHONY: lint format typecheck check fix test eval\:rag clean dev

# Lint code with ruff
lint:
	uv run ruff check .

# Format code with ruff
format:
	uv run ruff format .

# Type-check with ty
typecheck:
	uv run ty check .

# Run all checks (lint + format check + typecheck)
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .

# Fix linting issues and format code
fix:
	uv run ruff check . --fix
	uv run ruff format .

# Run tests
test:
	uv run pytest

# Run the smoke RAG subset and fail if the score drops below the recorded floor
eval\:rag:
	.venv/bin/python yar/evaluation/eval_rag_quality.py --dataset eval_docs/qa_pairs_smoke.json --ragendpoint $${RAG_EVAL_ENDPOINT:-http://localhost:9621}
	.venv/bin/python -c "import json, sys; from pathlib import Path; latest = max(Path('yar/evaluation/results').glob('results_*.json'), key=lambda path: path.stat().st_mtime); payload = json.loads(latest.read_text()); score = float(payload['benchmark_stats']['average_metrics']['ragas_score']); floor = 0.72; print(f'Smoke subset score: {score:.4f} (floor {floor:.4f}) from {latest.name}'); sys.exit(1 if score < floor else 0)"

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Start development server
dev:
	uv run yar-server

# Install dependencies
install:
	uv sync --extra api --extra test --extra lint

# Update lockfile
lock:
	uv lock --upgrade
