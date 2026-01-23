# syntax=docker/dockerfile:1

# Frontend build stage
FROM oven/bun:latest AS frontend-builder

WORKDIR /app

# Copy frontend source code
COPY yar_webui/ ./yar_webui/

# Build frontend assets for inclusion in the API package
RUN --mount=type=cache,target=/root/.bun/install/cache \
    cd yar_webui \
    && bun install --frozen-lockfile \
    && bun run build

# Python build stage - using uv for faster package installation
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system deps (Rust is required by some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Ensure shared data directory exists for uv caches
RUN mkdir -p /root/.local/share/uv

# Copy project metadata and sources
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Install base + API extras without the project to improve caching
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --no-install-project --no-editable

# Copy project sources after dependency layer
COPY yar/ ./yar/

# Include pre-built frontend assets from the previous stage
COPY --from=frontend-builder /app/yar/api/webui ./yar/api/webui

# Sync project in non-editable mode and ensure pip is available for runtime installs
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# Prepare tiktoken cache directory and pre-populate tokenizer data
# Use uv run to execute commands from the virtual environment
RUN mkdir -p /app/data/tiktoken \
    && uv run yar-download-cache --cache-dir /app/data/tiktoken || status=$?; \
    if [ -n "${status:-}" ] && [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi

# Setup pdfium symlink for Kreuzberg PDF support
# Kreuzberg needs libpdfium which is bundled with pypdfium2
RUN uv run python -c "from yar.document.kreuzberg_adapter import _setup_pdfium_for_kreuzberg; _setup_pdfium_for_kreuzberg()"

# Final stage
FROM python:3.13-slim

WORKDIR /app

# Add curl for runtime healthchecks and simple diagnostics
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Copy installed packages and application code
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/yar ./yar
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Ensure the installed scripts are on PATH
ENV PATH=/app/.venv/bin:/root/.local/bin:$PATH

# Install dependencies with uv sync (uses locked versions from uv.lock)
# And ensure pip is available for runtime installs
# Also setup pdfium symlink for Kreuzberg PDF support
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade \
    && /app/.venv/bin/python -c "from yar.document.kreuzberg_adapter import _setup_pdfium_for_kreuzberg; _setup_pdfium_for_kreuzberg()"

# Create persistent data directories AFTER package installation
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/tiktoken

# Copy tiktoken cache into the newly created directory
COPY --from=builder /app/data/tiktoken /app/data/tiktoken

# Point to the prepared cache
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# Expose API port
EXPOSE 9621

ENTRYPOINT ["python", "-m", "yar.api.yar_server"]
