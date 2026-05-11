# syntax=docker/dockerfile:1.7

# ──────────────────────────────────────────────────────────────────────────────
# Frontend dependency stage
# ──────────────────────────────────────────────────────────────────────────────
FROM oven/bun:1.3-debian AS frontend-deps

WORKDIR /app
COPY yar_webui/ ./yar_webui/

RUN --mount=type=cache,target=/root/.bun/install/cache \
    cd yar_webui \
    && bun install --frozen-lockfile

# ──────────────────────────────────────────────────────────────────────────────
# Frontend build stage (Node/Vite avoids the Bun build OOM)
# ──────────────────────────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /app
COPY --from=frontend-deps /app/yar_webui ./yar_webui

RUN cd yar_webui && node ./node_modules/vite/bin/vite.js build --emptyOutDir

# ──────────────────────────────────────────────────────────────────────────────
# Python build stage
# ──────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.11.8-python3.13-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Build deps + Rust toolchain (needed by some wheels).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Resolve dependencies first so the layer is cacheable across source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --no-install-project --no-editable

# Now bring in source and built frontend assets, then install the project itself.
COPY yar/ ./yar/
COPY --from=frontend-builder /app/yar/api/webui ./yar/api/webui

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# Pre-fetch the tiktoken cache (exit 2 means "no cacheable tokenizers", treat as success).
RUN mkdir -p /app/data/tiktoken \
    && (uv run yar-download-cache --cache-dir /app/data/tiktoken; status=$?; \
        if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi)


# ──────────────────────────────────────────────────────────────────────────────
# Final stage
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Runtime tools: curl for healthchecks, poppler for PDF rasterisation, LibreOffice for office→PDF.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        poppler-utils \
        libreoffice-common \
        libreoffice-core \
        libreoffice-writer \
        libreoffice-impress \
        fonts-dejavu-core \
        fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

# Application: pre-built venv (deps + entry-points + pdfium symlink) and sources.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/yar ./yar

ENV PATH=/app/.venv/bin:$PATH

# Persistent data directories + pre-fetched tiktoken cache.
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/tiktoken
COPY --from=builder /app/data/tiktoken /app/data/tiktoken

ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken \
    WORKING_DIR=/app/data/rag_storage \
    INPUT_DIR=/app/data/inputs

EXPOSE 9621

ENTRYPOINT ["python", "-m", "yar.api.yar_server"]
