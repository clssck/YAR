#!/bin/bash
# Start LightRAG API server (dev mode via uv)
# Requires: ./setup.sh to be run first (starts infra containers)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Load .env if exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Detect Docker gateway IP for service connections
# In K8s/code-server environments, localhost doesn't reach Docker containers
if [ -n "$DOCKER_GATEWAY_IP" ] && [ "$DOCKER_GATEWAY_IP" != "127.0.0.1" ]; then
    SERVICE_HOST="$DOCKER_GATEWAY_IP"
    echo -e "${YELLOW}Note: Using Docker gateway IP ($SERVICE_HOST) for services${NC}"
else
    SERVICE_HOST="localhost"
fi

# Check if infra is running
if ! docker compose ps --format "{{.Service}}" 2>/dev/null | grep -q "postgres"; then
    echo -e "${RED}Error: Infrastructure not running${NC}"
    echo -e "Run ${BLUE}./setup.sh${NC} first to start PostgreSQL, LiteLLM, and RustFS"
    exit 1
fi

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Starting LightRAG API Server                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Export environment variables for LightRAG
# Server
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-9621}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Reverse proxy support (set ROOT_PATH for proxied environments)
# Example: ROOT_PATH="/oneai-rnd-transformerscmc-genai/janos/proxy/9621"
export ROOT_PATH="${ROOT_PATH:-}"

# LLM via LiteLLM proxy
export LLM_BINDING="${LLM_BINDING:-openai}"
export LLM_MODEL="${LLM_MODEL:-beepboop}"
export LLM_BINDING_HOST="${LLM_BINDING_HOST:-http://${SERVICE_HOST}:4000/v1}"
export LLM_BINDING_API_KEY="${LLM_BINDING_API_KEY:-${LITELLM_MASTER_KEY:-sk-litellm-master-key}}"

# Embedding via LiteLLM proxy
export EMBEDDING_BINDING="${EMBEDDING_BINDING:-openai}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-titan-embed}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
export EMBEDDING_BINDING_HOST="${EMBEDDING_BINDING_HOST:-http://${SERVICE_HOST}:4000/v1}"
export EMBEDDING_BINDING_API_KEY="${EMBEDDING_BINDING_API_KEY:-${LITELLM_MASTER_KEY:-sk-litellm-master-key}}"

# Storage - PostgreSQL
export LIGHTRAG_KV_STORAGE="${LIGHTRAG_KV_STORAGE:-PGKVStorage}"
export LIGHTRAG_VECTOR_STORAGE="${LIGHTRAG_VECTOR_STORAGE:-PGVectorStorage}"
export LIGHTRAG_GRAPH_STORAGE="${LIGHTRAG_GRAPH_STORAGE:-PGGraphStorage}"
export LIGHTRAG_DOC_STATUS_STORAGE="${LIGHTRAG_DOC_STATUS_STORAGE:-PGDocStatusStorage}"
export POSTGRES_HOST="${POSTGRES_HOST:-${SERVICE_HOST}}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_USER="${POSTGRES_USER:-lightrag}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-lightrag_pass}"
export POSTGRES_DATABASE="${POSTGRES_DATABASE:-lightrag}"

# Workspace
export WORKSPACE="${WORKSPACE:-default}"

# Entity Resolution
export ENTITY_RESOLUTION_ENABLED="${ENTITY_RESOLUTION_ENABLED:-true}"

# Chunking
export CHUNKING_PRESET="${CHUNKING_PRESET:-semantic}"
export CHUNK_SIZE="${CHUNK_SIZE:-1600}"
export CHUNK_OVERLAP_SIZE="${CHUNK_OVERLAP_SIZE:-100}"

# S3/RustFS (port 9100 to avoid conflicts)
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://${SERVICE_HOST}:9100}"
export S3_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID:-rustfsadmin}"
export S3_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY:-rustfsadmin}"
export S3_BUCKET_NAME="${S3_BUCKET_NAME:-lightrag}"
export S3_REGION="${S3_REGION:-us-east-1}"

# Processing
export MAX_ASYNC="${MAX_ASYNC:-96}"
export MAX_PARALLEL_INSERT="${MAX_PARALLEL_INSERT:-10}"

# Data directories
export WORKING_DIR="${WORKING_DIR:-./data/rag_storage}"
export INPUT_DIR="${INPUT_DIR:-./data/inputs}"
mkdir -p "$WORKING_DIR" "$INPUT_DIR"

echo -e "  ${GREEN}Configuration:${NC}"
echo -e "    LLM:        $LLM_MODEL via LiteLLM @ ${SERVICE_HOST}:4000"
echo -e "    Embedding:  $EMBEDDING_MODEL ($EMBEDDING_DIM dims)"
echo -e "    Chunking:   $CHUNKING_PRESET (max $CHUNK_SIZE tokens)"
echo -e "    Storage:    PostgreSQL @ ${SERVICE_HOST}:$POSTGRES_PORT"
echo -e "    S3:         RustFS @ ${SERVICE_HOST}:9100"
echo ""
echo -e "  ${YELLOW}Starting server on http://localhost:$PORT${NC}"
echo -e "  ${BLUE}Press Ctrl+C to stop${NC}"
echo ""

# Sync dependencies (api + extras needed for PostgreSQL/S3)
echo -e "  ${BLUE}Installing dependencies...${NC}"
uv sync --extra api --quiet

# Install packages that are dynamically loaded via pipmaster but may fail in uv
# These are: aioboto3 (S3 client), pgvector (PostgreSQL vectors), asyncpg (PostgreSQL)
echo -e "  ${BLUE}Installing runtime dependencies...${NC}"
uv pip install aioboto3 pgvector asyncpg --quiet || {
    echo -e "  ${YELLOW}Warning: Some packages may not have installed correctly${NC}"
    echo -e "  ${YELLOW}Trying with pip directly...${NC}"
    uv run pip install aioboto3 pgvector asyncpg --quiet || true
}

# Start LightRAG
echo ""
echo -e "  ${GREEN}Launching server...${NC}"
exec uv run python -m lightrag.api.lightrag_server
