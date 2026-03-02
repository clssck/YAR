#!/bin/bash
# Start YAR API server (dev mode via uv)
# Requires: ./setup.sh to be run first (starts infra containers)
#
# Usage:
#   ./start.sh          # Dev profile (direct OpenRouter/OpenAI)
#   ./start.sh --work   # Work profile (AWS Bedrock via LiteLLM)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# ══════════════════════════════════════════════════════════════════════════════
# Parse arguments and detect profile
# ══════════════════════════════════════════════════════════════════════════════

PROFILE=""
for arg in "$@"; do
    case $arg in
        --work)  PROFILE="work" ;;
        --dev)   PROFILE="dev" ;;
        --help|-h)
            echo "Usage: ./start.sh [--dev|--work]"
            exit 0
            ;;
    esac
done

# Load .env if exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Resolve profile: CLI flag > .env > interactive
if [ -z "$PROFILE" ]; then
    if [ -n "${YAR_PROFILE}" ]; then
        PROFILE="$YAR_PROFILE"
    else
        echo -e "${YELLOW}Profile:${NC}  ${GREEN}1)${NC} dev  ${GREEN}2)${NC} work"
        read -p "Choose [1/2] (default: 1): " PROFILE_CHOICE
        case "$PROFILE_CHOICE" in
            2|work)  PROFILE="work" ;;
            *)       PROFILE="dev" ;;
        esac
    fi
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
    echo -e "Run ${BLUE}./setup.sh${NC} first to start PostgreSQL and RustFS"
    exit 1
fi

echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo -e "${BLUE}|          Starting YAR API Server  [${YELLOW}${PROFILE}${BLUE}]                      |${NC}"
echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Shared environment (both profiles)
# ══════════════════════════════════════════════════════════════════════════════

# Server
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-9621}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Storage - PostgreSQL
export YAR_KV_STORAGE="${YAR_KV_STORAGE:-PGKVStorage}"
export YAR_VECTOR_STORAGE="${YAR_VECTOR_STORAGE:-PGVectorStorage}"
export YAR_GRAPH_STORAGE="${YAR_GRAPH_STORAGE:-PGGraphStorage}"
export YAR_DOC_STATUS_STORAGE="${YAR_DOC_STATUS_STORAGE:-PGDocStatusStorage}"
export POSTGRES_HOST="${POSTGRES_HOST:-${SERVICE_HOST}}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_USER="${POSTGRES_USER:-yar}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-yar_pass}"
export POSTGRES_DATABASE="${POSTGRES_DATABASE:-yar}"

# Workspace
export WORKSPACE="${WORKSPACE:-default}"

# Entity Resolution
export ENTITY_RESOLUTION_ENABLED="${ENTITY_RESOLUTION_ENABLED:-true}"

# S3/RustFS
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://${SERVICE_HOST}:9100}"
export S3_ACCESS_KEY_ID="${S3_ACCESS_KEY_ID:-rustfsadmin}"
export S3_SECRET_ACCESS_KEY="${S3_SECRET_ACCESS_KEY:-rustfsadmin}"
export S3_BUCKET_NAME="${S3_BUCKET_NAME:-yar}"
export S3_REGION="${S3_REGION:-us-east-1}"

# Data directories
export WORKING_DIR="${WORKING_DIR:-./data/rag_storage}"
export INPUT_DIR="${INPUT_DIR:-./data/inputs}"
mkdir -p "$WORKING_DIR" "$INPUT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# Profile-specific environment
# ══════════════════════════════════════════════════════════════════════════════

if [ "$PROFILE" = "dev" ]; then
    # ── Dev: direct API calls to OpenRouter (LLM) and OpenAI (embeddings) ──
    export LLM_BINDING="${LLM_BINDING:-openai}"
    export LLM_MODEL="${LLM_MODEL:-x-ai/grok-4.1-fast}"
    export LLM_BINDING_HOST="${LLM_BINDING_HOST:-https://openrouter.ai/api/v1}"
    export LLM_TIMEOUT="${LLM_TIMEOUT:-300}"
    export OPENAI_LLM_MAX_COMPLETION_TOKENS="${OPENAI_LLM_MAX_COMPLETION_TOKENS:-9000}"
    # LLM_BINDING_API_KEY must be set in .env

    export EMBEDDING_BINDING="${EMBEDDING_BINDING:-openai}"
    export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
    export EMBEDDING_DIM="${EMBEDDING_DIM:-1532}"
    export EMBEDDING_SEND_DIM="${EMBEDDING_SEND_DIM:-true}"
    export EMBEDDING_TOKEN_LIMIT="${EMBEDDING_TOKEN_LIMIT:-8192}"
    export EMBEDDING_BINDING_HOST="${EMBEDDING_BINDING_HOST:-https://api.openai.com/v1}"
    # EMBEDDING_BINDING_API_KEY must be set in .env

    export RERANK_BINDING="${RERANK_BINDING:-deepinfra}"
    export RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-8B}"
    # RERANK_BINDING_API_KEY must be set in .env

    export CHUNK_SIZE="${CHUNK_SIZE:-1000}"
    export MAX_ASYNC="${MAX_ASYNC:-4}"
    export MAX_PARALLEL_INSERT="${MAX_PARALLEL_INSERT:-2}"

    echo -e "  LLM: ${GREEN}$LLM_MODEL${NC}  Embed: ${GREEN}$EMBEDDING_MODEL${NC}  Rerank: ${GREEN}$RERANK_MODEL${NC}"

else
    # ── Work: LiteLLM proxy for LLM + embeddings (AWS Bedrock) ──
    export LLM_BINDING="${LLM_BINDING:-openai}"
    export LLM_MODEL="${LLM_MODEL:-beepboop}"
    export LLM_BINDING_HOST="${LLM_BINDING_HOST:-http://${SERVICE_HOST}:4000/v1}"
    export LLM_BINDING_API_KEY="${LLM_BINDING_API_KEY:-${LITELLM_MASTER_KEY:-sk-litellm-master-key}}"

    export EMBEDDING_BINDING="${EMBEDDING_BINDING:-openai}"
    export EMBEDDING_MODEL="${EMBEDDING_MODEL:-titan-embed}"
    export EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
    export EMBEDDING_BINDING_HOST="${EMBEDDING_BINDING_HOST:-http://${SERVICE_HOST}:4000/v1}"
    export EMBEDDING_BINDING_API_KEY="${EMBEDDING_BINDING_API_KEY:-${LITELLM_MASTER_KEY:-sk-litellm-master-key}}"

    export CHUNKING_PRESET="${CHUNKING_PRESET:-semantic}"
    export CHUNK_SIZE="${CHUNK_SIZE:-1600}"
    export CHUNK_OVERLAP_SIZE="${CHUNK_OVERLAP_SIZE:-100}"
    export MAX_ASYNC="${MAX_ASYNC:-96}"
    export MAX_PARALLEL_INSERT="${MAX_PARALLEL_INSERT:-10}"

    # Reverse proxy support (set ROOT_PATH for proxied environments)
    export ROOT_PATH="${ROOT_PATH:-}"

    # Port 9622 internally - HonoHub proxies 9621 -> 9622 for path rewriting
    export PORT="${PORT:-9622}"

    echo -e "  LLM: ${GREEN}$LLM_MODEL${NC}  Embed: ${GREEN}$EMBEDDING_MODEL${NC}  via LiteLLM @ ${SERVICE_HOST}:4000"
fi

echo ""
echo -e "  ${YELLOW}Starting server on http://localhost:$PORT${NC}"
echo -e "  ${BLUE}Press Ctrl+C to stop${NC}"
echo ""

# Sync dependencies (api + extras needed for PostgreSQL/S3)
echo -e "  ${BLUE}Installing dependencies...${NC}"
uv sync --extra api --quiet


# Start YAR
echo ""
echo -e "  ${GREEN}Launching server...${NC}"
exec uv run python -m yar.api.yar_server
