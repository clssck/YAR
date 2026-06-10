#!/bin/bash
# YAR Environment Setup
# Sets up Docker services, environment variables, and starts the stack
#
# Usage:
#   ./setup.sh              # Dev profile (LiteLLM with lower concurrency)
#   ./setup.sh --work       # Work profile (LiteLLM with higher concurrency)
#   ./setup.sh --proxy      # Also start HonoHub proxy after stack is up

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ══════════════════════════════════════════════════════════════════════════════
# Parse arguments
# ══════════════════════════════════════════════════════════════════════════════

PROFILE=""
START_PROXY=false

for arg in "$@"; do
    case $arg in
        --work)   PROFILE="work" ;;
        --dev)    PROFILE="dev" ;;
        --proxy)  START_PROXY=true ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "  --dev       Dev environment"
            echo "  --work      Work environment"
            echo "  --proxy     Start HonoHub reverse proxy after stack is up"
            echo "  (none)      Interactive profile selection"
            echo ""
            exit 0
            ;;
    esac
done

echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo -e "${BLUE}|              YAR Environment Setup                                |${NC}"
echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Select Profile
# ══════════════════════════════════════════════════════════════════════════════

if [ -z "$PROFILE" ]; then
    echo -e "${YELLOW}Profile:${NC}  ${GREEN}1)${NC} dev  ${GREEN}2)${NC} work"
    read -p "Choose [1/2] (default: 1): " PROFILE_CHOICE
    case "$PROFILE_CHOICE" in
        2|work)  PROFILE="work" ;;
        *)       PROFILE="dev" ;;
    esac
fi

echo -e "Profile: ${GREEN}${PROFILE}${NC}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Detect Environment
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}Detecting environment...${NC}"

if [ -f .env ]; then
    source .env 2>/dev/null
fi

MISSING_KEYS=()
if [ -z "${OPENROUTER_API_KEY}" ]; then
    MISSING_KEYS+=("OPENROUTER_API_KEY")
fi
if [ -z "${OPENAI_API_KEY}" ]; then
    MISSING_KEYS+=("OPENAI_API_KEY")
fi

echo -e "${GREEN}* ${PROFILE} profile: routing LLM and embeddings through LiteLLM${NC}"

if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}! Missing API keys in .env:${NC}"
    for key in "${MISSING_KEYS[@]}"; do
        echo -e "    - $key"
    done
    echo ""
    echo -e "  Add them to ${BLUE}.env${NC} before running ${BLUE}./start.sh${NC}"
    echo ""
    read -p "Continue setup anyway? [Y/n]: " CONTINUE
    if [[ "$CONTINUE" =~ ^[Nn]$ ]]; then
        echo "Exiting. Add your API keys to .env and re-run."
        exit 1
    fi
else
    echo -e "${GREEN}* API keys found in .env${NC}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Vision Pipeline System Dependencies
# ══════════════════════════════════════════════════════════════════════════════
# YAR's vision adapter (yar/document/vision_adapter.py) shells out to:
#   - soffice  (LibreOffice)   .docx/.pptx/.odt -> PDF
#   - pdftoppm (poppler-utils) PDF pages -> JPG for the vision LLM
# Without both, every PDF and Office upload fails at extraction time.

echo ""
echo -e "${YELLOW}Checking vision pipeline dependencies...${NC}"

MISSING_BIN=()
command -v soffice  >/dev/null 2>&1 || MISSING_BIN+=("soffice")
command -v pdftoppm >/dev/null 2>&1 || MISSING_BIN+=("pdftoppm")

APT_PKGS="poppler-utils libreoffice-common libreoffice-core libreoffice-writer libreoffice-impress fonts-dejavu-core fonts-liberation2"

install_vision_deps_apt() {
    local sudo_cmd=""
    if [ "$(id -u)" != "0" ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd="sudo"
        else
            echo -e "${RED}x No sudo available. Install manually:${NC}"
            echo -e "  ${BLUE}apt-get install -y --no-install-recommends $APT_PKGS${NC}"
            return 1
        fi
    fi
    echo -e "  Installing: $APT_PKGS"
    if $sudo_cmd apt-get update && $sudo_cmd DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $APT_PKGS; then
        return 0
    fi
    echo -e "${RED}x apt-get install failed${NC}"
    echo -e "  Run manually: ${BLUE}sudo apt-get install -y --no-install-recommends $APT_PKGS${NC}"
    return 1
}

if [ ${#MISSING_BIN[@]} -eq 0 ]; then
    echo -e "${GREEN}* soffice and pdftoppm available${NC}"
elif command -v apt-get >/dev/null 2>&1; then
    echo -e "${YELLOW}! Missing: ${MISSING_BIN[*]}${NC}"
    if install_vision_deps_apt; then
        echo -e "${GREEN}* Vision pipeline dependencies installed${NC}"
    else
        read -p "Continue without these? PDF/Office uploads will fail. [y/N]: " SKIP_DEPS
        if [[ ! "$SKIP_DEPS" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
elif [[ "$OSTYPE" == darwin* ]]; then
    echo -e "${YELLOW}! Missing: ${MISSING_BIN[*]}${NC}"
    echo -e "  Install via Homebrew:"
    echo -e "  ${BLUE}brew install --cask libreoffice && brew install poppler${NC}"
    read -p "Continue without these? PDF/Office uploads will fail. [y/N]: " SKIP_DEPS
    if [[ ! "$SKIP_DEPS" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${YELLOW}! Missing: ${MISSING_BIN[*]}${NC}"
    echo -e "  Install LibreOffice (provides soffice) and poppler-utils (provides pdftoppm)"
    echo -e "  for your platform; PDF/Office uploads will fail without them."
    read -p "Continue without these? [y/N]: " SKIP_DEPS
    if [[ ! "$SKIP_DEPS" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Docker Network Configuration
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}Configuring Docker network...${NC}"

NETWORK_NAME="yar-stack_yar-network"

# Check if network already exists (from previous run)
if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    GATEWAY_IP=$(docker network inspect "$NETWORK_NAME" --format='{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null)
    if [ -n "$GATEWAY_IP" ] && [ "$GATEWAY_IP" != "null" ]; then
        echo -e "${GREEN}* Network exists: $NETWORK_NAME${NC}"
        echo -e "  Gateway IP: $GATEWAY_IP"
    else
        GATEWAY_IP="172.19.0.1"
        echo -e "${GREEN}* Network exists, using default gateway: $GATEWAY_IP${NC}"
    fi
else
    # Network will be created by docker compose with fixed IPAM (172.28.0.0/16)
    GATEWAY_IP="172.28.0.1"
    echo -e "${BLUE}i Network will be created by docker compose${NC}"
    echo -e "  Gateway IP: $GATEWAY_IP (fixed in docker-compose.yml)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Environment Configuration
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}Configuring environment...${NC}"

# Create or update .env file
touch .env

# Helper function to set env var (updates if exists, appends if not)
set_env() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" .env 2>/dev/null; then
        if [[ "$OSTYPE" == darwin* ]]; then
            sed -i '' "s|^${key}=.*|${key}=${value}|" .env
        else
            sed -i "s|^${key}=.*|${key}=${value}|" .env
        fi
    else
        echo "${key}=${value}" >> .env
    fi
}

# Save active profile
set_env "YAR_PROFILE" "$PROFILE"

# Docker gateway
set_env "DOCKER_GATEWAY_IP" "$GATEWAY_IP"

# PostgreSQL (shared across profiles)
set_env "POSTGRES_PASSWORD" "${POSTGRES_PASSWORD:-yar_pass}"

# S3/RustFS (shared across profiles)
set_env "S3_ACCESS_KEY_ID" "${S3_ACCESS_KEY_ID:-rustfsadmin}"
set_env "S3_SECRET_ACCESS_KEY" "${S3_SECRET_ACCESS_KEY:-rustfsadmin}"
set_env "S3_BUCKET_NAME" "${S3_BUCKET_NAME:-yar}"

# Workspace
set_env "WORKSPACE" "${WORKSPACE:-default}"

# Logging
set_env "LOG_LEVEL" "${LOG_LEVEL:-INFO}"

# Observability - Phoenix/OpenTelemetry is enabled by default for troubleshooting.
# Prompt/context text stays redacted unless capture env vars are explicitly set.
set_env "YAR_TRACE_ENABLED" "${YAR_TRACE_ENABLED:-true}"
set_env "YAR_TRACE_PROJECT" "${YAR_TRACE_PROJECT:-yar-app}"
set_env "PHOENIX_COLLECTOR_ENDPOINT" "${PHOENIX_COLLECTOR_ENDPOINT:-http://${GATEWAY_IP}:6006/v1/traces}"

set_env "LITELLM_MASTER_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

if [ "$PROFILE" = "dev" ]; then
    # ── Dev profile: LiteLLM proxy with lower concurrency ──
    set_env "LLM_BINDING" "openai"
    set_env "LLM_MODEL" "tuna"
    set_env "LLM_BINDING_HOST" "http://${GATEWAY_IP}:4000/v1"
    set_env "LLM_BINDING_API_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

    set_env "EMBEDDING_BINDING" "openai"
    set_env "EMBEDDING_MODEL" "shrimp"
    set_env "EMBEDDING_DIM" "1024"
    set_env "EMBEDDING_SEND_DIM" "false"
    set_env "EMBEDDING_TOKEN_LIMIT" "8192"
    set_env "EMBEDDING_BINDING_HOST" "http://${GATEWAY_IP}:4000/v1"
    set_env "EMBEDDING_BINDING_API_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

    set_env "RERANK_MODEL" "Qwen/Qwen3-Reranker-8B"

    set_env "CHUNK_SIZE" "1000"
    set_env "MAX_ASYNC" "${MAX_ASYNC:-4}"
    set_env "MAX_PARALLEL_INSERT" "${MAX_PARALLEL_INSERT:-2}"
    set_env "LLM_TIMEOUT" "300"
    set_env "OPENAI_LLM_MAX_COMPLETION_TOKENS" "9000"
else
    # ── Work profile: Bedrock via LiteLLM proxy ──
    set_env "LLM_BINDING" "openai"
    set_env "LLM_MODEL" "tuna"
    set_env "LLM_BINDING_HOST" "http://${GATEWAY_IP}:4000/v1"
    set_env "LLM_BINDING_API_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

    set_env "EMBEDDING_BINDING" "openai"
    set_env "EMBEDDING_MODEL" "shrimp"
    set_env "EMBEDDING_DIM" "1024"
    set_env "EMBEDDING_SEND_DIM" "false"
    set_env "EMBEDDING_TOKEN_LIMIT" "8192"
    set_env "EMBEDDING_BINDING_HOST" "http://${GATEWAY_IP}:4000/v1"
    set_env "EMBEDDING_BINDING_API_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

    set_env "CHUNKING_PRESET" "semantic"
    set_env "CHUNK_SIZE" "1600"
    set_env "CHUNK_OVERLAP_SIZE" "100"
    set_env "MAX_ASYNC" "${MAX_ASYNC:-96}"
    set_env "MAX_PARALLEL_INSERT" "${MAX_PARALLEL_INSERT:-10}"

    # Reverse proxy path prefix (for K8s/code-server environments)
    RP="${ROOT_PATH:-/oneai-rnd-transformerscmc-genai/janos/proxy/9621}"
    set_env "ROOT_PATH" "$RP"
    # Phoenix UI sits on 6006 behind the same proxy; serve it under the matching
    # sub-path so its SPA assets resolve through the proxy (avoids the white page).
    set_env "PHOENIX_HOST_ROOT_PATH" "${RP%/*}/6006"
fi

./scripts/generate_litellm_config.sh "$PROFILE"
echo -e "  ${GREEN}*${NC} Generated LiteLLM config"

echo -e "${GREEN}* Environment configured${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Create data directories
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}Creating data directories...${NC}"

mkdir -p data/rag_storage data/inputs data/postgres data/rustfs
echo -e "${GREEN}* Created data/rag_storage, data/inputs, data/postgres, data/rustfs${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Build WebUI and install HonoHub dependencies
# ══════════════════════════════════════════════════════════════════════════════

if command -v bun &> /dev/null; then
    # Build WebUI
    echo ""
    echo -e "${YELLOW}Building WebUI frontend...${NC}"
    cd yar_webui
    bun install || { echo -e "${RED}x Failed to install WebUI dependencies${NC}"; cd ..; }
    if bun run build; then
        echo -e "${GREEN}* WebUI built${NC}"
    else
        echo -e "${RED}x WebUI build failed${NC}"
    fi
    cd ..

    # Install HonoHub dependencies (work profile uses reverse proxy)
    if [ "$PROFILE" = "work" ]; then
        echo ""
        echo -e "${YELLOW}Installing HonoHub proxy dependencies...${NC}"
        cd scripts
        bun install --silent 2>/dev/null || bun install
        cd ..
        echo -e "${GREEN}* HonoHub dependencies installed${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}i Bun not found - skipping WebUI build${NC}"
    echo -e "  Install bun: curl -fsSL https://bun.sh/install | bash"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 7.5: Python dependencies (api + observability so OTEL tracing is on)
# ══════════════════════════════════════════════════════════════════════════════

if command -v uv &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Installing Python dependencies (api + observability)...${NC}"
    if uv sync --extra api --extra observability --extra dev --extra test --quiet; then
        echo -e "${GREEN}* Python deps synced; Phoenix/OpenTelemetry available${NC}"
    else
        echo -e "${RED}x uv sync failed - tracing will be disabled until resolved${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}i uv not found - skipping Python dep install${NC}"
    echo -e "  Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo -e "  Then run: uv sync --extra api --extra observability --extra dev --extra test"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Summary
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${GREEN}Profile:${NC} $PROFILE  ${GREEN}Gateway:${NC} $GATEWAY_IP"

SERVICES_LIST="postgres, rustfs, litellm, phoenix"
echo -e "${GREEN}Services:${NC} $SERVICES_LIST"
echo ""

read -p "Build and start? [Y/n]: " PROCEED
if [[ "$PROCEED" =~ ^[Nn]$ ]]; then
    echo ""
    echo -e "${GREEN}* Environment configured. Run ${BLUE}./setup.sh${GREEN} again when ready.${NC}"
    exit 0
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Build and start Docker containers
# ══════════════════════════════════════════════════════════════════════════════

COMPOSE_PROFILE_FLAG="--profile observability"
if [ "$PROFILE" = "work" ]; then
    COMPOSE_PROFILE_FLAG="--profile work --profile observability"
fi

echo -e "${YELLOW}[Step 1/3] Building Docker images...${NC}"
echo -e "  This may take a few minutes on first run."
echo ""

docker compose $COMPOSE_PROFILE_FLAG build 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"Building"* ]] || [[ "$line" == *"built"* ]] || [[ "$line" == *"CACHED"* ]] || [[ "$line" == *"exporting"* ]]; then
        echo -e "  ${BLUE}>${NC} $line"
    fi
done

echo ""
echo -e "${GREEN}* Images built${NC}"

echo ""
echo -e "${YELLOW}[Step 2/3] Starting containers...${NC}"
echo ""

docker compose $COMPOSE_PROFILE_FLAG up -d 2>&1 | while IFS= read -r line; do
    echo -e "  ${BLUE}>${NC} $line"
done

echo ""
echo -e "${GREEN}* Containers started${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 10: Wait for Health Checks
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}[Step 3/3] Waiting for services to be healthy...${NC}"
echo ""

SERVICES=("postgres" "rustfs" "litellm" "phoenix")
REQUIRED_HEALTHY=${#SERVICES[@]}

get_service_status() {
    local service=$1
    local status=$(docker compose ps --format "{{.Service}}:{{.Health}}" 2>/dev/null | grep "^${service}:" | cut -d: -f2)
    echo "${status:-starting}"
}

print_status_line() {
    local elapsed=$1
    echo -ne "\r  "
    for svc in "${SERVICES[@]}"; do
        local status=$(get_service_status "$svc")
        case "$status" in
            healthy)   echo -ne "${GREEN}*${NC} $svc  " ;;
            unhealthy) echo -ne "${RED}x${NC} $svc  " ;;
            *)         echo -ne "${YELLOW}o${NC} $svc  " ;;
        esac
    done
    echo -ne " [${elapsed}s]    "
}

MAX_WAIT=180
WAITED=0
ALL_HEALTHY=false

while [ $WAITED -lt $MAX_WAIT ]; do
    print_status_line $WAITED

    HEALTHY_COUNT=0
    for svc in "${SERVICES[@]}"; do
        if [ "$(get_service_status "$svc")" == "healthy" ]; then
            HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
        fi
    done

    if [ $HEALTHY_COUNT -ge $REQUIRED_HEALTHY ]; then
        ALL_HEALTHY=true
        break
    fi

    sleep 3
    WAITED=$((WAITED + 3))
done

echo ""
echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}* All services healthy!${NC}"
else
    echo -e "${YELLOW}! Some services not healthy after ${MAX_WAIT}s${NC}"
    echo ""
    echo -e "  Service status:"
    for svc in "${SERVICES[@]}"; do
        svc_status=$(get_service_status "$svc")
        case "$svc_status" in
            healthy)   echo -e "    ${GREEN}*${NC} $svc: healthy" ;;
            unhealthy) echo -e "    ${RED}x${NC} $svc: unhealthy" ;;
            *)         echo -e "    ${YELLOW}o${NC} $svc: $svc_status" ;;
        esac
    done
    echo ""
    echo -e "  Check logs: ${BLUE}docker compose logs -f${NC}"
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 11: Start HonoHub proxy (if --proxy flag)
# ══════════════════════════════════════════════════════════════════════════════

if [ "$START_PROXY" = true ] && command -v bun &> /dev/null; then
    echo -e "${YELLOW}Starting HonoHub reverse proxy...${NC}"
    echo ""
    cd scripts
    HONOHUB_FORCE=true bun run honohub.ts &
    PROXY_PID=$!
    cd ..
    echo ""
    echo -e "${GREEN}* HonoHub proxy started (PID: $PROXY_PID)${NC}"
    echo -e "  Stop with: kill $PROXY_PID"
fi

echo ""
echo -e "${GREEN}Infrastructure ready.${NC} Run ${BLUE}./start.sh${NC} to start YAR."
echo ""
