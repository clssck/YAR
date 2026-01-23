#!/bin/bash
# YAR Work Environment Setup
# Sets up Docker network, environment variables, and starts the stack
#
# Usage:
#   ./setup.sh          # Interactive setup + start
#   ./setup.sh --proxy  # Also start HonoHub proxy after stack is up

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          YAR Work Environment Setup                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Detect Environment
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}🔍 Detecting environment...${NC}"

if [ -n "${AWS_WEB_IDENTITY_TOKEN_FILE}" ] || [ -n "${AWS_ROLE_ARN}" ]; then
    echo -e "${GREEN}✓ AWS credentials detected (IRSA/IAM Role)${NC}"
    AWS_ENV=true
elif [ -n "${AWS_ACCESS_KEY_ID}" ]; then
    echo -e "${GREEN}✓ AWS credentials detected (Access Keys)${NC}"
    AWS_ENV=true
else
    echo -e "${RED}⚠ No AWS credentials found${NC}"
    echo -e "  This setup is designed for AWS Bedrock environments."
    echo -e "  Set AWS_WEB_IDENTITY_TOKEN_FILE/AWS_ROLE_ARN or AWS_ACCESS_KEY_ID."
    echo ""
    read -p "Continue anyway? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 1
    fi
    AWS_ENV=false
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Docker Network Configuration
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}🔧 Configuring Docker network...${NC}"

NETWORK_NAME="yar-stack_yar-network"

# Check if network already exists (from previous run)
if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    GATEWAY_IP=$(docker network inspect "$NETWORK_NAME" --format='{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null)
    if [ -n "$GATEWAY_IP" ] && [ "$GATEWAY_IP" != "null" ]; then
        echo -e "${GREEN}✓ Network exists: $NETWORK_NAME${NC}"
        echo -e "  Gateway IP: $GATEWAY_IP"
    else
        GATEWAY_IP="172.19.0.1"
        echo -e "${GREEN}✓ Network exists, using default gateway: $GATEWAY_IP${NC}"
    fi
else
    # Network will be created by docker compose with fixed IPAM (172.28.0.0/16)
    GATEWAY_IP="172.28.0.1"
    echo -e "${BLUE}ℹ Network will be created by docker compose${NC}"
    echo -e "  Gateway IP: $GATEWAY_IP (fixed in docker-compose.yml)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Environment Configuration
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}📝 Configuring environment...${NC}"

# Create or update .env file
touch .env

# Helper function to set env var (updates if exists, appends if not)
set_env() {
    local key="$1"
    local value="$2"
    if grep -q "^${key}=" .env 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        echo "${key}=${value}" >> .env
    fi
}

# Docker gateway
set_env "DOCKER_GATEWAY_IP" "$GATEWAY_IP"

# AWS defaults
set_env "AWS_REGION" "${AWS_REGION:-us-east-1}"
set_env "AWS_DEFAULT_REGION" "${AWS_DEFAULT_REGION:-us-east-1}"

# IRSA credentials (for EKS pods running Docker)
if [ -n "${AWS_ROLE_ARN}" ]; then
    set_env "AWS_ROLE_ARN" "${AWS_ROLE_ARN}"
    echo -e "  ${GREEN}✓${NC} Captured AWS_ROLE_ARN for Docker containers"
fi

# LiteLLM
set_env "LITELLM_MASTER_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

# LLM Model (Bedrock Claude 3.5 Sonnet)
set_env "LLM_MODEL" "beepboop"

# Embedding (Bedrock Titan Embed v2 - 1024 dims, 8192 token limit)
set_env "EMBEDDING_MODEL" "titan-embed"
set_env "EMBEDDING_DIM" "1024"

# Chunking (Kreuzberg semantic chunking by default)
set_env "CHUNKING_PRESET" "semantic"  # 'semantic', 'recursive', or '' for basic
set_env "CHUNK_SIZE" "1600"           # Max tokens per semantic chunk
set_env "CHUNK_OVERLAP_SIZE" "100"    # Overlap between chunks

# PostgreSQL
set_env "POSTGRES_PASSWORD" "${POSTGRES_PASSWORD:-yar_pass}"

# S3/RustFS
set_env "S3_ACCESS_KEY_ID" "${S3_ACCESS_KEY_ID:-rustfsadmin}"
set_env "S3_SECRET_ACCESS_KEY" "${S3_SECRET_ACCESS_KEY:-rustfsadmin}"
set_env "S3_BUCKET_NAME" "${S3_BUCKET_NAME:-yar}"

# Workspace
set_env "WORKSPACE" "${WORKSPACE:-default}"

# Processing defaults
set_env "LOG_LEVEL" "${LOG_LEVEL:-INFO}"
set_env "MAX_ASYNC" "${MAX_ASYNC:-96}"
set_env "MAX_PARALLEL_INSERT" "${MAX_PARALLEL_INSERT:-10}"

# Reverse proxy path prefix (for K8s/code-server environments)
set_env "ROOT_PATH" "${ROOT_PATH:-/oneai-rnd-transformerscmc-genai/janos/proxy/9621}"

echo -e "${GREEN}✓ Environment configured${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Create data directories
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}📁 Creating data directories...${NC}"

mkdir -p data/rag_storage data/inputs
echo -e "${GREEN}✓ Created data/rag_storage and data/inputs${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Build WebUI and install HonoHub dependencies
# ══════════════════════════════════════════════════════════════════════════════

if command -v bun &> /dev/null; then
    # Build WebUI
    echo ""
    echo -e "${YELLOW}🔨 Building WebUI frontend...${NC}"
    cd yar_webui
    bun install || { echo -e "${RED}✗ Failed to install WebUI dependencies${NC}"; cd ..; }
    if bun run build; then
        echo -e "${GREEN}✓ WebUI built${NC}"
    else
        echo -e "${RED}✗ WebUI build failed${NC}"
    fi
    cd ..

    # Install HonoHub dependencies
    echo ""
    echo -e "${YELLOW}📦 Installing HonoHub proxy dependencies...${NC}"
    cd scripts
    bun install --silent 2>/dev/null || bun install
    cd ..
    echo -e "${GREEN}✓ HonoHub dependencies installed${NC}"
else
    echo ""
    echo -e "${YELLOW}ℹ Bun not found - skipping WebUI build and HonoHub setup${NC}"
    echo -e "  Install bun: curl -fsSL https://bun.sh/install | bash"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Summary & Next Steps
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Configuration Summary                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Docker Gateway:${NC}    $GATEWAY_IP"
echo -e "  ${GREEN}AWS Environment:${NC}   ${AWS_ENV:-false}"
echo -e "  ${GREEN}LLM Model:${NC}         beepboop (Bedrock Claude 3.5 Sonnet)"
echo -e "  ${GREEN}Embedding Model:${NC}   titan-embed (Bedrock Titan v2, 1024 dims)"
echo -e "  ${GREEN}Chunking:${NC}          semantic (Kreuzberg, max 1600 tokens)"
echo ""
echo -e "${BLUE}Services:${NC}"
echo -e "  • YAR API + WebUI  → http://localhost:9621 (via ./start.sh)"
echo -e "  • LiteLLM Proxy         → http://localhost:4000"
echo -e "  • PostgreSQL            → localhost:5432"
echo -e "  • RustFS S3             → http://localhost:9100"
echo -e "  • RustFS Console        → http://localhost:9101"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Build Images
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}🔨 [Step 1/3] Building Docker images...${NC}"
echo -e "  This may take a few minutes on first run."
echo ""

# Build with progress output
docker compose build 2>&1 | while IFS= read -r line; do
    # Show key build events
    if [[ "$line" == *"Building"* ]] || [[ "$line" == *"built"* ]] || [[ "$line" == *"CACHED"* ]] || [[ "$line" == *"exporting"* ]]; then
        echo -e "  ${BLUE}▸${NC} $line"
    fi
done

echo ""
echo -e "${GREEN}✓ Images built${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Start Containers
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}🚀 [Step 2/3] Starting containers...${NC}"
echo ""

docker compose up -d 2>&1 | while IFS= read -r line; do
    echo -e "  ${BLUE}▸${NC} $line"
done

echo ""
echo -e "${GREEN}✓ Containers started${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Wait for Health Checks
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}⏳ [Step 3/3] Waiting for services to be healthy...${NC}"
echo ""

# Service list for status tracking (infra only, YAR runs locally)
SERVICES=("postgres" "rustfs" "litellm")

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
            healthy)
                echo -ne "${GREEN}✓${NC} $svc  "
                ;;
            unhealthy)
                echo -ne "${RED}✗${NC} $svc  "
                ;;
            *)
                echo -ne "${YELLOW}◦${NC} $svc  "
                ;;
        esac
    done
    echo -ne " [${elapsed}s]    "
}

MAX_WAIT=180
WAITED=0
ALL_HEALTHY=false

while [ $WAITED -lt $MAX_WAIT ]; do
    print_status_line $WAITED

    # Count healthy services
    HEALTHY_COUNT=0
    for svc in "${SERVICES[@]}"; do
        if [ "$(get_service_status "$svc")" == "healthy" ]; then
            HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
        fi
    done

    if [ $HEALTHY_COUNT -ge 3 ]; then
        ALL_HEALTHY=true
        break
    fi

    sleep 3
    WAITED=$((WAITED + 3))
done

echo ""
echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}✓ All services healthy!${NC}"
else
    echo -e "${YELLOW}⚠ Some services not healthy after ${MAX_WAIT}s${NC}"
    echo ""
    echo -e "  Service status:"
    for svc in "${SERVICES[@]}"; do
        svc_status=$(get_service_status "$svc")
        case "$svc_status" in
            healthy)
                echo -e "    ${GREEN}✓${NC} $svc: healthy"
                ;;
            unhealthy)
                echo -e "    ${RED}✗${NC} $svc: unhealthy"
                ;;
            *)
                echo -e "    ${YELLOW}◦${NC} $svc: $svc_status"
                ;;
        esac
    done
    echo ""
    echo -e "  Check logs: ${BLUE}docker compose logs -f${NC}"
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Start HonoHub proxy (if --proxy flag)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$1" == "--proxy" ]] && command -v bun &> /dev/null; then
    echo -e "${YELLOW}🌐 Starting HonoHub reverse proxy...${NC}"
    echo ""
    cd scripts
    HONOHUB_FORCE=true bun run honohub.ts &
    PROXY_PID=$!
    cd ..
    echo ""
    echo -e "${GREEN}✓ HonoHub proxy started (PID: $PROXY_PID)${NC}"
    echo -e "  Stop with: kill $PROXY_PID"
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Infrastructure Ready!                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Infra Services:${NC}"
echo -e "    • PostgreSQL:  localhost:5432"
echo -e "    • LiteLLM:     http://localhost:4000"
echo -e "    • RustFS S3:   http://localhost:9100"
echo ""
echo -e "  ${YELLOW}Start YAR:${NC}"
echo -e "    ./start.sh                # Start YAR API server"
echo ""
echo -e "  ${BLUE}Other Commands:${NC}"
echo -e "    docker compose logs -f    # View infra logs"
echo -e "    docker compose down       # Stop infra"
echo -e "    ./cleanup.sh              # Full cleanup"
echo ""
if command -v bun &> /dev/null && [[ "$1" != "--proxy" ]]; then
    echo -e "  ${BLUE}Proxy (for K8s/code-server):${NC}"
    echo -e "    cd scripts && bun run honohub"
    echo ""
fi
