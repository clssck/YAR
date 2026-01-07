#!/bin/bash
# LightRAG Work Environment Setup
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
echo -e "${BLUE}║          LightRAG Work Environment Setup                        ║${NC}"
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
# Step 2: Docker Network Setup
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}🔧 Setting up Docker network...${NC}"

NETWORK_NAME="lightrag-stack_lightrag-network"

# Check if network exists
if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    GATEWAY_IP=$(docker network inspect "$NETWORK_NAME" --format='{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null)
    if [ -n "$GATEWAY_IP" ] && [ "$GATEWAY_IP" != "null" ]; then
        echo -e "${GREEN}✓ Existing network found: $NETWORK_NAME${NC}"
        echo -e "  Gateway: $GATEWAY_IP"
    else
        echo -e "${YELLOW}⚠ Network exists but no gateway found${NC}"
        GATEWAY_IP="127.0.0.1"
    fi
else
    # Create network with fixed subnet for consistent gateway IP
    echo -e "  Creating network with fixed subnet..."
    docker network create "$NETWORK_NAME" \
        --driver bridge \
        --subnet=172.19.0.0/16 \
        --gateway=172.19.0.1 \
        --label com.docker.compose.project=lightrag-stack \
        --label com.docker.compose.network=lightrag-network \
        >/dev/null 2>&1 || true

    GATEWAY_IP=$(docker network inspect "$NETWORK_NAME" --format='{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null)
    if [ -n "$GATEWAY_IP" ] && [ "$GATEWAY_IP" != "null" ]; then
        echo -e "${GREEN}✓ Created network with gateway: $GATEWAY_IP${NC}"
    else
        echo -e "${YELLOW}⚠ Could not detect gateway, using: 127.0.0.1${NC}"
        GATEWAY_IP="127.0.0.1"
    fi
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

# LiteLLM
set_env "LITELLM_MASTER_KEY" "${LITELLM_MASTER_KEY:-sk-litellm-master-key}"

# LLM Model (Bedrock Claude 3.5 Sonnet)
set_env "LLM_MODEL" "beepboop"

# Embedding (Bedrock Titan v2 - 1024 dims, 8192 token limit)
set_env "EMBEDDING_MODEL" "bedrock-titan-v2"
set_env "EMBEDDING_DIM" "1024"
set_env "CHUNK_SIZE" "1600"  # Titan v2 supports up to 8192 tokens

# PostgreSQL
set_env "POSTGRES_PASSWORD" "${POSTGRES_PASSWORD:-lightrag_pass}"

# S3/RustFS
set_env "S3_ACCESS_KEY_ID" "${S3_ACCESS_KEY_ID:-rustfsadmin}"
set_env "S3_SECRET_ACCESS_KEY" "${S3_SECRET_ACCESS_KEY:-rustfsadmin}"
set_env "S3_BUCKET_NAME" "${S3_BUCKET_NAME:-lightrag}"

# Workspace
set_env "WORKSPACE" "${WORKSPACE:-default}"

# Processing defaults
set_env "LOG_LEVEL" "${LOG_LEVEL:-INFO}"
set_env "MAX_ASYNC" "${MAX_ASYNC:-96}"
set_env "MAX_PARALLEL_INSERT" "${MAX_PARALLEL_INSERT:-10}"

echo -e "${GREEN}✓ Environment configured${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Create data directories
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}📁 Creating data directories...${NC}"

mkdir -p data/rag_storage data/inputs
echo -e "${GREEN}✓ Created data/rag_storage and data/inputs${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Install HonoHub dependencies (optional)
# ══════════════════════════════════════════════════════════════════════════════

if command -v bun &> /dev/null; then
    echo ""
    echo -e "${YELLOW}📦 Installing HonoHub proxy dependencies...${NC}"
    cd scripts
    bun install --silent 2>/dev/null || bun install
    cd ..
    echo -e "${GREEN}✓ HonoHub dependencies installed${NC}"
else
    echo ""
    echo -e "${YELLOW}ℹ Bun not found - skipping HonoHub setup${NC}"
    echo -e "  Install bun to use the reverse proxy: curl -fsSL https://bun.sh/install | bash"
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
echo -e "  ${GREEN}Embedding Model:${NC}   bedrock-titan-v2 (1024 dims)"
echo -e "  ${GREEN}Chunk Size:${NC}        1600 tokens"
echo ""
echo -e "${BLUE}Services:${NC}"
echo -e "  • LightRAG API + WebUI  → http://localhost:9621"
echo -e "  • LiteLLM Proxy         → http://localhost:4000"
echo -e "  • PostgreSQL            → localhost:5432"
echo -e "  • RustFS S3             → http://localhost:9000"
echo -e "  • RustFS Console        → http://localhost:9001"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Start Stack
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}🚀 Starting LightRAG stack...${NC}"
echo ""

docker compose build --quiet
docker compose up -d

echo ""
echo -e "${GREEN}✓ Stack started!${NC}"
echo ""

# Wait for health checks
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"

MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    HEALTHY=$(docker compose ps --format json 2>/dev/null | grep -c '"Health": "healthy"' || echo "0")
    TOTAL=$(docker compose ps --format json 2>/dev/null | grep -c '"Service"' || echo "4")

    if [ "$HEALTHY" -ge 4 ]; then
        echo -e "${GREEN}✓ All services healthy!${NC}"
        break
    fi

    echo -ne "  Healthy: $HEALTHY/4 (${WAITED}s)...\r"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${YELLOW}⚠ Timeout waiting for services. Check: docker compose logs${NC}"
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
echo -e "${BLUE}║                         Ready!                                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}WebUI:${NC}     http://localhost:9621/webui"
echo -e "  ${GREEN}API Docs:${NC}  http://localhost:9621/docs"
echo -e "  ${GREEN}Health:${NC}    curl http://localhost:9621/health"
echo ""
echo -e "  ${BLUE}Commands:${NC}"
echo -e "    docker compose logs -f lightrag    # View logs"
echo -e "    docker compose down                # Stop stack"
echo -e "    docker compose restart lightrag    # Restart API"
echo ""
if command -v bun &> /dev/null && [[ "$1" != "--proxy" ]]; then
    echo -e "  ${BLUE}Proxy (for K8s/code-server):${NC}"
    echo -e "    cd scripts && bun run honohub"
    echo ""
fi
