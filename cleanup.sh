#!/bin/bash
# LightRAG Cleanup Script
# Removes LightRAG Docker resources WITHOUT touching other stuff
#
# Usage:
#   ./cleanup.sh           # Remove containers, images, network (keeps data)
#   ./cleanup.sh --volumes # Also remove volumes (DELETES ALL DATA)
#   ./cleanup.sh --all     # Nuclear option: remove everything including data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              LightRAG Cleanup                                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse args
REMOVE_VOLUMES=false
REMOVE_ALL=false

for arg in "$@"; do
    case $arg in
        --volumes)
            REMOVE_VOLUMES=true
            ;;
        --all)
            REMOVE_ALL=true
            REMOVE_VOLUMES=true
            ;;
        --help|-h)
            echo "Usage: ./cleanup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (none)      Remove containers, images, network (keeps data volumes)"
            echo "  --volumes   Also remove Docker volumes (DELETES RAG DATA)"
            echo "  --all       Remove everything including local data/ directory"
            echo ""
            exit 0
            ;;
    esac
done

# ══════════════════════════════════════════════════════════════════════════════
# Show what will be removed
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}🔍 Scanning for LightRAG resources...${NC}"
echo ""

# Containers (lightrag-stack project)
CONTAINERS=$(docker ps -a --filter "label=com.docker.compose.project=lightrag-stack" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$CONTAINERS" ]; then
    echo -e "${BLUE}Containers:${NC}"
    echo "$CONTAINERS" | while read -r c; do echo "  • $c"; done
else
    echo -e "${BLUE}Containers:${NC} (none found)"
fi

# Images
IMAGES=$(docker images --filter "reference=lightrag-stack*" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null || true)
if [ -n "$IMAGES" ]; then
    echo -e "${BLUE}Images:${NC}"
    echo "$IMAGES" | while read -r i; do echo "  • $i"; done
else
    echo -e "${BLUE}Images:${NC} (none found)"
fi

# Network
NETWORK="lightrag-stack_lightrag-network"
if docker network inspect "$NETWORK" >/dev/null 2>&1; then
    echo -e "${BLUE}Network:${NC}"
    echo "  • $NETWORK"
else
    echo -e "${BLUE}Network:${NC} (none found)"
fi

# Volumes
VOLUMES=$(docker volume ls --filter "label=com.docker.compose.project=lightrag-stack" --format "{{.Name}}" 2>/dev/null || true)
if [ -z "$VOLUMES" ]; then
    # Try by name pattern
    VOLUMES=$(docker volume ls --format "{{.Name}}" 2>/dev/null | grep -E "^lightrag-stack_" || true)
fi
if [ -n "$VOLUMES" ]; then
    echo -e "${BLUE}Volumes:${NC}"
    echo "$VOLUMES" | while read -r v; do echo "  • $v"; done
    if [ "$REMOVE_VOLUMES" = false ]; then
        echo -e "  ${YELLOW}(will be kept - use --volumes to remove)${NC}"
    fi
else
    echo -e "${BLUE}Volumes:${NC} (none found)"
fi

# Local data
if [ -d "data" ]; then
    DATA_SIZE=$(du -sh data 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "${BLUE}Local data/:${NC}"
    echo "  • data/rag_storage"
    echo "  • data/inputs"
    echo "  • Size: $DATA_SIZE"
    if [ "$REMOVE_ALL" = false ]; then
        echo -e "  ${YELLOW}(will be kept - use --all to remove)${NC}"
    fi
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Confirmation
# ══════════════════════════════════════════════════════════════════════════════

if [ "$REMOVE_VOLUMES" = true ] || [ "$REMOVE_ALL" = true ]; then
    echo -e "${RED}⚠️  WARNING: This will delete data!${NC}"
    if [ "$REMOVE_VOLUMES" = true ]; then
        echo -e "  • Docker volumes (PostgreSQL database, RustFS files)"
    fi
    if [ "$REMOVE_ALL" = true ]; then
        echo -e "  • Local data/ directory (RAG storage, uploaded documents)"
    fi
    echo ""
    read -p "Are you sure? Type 'yes' to confirm: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
else
    read -p "Proceed with cleanup? [Y/n]: " CONFIRM
    if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Stop and remove containers
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}🛑 [Step 1/4] Stopping containers...${NC}"

if [ -f "docker-compose.yml" ]; then
    docker compose down 2>&1 | while read -r line; do
        echo -e "  ${BLUE}▸${NC} $line"
    done
else
    # Fallback: stop by name
    if [ -n "$CONTAINERS" ]; then
        echo "$CONTAINERS" | xargs -r docker stop 2>/dev/null || true
        echo "$CONTAINERS" | xargs -r docker rm 2>/dev/null || true
    fi
fi

echo -e "${GREEN}✓ Containers removed${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Remove images
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}🗑️  [Step 2/4] Removing images...${NC}"

# Get image IDs for lightrag-stack
IMAGE_IDS=$(docker images --filter "reference=lightrag-stack*" -q 2>/dev/null || true)

if [ -n "$IMAGE_IDS" ]; then
    echo "$IMAGE_IDS" | xargs -r docker rmi -f 2>&1 | while read -r line; do
        echo -e "  ${BLUE}▸${NC} $line"
    done
    echo -e "${GREEN}✓ Images removed${NC}"
else
    echo -e "  (no images to remove)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Remove network
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}🌐 [Step 3/4] Removing network...${NC}"

if docker network inspect "$NETWORK" >/dev/null 2>&1; then
    docker network rm "$NETWORK" 2>&1 | while read -r line; do
        echo -e "  ${BLUE}▸${NC} $line"
    done
    echo -e "${GREEN}✓ Network removed${NC}"
else
    echo -e "  (network already removed)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Remove volumes (if requested)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${YELLOW}💾 [Step 4/4] Handling volumes...${NC}"

if [ "$REMOVE_VOLUMES" = true ]; then
    # Re-fetch volumes
    VOLUMES=$(docker volume ls --format "{{.Name}}" 2>/dev/null | grep -E "^lightrag-stack_" || true)

    if [ -n "$VOLUMES" ]; then
        echo "$VOLUMES" | while read -r vol; do
            docker volume rm "$vol" 2>&1 | while read -r line; do
                echo -e "  ${BLUE}▸${NC} Removed $vol"
            done
        done
        echo -e "${GREEN}✓ Volumes removed${NC}"
    else
        echo -e "  (no volumes to remove)"
    fi
else
    echo -e "  ${BLUE}ℹ${NC} Volumes preserved (use --volumes to remove)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Remove local data (if --all)
# ══════════════════════════════════════════════════════════════════════════════

if [ "$REMOVE_ALL" = true ] && [ -d "data" ]; then
    echo ""
    echo -e "${YELLOW}📂 Removing local data/...${NC}"
    rm -rf data
    echo -e "${GREEN}✓ Local data removed${NC}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Cleanup Complete                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$REMOVE_VOLUMES" = false ]; then
    echo -e "  ${GREEN}✓${NC} Containers, images, network removed"
    echo -e "  ${BLUE}ℹ${NC} Volumes preserved (your data is safe)"
    echo ""
    echo -e "  To also remove volumes: ${YELLOW}./cleanup.sh --volumes${NC}"
else
    echo -e "  ${GREEN}✓${NC} All Docker resources removed"
fi

if [ "$REMOVE_ALL" = true ]; then
    echo -e "  ${GREEN}✓${NC} Local data/ directory removed"
fi

echo ""
echo -e "  To start fresh: ${BLUE}./setup.sh${NC}"
echo ""
