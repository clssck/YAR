#!/bin/bash
# YAR Cleanup Script
# Removes YAR Docker resources WITHOUT touching other stuff
#
# Usage:
#   ./cleanup.sh               # Interactive menu
#   ./cleanup.sh --volumes     # Preselect cleanup plus Docker volumes
#   ./cleanup.sh --all         # Preselect cleanup plus all local data
#   ./cleanup.sh --eval-corpus # Preselect eval corpus reset flow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo -e "${BLUE}|              YAR Cleanup                                      |${NC}"
echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo ""

# Parse args
REMOVE_VOLUMES=false
REMOVE_ALL=false
EVAL_CORPUS=false
EVAL_ARGS=()
EVAL_ARG_SEEN=false

show_help() {
    echo "Usage: ./cleanup.sh [OPTIONS]"
    echo ""
    echo "Run without options for the interactive menu."
    echo ""
    echo "Options:"
    echo "  --volumes     Remove containers, images, network, and Docker volumes"
    echo "  --all         Remove containers, images, network, Docker volumes, and local data/"
    echo "  --eval-corpus Reset Postgres/RustFS corpus and ingest Phoenix eval docs"
    echo ""
    echo "Eval corpus options passed to scripts/reset_eval_corpus.sh:"
    echo "  --skip-wipe --skip-ingest --rag-url URL --profile NAME --server-log PATH"
    echo ""
}

choose_interactive_mode() {
    local choice
    echo -e "${YELLOW}Choose cleanup mode:${NC}"
    echo "  1) Remove containers, images, network (keep data)"
    echo "  2) Remove containers, images, network, and Docker volumes"
    echo "  3) Remove everything including local data/"
    echo "  4) Reset eval corpus and ingest Phoenix eval docs"
    echo "  5) Exit"
    echo ""
    read -r -p "Choice [1]: " choice

    case "${choice:-1}" in
        1)
            ;;
        2)
            REMOVE_VOLUMES=true
            ;;
        3)
            REMOVE_ALL=true
            REMOVE_VOLUMES=true
            ;;
        4)
            EVAL_CORPUS=true
            ;;
        5|q|Q|quit|exit)
            echo "Aborted."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice:${NC} $choice" >&2
            exit 2
            ;;
    esac

    if [ "$EVAL_CORPUS" = true ]; then
        local reset_choice
        local ingest_choice
        local profile
        local rag_url

        echo ""
        read -r -p "Wipe Postgres/RustFS datastore before ingest? [Y/n]: " reset_choice
        if [[ "$reset_choice" =~ ^[Nn]$ ]]; then
            EVAL_ARGS+=(--skip-wipe)
        fi

        read -r -p "Upload Phoenix eval documents after reset? [Y/n]: " ingest_choice
        if [[ "$ingest_choice" =~ ^[Nn]$ ]]; then
            EVAL_ARGS+=(--skip-ingest)
        fi

        read -r -p "YAR profile [${YAR_PROFILE:-dev}]: " profile
        profile="${profile:-${YAR_PROFILE:-dev}}"
        case "$profile" in
            dev|work)
                EVAL_ARGS+=(--profile "$profile")
                ;;
            *)
                echo -e "${RED}Invalid profile:${NC} $profile (expected dev or work)" >&2
                exit 2
                ;;
        esac

        read -r -p "YAR API URL [${YAR_API_URL:-http://127.0.0.1:${PORT:-9621}}]: " rag_url
        if [ -n "$rag_url" ]; then
            EVAL_ARGS+=(--rag-url "$rag_url")
        fi
    fi
}

if [ "$#" -eq 0 ]; then
    choose_interactive_mode
else
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --volumes)
                REMOVE_VOLUMES=true
                ;;
            --all)
                REMOVE_ALL=true
                REMOVE_VOLUMES=true
                ;;
            --eval-corpus)
                EVAL_CORPUS=true
                ;;
            --skip-wipe|--skip-ingest)
                EVAL_ARGS+=("$1")
                EVAL_ARG_SEEN=true
                ;;
            --rag-url|--profile|--server-log)
                EVAL_ARGS+=("$1" "${2:?missing value for $1}")
                EVAL_ARG_SEEN=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option:${NC} $1" >&2
                echo "Run './cleanup.sh --help' for usage." >&2
                exit 2
                ;;
        esac
        shift
    done
fi

if [ "$EVAL_CORPUS" = true ]; then
    if [ "$REMOVE_VOLUMES" = true ] || [ "$REMOVE_ALL" = true ]; then
        echo -e "${RED}--eval-corpus cannot be combined with --volumes or --all.${NC}" >&2
        exit 2
    fi
    if [ ! -x "scripts/reset_eval_corpus.sh" ]; then
        echo -e "${RED}Missing executable script:${NC} scripts/reset_eval_corpus.sh" >&2
        exit 1
    fi
    exec "scripts/reset_eval_corpus.sh" "${EVAL_ARGS[@]}"
fi

if [ "$EVAL_ARG_SEEN" = true ]; then
    echo -e "${RED}Eval corpus options require --eval-corpus.${NC}" >&2
    exit 2
fi

# ══════════════════════════════════════════════════════════════════════════════
# Show what will be removed
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}🔍 Scanning for YAR resources...${NC}"
echo ""

# Containers (yar-stack project)
CONTAINERS=$(docker ps -a --filter "label=com.docker.compose.project=yar-stack" --format "{{.Names}}" 2>/dev/null || true)
if [ -n "$CONTAINERS" ]; then
    echo -e "${BLUE}Containers:${NC}"
    echo "$CONTAINERS" | while read -r c; do echo "  • $c"; done
else
    echo -e "${BLUE}Containers:${NC} (none found)"
fi

# Images
IMAGES=$(docker images --filter "reference=yar-stack*" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null || true)
if [ -n "$IMAGES" ]; then
    echo -e "${BLUE}Images:${NC}"
    echo "$IMAGES" | while read -r i; do echo "  • $i"; done
else
    echo -e "${BLUE}Images:${NC} (none found)"
fi

# Network
NETWORK="yar-stack_yar-network"
if docker network inspect "$NETWORK" >/dev/null 2>&1; then
    echo -e "${BLUE}Network:${NC}"
    echo "  • $NETWORK"
else
    echo -e "${BLUE}Network:${NC} (none found)"
fi

# Volumes
VOLUMES=$(docker volume ls --filter "label=com.docker.compose.project=yar-stack" --format "{{.Name}}" 2>/dev/null || true)
if [ -z "$VOLUMES" ]; then
    # Try by name pattern
    VOLUMES=$(docker volume ls --format "{{.Name}}" 2>/dev/null | grep -E "^yar-stack_" || true)
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
    read -r -p "Are you sure? Type 'yes' to confirm: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
else
    read -r -p "Proceed with cleanup? [Y/n]: " CONFIRM
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
    # Stop all services including work profile (litellm)
    docker compose --profile work down 2>&1 | while read -r line; do
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

# Get image IDs for yar-stack
IMAGE_IDS=$(docker images --filter "reference=yar-stack*" -q 2>/dev/null || true)

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
    VOLUMES=$(docker volume ls --format "{{.Name}}" 2>/dev/null | grep -E "^yar-stack_" || true)

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
echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo -e "${BLUE}|                    Cleanup Complete                              |${NC}"
echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
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
