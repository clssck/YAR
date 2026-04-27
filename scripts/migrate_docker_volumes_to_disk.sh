#!/bin/bash
# Migrate existing Docker named volumes (pgdata, rustfs_data) to bind mounts on disk.
#
# When to use: ONLY if you have an existing yar-stack deployment with data in the
# legacy named volumes and you've just pulled docker-compose.yml that switched to
# bind mounts under ${YAR_DATA_DIR:-./data}. New installs do not need this.
#
# What it does:
#   1. Stops yar-postgres + yar-rustfs containers
#   2. Copies the contents of the named volumes pgdata and rustfs_data into
#      ${YAR_DATA_DIR:-./data}/postgres and ${YAR_DATA_DIR:-./data}/rustfs
#   3. Restarts the containers (which now bind-mount the on-disk dirs)
#   4. Leaves the original named volumes intact so you can roll back

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

DATA_DIR="${YAR_DATA_DIR:-./data}"
POSTGRES_DIR="$DATA_DIR/postgres"
RUSTFS_DIR="$DATA_DIR/rustfs"

# Resolve actual named-volume names. compose prefixes them with the project name
# from the top-level `name:` field in docker-compose.yml (yar-stack).
PG_VOLUME="yar-stack_pgdata"
RUSTFS_VOLUME="yar-stack_rustfs_data"

echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo -e "${BLUE}|        YAR Docker Volume -> Disk Migration                       |${NC}"
echo -e "${BLUE}+------------------------------------------------------------------+${NC}"
echo ""
echo -e "Source volumes: ${YELLOW}$PG_VOLUME${NC}, ${YELLOW}$RUSTFS_VOLUME${NC}"
echo -e "Target dirs:    ${YELLOW}$POSTGRES_DIR${NC}, ${YELLOW}$RUSTFS_DIR${NC}"
echo ""

if ! docker volume inspect "$PG_VOLUME" >/dev/null 2>&1; then
    echo -e "${YELLOW}i Named volume $PG_VOLUME does not exist. Nothing to migrate for postgres.${NC}"
    PG_NEEDS_MIGRATION=false
else
    PG_NEEDS_MIGRATION=true
fi

if ! docker volume inspect "$RUSTFS_VOLUME" >/dev/null 2>&1; then
    echo -e "${YELLOW}i Named volume $RUSTFS_VOLUME does not exist. Nothing to migrate for rustfs.${NC}"
    RUSTFS_NEEDS_MIGRATION=false
else
    RUSTFS_NEEDS_MIGRATION=true
fi

if [ "$PG_NEEDS_MIGRATION" = "false" ] && [ "$RUSTFS_NEEDS_MIGRATION" = "false" ]; then
    echo -e "${GREEN}* No migration needed. Both volumes already absent.${NC}"
    exit 0
fi

if [ -d "$POSTGRES_DIR" ] && [ -n "$(ls -A "$POSTGRES_DIR" 2>/dev/null || true)" ]; then
    echo -e "${RED}x $POSTGRES_DIR is not empty. Refusing to overwrite.${NC}"
    echo -e "  Either remove it or set YAR_DATA_DIR to a fresh location."
    exit 1
fi

if [ -d "$RUSTFS_DIR" ] && [ -n "$(ls -A "$RUSTFS_DIR" 2>/dev/null || true)" ]; then
    echo -e "${RED}x $RUSTFS_DIR is not empty. Refusing to overwrite.${NC}"
    echo -e "  Either remove it or set YAR_DATA_DIR to a fresh location."
    exit 1
fi

read -p "Stop yar-postgres + yar-rustfs and migrate? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}[1/4] Stopping containers...${NC}"
docker compose stop postgres rustfs 2>/dev/null || true

mkdir -p "$POSTGRES_DIR" "$RUSTFS_DIR"

if [ "$PG_NEEDS_MIGRATION" = "true" ]; then
    echo -e "${YELLOW}[2/4] Copying $PG_VOLUME -> $POSTGRES_DIR ...${NC}"
    docker run --rm \
        -v "$PG_VOLUME:/from:ro" \
        -v "$(realpath "$POSTGRES_DIR"):/to" \
        alpine sh -c 'cp -a /from/. /to/'
    echo -e "${GREEN}* Postgres data copied${NC}"
else
    echo -e "${YELLOW}[2/4] Skipping postgres (no source volume)${NC}"
fi

if [ "$RUSTFS_NEEDS_MIGRATION" = "true" ]; then
    echo -e "${YELLOW}[3/4] Copying $RUSTFS_VOLUME -> $RUSTFS_DIR ...${NC}"
    docker run --rm \
        -v "$RUSTFS_VOLUME:/from:ro" \
        -v "$(realpath "$RUSTFS_DIR"):/to" \
        alpine sh -c 'cp -a /from/. /to/'
    echo -e "${GREEN}* RustFS data copied${NC}"
else
    echo -e "${YELLOW}[3/4] Skipping rustfs (no source volume)${NC}"
fi

echo -e "${YELLOW}[4/4] Restarting containers...${NC}"
docker compose up -d postgres rustfs

echo ""
echo -e "${GREEN}* Migration complete.${NC}"
echo ""
echo -e "  Verify with: ${BLUE}docker compose ps${NC}"
echo -e "  Inspect data: ${BLUE}sudo du -sh $POSTGRES_DIR $RUSTFS_DIR${NC}"
echo ""
echo -e "${YELLOW}Original named volumes are still present as backup.${NC}"
echo -e "  Once you've verified everything works, remove them with:"
echo -e "    ${BLUE}docker volume rm $PG_VOLUME $RUSTFS_VOLUME${NC}"
