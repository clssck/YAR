#!/bin/bash
# Reset PostgreSQL/RustFS corpus data and ingest the Phoenix eval documents.
#
# Usage:
#   scripts/reset_eval_corpus.sh --yes
#   scripts/reset_eval_corpus.sh --yes --skip-wipe
#   YAR_API_URL=http://127.0.0.1:9621 scripts/reset_eval_corpus.sh --yes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ -f .env ]; then
  set -a
  set +u
  # shellcheck disable=SC1091
  source .env
  set -u
  set +a
fi

RAG_URL="${YAR_API_URL:-http://127.0.0.1:${PORT:-9621}}"
PROFILE="${YAR_PROFILE:-dev}"
DATA_DIR="${YAR_DATA_DIR:-./data}"
SERVER_LOG="${SERVER_LOG:-/tmp/yar-reset-eval-server.log}"
YES=false
SKIP_WIPE=false
SKIP_INGEST=false

DOCS=(
  "inputs/2024-02-21_Fitusiran PMG Green Light Presentation.pdf"
  "inputs/default/__enqueued__/2023_09_07_PKU IND CMC Dossier Lessons Learned .pdf"
  "inputs/default/__enqueued__/190917_Lessons Learned - Project freeze.pptx"
  "inputs/default/__enqueued__/16-LLsession_12-TechTransfer implementation-final.pptx"
  "inputs/default/__enqueued__/16-LLsession-09- outcome_Jan 18 2017.pptx"
  "inputs/default/__enqueued__/18-LLsession-02-Devpt and supply of blinded comparator- outcome_Oct 18_VF.pptx"
  "inputs/default/__enqueued__/17-LLsession-01-Risk Review CIR 15 march 2017.pdf"
  "inputs/default/__enqueued__/2020-06-26 CSTD Strategy WG  FINAL_for cross-share_CT_VF.pptx"
)

usage() {
  cat <<EOF
Usage: scripts/reset_eval_corpus.sh [OPTIONS]

Deletes persistent PostgreSQL and RustFS bind-mounted data, restarts infra,
starts yar-server, and uploads the Phoenix eval corpus.

Options:
  --yes              Required for non-interactive destructive reset.
  --skip-wipe        Do not delete ${DATA_DIR}/postgres or ${DATA_DIR}/rustfs.
  --skip-ingest      Reset/restart only; do not upload documents.
  --rag-url URL      YAR API URL (default: $RAG_URL).
  --profile NAME     start.sh profile: dev or work (default: $PROFILE).
  --server-log PATH  Server log path (default: $SERVER_LOG).
  -h, --help         Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes)
      YES=true
      ;;
    --skip-wipe)
      SKIP_WIPE=true
      ;;
    --skip-ingest)
      SKIP_INGEST=true
      ;;
    --rag-url)
      RAG_URL="${2:?missing value for --rag-url}"
      shift
      ;;
    --profile)
      PROFILE="${2:?missing value for --profile}"
      shift
      ;;
    --server-log)
      SERVER_LOG="${2:?missing value for --server-log}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option:${NC} $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$PROFILE" in
  dev|work) ;;
  *)
    echo -e "${RED}Invalid profile:${NC} $PROFILE (expected dev or work)" >&2
    exit 2
    ;;
esac

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo -e "${RED}Missing required command:${NC} $1" >&2
    exit 1
  fi
}

require_cmd docker
require_cmd uv
require_cmd curl
require_cmd lsof


missing=()
for doc in "${DOCS[@]}"; do
  if [ ! -f "$doc" ]; then
    missing+=("$doc")
  fi
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo -e "${RED}Missing eval document(s):${NC}" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

if [ "$SKIP_WIPE" = false ] && [ "$YES" = false ]; then
  echo -e "${YELLOW}This will delete:${NC}"
  echo "  ${DATA_DIR}/postgres"
  echo "  ${DATA_DIR}/rustfs"
  echo "PostgreSQL includes the Phoenix schema because Phoenix shares this database."
  echo "Eval source files under inputs/ are preserved."
  read -r -p "Type 'yes' to continue: " CONFIRM
  if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
  fi
fi

stop_yar_server() {
  local port
  port="${RAG_URL##*:}"
  port="${port%%/*}"
  local pids
  pids="$(lsof -ti "tcp:$port" || true)"
  if [ -n "$pids" ]; then
    echo -e "${BLUE}Stopping yar-server on port $port:${NC} $pids"
    printf '%s\n' "$pids" | xargs /bin/kill
    sleep 2
  fi
}

wait_container_healthy() {
  local container="$1"
  local label="$2"
  local state=""
  for _ in $(seq 1 120); do
    state="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)"
    if [ "$state" = "healthy" ] || [ "$state" = "running" ]; then
      echo -e "  ${GREEN}$label ready${NC}"
      return 0
    fi
    sleep 2
  done
  echo -e "${RED}$label did not become ready; last state: ${state:-unknown}${NC}" >&2
  docker compose --profile observability ps >&2 || true
  exit 1
}

wait_yar_health() {
  for _ in $(seq 1 180); do
    if curl -fsS "$RAG_URL/health" >/dev/null 2>&1; then
      echo -e "  ${GREEN}yar-server ready:${NC} $RAG_URL"
      return 0
    fi
    sleep 1
  done
  echo -e "${RED}yar-server did not become healthy.${NC}" >&2
  echo "Last 120 server log lines from $SERVER_LOG:" >&2
  tail -n 120 "$SERVER_LOG" >&2 || true
  exit 1
}

wipe_data_dirs() {
  echo -e "${BLUE}Stopping infra containers...${NC}"
  docker compose --profile observability down
  echo -e "${BLUE}Deleting persistent datastore dirs...${NC}"
  rm -rf "${DATA_DIR}/postgres" "${DATA_DIR}/rustfs"
}

start_infra() {
  echo -e "${BLUE}Starting PostgreSQL, RustFS, LiteLLM, Phoenix...${NC}"
  docker compose --profile observability up -d postgres rustfs litellm phoenix
  wait_container_healthy yar-postgres PostgreSQL
  wait_container_healthy yar-rustfs RustFS
  wait_container_healthy yar-litellm LiteLLM
}

start_yar_server() {
  echo -e "${BLUE}Starting yar-server (${PROFILE})...${NC}"
  : > "$SERVER_LOG"
  ./start.sh "--${PROFILE}" > "$SERVER_LOG" 2>&1 &
  echo "$!" > /tmp/yar-reset-eval-server.pid
  wait_yar_health
}

ingest_eval_docs() {
  echo -e "${BLUE}Uploading ${#DOCS[@]} eval documents...${NC}"
  local doc_list
  doc_list="$(printf '%s\n' "${DOCS[@]}")"
  export RAG_URL
  export DOC_LIST="$doc_list"
  uv run python - <<'PY'
from __future__ import annotations

import asyncio
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx

rag_url = os.environ['RAG_URL'].rstrip('/')
docs = [Path(line) for line in os.environ['DOC_LIST'].splitlines() if line.strip()]
api_key = os.getenv('YAR_API_KEY')
headers = {'X-API-Key': api_key} if api_key else {}


def status_summary(payload: dict[str, Any]) -> dict[str, int]:
    summary = {'pending': 0, 'processing': 0, 'processed': 0, 'failed': 0}
    raw = payload.get('status_summary')
    if isinstance(raw, dict) and raw:
        for key, value in raw.items():
            summary[str(key).split('.')[-1].lower()] = int(value)
        return summary
    for doc in payload.get('documents') or []:
        if isinstance(doc, dict):
            key = str(doc.get('status', '')).split('.')[-1].lower()
            summary[key] = summary.get(key, 0) + 1
    return summary


async def wait_track(client: httpx.AsyncClient, track_id: str) -> dict[str, int]:
    last: dict[str, int] | None = None
    for _ in range(720):
        response = await client.get(f'{rag_url}/documents/track_status/{track_id}', headers=headers)
        response.raise_for_status()
        summary = status_summary(response.json())
        if summary != last:
            print(f'  {track_id}: {summary}', flush=True)
            last = summary
        if summary['pending'] == 0 and summary['processing'] == 0 and (summary['processed'] + summary['failed']) > 0:
            return summary
        await asyncio.sleep(5)
    raise TimeoutError(f'timed out waiting for {track_id}')


async def main() -> None:
    timeout = httpx.Timeout(180.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(f'{rag_url}/health', headers=headers)
        health.raise_for_status()
        tracks: list[tuple[str, str]] = []
        start = time.time()
        for path in docs:
            content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            with path.open('rb') as handle:
                response = await client.post(
                    f'{rag_url}/documents/upload',
                    files={'file': (path.name, handle, content_type)},
                    headers=headers,
                )
            response.raise_for_status()
            payload = response.json()
            track_id = str(payload.get('track_id') or '')
            print(f'uploaded {path.name}: {payload.get("status")} track={track_id}', flush=True)
            if not track_id:
                raise RuntimeError(f'missing track_id for {path}')
            tracks.append((path.name, track_id))

        failures: list[tuple[str, dict[str, int]]] = []
        for name, track_id in tracks:
            summary = await wait_track(client, track_id)
            if summary.get('failed', 0):
                failures.append((name, summary))

        docs_response = await client.post(
            f'{rag_url}/documents/paginated',
            json={'page': 1, 'page_size': 50, 'sort_field': 'updated_at', 'sort_direction': 'desc'},
            headers=headers,
        )
        docs_response.raise_for_status()
        docs_payload = docs_response.json()
        print(f'total documents: {docs_payload.get("pagination", {}).get("total_count")}')
        print(f'status counts: {docs_payload.get("status_counts")}')
        print(f'elapsed: {time.time() - start:.1f}s')
        if failures:
            raise RuntimeError(f'ingest failures: {failures}')


asyncio.run(main())
PY
}

print_counts() {
  echo -e "${BLUE}PostgreSQL corpus counts:${NC}"
  docker exec yar-postgres psql -U "${POSTGRES_USER:-yar}" -d "${POSTGRES_DATABASE:-yar}" -At -c \
    "SELECT status || '=' || COUNT(*) FROM public.yar_doc_status GROUP BY status ORDER BY status;
     SELECT 'doc_chunks=' || COUNT(*) FROM public.yar_doc_chunks;
     SELECT 'vdb_chunks=' || COUNT(*) FROM public.yar_vdb_chunks;
     SELECT 'entities=' || COUNT(*) FROM public.yar_vdb_entity;
     SELECT 'relations=' || COUNT(*) FROM public.yar_vdb_relation;"
}

stop_yar_server
if [ "$SKIP_WIPE" = false ]; then
  wipe_data_dirs
fi
start_infra
start_yar_server
if [ "$SKIP_INGEST" = false ]; then
  ingest_eval_docs
fi
print_counts

echo -e "${GREEN}Done.${NC}"
