#!/bin/bash
# Quick script to populate YAR with diverse test documents
#
# This downloads Wikipedia articles across 4 domains (Medical, Finance, Climate, Sports)
# and ingests them into YAR. The articles are chosen to have entity overlap
# (WHO, Carbon/Emissions, Organizations) to test entity merging and summarization.
#
# Usage:
#   ./yar/evaluation/populate_test_data.sh
#   LIGHTRAG_API_URL=http://localhost:9622 ./yar/evaluation/populate_test_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_URL="${LIGHTRAG_API_URL:-http://localhost:9622}"

echo "=== YAR Test Data Population ==="
echo "RAG URL: $RAG_URL"
echo ""

# Check if YAR is running
if ! curl -s "$RAG_URL/health" > /dev/null 2>&1; then
    echo "✗ Cannot connect to YAR at $RAG_URL"
    echo "  Make sure YAR is running first"
    exit 1
fi

# 1. Download Wikipedia articles
echo "[1/2] Downloading Wikipedia articles..."
python3 "$SCRIPT_DIR/download_wikipedia.py"

# 2. Ingest into YAR
echo ""
echo "[2/2] Ingesting documents..."
python3 "$SCRIPT_DIR/ingest_test_docs.py" --rag-url "$RAG_URL"

echo ""
echo "=== Done! ==="
echo "Documents ingested into YAR."
echo ""
echo "Next steps:"
echo "  - Check graph stats: curl $RAG_URL/graph/statistics"
echo "  - Query the data: curl '$RAG_URL/query?mode=global&query=What+is+climate+change'"
