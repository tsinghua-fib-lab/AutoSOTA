#!/bin/bash
# Complete DBpedia1536 benchmark workflow for PAG
set -euo pipefail

DATA_DIR="/datasets/dbpedia1536"
BASE_COUNT=999000
QUERY_COUNT=1000
DIM=1536
TOPK=100
MAX_SEARCH_K=1000
EF_CONSTRUCTION=1000
TARGET_DEGREE=32
PROJECTION_LEVELS=128
METRIC=l2
INDEX_DIR="${DATA_DIR}/index_pag_base"

echo "=== PAG DBpedia1536 Benchmark ==="
echo "Parameters:"
echo "  ef_construction=${EF_CONSTRUCTION}"
echo "  target_degree (M)=${TARGET_DEGREE}"
echo "  projection_levels (L)=${PROJECTION_LEVELS}"
echo "  metric=${METRIC}"
echo "  K (topk)=${TOPK}"
echo "  max_search_k=${MAX_SEARCH_K}"
echo ""

cd /repo

if [ -d "${INDEX_DIR}" ]; then
    echo "Index exists at ${INDEX_DIR}, running search benchmark..."
else
    echo "Building index at ${INDEX_DIR}..."
fi

./build/PAG \
    "${DATA_DIR}/base.fbin" \
    "${DATA_DIR}/query.fbin" \
    "${DATA_DIR}/gt1000.ibin" \
    "${INDEX_DIR}" \
    "${BASE_COUNT}" \
    "${QUERY_COUNT}" \
    "${DIM}" \
    "${TOPK}" \
    "${EF_CONSTRUCTION}" \
    "${TARGET_DEGREE}" \
    "${PROJECTION_LEVELS}" \
    "${METRIC}" \
    "${MAX_SEARCH_K}"

echo ""
echo "=== Benchmark Complete ==="
