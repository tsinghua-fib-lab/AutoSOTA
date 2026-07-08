#!/usr/bin/env bash
set -euo pipefail
# PAG reproduction evaluation for DBpedia1536
# Paper settings: PAG-Base, ef_C=1000, M=32, L=128, K=100, Euclidean

DATA_DIR="${DATA_DIR:-./data/dbpedia1536}"
BASE_COUNT="${BASE_COUNT:-999000}"
QUERY_COUNT="${QUERY_COUNT:-1000}"
DIM="${DIM:-1536}"
TOPK="${TOPK:-100}"
MAX_SEARCH_K="${MAX_SEARCH_K:-100}"

EF_CONSTRUCTION="${EF_CONSTRUCTION:-1000}"
TARGET_DEGREE="${TARGET_DEGREE:-32}"
PROJECTION_LEVELS="${PROJECTION_LEVELS:-128}"
METRIC="${METRIC:-l2}"

BINARY="${BINARY:-./build/PAG}"
INDEX_DIR="${INDEX_DIR:-${DATA_DIR}/index_pag}"

exec "${BINARY}" \
  "${DATA_DIR}/base.fbin" "${DATA_DIR}/query.fbin" "${DATA_DIR}/gt100.ibin" "${INDEX_DIR}" \
  "${BASE_COUNT}" "${QUERY_COUNT}" "${DIM}" "${TOPK}" \
  "${EF_CONSTRUCTION}" "${TARGET_DEGREE}" "${PROJECTION_LEVELS}" "${METRIC}" \
  "${MAX_SEARCH_K}"
