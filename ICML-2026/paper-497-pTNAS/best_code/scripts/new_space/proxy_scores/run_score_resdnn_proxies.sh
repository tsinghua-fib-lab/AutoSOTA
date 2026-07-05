#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${PROJECT_ROOT}/scripts/new_space/proxy_scores/score_resdnn_proxies.py"
OUTPUT_DIR="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/proxy_score/ptproxy"
CACHE_DIR="${OUTPUT_DIR}/batch_cache"
LOG_DIR="${OUTPUT_DIR}/logs"
RESULTS_CSV="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${LOG_DIR}"

DATASETS=(
  avito-user-clicks
  event-user-attendance
  avito-ad-ctr
  hm-user-churn
)

for IDX in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$IDX]}"
  GPU_ID="${IDX}"
  LOG_FILE="${LOG_DIR}/${DATASET}.log"
  echo "[launch] dataset=${DATASET} gpu=${GPU_ID} log=${LOG_FILE}"
  bash -lc "cd '${PROJECT_ROOT}' && PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' CUDA_VISIBLE_DEVICES='${GPU_ID}' '${PYTHON_BIN}' '${SCRIPT}' --dataset '${DATASET}' --results_csv '${RESULTS_CSV}' --output_dir '${OUTPUT_DIR}' --cache_dir '${CACHE_DIR}' --device cuda:0 --variants v1" > "${LOG_FILE}" 2>&1 &
done

wait
echo "[done] all resnet pool proxy runs finished"
