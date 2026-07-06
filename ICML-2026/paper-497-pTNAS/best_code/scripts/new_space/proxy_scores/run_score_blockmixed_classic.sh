#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
SCRIPT="${PROJECT_ROOT}/scripts/new_space/proxy_scores/score_blockmixed_classic_proxies.py"
SPACE_FILE="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/architecture/blockmixed.txt"
OUT_DIR="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/proxy_score/baseline"
CACHE_DIR="${OUT_DIR}/batch_cache"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

DATASETS=(
  avito-user-clicks
  event-user-attendance
  avito-ad-ctr
  hm-user-churn
  trial-site-success
)

for IDX in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$IDX]}"
  GPU="${IDX}"
  LOG_FILE="${LOG_DIR}/${DATASET}.log"
  echo "[launch] dataset=${DATASET} gpu=${GPU} log=${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT}" \
    --dataset "${DATASET}" \
    --space_file "${SPACE_FILE}" \
    --output_dir "${OUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --device "cuda:0" \
    > "${LOG_FILE}" 2>&1 &
done

wait
echo "[done] all baseline proxy runs finished"
