#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPACE_FILE="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/architecture/blockmixed.txt"
OUTPUT_DIR="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/proxy_score/ptproxy"
CACHE_DIR="${OUTPUT_DIR}/batch_cache"
LOG_DIR="${OUTPUT_DIR}/logs"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${LOG_DIR}"
DATASETS=(
  avito-ad-ctr
  event-user-repeat
  avito-user-clicks
  trial-study-outcome
  event-user-attendance
  ratebeer-user-active
  trial-site-success
  ratebeer-beer-positive
  hm-user-churn
  hm-item-sales
)
MAX_JOBS=8
RUNNING_PIDS=()
wait_for_slot() {
  while [[ ${#RUNNING_PIDS[@]} -ge ${MAX_JOBS} ]]; do
    NEW_PIDS=()
    for PID in "${RUNNING_PIDS[@]}"; do
      if kill -0 "${PID}" 2>/dev/null; then
        NEW_PIDS+=("${PID}")
      fi
    done
    RUNNING_PIDS=("${NEW_PIDS[@]}")
    sleep 2
  done
}
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_scoring_${RUN_TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[run] $(date)"
echo "[space] ${SPACE_FILE}"
echo "[out] ${OUTPUT_DIR}"
echo "[cache] ${CACHE_DIR}"
IDX=0
for DATASET in "${DATASETS[@]}"; do
  GPU_ID=$(( IDX % 8 ))
  IDX=$(( IDX + 1 ))
  wait_for_slot
  echo "[launch] dataset=${DATASET} gpu=${GPU_ID}"
  bash -lc "cd '${PROJECT_ROOT}' && PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' '${PYTHON_BIN}' scripts/new_space/proxy_scores/score_blockmixed_ptproxy.py --dataset '${DATASET}' --space_file '${SPACE_FILE}' --output_dir '${OUTPUT_DIR}' --cache_dir '${CACHE_DIR}' --device cuda:${GPU_ID} --channels 32 --batch_size 32 --variants 'v1'" &
  PID=$!
  RUNNING_PIDS+=("${PID}")
  echo "[pid] dataset=${DATASET} pid=${PID} gpu=${GPU_ID}"
done
if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
  wait "${RUNNING_PIDS[@]}" || true
fi
echo "[done] $(date)"
