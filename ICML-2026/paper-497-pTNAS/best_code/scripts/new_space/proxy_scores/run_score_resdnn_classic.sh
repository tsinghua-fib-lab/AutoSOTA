#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
SCRIPT="${PROJECT_ROOT}/scripts/new_space/proxy_scores/score_resdnn_proxies.py"
OUTPUT_DIR="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/proxy_score/baseline"
CACHE_DIR="${OUTPUT_DIR}/batch_cache"
LOG_DIR="${OUTPUT_DIR}/logs"
RESULTS_CSV="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv"
VARIANTS="GradNorm,NASWOT,NTKCond,NTKTrace,NTKTrAppx,Fisher,GraSP,SNIP,SynFlow"

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
  bash -lc "cd '${PROJECT_ROOT}' && PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' CUDA_VISIBLE_DEVICES='${GPU_ID}' '${PYTHON}' '${SCRIPT}' --dataset '${DATASET}' --results_csv '${RESULTS_CSV}' --output_dir '${OUTPUT_DIR}' --cache_dir '${CACHE_DIR}' --device cuda:0 --variants '${VARIANTS}' --merge_variants" > "${LOG_FILE}" 2>&1 &
done

wait
echo "[done] all ResDNN baseline proxy runs finished"
