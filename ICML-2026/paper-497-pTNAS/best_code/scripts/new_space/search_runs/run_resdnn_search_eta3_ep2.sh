#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CSV="${PROJECT_ROOT}/run_outputs/data/new_space/search_runs/resdnn/ptnas_pool_search_stratified_eta3_ep2_results.csv"
SH_CSV="${PROJECT_ROOT}/run_outputs/data/new_space/search_runs/resdnn/ptnas_pool_search_stratified_eta3_ep2_sh_detail.csv"
LOG_DIR="${PROJECT_ROOT}/run_outputs/data/new_space/search_runs/resdnn/logs"
PYTHON="${PYTHON:-python}"
mkdir -p "${LOG_DIR}"

SEED=42
GPU=2

for DATASET in avito-user-clicks hm-user-churn; do
  LOG="${LOG_DIR}/resnet_strat_eta3_ep2_${DATASET}.log"
  bash -c "
    cd '${PROJECT_ROOT}'
    PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' '${PYTHON}' scripts/new_space/search_runs/ptnas_resdnn_search.py \
      --data_dir '${PROJECT_ROOT}/datasets/fit-medium-table/${DATASET}' \
      --space_file '${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/architecture/random_sampled_arch_resdnn_classification.txt' \
      --seed ${SEED} --sample_method stratified --sh_min_epochs 2 --eta 3 \
      --device cuda:$((GPU % 8)) \
      --output_csv '${CSV}' --sh_detail_csv '${SH_CSV}'
  " > "${LOG}" 2>&1 &
  echo "[launch] ${DATASET} gpu=$((GPU % 8)) pid=$!"
  GPU=$((GPU + 1))
done

for DATASET in event-user-attendance avito-ad-ctr; do
  LOG="${LOG_DIR}/resnet_strat_eta3_ep2_${DATASET}.log"
  bash -c "
    cd '${PROJECT_ROOT}'
    PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' '${PYTHON}' scripts/new_space/search_runs/ptnas_resdnn_search.py \
      --data_dir '${PROJECT_ROOT}/datasets/fit-medium-table/${DATASET}' \
      --space_file '${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/architecture/random_sampled_arch_resdnn_regression.txt' \
      --seed ${SEED} --sample_method stratified --sh_min_epochs 2 --eta 3 \
      --device cuda:$((GPU % 8)) \
      --output_csv '${CSV}' --sh_detail_csv '${SH_CSV}'
  " > "${LOG}" 2>&1 &
  echo "[launch] ${DATASET} gpu=$((GPU % 8)) pid=$!"
  GPU=$((GPU + 1))
done

echo "[wait] 5 processes..."
wait
echo "[done] CSV: ${CSV}"
