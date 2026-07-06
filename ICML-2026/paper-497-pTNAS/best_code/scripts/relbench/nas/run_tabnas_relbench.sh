#!/usr/bin/env bash
# TabNAS baseline: sequential, one dataset at a time
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="run_outputs/data/relbench/baselines"
CSV="${OUT_DIR}/tabnas_relbench_results.csv"
LOG_DIR="${OUT_DIR}/tabnas_logs"
mkdir -p "${LOG_DIR}"
CONDA_EXE="${CONDA_EXE:-conda}"

DATASETS=(
  avito-ad-ctr
  event-user-repeat
  avito-user-clicks
  event-user-attendance
  trial-study-outcome
  trial-site-success
  ratebeer-user-active
  ratebeer-beer-positive
  hm-user-churn
  hm-item-sales
)

GPU=0
for DS in "${DATASETS[@]}"; do
  LOG="${LOG_DIR}/tabnas_${DS}.log"
  echo "[run] ${DS} gpu=$((GPU % 8))"

  "${CONDA_EXE}" run -n ptnas --no-capture-output bash -c "
    OMP_NUM_THREADS=4 PYTHONPATH='src:.' python scripts/relbench/nas/tab_nas_relbench.py \
      --data_dir 'datasets/fit-medium-table/${DS}' \
      --device cuda:$((GPU % 8)) \
      --max_iter 500 --time_budget 8 --n_reps 3 \
      --output_csv '${CSV}'
  " > "${LOG}" 2>&1

  echo "[done] ${DS}"
  GPU=$((GPU + 1))
done

echo "[all done] CSV: ${CSV}"
