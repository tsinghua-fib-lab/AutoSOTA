#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

BASE_DATA_DIR="datasets/fit-medium-table"
DEVICE="${DEVICE:-cuda:7}"
SPACE="${SPACE:-resnet}"
SEED="${SEED:-42}"
TOP_K="${TOP_K:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-run_outputs/data/relbench/ptnas}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/ptnas_full_relbench_results_${TIMESTAMP}.csv}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run_ptnas_full_relbench_${TIMESTAMP}.log}"
CONDA_EXE="${CONDA_EXE:-conda}"

DATASETS=(
  # RelBench datasets used by the full per-dataset result table:
  avito-user-clicks
  event-user-attendance
  event-user-repeat
  hm-item-sales
  ratebeer-beer-positive
  ratebeer-user-active
  trial-site-success
  trial-study-outcome

  # Newly added datasets:
  # avito-ad-ctr
  # hm-user-churn
)

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "[run] $(date)"
echo "[run] root=${ROOT_DIR}"
echo "[run] space=${SPACE} device=${DEVICE} seed=${SEED}"
echo "[run] top_k=${TOP_K:-default}"
echo "[run] output_csv=${OUTPUT_CSV}"
echo "[run] log=${LOG_FILE}"
echo "[run] datasets=${#DATASETS[@]}"
echo "================================================================"

for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${BASE_DATA_DIR}/${DATASET}"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[skip] ${DATASET}: not found"
    continue
  fi

  EXTRA_ARGS=()
  if [[ -n "${TOP_K}" ]]; then
    EXTRA_ARGS+=(--top_k "${TOP_K}")
  fi
  if [[ "${DATASET}" == "trial-site-success" ]]; then
    EXTRA_ARGS+=(
      --population 120
      --generations 30
      --sh_min_epochs 1
      --sh_max_epochs 50
      --eta 3
      --final_lr 0.0005
      --final_epochs 200
      --final_batch_size 512
      --final_dropout 0.0
      --final_max_batches 20
      --final_early_stop 30
    )
    if [[ -z "${TOP_K}" ]]; then
      EXTRA_ARGS+=(--top_k 30)
    fi
  fi

  echo ""
  echo "================================================================"
  echo "[dataset] ${DATASET} $(date)"
  echo "================================================================"
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "[dataset] extra args: ${EXTRA_ARGS[*]}"
  fi

  "${CONDA_EXE}" run -n ptnas --no-capture-output bash -c "
    PYTHONPATH='src:.' python3 scripts/ptnas_full.py \
      --data_dir '${DATA_DIR}' \
      --space_name '${SPACE}' \
      --output_csv '${OUTPUT_CSV}' \
      --device '${DEVICE}' \
      --seed '${SEED}' \
      ${EXTRA_ARGS[*]}
  "

  echo "[done] ${DATASET} $(date)"
done

echo ""
echo "================================================================"
echo "[all done] $(date)"
echo "================================================================"
