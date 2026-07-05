#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

BASE_DATA_DIR="datasets/fit-medium-table"
DEVICE="${DEVICE:-cuda:7}"
SPACE="${SPACE:-resnet}"
SEED="${SEED:-42}"
TIME_BUDGET="${TIME_BUDGET:-10}"
DEFAULT_MK_RATIO="${DEFAULT_MK_RATIO:-60}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-run_outputs/data/relbench/ptnas}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/ptnas_full_timebudget_${TIME_BUDGET}s_${TIMESTAMP}.csv}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run_ptnas_full_timebudget_${TIME_BUDGET}s_${TIMESTAMP}.log}"
CONDA_EXE="${CONDA_EXE:-conda}"

DATASETS=(
  avito-user-clicks
  event-user-attendance
  event-user-repeat
  hm-item-sales
  ratebeer-beer-positive
  ratebeer-user-active
  trial-site-success
  trial-study-outcome
)

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "[run] $(date)"
echo "[run] root=${ROOT_DIR}"
echo "[run] space=${SPACE} device=${DEVICE} seed=${SEED}"
echo "[run] given_time_budget=${TIME_BUDGET}"
echo "[run] default_mk_ratio=${DEFAULT_MK_RATIO}"
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

  EXTRA_ARGS=(
    --given_time_budget "${TIME_BUDGET}"
    --mk_ratio "${DEFAULT_MK_RATIO}"
  )

  # Dataset-specific best settings validated from current time-budget runs.
  case "${DATASET}" in
    avito-user-clicks)
      EXTRA_ARGS+=(
        --mk_ratio 90
      )
      ;;
    hm-item-sales)
      EXTRA_ARGS+=(
        --mk_ratio 90
      )
      ;;
    trial-study-outcome)
      EXTRA_ARGS+=(
        --mk_ratio 90
      )
      ;;
    trial-site-success)
      EXTRA_ARGS+=(
        --mk_ratio 120
      )
      ;;
  esac

  echo ""
  echo "================================================================"
  echo "[dataset] ${DATASET} $(date)"
  echo "================================================================"
  echo "[dataset] extra args: ${EXTRA_ARGS[*]}"

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
