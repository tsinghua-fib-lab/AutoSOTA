#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train_blockmixed.sh
#
# Runs all PTNASBlockMixed models from datasets/nas_bench_tabular/space_blockmixed/architecture/blockmixed.txt.
#
# Schedule:
#   - datasets processed SEQUENTIALLY
#   - models split across PROCS_PER_GPU * NUM_GPUS processes
#   - each process loads the dataset ONCE, trains its chunk sequentially
#   - already-completed (dataset, model_index) pairs in CSV are SKIPPED
#
# One log file per sh invocation under datasets/nas_bench_tabular/space_blockmixed/training/logs/.
#
# Usage:
#   bash scripts/new_space/training/run_train_blockmixed.sh [--procs_per_gpu N] [--model_start I] [--model_end J]
# ---------------------------------------------------------------------------

set -uo pipefail
trap '' ERR

CHANNELS=32
MODEL_START=0
MODEL_END=899
NUM_EPOCHS=200
EARLY_STOP=10
MAX_ROUND_EPOCH=20
BATCH_SIZE=256
LR=0.001
PROCS_PER_GPU=2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DATA_DIR="${PROJECT_ROOT}/datasets/fit-medium-table"
SPACE_FILE="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/architecture/blockmixed.txt"
LOG_DIR="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/training/logs"
CSV_FILE="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_blockmixed/training/block_mixed_diverse_results.csv"

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_start)   MODEL_START="$2";   shift 2 ;;
    --model_end)     MODEL_END="$2";     shift 2 ;;
    --channels)      CHANNELS="$2";      shift 2 ;;
    --procs_per_gpu) PROCS_PER_GPU="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---- detect GPUs ------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=1
else
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# ---- single log file for this run ------------------------------------------
mkdir -p "${LOG_DIR}"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_block_mixed_batch_${RUN_TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "[run] $(date)"
TOTAL_PROCS=$(( NUM_GPUS * PROCS_PER_GPU ))
echo "[run] GPUs=${NUM_GPUS}  procs_per_gpu=${PROCS_PER_GPU}  total_procs=${TOTAL_PROCS}  models=${MODEL_START}-${MODEL_END}  channels=${CHANNELS}"
echo "[run] log=${LOG_FILE}"
echo "[run] csv=${CSV_FILE}"
echo "================================================================"

# ---- split model indices by slot --------------------------------------------
# TOTAL_PROCS = NUM_GPUS * PROCS_PER_GPU
# slot p (0-indexed) → gpu = p // PROCS_PER_GPU, gets indices where index % TOTAL_PROCS == p

split_indices_for_slot() {
  local slot="$1"
  local indices=""
  for idx in $(seq "${MODEL_START}" "${MODEL_END}"); do
    if (( idx % TOTAL_PROCS == slot )); then
      if [[ -z "$indices" ]]; then
        indices="${idx}"
      else
        indices="${indices},${idx}"
      fi
    fi
  done
  echo "$indices"
}

# ---- main loop --------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
  DATA_DIR="${BASE_DATA_DIR}/${DATASET}"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[skip] ${DATASET}: not found"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "[dataset] ${DATASET}  $(date)"
  echo "================================================================"

  declare -a RUNNING_PIDS=()

  for SLOT in $(seq 0 $(( TOTAL_PROCS - 1 ))); do
    GPU_ID=$(( SLOT / PROCS_PER_GPU ))
    INDICES=$(split_indices_for_slot "${SLOT}")
    if [[ -z "${INDICES}" ]]; then
      continue
    fi

    echo "[launch] ${DATASET}  gpu=${GPU_ID}  slot=${SLOT}  indices=${INDICES:0:40}..."

    bash -c "
      cd '${PROJECT_ROOT}'
      PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' python scripts/new_space/training/train_blockmixed_batch.py \
        --data_dir '${DATA_DIR}' \
        --space_file '${SPACE_FILE}' \
        --model_indices '${INDICES}' \
        --device cuda:${GPU_ID} \
        --channels ${CHANNELS} \
        --num_epochs ${NUM_EPOCHS} \
        --early_stop_threshold ${EARLY_STOP} \
        --max_round_epoch ${MAX_ROUND_EPOCH} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --log_dir '${LOG_DIR}' \
        --csv_file '${CSV_FILE}'
    " &

    PID=$!
    RUNNING_PIDS+=($PID)
    echo "[pid] gpu=${GPU_ID}  slot=${SLOT}  pid=${PID}"
  done

  echo "[wait] ${DATASET}: waiting for ${#RUNNING_PIDS[@]} GPU processes..."
  if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
    wait "${RUNNING_PIDS[@]}" || true
  fi
  RUNNING_PIDS=()
  echo "[done] ${DATASET} finished at $(date)"
done

echo ""
echo "================================================================"
echo "[all done] $(date)"
echo "[csv] ${CSV_FILE}"
echo "================================================================"
