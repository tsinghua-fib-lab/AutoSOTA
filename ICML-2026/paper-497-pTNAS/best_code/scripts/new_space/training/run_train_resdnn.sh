#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train_resdnn.sh
#
# Trains all PTNASResNet architectures (blocks=[2,3,4,5,6], channels=[32,64,128,256])
# on 4 datasets in parallel across 8 GPUs × 3 procs = 24 parallel.
# Each slot gets a shuffled chunk; already-done archs are skipped via CSV.
#
# Usage:
#   bash scripts/new_space/training/run_train_resdnn.sh
# ---------------------------------------------------------------------------

set -uo pipefail
trap '' ERR

SPACE_NAME="resnet"
PROCS_PER_GPU=5
NUM_GPUS=8
TOTAL_PROCS=$(( NUM_GPUS * PROCS_PER_GPU ))   # 24

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DATA_DIR="${PROJECT_ROOT}/datasets/fit-medium-table"
LOG_DIR="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/training/logs"
CSV_FILE="${PROJECT_ROOT}/datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv"

DATASETS=(
  avito-user-clicks
  event-user-attendance
  avito-ad-ctr
  hm-user-churn
  trial-site-success
)

NUM_EPOCHS=200
EARLY_STOP=10
MAX_ROUND_EPOCH=20
BATCH_SIZE=256
LR=0.001

# ---- generate ALL archs, shuffle, split into TOTAL_PROCS chunks -----------
TMPDIR_ARCHS=$(mktemp -d)
python3 -c "
import itertools, random

blocks_choices = [2, 3, 4, 5, 6]
channel_choices = [32, 64, 128, 256]
total_procs = ${TOTAL_PROCS}

all_archs = []
for depth in blocks_choices:
    for combo in itertools.product(channel_choices, repeat=depth):
        all_archs.append('-'.join(map(str, combo)))

random.seed(42)
random.shuffle(all_archs)

# split into chunks round-robin
chunks = [[] for _ in range(total_procs)]
for i, arch in enumerate(all_archs):
    chunks[i % total_procs].append(arch)

for slot, chunk in enumerate(chunks):
    with open(f'${TMPDIR_ARCHS}/slot_{slot}.txt', 'w') as f:
        f.write(','.join(chunk))

print(f'total={len(all_archs)} archs, {total_procs} slots')
"

TOTAL_ARCHS=$(python3 -c "
import itertools
n = sum(4**d for d in [2,3,4,5,6])
print(n)
")

# ---- log ------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_resnet_pool_${RUN_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "[run] $(date)"
echo "[run] GPUs=${NUM_GPUS}  procs_per_gpu=${PROCS_PER_GPU}  total_procs=${TOTAL_PROCS}"
echo "[run] total archs=${TOTAL_ARCHS}  space=${SPACE_NAME}"
echo "[run] datasets=${DATASETS[*]}"
echo "[run] csv=${CSV_FILE}"
echo "[run] log=${LOG_FILE}"
echo "================================================================"

# ---- main loop ------------------------------------------------------------
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
    ARCHS=$(cat "${TMPDIR_ARCHS}/slot_${SLOT}.txt")
    if [[ -z "${ARCHS}" ]]; then
      continue
    fi

    N_ARCHS=$(echo "${ARCHS}" | tr ',' '\n' | wc -l)
    echo "[launch] ${DATASET}  gpu=${GPU_ID}  slot=${SLOT}  n_archs=${N_ARCHS}"

    bash -c "
      cd '${PROJECT_ROOT}'
      PYTHONPATH='${PROJECT_ROOT}/src:${PROJECT_ROOT}' python scripts/new_space/training/train_resdnn_batch.py \
        --data_dir '${DATA_DIR}' \
        --space_name ${SPACE_NAME} \
        --architectures '${ARCHS}' \
        --device cuda:${GPU_ID} \
        --output_csv '${CSV_FILE}' \
        --num_epochs ${NUM_EPOCHS} \
        --early_stop_threshold ${EARLY_STOP} \
        --max_round_epoch ${MAX_ROUND_EPOCH} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR}
    " &

    PID=$!
    RUNNING_PIDS+=($PID)
    echo "[pid] gpu=${GPU_ID}  slot=${SLOT}  pid=${PID}"
  done

  echo "[wait] ${DATASET}: ${#RUNNING_PIDS[@]} processes on ${TOTAL_PROCS} slots..."
  if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
    wait "${RUNNING_PIDS[@]}" || true
  fi
  RUNNING_PIDS=()
  echo "[done] ${DATASET} finished at $(date)"
done

rm -rf "${TMPDIR_ARCHS}"

echo ""
echo "================================================================"
echo "[all done] $(date)"
echo "[csv] ${CSV_FILE}"
echo "================================================================"
