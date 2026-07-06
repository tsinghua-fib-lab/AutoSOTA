#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration
###############################################

PYTHON="python3"
MAIN_SCRIPT="./src/real_data.py"   # change to your actual main script

DATASETS=("CausalTime")
LAG_LIST=(2 3 5)

# Per-dataset series lists (adjust if needed)
CAUSALTIME_SERIES=(1 2 3)

# Subject index (adjust if your datasets use multiple subjects)
SUBJECT=1

PENALTY_TYPE="Shapley"

NUM_WORKERS=48          # workers per (dataset, lag, series, subject)
MAX_PARALLEL_JOBS=48    # max background jobs at once
SEED=2025

LOG_DIR="logs_shapley"
mkdir -p "${LOG_DIR}"

###############################################
# Launch jobs
###############################################

for DATASET in "${DATASETS[@]}"; do

  for LAG in "${LAG_LIST[@]}"; do
    for SERIES in "${CAUSALTIME_SERIES[@]}"; do
      echo "Launching dataset=${DATASET}, lag=${LAG}, series=${SERIES}, subject=${SUBJECT} with ${NUM_WORKERS} workers (Shapley penalty)"

      # Launch NUM_WORKERS jobs for this combination with unique exec_idx
      for EXEC_IDX in $(seq 1 "${NUM_WORKERS}"); do
        LOG_FILE="${LOG_DIR}/${DATASET}_lag${LAG}_series${SERIES}_subject${SUBJECT}_Shapley_w${EXEC_IDX}.log"

        echo "  -> exec_idx=${EXEC_IDX}, log=${LOG_FILE}"

        ${PYTHON} "${MAIN_SCRIPT}" \
          --dataset "${DATASET}" \
          --series "${SERIES}" \
          --subject "${SUBJECT}" \
          --lag "${LAG}" \
          --seed "${SEED}" \
          --num_workers "${NUM_WORKERS}" \
          --exec_idx "${EXEC_IDX}" \
          --penalty_type "${PENALTY_TYPE}" \
          > "${LOG_FILE}" 2>&1 &

        # Limit max parallel jobs
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
          sleep 1
        done
      done
    done
  done
done

# Wait for all background jobs to finish
wait
echo "All Shapley runs finished."
