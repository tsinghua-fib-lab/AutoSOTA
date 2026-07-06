#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration
###############################################

DATASET="VAR3"

SERIES_LIST=(1 2 3 4)
SUBJECT_LIST=(1 2 3 4 5)
PENALTY_LIST=("Fast_Shap" "Shapley" "Jacob_F" "Jacob_L1")

NUM_WORKERS=24          # workers per (series, subject, penalty)
MAX_PARALLEL_JOBS=24    # max background jobs at once

SEED=2025
LAG=3

PYTHON="python3"
MAIN_SCRIPT="src/simulation.py"   # change if your entry file is elsewhere
LOG_DIR="logs_var3"
mkdir -p "${LOG_DIR}"

###############################################
# Launch jobs
###############################################

for S in "${SERIES_LIST[@]}"; do
  for SUBJ in "${SUBJECT_LIST[@]}"; do
    for PENALTY in "${PENALTY_LIST[@]}"; do
      echo "Launching dataset=${DATASET}, series=${S}, subject=${SUBJ}, penalty=${PENALTY} with ${NUM_WORKERS} workers"

      # Launch NUM_WORKERS jobs for this combination with unique exec_idx
      for EXEC_IDX in $(seq 1 "${NUM_WORKERS}"); do
        LOG_FILE="${LOG_DIR}/${DATASET}_series${S}_subject${SUBJ}_${PENALTY}_w${EXEC_IDX}.log"

        echo "  -> exec_idx=${EXEC_IDX}, log=${LOG_FILE}"

        ${PYTHON} "${MAIN_SCRIPT}" \
          --dataset "${DATASET}" \
          --series "${S}" \
          --subject "${SUBJ}" \
          --lag "${LAG}" \
          --seed "${SEED}" \
          --num_workers "${NUM_WORKERS}" \
          --exec_idx "${EXEC_IDX}" \
          --penalty_type "${PENALTY}" \
          > "${LOG_FILE}" 2>&1 &

        # Limit max parallel jobs at the system level
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
          sleep 1
        done
      done
    done
  done
done

# Wait for all background jobs to finish
wait
echo "All jobs finished."
