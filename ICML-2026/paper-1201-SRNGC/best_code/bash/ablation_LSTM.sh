#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration
###############################################

DATASET="fMRI"

SERIES_LIST=(1)             # or (1 2 3 4) if you want multiple series
SUBJECT_LIST=(0 1 2 3 4)    # subject indices
MODEL_TYPES=("cLSTM" "LSTM")
PENALTY_LIST=("Fast_Shap" "Shapley" "Jacob_F" "Jacob_L1" "Layer_Weight")

NUM_WORKERS=24              # workers per (series, subject, model, penalty)
MAX_PARALLEL_JOBS=24        # max background jobs at once

SEED=2025
LAG=5

PYTHON="python3"
MAIN_SCRIPT="src/ablation.py"
LOG_DIR="logs_fmri"
mkdir -p "${LOG_DIR}"

###############################################
# Launch jobs
###############################################

for S in "${SERIES_LIST[@]}"; do
  for SUBJ in "${SUBJECT_LIST[@]}"; do
    for MODEL in "${MODEL_TYPES[@]}"; do
      for PENALTY in "${PENALTY_LIST[@]}"; do

        # ---------- Enforce constraints ----------
        # Layer_Weight penalty only valid for cLSTM / cMLP (so skip LSTM)
        if [[ "$PENALTY" == "Layer_Weight" && "$MODEL" == "LSTM" ]]; then
          echo "[SKIP] model_type=$MODEL | penalty_type=$PENALTY (Layer_Weight only for cLSTM/cMLP)"
          continue
        fi

        # Deduce importance_type from penalty_type (for logging only)
        if [[ "$PENALTY" == "Fast_Shap" || "$PENALTY" == "Shapley" ]]; then
          IMPORTANCE="Shapley"
        elif [[ "$PENALTY" == "Jacob_F" || "$PENALTY" == "Jacob_L1" ]]; then
          IMPORTANCE="Jacobian"
        elif [[ "$PENALTY" == "Layer_Weight" ]]; then
          IMPORTANCE="Layer_Weight"
        else
          echo "[SKIP] Unknown penalty_type=$PENALTY"
          continue
        fi
        # -----------------------------------------

        echo "Launching dataset=${DATASET}, series=${S}, subject=${SUBJ}, model=${MODEL}, penalty=${PENALTY} with ${NUM_WORKERS} workers"

        # Launch NUM_WORKERS jobs for this combination with unique exec_idx
        for EXEC_IDX in $(seq 1 "${NUM_WORKERS}"); do
          LOG_FILE="${LOG_DIR}/${DATASET}_series${S}_subject${SUBJ}_${MODEL}_${PENALTY}_w${EXEC_IDX}.log"

          echo "  -> exec_idx=${EXEC_IDX}, importance=${IMPORTANCE}, log=${LOG_FILE}"

          ${PYTHON} "${MAIN_SCRIPT}" \
            --dataset "${DATASET}" \
            --series "${S}" \
            --subject "${SUBJ}" \
            --lag "${LAG}" \
            --seed "${SEED}" \
            --num_workers "${NUM_WORKERS}" \
            --exec_idx "${EXEC_IDX}" \
            --model_type "${MODEL}" \
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
done

# Wait for all background jobs to finish
wait
echo "All jobs finished."
