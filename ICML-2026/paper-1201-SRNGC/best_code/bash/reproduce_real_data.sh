#!/usr/bin/env bash
set -euo pipefail

########################################
# User configuration
########################################

PYTHON="python3"
SCRIPT="./src/real_data.py"

SEED=2025
NUM_WORKERS=1
EXEC_IDX=1
SUBJECT=1       # real data table has no subject; adjust if needed
PENALTY_TYPES=("Fast_Shap" "Shapley")

# Datasets / series for which you have tuned configs
DATASETS=("CausalTime" "DREAM3" "DREAM4")

# Series lists per dataset, matching the table you provided
CAUSALTIME_SERIES=(1 2 3)
DREAM3_SERIES=(1 2 3 4 5)
DREAM4_SERIES=(1 2 3 4 5)

########################################
# Main loop
########################################

for DATASET in "${DATASETS[@]}"; do
  case "$DATASET" in
    "CausalTime")
      SERIES_ARR=("${CAUSALTIME_SERIES[@]}")
      ;;
    "DREAM3")
      SERIES_ARR=("${DREAM3_SERIES[@]}")
      ;;
    "DREAM4")
      SERIES_ARR=("${DREAM4_SERIES[@]}")
      ;;
    *)
      echo "[WARN] Unknown dataset: $DATASET, skipping."
      continue
      ;;
  esac

  for SERIES in "${SERIES_ARR[@]}"; do
    for PENALTY_TYPE in "${PENALTY_TYPES[@]}"; do
      echo "============================================================"
      echo "Running dataset=$DATASET | series=$SERIES | penalty=$PENALTY_TYPE"
      echo "Using tuned hyperparameters from server_results/real_data_${PENALTY_TYPE}.csv"
      echo "============================================================"

      ${PYTHON} "${SCRIPT}" \
        --dataset "${DATASET}" \
        --series "${SERIES}" \
        --subject "${SUBJECT}" \
        --seed "${SEED}" \
        --num_workers "${NUM_WORKERS}" \
        --exec_idx "${EXEC_IDX}" \
        --penalty_type "${PENALTY_TYPE}" \
        --use_best

      echo ""
    done
  done
done

echo "All reproduction runs finished."
