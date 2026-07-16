#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTISEED_ROOT="${MULTISEED_ROOT:-${SCRIPT_DIR}/data_multiseed}"
MULTISEED_RESULTS_ROOT="${MULTISEED_RESULTS_ROOT:-${SCRIPT_DIR}/results_multiseed}"
LOG_PATH="${LOG_PATH:-${SCRIPT_DIR}/run_multiseed.log}"

exec > >(tee -a "${LOG_PATH}") 2>&1

SEEDS=($(seq 0 99))
IRT_MODELS=(1pl 2pl)

mkdir -p "${MULTISEED_ROOT}"
mkdir -p "${MULTISEED_RESULTS_ROOT}"

for seed in "${SEEDS[@]}"; do
  DATA_ROOT="${MULTISEED_ROOT}/${seed}"
  RESULTS_ROOT="${MULTISEED_RESULTS_ROOT}/${seed}"
  mkdir -p "${DATA_ROOT}"
  mkdir -p "${RESULTS_ROOT}"

  for irt_model in "${IRT_MODELS[@]}"; do
    python "${SCRIPT_DIR}/1_calibrate.py" \
      --input-hf-repo irsl_testtime_resmat2 \
      --irt-model "${irt_model}" \
      --split-seed "${seed}" \
      --data-root "${DATA_ROOT}"

    python "${SCRIPT_DIR}/2_cat.py" \
      --input-hf-repo irsl_testtime_resmat2 \
      --irt-model "${irt_model}" \
      --data-root "${DATA_ROOT}" \
      --results-root "${RESULTS_ROOT}"
  done
done

python "${SCRIPT_DIR}/3_law_curve_multiseed.py" \
  --multiseed-root "${MULTISEED_ROOT}"
