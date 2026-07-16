#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTISEED_ROOT="${MULTISEED_ROOT:-${SCRIPT_DIR}/data_multiseed}"
MULTISEED_RESULTS_ROOT="${MULTISEED_RESULTS_ROOT:-${SCRIPT_DIR}/results_multiseed}"
LOG_PATH="${LOG_PATH:-${SCRIPT_DIR}/run_multiseed_seed5.log}"

exec > >(tee -a "${LOG_PATH}") 2>&1

SEEDS=(5)

mkdir -p "${MULTISEED_ROOT}"
mkdir -p "${MULTISEED_RESULTS_ROOT}"

echo "Logging to ${LOG_PATH}"
echo "Using seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  DATA_ROOT="${MULTISEED_ROOT}/${seed}"
  RESULTS_ROOT="${MULTISEED_RESULTS_ROOT}/${seed}"
  mkdir -p "${DATA_ROOT}"
  mkdir -p "${RESULTS_ROOT}"

  # python "${SCRIPT_DIR}/3_clean_and_pivot.py" \
  #   --split-seed "${seed}" \
  #   --output-root "${DATA_ROOT}" \
  #   --results-root "${RESULTS_ROOT}"

  python "${SCRIPT_DIR}/4_calibrate_1pl.py" --data-root "${DATA_ROOT}" --loss-kind beta
  python "${SCRIPT_DIR}/4_calibrate_1pl.py" --data-root "${DATA_ROOT}" --loss-kind binary
  python "${SCRIPT_DIR}/4_calibrate_2pl.py" --data-root "${DATA_ROOT}" --loss-kind beta
  python "${SCRIPT_DIR}/4_calibrate_2pl.py" --data-root "${DATA_ROOT}" --loss-kind binary

  python "${SCRIPT_DIR}/5_cat.py" --data-root "${DATA_ROOT}" --results-root "${RESULTS_ROOT}" --loss-kind beta --irt-model 1pl
  python "${SCRIPT_DIR}/5_cat.py" --data-root "${DATA_ROOT}" --results-root "${RESULTS_ROOT}" --loss-kind binary --irt-model 1pl
  python "${SCRIPT_DIR}/5_cat.py" --data-root "${DATA_ROOT}" --results-root "${RESULTS_ROOT}" --loss-kind beta --irt-model 2pl
  python "${SCRIPT_DIR}/5_cat.py" --data-root "${DATA_ROOT}" --results-root "${RESULTS_ROOT}" --loss-kind binary --irt-model 2pl

  python "${SCRIPT_DIR}/6_organize_data.py" --data-root "${DATA_ROOT}"
  python "${SCRIPT_DIR}/7_filt_laws.py" --data-root "${DATA_ROOT}"
done

python "${SCRIPT_DIR}/8_plot_multiseed.py"
