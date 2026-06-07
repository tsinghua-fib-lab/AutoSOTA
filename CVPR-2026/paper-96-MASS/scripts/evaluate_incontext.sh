#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-./mass_base.pth}
DATA_ROOT=${DATA_ROOT:-./data/mass_h5}
DATASET=${DATASET:-bcv}
GPUS=${GPUS:-0}
REFERENCE_MODE=${REFERENCE_MODE:-fixed}
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-1}

python evaluate.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --reference-mode "${REFERENCE_MODE}" \
  --ensemble-size "${ENSEMBLE_SIZE}" \
  --gpus "${GPUS}" \
  --use-ema
