#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-./mass_base.pth}
GPUS=${GPUS:-0}

python train.py \
  --config config/downstream/classification_linear_probe_example.yaml \
  --gpu "${GPUS}" \
  --name classification_linear_probe_example \
  --override classification.encoder.pretrained_checkpoint="${CHECKPOINT}"
