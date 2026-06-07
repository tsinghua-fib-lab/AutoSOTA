#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-./mass_base.pth}
GPUS=${GPUS:-0}

python train.py \
  --config config/downstream/segmentation_finetune_example.yaml \
  --gpu "${GPUS}" \
  --name segmentation_finetune_example \
  --override finetuning.pretrained_checkpoint="${CHECKPOINT}"
