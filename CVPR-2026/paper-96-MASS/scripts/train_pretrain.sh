#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-0}

python train.py \
  --config config/pretrain/mask_guided_self_supervised.yaml \
  --gpu "${GPUS}" \
  --name mass_pretrain_example
