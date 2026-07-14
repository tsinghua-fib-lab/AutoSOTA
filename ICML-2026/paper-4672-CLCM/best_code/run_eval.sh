#!/usr/bin/env bash
set -euo pipefail
cd /repo
LOG="/tmp/eval_${1:-default}.log"
shift || true
echo "=== Running: WANDB_MODE=disabled python3 train.py --config eval_config.yaml $@ ===" | tee "$LOG"
WANDB_MODE=disabled timeout 3600 python3 train.py --config eval_config.yaml "$@" 2>&1 | tee -a "$LOG"
ACC=$(grep -oP "Full: \d+\.\d+" "$LOG" | tail -1 | sed "s/Full: //")
echo "EXTRACTED_ACCURACY=$ACC"
