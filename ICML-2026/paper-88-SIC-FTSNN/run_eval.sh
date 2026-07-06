#!/usr/bin/env bash
# Usage: run_eval.sh <log_path> [extra_args...]
# Runs the evaluation and captures output + parsed accuracy
set -euo pipefail
LOGFILE="$1"
shift
cd /repo
mkdir -p propdata
ln -sfn /datasets/fmnist propdata/FMNIST
python3 -u simple_snn.py \
  --data_path propdata/FMNIST \
  --Fault True \
  --fault_type stuck \
  --fault_ratio 0.3 \
  --Dynamic True \
  --num_steps 8 \
  --num_epochs 50 \
  --batch_size 100 \
  --learning_rate 0.001 \
  --gpu_num 0 \
  --plot False \
  "$@" 2>&1 | tee "$LOGFILE"
# Parse accuracy
ACC=$(grep -oP "Test Set Accuracy: \K[0-9.]+" "$LOGFILE" || echo "0.0")
echo "PARSED_ACCURACY=$ACC"
