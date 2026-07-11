#!/usr/bin/env bash
# Quick evaluation runner for SOTA optimization
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate gift
cd /repo
CONFIG="${1:-configs/chair_llava_1.5_7b.yaml}"
LOG="${2:-/tmp/eval_output.log}"
echo "[run_eval] Starting evaluation with config: $CONFIG"
echo "[run_eval] Log: $LOG"
timeout 3600 python3 eval_chair.py --config "$CONFIG" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[run_eval] Exit code: $RC"
exit $RC
