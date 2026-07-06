#!/bin/bash
# Run a single optimization iteration and record score
# Usage: run_optimization_iter.sh <iter_num> <idea_id> <title> <config_file> [output_subdir]

set -euo pipefail

ITER="${1:?}"
IDEA_ID="${2:?}"
TITLE="${3:?}"
CONFIG="${4:?}"
SUBDIR="${5:-cpu_iter${ITER}}"

source /opt/conda/bin/activate py312
cd /repo

echo "============================================"
echo "Iteration $ITER: $TITLE"
echo "Config: $CONFIG"
echo "Output: output/$SUBDIR"
echo "Started at: $(date)"
echo "============================================"

# Run evaluation
python run_cpu_eval.py \
  "$CONFIG" \
  /tmp/subset_10mol.csv \
  "output/$SUBDIR" \
  2>&1 | tee "/tmp/iter_${ITER}.log"

# Parse metrics
METRICS_JSON="output/$SUBDIR/metrics.json"
if [ -f "$METRICS_JSON" ]; then
  HV=$(python -c "import json; print(json.load(open('$METRICS_JSON'))['HV'])")
  R2=$(python -c "import json; print(json.load(open('$METRICS_JSON'))['R2'])")
  SR=$(python -c "import json; print(json.load(open('$METRICS_JSON'))['Success Rate'])")
  NPROC=$(python -c "import json; print(json.load(open('$METRICS_JSON'))['n_processed'])")
  NSUCC=$(python -c "import json; print(json.load(open('$METRICS_JSON'))['n_success'])")

  echo ""
  echo "===== METRICS ====="
  echo "HV: $HV, R2: $R2, Success Rate: $SR%, Processed: $NPROC, Success: $NSUCC"

  # Build metrics JSON safely
  METRICS_STR="{\"HV\": $HV, \"R2\": $R2, \"Success Rate\": $SR}"

  # Record score
  /tools/record_score.sh \
    --scores /autosota_artifacts/paper-5397/sota/scores.jsonl \
    --iter "$ITER" \
    --idea-id "$IDEA_ID" \
    --title "$TITLE" \
    --status success \
    --primary "$HV" \
    --metrics "$METRICS_STR" \
    --notes "CPU eval (10 mol, 100 iters). Config: $CONFIG. $NPROC processed, $NSUCC success."

  echo "Score recorded for iteration $ITER"
else
  echo "ERROR: No metrics.json found at $METRICS_JSON"
  /tools/record_score.sh \
    --scores /autosota_artifacts/paper-5397/sota/scores.jsonl \
    --iter "$ITER" \
    --idea-id "$IDEA_ID" \
    --title "$TITLE" \
    --status failed \
    --primary 0.0 \
    --metrics '{}' \
    --notes "CPU eval failed: no metrics.json. Config: $CONFIG."
fi

echo "Finished at: $(date)"
