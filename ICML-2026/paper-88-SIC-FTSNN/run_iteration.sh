#!/usr/bin/env bash
# Usage: run_iteration.sh <iter_num> <idea_id> <patch_script> <description>
set -euo pipefail
ITER="$1"
IDEA_ID="$2"
PATCH="$3"
DESC="$4"
LOGFILE="/repo/eval_iter${ITER}.log"

cd /repo

# Restore baseline
git checkout -- simple_snn.py learnable_fragmentation.py 2>/dev/null || true

# Re-apply IDEA-03 (cosine LR) as base improvement
python3 /repo/patches/apply_cosine_lr.py

# Apply specific patch
if [ -n "$PATCH" ] && [ "$PATCH" != "none" ]; then
    python3 "/repo/patches/${PATCH}"
fi

# Verify syntax
python3 -c "compile(open('/repo/simple_snn.py').read(), 'simple_snn.py', 'exec'); print('Syntax OK')"

# Run evaluation
echo "=== Iteration ${ITER}: ${IDEA_ID} - ${DESC} ==="
echo "Started at $(date)"

mkdir -p propdata
ln -sfn /datasets/fmnist propdata/FMNIST

timeout 3600 python3 -u simple_snn.py \
  --data_path propdata/FMNIST \
  --Fault True --fault_type stuck --fault_ratio 0.3 \
  --Dynamic True --num_steps 8 --num_epochs 50 \
  --batch_size 100 --learning_rate 0.001 \
  --gpu_num 0 --plot False \
  2>&1 | tee "$LOGFILE"

EXIT_CODE=$?
echo "Exit: ${EXIT_CODE}"
echo "Finished at $(date)"

# Parse accuracy
ACC=$(grep -oP "Test Set Accuracy: \K[0-9.]+" "$LOGFILE" || echo "0.0")
echo "PARSED_ACCURACY=${ACC}"

# Record score
if [ "$EXIT_CODE" = "0" ] && [ "$ACC" != "0.0" ]; then
    /tools/record_score.sh \
      --scores /autosota_artifacts/paper-88/sota/scores.jsonl \
      --iter "$ITER" \
      --idea-id "$IDEA_ID" \
      --title "${DESC}" \
      --status success \
      --primary "$ACC" \
      --metrics "{\"Accuracy\": ${ACC}}" \
      --notes "Iteration ${ITER}: ${IDEA_ID}. Accuracy=${ACC}%"
    echo "Score recorded: ${ACC}%"
else
    /tools/record_score.sh \
      --scores /autosota_artifacts/paper-88/sota/scores.jsonl \
      --iter "$ITER" \
      --idea-id "$IDEA_ID" \
      --title "${DESC}" \
      --status failed \
      --primary 0.0 \
      --metrics "{}" \
      --notes "Iteration ${ITER}: ${IDEA_ID} FAILED. Exit=${EXIT_CODE}, ACC=${ACC}"
    echo "Score recorded as FAILED"
fi
