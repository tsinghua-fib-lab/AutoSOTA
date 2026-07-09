#!/bin/bash
# Full HEDP CDDB-Hard reproduction pipeline
set -e
cd /repo

echo "================================================"
echo "HEDP CDDB-Hard Full Pipeline"
echo "================================================"
echo "Start: $(date)"
echo ""

# Verify data
if [ ! -d "/repo/data/CDDB/gaugan/train/0_real" ]; then
    echo "FATAL: CDDB dataset not found."
    echo "Run: bash /repo/extract_data.sh first"
    exit 1
fi

# Train
echo "=== STEP 1: Training ==="
python3 main.py --config configs/train/cddb-hard.json 2>&1 | tee /repo/logs/train/cddb.log
echo "Training complete"

# Find checkpoint
LOGDIR=$(ls -td /repo/logs/hybrid_energy_distance_prompt_trainer_cddb_* 2>/dev/null | head -1)
if [ -z "$LOGDIR" ] || [ ! -f "$LOGDIR/task_4.pth" ]; then
    echo "FATAL: Checkpoint not found in $LOGDIR"
    exit 1
fi
cp "$LOGDIR/task_4.pth" /repo/logsmodel/cddb/task_4.pth
echo "Checkpoint: $LOGDIR/task_4.pth"

# Eval known
echo ""
echo "=== STEP 2: Known Domain Evaluation ==="
python3 main.py --config configs/eval/known/cddb-hard.json 2>&1 | tee /repo/logs/eval/known/cddb.log

# Eval unknown
echo ""
echo "=== STEP 3: Unknown Domain Evaluation ==="
python3 main.py --config configs/eval/unknown/cddb-hard.json 2>&1 | tee /repo/logs/eval/unknown/cddb.log

# Parse results
echo ""
echo "=== STEP 4: Results ==="
python3 /repo/host_tmp/parse_metrics.py \
  --train-log /repo/logs/train/cddb.log \
  --known-log /repo/logs/eval/known/cddb.log \
  --unknown-log /repo/logs/eval/unknown/cddb.log \
  --output /repo/results.json

cat /repo/results.json
echo ""
echo "Pipeline complete at: $(date)"
