#!/bin/bash
# Usage: bash eval_checkpoint.sh <checkpoint_path> <output_dir>
CKPT="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"
echo "Evaluating: $CKPT"
python3 evaluate_potts.py \
  --ckpt "$CKPT" \
  --L 4 --q 3 --beta 1.2 --J 1.0 \
  --num-samples 10000 --batch-size 1024 \
  --device cuda:0 --output-dir "$OUTDIR" \
  --seed 42 --sw-blocks 40 --sw-steps-per-block 500 \
  --sw-burn-in 2048 --sw-n-configs 10240 \
  2>&1 | tee "$OUTDIR/eval.log"
echo "Done. Metrics at $OUTDIR/metrics.json"
