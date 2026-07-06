#!/bin/bash
# Evaluation script for REPA-P reproduction
# Run from /repo with: bash eval_script.sh

set -e

NAME="darcy.repap.repro"
STEP="${1:-120000}"
GPU="${2:-0}"
OUTDIR="trained_models/${NAME}/evaluation/step_${STEP}"

echo "=== REPA-P Evaluation ==="
echo "Run: ${NAME}, Step: ${STEP}, GPU: ${GPU}"
echo "Output: ${OUTDIR}"

# Reconstruction evaluation (Data Loss / MSE)
echo ""
echo "--- Reconstruction Evaluation ---"
python3 sample.py --name "${NAME}" --step "${STEP}" --gpu "${GPU}" --mode reconstruction --use-ema --batch-size 32 --num-batches 5

echo ""
echo "--- Generative Evaluation (Physics Residual) ---"
python3 sample.py --name "${NAME}" --step "${STEP}" --gpu "${GPU}" --mode generative --use-ema --num-samples 20

echo ""
echo "=== Evaluation Complete ==="
echo "Reconstruction metrics:"
cat "${OUTDIR}/reconstruction_metrics.csv" 2>/dev/null | python3 -c "
import sys, pandas as pd
df = pd.read_csv(sys.stdin)
mse_mean = df[df['sample'] != 'mean']['mse'].mean() if 'sample' in df.columns else df['mse'].mean()
print(f'Data Loss (MSE): {mse_mean:.6f}')
"

echo "Generative physics residual:"
cat "${OUTDIR}/generative_sample_statistics.csv" 2>/dev/null | head -3
