#!/usr/bin/env bash
set -euo pipefail
# Generate sample figures for multiple sigmas and training sizes.
# Only compute metrics table for sigma=0.01 and num_samples=10000.

PY=python
SCRIPT=plotting/plotting_mnist.py
OUTDIR=results/mnist
mkdir -p "$OUTDIR"

SIGMAS=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0)
# Loop through all training checkpoint sizes, but only generate/evaluate 100 samples per run
NUM_SAMPLES=(100 1000 10000)

# Loop and run plotting; only pass --compute_table for the metrics-canonical run
for sigma in "${SIGMAS[@]}"; do
  for n in "${NUM_SAMPLES[@]}"; do
    echo "Running sigma=${sigma}, num_samples=${n}..."
    # Generate samples/figures only, use 1 trial, and always generate/evaluate 100 samples
    "$PY" "$SCRIPT" --noise_level "$sigma" --num_samples "$n" --num_eval_samples 100 --n_trials 1
  done
done

echo "Done: plots in $OUTDIR" 
