#!/bin/bash
#OAR -n benchmark_evaluate
#OAR -l /nodes=1/gpu=1,walltime=4:00:00
#OAR -p "gpu='YES'"
#OAR -t besteffort
#OAR -t idempotent
#OAR --stdout logs/evaluate_%jobid%.out
#OAR --stderr logs/evaluate_%jobid%.err

# Benchmark Evaluation Script
#
# This script evaluates all methods and variants on the test set.
#
# Usage:
#   oarsub -S ./scripts/cluster/run_benchmark_evaluate.sh
#
# Or to evaluate specific variant:
#   VARIANT_NAME=fm_pt_base oarsub -S ./scripts/cluster/run_benchmark_evaluate.sh

set -e

# Setup environment
source ~/.bashrc
conda activate fmcpe

# Create log directory
mkdir -p logs

# Change to project directory
cd /home/pruhlman/Projects/ropefm

# Get variant name from environment (optional)
VARIANT_NAME=${VARIANT_NAME:-""}

echo "=============================================="
echo "Benchmark Evaluation"
echo "=============================================="
echo "Job ID: $OAR_JOB_ID"
echo "Node: $(hostname)"
echo "Variant: ${VARIANT_NAME:-'all'}"
echo "Date: $(date)"
echo "=============================================="

if [ -n "$VARIANT_NAME" ]; then
    # Evaluate specific variant
    python run_benchmark.py evaluate \
        +experiment=comparison_benchmark \
        variant_name=${VARIANT_NAME} \
        skip_existing=true
else
    # Evaluate all
    python run_benchmark.py evaluate_all \
        +experiment=comparison_benchmark \
        skip_existing=true
fi

echo "=============================================="
echo "Evaluation completed!"
echo "Date: $(date)"
echo "=============================================="
