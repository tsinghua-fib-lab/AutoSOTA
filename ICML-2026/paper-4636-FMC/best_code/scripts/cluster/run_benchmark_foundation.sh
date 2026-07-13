#!/bin/bash
#OAR -n benchmark_foundation
#OAR -l /nodes=1/gpu=1,walltime=12:00:00
#OAR -p "gpu='YES'"
#OAR -t besteffort
#OAR -t idempotent
#OAR --stdout logs/foundation_%jobid%.out
#OAR --stderr logs/foundation_%jobid%.err

# Benchmark Foundation Training Script
#
# This script trains all foundation models (NPE, FMPE) and baselines
# for the comparison benchmark.
#
# Usage:
#   oarsub -S ./scripts/cluster/run_benchmark_foundation.sh
#
# Or with specific tasks:
#   TASKS="pendulum,gaussian" oarsub -S ./scripts/cluster/run_benchmark_foundation.sh

set -e

# Setup environment
source ~/.bashrc
conda activate fmcpe

# Create log directory
mkdir -p logs

# Change to project directory
cd /home/pruhlman/Projects/ropefm

# Get tasks from environment or use defaults
TASKS=${TASKS:-"pure_gaussian,pendulum,ou_process,wind_tunnel"}

echo "=============================================="
echo "Benchmark Foundation Training"
echo "=============================================="
echo "Job ID: $OAR_JOB_ID"
echo "Node: $(hostname)"
echo "Tasks: $TASKS"
echo "Date: $(date)"
echo "=============================================="

# Run foundation training
python run_benchmark.py train_foundation \
    +experiment=comparison_benchmark \
    "tasks=[$TASKS]" \
    skip_existing=true

echo "=============================================="
echo "Foundation training completed!"
echo "Date: $(date)"
echo "=============================================="
