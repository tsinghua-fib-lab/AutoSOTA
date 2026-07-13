#!/bin/bash
#OAR -n benchmark_variant
#OAR -l /nodes=1/gpu=1,walltime=2:00:00
#OAR -p "gpu='YES'"
#OAR -t besteffort
#OAR -t idempotent
#OAR --array-param-file tasks.txt
#OAR --stdout logs/variant_%jobid%_%arrayid%.out
#OAR --stderr logs/variant_%jobid%_%arrayid%.err

# Benchmark Variant Training Script (Array Job)
#
# This script trains a specific variant configuration for one task/seed/ncal
# combination. It's designed to be used with OAR array jobs.
#
# Usage:
#   1. Create tasks.txt with format: task,seed,ncal
#   2. Set VARIANT_NAME environment variable
#   3. Submit: VARIANT_NAME=fm_pt_base oarsub -S ./scripts/cluster/run_benchmark_variant.sh
#
# tasks.txt example:
#   pendulum,0,10
#   pendulum,0,50
#   pendulum,0,200
#   pendulum,0,1000
#   gaussian,0,10
#   ...

set -e

# Setup environment
source ~/.bashrc
conda activate fmcpe

# Create log directory
mkdir -p logs

# Change to project directory
cd /home/pruhlman/Projects/ropefm

# Parse array job parameter (format: task,seed,ncal)
# $1 is passed from --array-param-file
PARAMS="$1"
TASK=$(echo "$PARAMS" | cut -d',' -f1)
SEED=$(echo "$PARAMS" | cut -d',' -f2)
NCAL=$(echo "$PARAMS" | cut -d',' -f3)

# Get variant name from environment
VARIANT_NAME=${VARIANT_NAME:-"fm_pt_base"}

echo "=============================================="
echo "Benchmark Variant Training"
echo "=============================================="
echo "Job ID: $OAR_JOB_ID"
echo "Array ID: $OAR_ARRAY_ID"
echo "Node: $(hostname)"
echo "Variant: $VARIANT_NAME"
echo "Task: $TASK"
echo "Seed: $SEED"
echo "Ncal: $NCAL"
echo "Date: $(date)"
echo "=============================================="

# Run variant training for single configuration
python run_benchmark.py train_variant \
    +experiment=comparison_benchmark \
    variant_name=${VARIANT_NAME} \
    task=${TASK} \
    seed=${SEED} \
    ncal=${NCAL} \
    parallel.enabled=true \
    skip_existing=true

echo "=============================================="
echo "Variant training completed!"
echo "Date: $(date)"
echo "=============================================="
