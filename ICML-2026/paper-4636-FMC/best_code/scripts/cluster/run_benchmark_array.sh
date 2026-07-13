#!/bin/bash
#OAR -n benchmark_task
#OAR -l walltime=48:00:00
#OAR --array 4

# Array job script - runs one task per job
# Submit with: oarsub -S ./run_benchmark_array.sh

set -e

# Configuration
PROJECT_DIR="/home/pruhlman/project/ropefm"
CONDA_ENV="fmcpe"
EXPERIMENT="comparison_benchmark"

# Cluster paths
RESULTS_DIR="/scratch/clear/pruhlman/ropefm/results"
DATA_DIR="/scratch/clear/pruhlman/ropefm/data"

export PYTHONPATH="/home/pruhlman/project/npe_pfn:$PYTHONPATH"
export HF_HOME="/scratch/clear/pruhlman/huggingface_cache"                                        
mkdir -p $HF_HOME  

# Map array index to task
TASKS=("high_dim_gaussian" "pendulum" "wind_tunnel" "light_tunnel")
TASK_IDX=$((OAR_ARRAY_INDEX - 1))  # OAR arrays are 1-indexed
TASK="${TASKS[$TASK_IDX]}"

# Setup environment
cd "$PROJECT_DIR"
conda activate "$CONDA_ENV"

# Create directories
mkdir -p logs
mkdir -p "$RESULTS_DIR"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "ROPEFM Benchmark - Task: $TASK"
echo "============================================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Array Index: $OAR_ARRAY_INDEX"
echo "============================================================"

# Phase 1: Foundation Training
echo "[Phase 1/4] Foundation Training for $TASK..."
python run_benchmark.py command=train_foundation \
    +experiment="$EXPERIMENT" \
    ++"tasks=[$TASK]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

# Phase 2: Variant Training
echo "[Phase 2/4] Variant Training for $TASK..."
python run_benchmark.py command=train_variant \
    +experiment="$EXPERIMENT" \
    ++variant_name=fm_pt_fmpe \
    ++"tasks=[$TASK]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

# Phase 3: Evaluation
echo "[Phase 3/4] Evaluation for $TASK..."
python run_benchmark.py command=evaluate_all \
    +experiment="$EXPERIMENT" \
    ++"tasks=[$TASK]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

echo ""
echo "Task $TASK completed at $(date)"
