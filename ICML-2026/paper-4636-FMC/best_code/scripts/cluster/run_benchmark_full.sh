#!/bin/bash
#OAR -n benchmark_ropefm
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR -p gpumodel='V100'
#OAR -O /home/pruhlman/Projects/ropefm/logs/benchmark_%jobid%.out
#OAR -E /home/pruhlman/Projects/ropefm/logs/benchmark_%jobid%.err

# Benchmark script for NPE/FMPE comparison on cluster
# Runs all 4 phases for specified tasks

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/pruhlman/Projects/ropefm"
CONDA_ENV="fmcpe"
EXPERIMENT="comparison_benchmark"

# Cluster paths
RESULTS_DIR="/scratch/clear/pruhlman/ropefm/results"
DATA_DIR="/scratch/clear/pruhlman/ropefm/data"

# Tasks to run (can be overridden via OAR_TASK environment variable)
TASKS="${OAR_TASK:-high_dim_gaussian,pendulum,wind_tunnel,light_tunnel}"

# Setup environment
cd "$PROJECT_DIR"
conda activate "$CONDA_ENV"

# Create directories
mkdir -p logs
mkdir -p "$RESULTS_DIR"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "ROPEFM Benchmark - Full Pipeline"
echo "============================================================"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Tasks: $TASKS"
echo "Experiment: $EXPERIMENT"
echo "============================================================"

# Phase 1: Foundation Training
echo ""
echo "[Phase 1/4] Foundation Training..."
echo "============================================================"
python run_benchmark.py command=train_foundation \
    +experiment="$EXPERIMENT" \
    "tasks=[$TASKS]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

# Phase 2: Variant Training (fm_pt_base)
echo ""
echo "[Phase 2/4] Variant Training..."
echo "============================================================"
python run_benchmark.py command=train_variant \
    +experiment="$EXPERIMENT" \
    ++variant_name=fm_pt_base \
    "tasks=[$TASKS]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

# Phase 3: Evaluation
echo ""
echo "[Phase 3/4] Evaluation..."
echo "============================================================"
python run_benchmark.py command=evaluate_all \
    +experiment="$EXPERIMENT" \
    "tasks=[$TASKS]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

# Phase 4: Aggregation
echo ""
echo "[Phase 4/4] Aggregation..."
echo "============================================================"
python run_benchmark.py command=aggregate \
    +experiment="$EXPERIMENT" \
    "tasks=[$TASKS]" \
    results_dir="$RESULTS_DIR" \
    data_dir="$DATA_DIR"

echo ""
echo "============================================================"
echo "Benchmark completed at $(date)"
echo "============================================================"
