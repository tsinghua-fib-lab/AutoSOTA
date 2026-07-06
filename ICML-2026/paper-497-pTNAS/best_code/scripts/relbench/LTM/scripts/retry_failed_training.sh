#!/bin/bash

# Retry Failed Training Tasks
# This script retrains specific dataset-model combinations that had poor results
# Usage: ./retry_failed_training.sh

set -e  # Exit on error (but we'll handle errors in the loop)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${LTM_ROOT}/../../../.." && pwd)"
cd "${ROOT_DIR}"

# ============================================
# Configuration
# ============================================

# Data directory roots (full paths)
INPUT_DATA_DIR_ROOT="run_outputs/data/relbench/baselines/ltm/tpberta_table"
ORIGINAL_DATA_DIR_ROOT="datasets/fit-medium-table"

# Failed tasks to retry: (dataset, model) pairs
# Format: "dataset:model"
FAILED_TASKS=(
    "avito-user-clicks:nomic"
    "hm-user-churn:nomic"
)

# Seeds to try for each task (5 different seeds)
SEEDS=(123 456 789 2024 1024)

# TP-BERTa root can be overridden from the environment.
TPBERTA_ROOT="${TPBERTA_ROOT:-third_party/tp-berta}"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH=".:scripts/relbench/LTM:${TPBERTA_ROOT}:${PYTHONPATH}"

# Output directory for training results
RESULT_DIR="run_outputs/data/relbench/baselines/ltm/results"

# Logging setup
LOG_DIR="run_outputs/data/relbench/baselines/ltm/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/retry_failed_training_${TIMESTAMP}.log"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Retry Failed Training Tasks"
echo "=========================================="
echo ""
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# Set CUDA_VISIBLE_DEVICES to use GPU 5 (index 4)
export CUDA_VISIBLE_DEVICES=4

# ============================================
# Function to train on a single dataset with a model
# ============================================

train_dataset_model() {
    local dataset=$1
    local model=$2
    local seed=$3
    local input_dir="${INPUT_DATA_DIR_ROOT}/${model}/${dataset}"
    local original_data_dir="${ORIGINAL_DATA_DIR_ROOT}/${dataset}"
    # Add seed to output directory to avoid overwriting
    local output_dir="${RESULT_DIR}/${model}_head/${dataset}_seed${seed}"
    local target_col_txt="${original_data_dir}/target_col.txt"
    
    echo ""
    echo "=========================================="
    echo "Training Dataset: $dataset with Model: $model (seed=$seed)"
    echo "=========================================="
    echo "  INPUT_DIR: $input_dir"
    echo "  OUTPUT_DIR: $output_dir"
    echo "  TARGET_COL_TXT: $target_col_txt"
    echo "  SEED: $seed"
    echo ""

    # Check input directory exists
    if [ ! -d "$input_dir" ]; then
        echo "  ⚠️  Warning: Input directory not found: $input_dir"
        echo "  Skipping..."
        return 1
    fi
    
    # Check required files exist
    if [ ! -f "$input_dir/train.csv" ] || [ ! -f "$input_dir/val.csv" ] || [ ! -f "$input_dir/test.csv" ]; then
        echo "  ⚠️  Warning: Missing CSV files in: $input_dir"
        echo "  Skipping..."
        return 1
    fi
    
    if [ ! -f "$target_col_txt" ]; then
        echo "  ⚠️  Warning: target_col.txt not found: $target_col_txt"
        echo "  Skipping..."
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run training with specific seed
    if python "scripts/relbench/LTM/train.py" \
        --data_dir "$input_dir" \
        --output_dir "$output_dir" \
        --target_col_txt "$target_col_txt" \
        --seed "$seed"; then
        echo ""
        echo "  ✅ Completed: $dataset with $model (seed=$seed)"
        echo "     Results saved to: $output_dir"
        return 0
    else
        echo ""
        echo "  ❌ Error: Failed to train $dataset with $model (seed=$seed)"
        echo "  Continuing to next..."
        return 1
    fi
}

# ============================================
# Main - Loop through failed tasks with multiple seeds
# ============================================

echo "Retrying ${#FAILED_TASKS[@]} failed tasks with ${#SEEDS[@]} seeds each:"
for task in "${FAILED_TASKS[@]}"; do
    echo "  - $task (seeds: ${SEEDS[*]})"
done
echo ""

total_tasks=0
successful_tasks=0

for task in "${FAILED_TASKS[@]}"; do
    # Parse dataset:model format
    IFS=':' read -r dataset model <<< "$task"
    
    if [ -z "$dataset" ] || [ -z "$model" ]; then
        echo "  ⚠️  Warning: Invalid task format: $task (expected 'dataset:model')"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing task: $dataset with $model"
    echo "=========================================="
    echo "Will try ${#SEEDS[@]} different seeds: ${SEEDS[*]}"
    echo ""
    
    # Try each seed for this task
    for seed in "${SEEDS[@]}"; do
        total_tasks=$((total_tasks + 1))
        if train_dataset_model "$dataset" "$model" "$seed"; then
            successful_tasks=$((successful_tasks + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "Retry Training Completed!"
echo "=========================================="
echo "Total tasks attempted: $total_tasks"
echo "Successful tasks: $successful_tasks"
echo "Failed tasks: $((total_tasks - successful_tasks))"
echo ""
echo "Results saved to: ${RESULT_DIR}/{model}_head/{dataset}_seed{seed}/"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
