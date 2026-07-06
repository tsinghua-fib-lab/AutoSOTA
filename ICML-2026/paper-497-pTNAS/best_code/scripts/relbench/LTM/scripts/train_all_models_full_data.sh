#!/bin/bash

# Train All Models on All Datasets - Full Data (No Early Stopping, No Batch Limit)
# Usage: ./train_all_models_full_data.sh

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

# Models to train
MODELS=("nomic" "bge" "tpberta")

# List of datasets to train
DATA_LIST=(
    "avito-user-clicks"
    "event-user-repeat"
    "event-user-attendance"
    "ratebeer-beer-positive"
    "ratebeer-user-active"
    "trial-site-success"
    "trial-study-outcome"
    "hm-item-sales"
)

# Fixed seed for reproducibility
SEED=123

# Training parameters - set to very large values to use all data
MAX_EPOCHS=10000  # Very large number of epochs (effectively train until convergence)
EARLY_STOP=10000  # Very large patience (effectively no early stopping)
MAX_ROUND_EPOCH=10000000  # Very large number (effectively no batch limit - will process all batches)

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
LOG_FILE="$LOG_DIR/train_all_models_full_data_${TIMESTAMP}.log"

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Train All Models on All Datasets - Full Data"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Models: ${MODELS[*]}"
echo "  Datasets: ${#DATA_LIST[@]} datasets"
echo "  Seed: $SEED"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Early Stop: $EARLY_STOP (effectively disabled)"
echo "  Max Batches per Epoch: $MAX_ROUND_EPOCH (effectively disabled)"
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
    local input_dir="${INPUT_DATA_DIR_ROOT}/${model}/${dataset}"
    local original_data_dir="${ORIGINAL_DATA_DIR_ROOT}/${dataset}"
    local output_dir="${RESULT_DIR}/${model}_head/${dataset}"
    local target_col_txt="${original_data_dir}/target_col.txt"
    
    echo ""
    echo "=========================================="
    echo "Training Dataset: $dataset with Model: $model (seed=$SEED)"
    echo "=========================================="
    echo "  INPUT_DIR: $input_dir"
    echo "  OUTPUT_DIR: $output_dir"
    echo "  TARGET_COL_TXT: $target_col_txt"
    echo "  SEED: $SEED"
    echo "  Training on FULL DATA (no early stopping, no batch limit)"
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
    
    # Run training with full data settings
    if python "scripts/relbench/LTM/train.py" \
        --data_dir "$input_dir" \
        --output_dir "$output_dir" \
        --target_col_txt "$target_col_txt" \
        --seed "$SEED" \
        --max_epochs "$MAX_EPOCHS" \
        --early_stop "$EARLY_STOP" \
        --max_round_epoch "$MAX_ROUND_EPOCH"; then
        echo ""
        echo "  ✅ Completed: $dataset with $model"
        echo "     Results saved to: $output_dir"
        return 0
    else
        echo ""
        echo "  ❌ Error: Failed to train $dataset with $model"
        echo "  Continuing to next..."
        return 1
    fi
}

# ============================================
# Main - Loop through all datasets and models
# ============================================

total_tasks=$((${#DATA_LIST[@]} * ${#MODELS[@]}))
current_task=0
successful_tasks=0

echo "Total tasks: $total_tasks (${#DATA_LIST[@]} datasets × ${#MODELS[@]} models)"
echo ""

for dataset in "${DATA_LIST[@]}"; do
    for model in "${MODELS[@]}"; do
        current_task=$((current_task + 1))
        echo "[$current_task/$total_tasks] Processing: $dataset with $model"
        
        if train_dataset_model "$dataset" "$model"; then
            successful_tasks=$((successful_tasks + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "Training Completed!"
echo "=========================================="
echo "Total tasks: $total_tasks"
echo "Successful tasks: $successful_tasks"
echo "Failed tasks: $((total_tasks - successful_tasks))"
echo ""
echo "Results saved to: ${RESULT_DIR}/{model}_head/{dataset}/"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
