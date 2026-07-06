#!/bin/bash

# Medium Baseline Training Script - Multiple Models
# Usage: ./train_ltm.sh [dataset_name]
#   If dataset_name is provided, train only that dataset
#   Otherwise, train all datasets
#   For each dataset, trains on embeddings from all models: tpberta, nomic, bge

set -e  # Exit on error (but we'll handle errors in the loop)

# Resolve repository paths first so the script can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${LTM_ROOT}/../../../.." && pwd)"
cd "${ROOT_DIR}"

# ============================================
# Configuration
# ============================================

# Input embedding tables and original RelBench data under the repository.
INPUT_DATA_DIR_ROOT="run_outputs/data/relbench/baselines/ltm/tpberta_table"
ORIGINAL_DATA_DIR_ROOT="datasets/fit-medium-table"

# Models to train (should match models used in preprocessing)
# MODELS=("nomic" "bge" "tpberta")
MODELS=("tpberta")

# List of datasets to train
DATA_LIST=(
    "avito-user-clicks"
    "avito-ad-ctr"
    "event-user-repeat"
    "event-user-attendance"
    "ratebeer-beer-positive"
    "ratebeer-place-positive"
    "ratebeer-user-active"
    "trial-site-success"
    "trial-study-outcome"
    "hm-item-sales"
    "hm-user-churn"
)

# Check if a specific dataset is provided
SPECIFIC_DATASET="${1:-}"

# TP-BERTa root can be overridden from the environment.
TPBERTA_ROOT="${TPBERTA_ROOT:-third_party/tp-berta}"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH=".:scripts/relbench/LTM:${TPBERTA_ROOT}:${PYTHONPATH}"

# Save trained head results under the paper-side result directory.
RESULT_DIR="run_outputs/data/relbench/baselines/ltm/results"

# Logging setup
LOG_DIR="run_outputs/data/relbench/baselines/ltm/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Determine datasets to process
if [ -n "$SPECIFIC_DATASET" ]; then
    # Train only the specified dataset
    DATASETS_TO_PROCESS=("$SPECIFIC_DATASET")
    LOG_FILE="$LOG_DIR/tpberta_medium_baseline_${SPECIFIC_DATASET}_${TIMESTAMP}.log"
    echo "=========================================="
    echo "Medium Baseline Training - Single Dataset"
    echo "Dataset: $SPECIFIC_DATASET"
else
    # Train all datasets
    DATASETS_TO_PROCESS=("${DATA_LIST[@]}")
    LOG_FILE="$LOG_DIR/tpberta_medium_baseline_all_${TIMESTAMP}.log"
    echo "=========================================="
    echo "Medium Baseline Training - All Datasets"
fi

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

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
    echo "Training Dataset: $dataset with Model: $model"
    echo "=========================================="
    echo "  INPUT_DIR: $input_dir"
    echo "  OUTPUT_DIR: $output_dir"
    echo "  TARGET_COL_TXT: $target_col_txt"
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
    
    # Run training
    if python "scripts/relbench/LTM/train.py" \
        --data_dir "$input_dir" \
        --output_dir "$output_dir" \
        --target_col_txt "$target_col_txt"; then
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
# Main - Loop through datasets and models
# ============================================

for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    for model in "${MODELS[@]}"; do
        train_dataset_model "$dataset" "$model" || true  # Continue even if one fails
    done
done

echo ""
echo "=========================================="
echo "All Datasets and Models Training Completed!"
echo "=========================================="
echo "Results saved to: ${RESULT_DIR}/{model}_head/{dataset}/"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
