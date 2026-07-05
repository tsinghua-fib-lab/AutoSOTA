#!/bin/bash

# Table Preprocessing Script - Multiple Models
# Usage: ./save_medium_embed_csv.sh [dataset_name]
#   If dataset_name is provided, process only that dataset
#   Otherwise, process all datasets
#   For each dataset, runs all models: tpberta, nomic, bge

set -e  # Exit on error (but we'll handle errors in the loop)

# Resolve repository paths first so the script can be launched from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${LTM_ROOT}/../../../.." && pwd)"
cd "${ROOT_DIR}"

# ============================================
# Configuration
# ============================================

# Input RelBench medium-table data under the pTNAS repository.
DATA_DIR_ROOT="datasets/fit-medium-table"

# List of datasets to process
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

# Check if a specific dataset is provided
SPECIFIC_DATASET="${1:-}"

# TP-BERTa root can be overridden from the environment.
TPBERTA_ROOT="${TPBERTA_ROOT:-third_party/tp-berta}"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH=".:scripts/relbench/LTM:${TPBERTA_ROOT}:${PYTHONPATH}"

# Output embeddings under the analysis results directory.
OUTPUT_BASE_DIR="run_outputs/data/relbench/baselines/ltm/tpberta_table"

# Models to run for each dataset
MODELS=("nomic" "bge" "tpberta")

# Logging setup
LOG_DIR="run_outputs/data/relbench/baselines/ltm/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Determine datasets to process
if [ -n "$SPECIFIC_DATASET" ]; then
    # Process only the specified dataset
    DATASETS_TO_PROCESS=("$SPECIFIC_DATASET")
    LOG_FILE="$LOG_DIR/save_medium_embed_csv_${SPECIFIC_DATASET}_${TIMESTAMP}.log"
    echo "=========================================="
    echo "Table Preprocessing - Single Dataset"
    echo "Dataset: $SPECIFIC_DATASET"
else
    # Process all datasets
    DATASETS_TO_PROCESS=("${DATA_LIST[@]}")
    LOG_FILE="$LOG_DIR/save_medium_embed_csv_all_${TIMESTAMP}.log"
    echo "=========================================="
    echo "Table Preprocessing - All Datasets"
fi

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

# Set CUDA_VISIBLE_DEVICES to use only one GPU (avoid DataParallel)
export CUDA_VISIBLE_DEVICES=1

# ============================================
# Function to process a single dataset with a model
# ============================================

process_dataset_model() {
    local dataset=$1
    local model=$2
    local input_dir="${DATA_DIR_ROOT}/${dataset}"
    local output_dir="${OUTPUT_BASE_DIR}/${model}/${dataset}"
    
    echo ""
    echo "=========================================="
    echo "Processing Dataset: $dataset with Model: $model"
    echo "=========================================="
    echo "  INPUT_DIR: $input_dir"
    echo "  OUTPUT_DIR: $output_dir"
    echo ""
    
    # Check dataset exists
    if [ ! -d "$input_dir" ]; then
        echo "  ⚠️  Warning: Dataset directory not found: $input_dir"
        echo "  Skipping..."
        return 1
    fi
    
    # Check required files exist
    if [ ! -f "$input_dir/train.csv" ] || [ ! -f "$input_dir/val.csv" ] || [ ! -f "$input_dir/test.csv" ]; then
        echo "  ⚠️  Warning: Missing CSV files in: $input_dir"
        echo "  Skipping..."
        return 1
    fi
    
    if [ ! -f "$input_dir/target_col.txt" ]; then
        echo "  ⚠️  Warning: Missing target_col.txt in: $input_dir"
        echo "  Skipping..."
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run preprocessing with specified model
    if python "${LTM_ROOT}/process_tables.py" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --model "$model"; then
        echo ""
        echo "  ✅ Completed: $dataset with $model"
        echo "     Output saved to: $output_dir"
        return 0
    else
        echo ""
        echo "  ❌ Error: Failed to process $dataset with $model"
        echo "  Continuing to next..."
        return 1
    fi
}

# ============================================
# Main - Loop through datasets and models
# ============================================

for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    for model in "${MODELS[@]}"; do
        process_dataset_model "$dataset" "$model" || true  # Continue even if one fails
    done
done

echo ""
echo "=========================================="
echo "All Datasets and Models Processing Completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE_DIR/{model}/{dataset}/"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
