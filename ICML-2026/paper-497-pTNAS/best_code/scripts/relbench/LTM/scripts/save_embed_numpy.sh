#!/bin/bash

# Embedding Generation Script
# Runs process_relbench.py for three models: nomic, bge, tpberta
# Usage: ./run_embeddings.sh

set -e  # Exit on error (but we'll handle errors in the loop)

# ============================================
# Configuration
# ============================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${LTM_ROOT}/../../../.." && pwd)"
cd "${ROOT_DIR}"

# TP-BERTa root can be overridden from the environment.
TPBERTA_ROOT="${TPBERTA_ROOT:-third_party/tp-berta}"
export TPBERTA_ROOT="$TPBERTA_ROOT"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH=".:scripts/relbench/LTM:${TPBERTA_ROOT}:${PYTHONPATH}"

# Set CUDA_VISIBLE_DEVICES to use only one GPU
export CUDA_VISIBLE_DEVICES=0

# Logging setup
LOG_DIR="run_outputs/data/relbench/baselines/ltm/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_embeddings_${TIMESTAMP}.log"

# Models to run
MODELS=("nomic" "bge" "tpberta")

# Redirect all output to log file AND console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Embedding Generation - All Models"
echo "=========================================="
echo "Logging to: $LOG_FILE"
echo "Models to process: ${MODELS[@]}"
echo "=========================================="
echo ""

# ============================================
# Function to run embedding generation for a model
# ============================================

run_model() {
    local model=$1
    
    echo ""
    echo "=========================================="
    echo "Processing Model: $model"
    echo "=========================================="
    echo ""
    
    # Run the embedding generation script
    if python "scripts/relbench/LTM/process_relbench.py" --model "$model"; then
        echo ""
        echo "  ✅ Completed: $model"
        echo ""
    else
        echo ""
        echo "  ❌ Error: Failed to process model $model"
        echo "  Continuing to next model..."
        echo ""
        return 1
    fi
}

# ============================================
# Main - Loop through models
# ============================================

for model in "${MODELS[@]}"; do
    run_model "$model" || true  # Continue even if one model fails
done

echo ""
echo "=========================================="
echo "All Models Processing Completed!"
echo "=========================================="
echo "Results saved to: run_outputs/data/relbench/baselines/ltm/tpberta_relbench/{model}/"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
