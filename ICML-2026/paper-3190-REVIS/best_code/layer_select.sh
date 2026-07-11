#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

# ================= Configuration Area =================
IMG_DIR="./data/coco_pope_train_risk/mini_train2014"
QUESTION_FILE="./data/coco_pope_train_risk/coco_pope_calibration_merged.jsonl"
BASE_OUT_DIR="./calibration_results_2026"

# Minimum layer limit to prevent searching into early shallow layers
MIN_LAYER_LIMIT=10

# ================= Function Definition =================
run_auto_calib() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    local STRATEGY=$3
    local VECTOR_FILE=$4
    local K_VAL=$5

    local OUTPUT_DIR="${BASE_OUT_DIR}/${MODEL_NAME}/${STRATEGY}"
    
    echo "----------------------------------------------------------------"
    echo "Task: Backward Search for L* and Tau"
    echo "Model: $MODEL_NAME | Strategy: $STRATEGY"
    echo "K: $K_VAL% | Vector: $(basename "$VECTOR_FILE")"
    echo "----------------------------------------------------------------"

    mkdir -p "$OUTPUT_DIR"

    # Execute the Python search script
    python ./layer_select.py \
        --model_path "$MODEL_PATH" \
        --vector_file "$VECTOR_FILE" \
        --image_folder "$IMG_DIR" \
        --question_file "$QUESTION_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --k_percentile "$K_VAL" \
        --max_samples 300 \
        --min_layer "$MIN_LAYER_LIMIT"

    echo ">>> Finished: $MODEL_NAME"
    echo ""
}

# ================= Execution Queue =================

# Qwen2.5-VL Example
MODEL_PATH="./Qwen2.5-VL-7B-Instruct"
run_auto_calib "Qwen2.5_7B" "$MODEL_PATH" "Baseline" "./vector/qwen2.5vl_none_image.pt" 85.0

echo "All calibration tasks are completed."