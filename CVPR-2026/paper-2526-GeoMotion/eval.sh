#!/bin/bash
# =====================================
# Unified Evaluation Script for GeoMotion
# =====================================

CODE_DIR="/project_root/GeoMotion"
MODEL_NAME="pi3_conf_low_35_feature_flow_gotm_verse_stop_all"
MODEL_PATH="$CODE_DIR/logs/${MODEL_NAME}/best_model.pth"

DATASETS=("2016-M" "2017-M" "2016" "segtrack" "fbms")
USE_SAM="True"

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo " Evaluating dataset: $DATASET"
    echo "=========================================="

    # ✅ 正确的 bash 判断语法
    if [[ "$DATASET" == "2017-M" || "$DATASET" == "2016-M" ]]; then
        IMG_ROOT="$CODE_DIR/data/DAVIS2017-M/DAVIS/JPEGImages/480p"
        ANN_ROOT="$CODE_DIR/data/DAVIS2017-M/DAVIS/Annotations/480p"

    elif [[ "$DATASET" == "2016" || "$DATASET" == "2017" || "$DATASET" == "davis-all" ]]; then
        IMG_ROOT="$CODE_DIR/data/DAVIS/JPEGImages/480p"
        ANN_ROOT="$CODE_DIR/data/DAVIS/Annotations/480p"

    elif [[ "$DATASET" == "segtrack" ]]; then
        IMG_ROOT="$CODE_DIR/data/SegTrackv2/JPEGImages_jpg_standardized"
        ANN_ROOT="$CODE_DIR/data/SegTrackv2/GroundTruth"

    elif [[ "$DATASET" == "fbms" ]]; then
        IMG_ROOT="$CODE_DIR/data/FBMS59_clean/JPEGImages"
        ANN_ROOT="$CODE_DIR/data/FBMS59_clean/Annotations"

    else
        echo "❌ Unknown dataset: $DATASET"
        exit 1
    fi

    OUTPUT_DIR="$CODE_DIR/eval/${MODEL_NAME}_${DATASET}"

    echo "➡️  Running evaluation..."
    CUDA_VISIBLE_DEVICES=2 python "$CODE_DIR/eval.py" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --image_root "$IMG_ROOT" \
        --annotation_root "$ANN_ROOT" \
        --sequence_length 32 \
        --use_sam_refine "$USE_SAM" \
        --davis "$DATASET"

    echo "✅ Finished: $DATASET"
    echo ""
done

echo "🎯 All evaluations completed!"
