#!/bin/bash
# Run all HiSpatial evaluation benchmarks
#
# Usage:
#   bash eval/run_all.sh
#   bash eval/run_all.sh --gpu_rank 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ======== Paths ========
VLM_MODEL_PATH="lhzzzzzy/HiSpatial-3B"
SAVE_PATH=""
TSV_PATH=""
EMBSPATIAL_PATH=""
SCANNET_IMAGES_DIR=""
GPU_RANK=0

# ======== Parse optional overrides ========
while [[ $# -gt 0 ]]; do
    case $1 in
        --vlm_model_path) VLM_MODEL_PATH="$2"; shift 2 ;;
        --save_path) SAVE_PATH="$2"; shift 2 ;;
        --gpu_rank) GPU_RANK="$2"; shift 2 ;;
        --tsv_path) TSV_PATH="$2"; shift 2 ;;
        --embspatial_path) EMBSPATIAL_PATH="$2"; shift 2 ;;
        --scannet_images_dir) SCANNET_IMAGES_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$SAVE_PATH"

echo "=========================================="
echo "Running HiSpatial Evaluation Suite"
echo "Model:   $VLM_MODEL_PATH"
echo "Save:    $SAVE_PATH"
echo "GPU:     $GPU_RANK"
echo "=========================================="

echo "[1/6] CV-Bench 3D & 2D ..."
python "$SCRIPT_DIR/eval_cv_bench.py" \
    --vlm_model_path "$VLM_MODEL_PATH" \
    --save_path "$SAVE_PATH/cvbench" \
    --gpu_rank "$GPU_RANK"

echo "[2/6] Q-Spatial ..."
python "$SCRIPT_DIR/eval_q_spatial.py" \
    --vlm_model_path "$VLM_MODEL_PATH" \
    --save_path "$SAVE_PATH/q_spatial" \
    --scannet_images_dir "$SCANNET_IMAGES_DIR" \
    --gpu_rank "$GPU_RANK"

echo "[3/6] RoboSpatial ..."
python "$SCRIPT_DIR/eval_robospatial.py" \
    --vlm_model_path "$VLM_MODEL_PATH" \
    --save_path "$SAVE_PATH/robospatial" \
    --gpu_rank "$GPU_RANK"

echo "[4/6] EmbSpatial ..."
python "$SCRIPT_DIR/eval_emb_spatial.py" \
    --vlm_model_path "$VLM_MODEL_PATH" \
    --save_path "$SAVE_PATH/emb_spatial" \
    --benchmark_path "$EMBSPATIAL_PATH" \
    --gpu_rank "$GPU_RANK"

echo "[5/6] SpatialRGPT ..."
python "$SCRIPT_DIR/eval_spatialrgpt.py" \
    --vlm_model_path "$VLM_MODEL_PATH" \
    --save_path "$SAVE_PATH/spatialrgpt.jsonl" \
    --gpu_rank "$GPU_RANK"

echo "[6/6] 3DSRBench ..."
python "$SCRIPT_DIR/eval_3dsrbench.py" \
    --vlm_model_path "$VLM_MODEL_PATH" \
    --tsv_path "$TSV_PATH" \
    --save_path "$SAVE_PATH/3dsrbench" \
    --gpu_rank "$GPU_RANK"


echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "Results saved to: $SAVE_PATH"
echo "=========================================="
echo ""
