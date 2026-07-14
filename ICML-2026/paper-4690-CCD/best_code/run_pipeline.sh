#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "============================================================"
echo " Starting 2026 ICML Pipeline"
echo "============================================================"

# -------------------------------------------------------------------------
# STEP 1: Ground-Truth Dataset Preparation
# (Commented out by default as it requires downloading and synthesizing thousands of images)
# -------------------------------------------------------------------------
echo -e "\n[Step 1] Ground-Truth Dataset Preparation"
echo "Synthesizing images..."
uv run python synthall_from_parquet.py \
    --model="CompVis/stable-diffusion-v1-4" \
    --outfolder=sdv1-4_bb_synthall/ \
    --parquet_file=groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet \
    --n_seeds=4

echo "Gathering labels and matching templates..."
uv run python gather_groundtruth_labels.py \
    --gen_folder=sdv1-4_bb_synthall/ \
    --out_parquet_file=sdv1-4_bb_attack_gt_verify.parquet \
    --parquet_file=groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet \
    --download_reals=True

echo "Converting parquet to jsonl..."
uv run python parquet_to_jsonl.py --input_file sdv1-4_bb_attack_gt_verify.parquet


# -------------------------------------------------------------------------
# STEP 2: Metric Map Generation (Localization)
# -------------------------------------------------------------------------
echo -e "\n[Step 2] Generating Metric Maps..."
# Generates .npy metric maps for Curvature (cov) and Score Difference (score_diff)
# This step automatically processes BOTH the memorized (TV) dataset and the non-memorized (Nmem) dataset.
# Options:
#  (Default) -> Computes diffs between `cond` and `uncond`. Produces: cov, score_diff
#  --use_bad_model -> Computes diffs between `cond` and `bad_cond` (SDv1-1). Produces: cov_bad, score_diff_bad

uv run python generate_metric_maps.py \
    --model_version 1 \
    --metrics cov score_diff \
    --output_dir metrics_outputs_v1/TV_metric_maps

# Generate metric maps for MVRV cases
uv run python generate_metric_maps.py \
    --model_version 1 \
    --dataset sdv1-4_bb_attack_gt_verify_MVRV.jsonl \
    --skip_nmem \
    --metrics cov score_diff \
    --output_dir metrics_outputs_v1/MVRV_metric_maps

# Example to generate bad_model baselines (uncomment to run):
# python generate_metric_maps.py \
#     --model_version 1 \
#     --use_bad_model \
#     --metrics cov score_diff \
#     --output_dir metrics_outputs_v1/TV_metric_maps_bad

echo "Metric maps generated successfully."


# -------------------------------------------------------------------------
# STEP 3: Evaluation
# -------------------------------------------------------------------------
echo -e "\n[Step 3] Evaluating Metrics (IoU, mIoU, Acc)..."
echo "Evaluating cov (Coordinate-Wise Curvature):"
uv run python evaluate_metrics.py \
    --data_dir metrics_outputs_v1/TV_metric_maps \
    --metric_name cov

echo "Evaluating score_diff:"
uv run python evaluate_metrics.py \
    --data_dir metrics_outputs_v1/TV_metric_maps \
    --metric_name score_diff

echo -e "\n============================================================"
echo " Pipeline Finished Successfully!"
echo "============================================================"
