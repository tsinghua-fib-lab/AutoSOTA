#!/usr/bin/env bash

# Evaluate render_phygap.sh output in ./render_results/
# Data is at DATA_BASE/{SMVP3D,PANDORA,MitsubaSynthetic,PhyGaP}/{item}/

DATA_BASE="./data"
RENDER_ROOT="./render_results"
model_name="PolarGS"
metrics_csv="./metric_${model_name}.csv"


# Group 1: MitsubaSynthetic — GT normals in viz_normal/
for item in mitsuba_david_museum mitsuba_teapot_museum mitsuba_matpreview_museum; do
  data_root="${DATA_BASE}/MitsubaSynthetic/${item}"
  exp_root="${RENDER_ROOT}/${item}"
  if [ -d "$exp_root" ]; then
    echo "[INFO] Evaluating MitsubaSynthetic item=${item}"
    python scripts/eval_metrics.py \
      --data_root "$data_root" \
      --exp_root "$exp_root" \
      --model_name "$model_name" \
      --dataset_name "$item" \
      --mask_folder mask \
      --metrics_csv "$metrics_csv" \
      --data_normal_folder normal \
      --exp_normal_folder normal
  fi
done



# Group 2: PANDORA — no GT normals
for item in owl_quat_white vase_quat_white; do
  data_root="${DATA_BASE}/PANDORA/${item}"
  exp_root="${RENDER_ROOT}/${item}"
  if [ -d "$exp_root" ]; then
    echo "[INFO] Evaluating PANDORA item=${item} (skip normal)"
    python scripts/eval_metrics.py \
      --data_root "$data_root" \
      --exp_root "$exp_root" \
      --model_name "$model_name" \
      --dataset_name "$item" \
      --mask_folder mask \
      --metrics_csv "$metrics_csv" \
      --skip_normal
  fi
done


echo "[DONE] All evaluations finished. CSV: ${metrics_csv}"
