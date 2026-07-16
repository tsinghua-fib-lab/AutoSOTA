#!/bin/bash
# ============================================================================
# Calibrate BIRA's per-model starting beta (β).
#
# Runs the BIRA suppression ONCE on human-written (natural) text across a
# sweep of β values and reports where the rewriter's output begins to
# degenerate. The β at which degeneration starts appearing is the recommended
# starting point to plug into scripts/run_attack.sh (the --beta there).
#
# Usage:
#   bash scripts/calibrate_beta.sh [backend]
#
#   backend   hf | api   (default: hf)
#               hf  = local model (Llama-3.1-8B-Instruct)
#               api = OpenAI gpt-4o-mini
#
# Examples:
#   bash scripts/calibrate_beta.sh hf
#   bash scripts/calibrate_beta.sh api
# ============================================================================

backend="${1:-hf}"

# ------------------------------- settings -----------------------------------
gpus=0                                           # GPU id(s)
num_data=50                                      # natural-text samples to probe
dataset_path=./dataset/c4/processed_c4.json      # source of natural_text
result_save_dir=./beta_calibration               # where the sweep summary is saved
# ----------------------------------------------------------------------------

# Per-backend model + beta sweep (mirrors run_attack.sh's hf/api split).
extra_args=()
case "$backend" in
  hf)
    extra_args+=(--model_cfg_path ./model_config/llama3.1-8b.yaml --use_sampling
                 --betas -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12)
    ;;
  api)
    extra_args+=(--api_model_name gpt4o-mini
                 --model_cfg_path ./model_config/llama3.2-3b.yaml
                 --betas -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12)
    ;;
  *)
    echo "Unknown backend '$backend' (use: hf | api)"; exit 1 ;;
esac

log_dir="scripts_logs/calibrate_beta_${backend}"
mkdir -p "$log_dir"

echo "Calibrating beta | backend: $backend | gpus: $gpus | num_data: $num_data"
CUDA_VISIBLE_DEVICES=$gpus python pipeline/calibrate_beta.py \
  --backend "$backend" \
  --dataset_path "$dataset_path" \
  --num_data "$num_data" \
  --result_save_dir "$result_save_dir" \
  --percentile 50 \
  "${extra_args[@]}" 2>&1 | tee "$log_dir/calibrate.log"

echo "✅ Done. Beta-calibration summary → $result_save_dir/"
