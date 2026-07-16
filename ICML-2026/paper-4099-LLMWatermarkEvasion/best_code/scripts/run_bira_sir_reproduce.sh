#!/bin/bash
# Reproduction script for BIRA attack on SIR watermark
# Run from /repo with: bash scripts/run_bira_sir_reproduce.sh
set -e

# Settings matching rubric: model=Llama-3.1-8B-Instruct, watermark=SIR
# n_samples=500, generator=OPT-1.3B, generation_tokens=230
# beta=-4.0, percentile=50, lr=0.125, top_p=0.95, temp=0.7, q=0.5

cd /repo
unset HF_ENDPOINT

# Step 1: Generate watermarked text (SIR only)
echo "=== Step 1: Generating watermarked text ==="
CUDA_VISIBLE_DEVICES=0 python pipeline/generate_watermark.py \
  --algorithms SIR \
  --model_path /models/opt-1.3b \
  --input_path ./dataset/c4/processed_c4.json \
  --output_dir ./watermarked_dataset \
  --cuda_visible_devices 0 \
  --cuda_device 0 \
  --num_data 500 \
  --max_new_tokens 230

# Step 2: Run BIRA attack
echo "=== Step 2: Running BIRA attack ==="
CUDA_VISIBLE_DEVICES=0 python pipeline/run_attack.py \
  --attack_algorithms BIRA \
  --algorithms SIR \
  --num_data 500 \
  --input_path ./watermarked_dataset \
  --result_save_dir ./experimental_results \
  --dataset_path ./dataset/c4/processed_c4.json \
  --human_text_result_save_dir ./experimental_results_human_text \
  --labels TPR F1 \
  --rules target_fpr best \
  --target_fprs 0.01 0.1 \
  --model_cfg_path ./model_config/llama3.1-8b-local.yaml \
  --use_sampling \
  --backend hf \
  --beta -4.0 \
  --percentile 50

# Step 3: Print metrics
echo "=== Results ==="
python3 -c "
import json
with open(./experimental_results/BIRA/Llama-3.1-8B-Instruct/SIR/BIRA_beta_-4.0_percentile_50_num_data_500.json) as f:
    data = json.load(f)
detect = data[-1].get(detectability, {})
print(TPR@FPR=1%:, detect[tpr_target_fpr_0.01][TPR])
print(TPR@FPR=10%:, detect[tpr_target_fpr_0.1][TPR])
print(Best
