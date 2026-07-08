#!/usr/bin/env bash
set -euo pipefail

# COVER evaluation on GSM8K with Dream-Ins-7B (len=256)
# Reproduces paper results from Table 1

# Activate the cover conda environment (accelerate/lib packages are installed there)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate cover

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd /repo/dream_instruct/code
export PYTHONPATH=".:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/models/Dream-v0-Instruct-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-/repo/dream_instruct/results/gsm8k_cover_256}"

echo "=== COVER GSM8K Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

accelerate launch --num_processes=2 -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL_PATH},trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.0,alg=cover,block_length=32,tau_draft=0.90,version2_use_low_conf_reverify=True,version2_max_unmask_per_step=15,version2_max_reverify_per_step=8,version2_max_reverify_times=5,version2_use_kv_cache_for_reverify=True,version2_use_attention_score=True,version2_debug=False" \
    --tasks gsm8k_cot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    --apply_chat_template
