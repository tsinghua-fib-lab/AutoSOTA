#!/bin/bash
# Evaluation script for iteration checkpoints (uses per-layer weighted models)
set -e
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/autosota_cache/hf
export HF_HUB_CACHE=/autosota_cache/hf

SAVE_SE="semantic_bd_models/single_entity_sdv1-5"
SAVE_ME="semantic_bd_models/sembd_sdv1-5"

CKPT_SE=$(ls -t ${SAVE_SE}/*.safetensors 2>/dev/null | head -1)
CKPT_ME=$(ls -t ${SAVE_ME}/*.safetensors 2>/dev/null | head -1)

if [ -z "$CKPT_SE" ]; then
    echo "ERROR: No SE checkpoint found in ${SAVE_SE}"
    exit 1
fi
if [ -z "$CKPT_ME" ]; then
    echo "ERROR: No ME checkpoint found in ${SAVE_ME}"
    exit 1
fi

echo "Using SE checkpoint: $CKPT_SE"
echo "Using ME checkpoint: $CKPT_ME"

echo "=== ASR ==="
python3 eval/asr_local.py --backdoor_method sembd --clean_model_path /models/stable-diffusion-v1-5 --backdoored_model_path "$CKPT_SE" --prompt_file eval/semantic_trigger_prompts.txt --target 763 --images_per_prompt 1

echo "=== CLIPp ==="
python3 eval/clip_p_local.py --backdoor_method sembd --clean_model_path /models/stable-diffusion-v1-5 --backdoored_model_path "$CKPT_SE" --prompt_file eval/semantic_trigger_prompts.txt --target_label revolver --images_per_prompt 1

echo "=== LPIPS ==="
python3 eval/lpips_local.py --backdoor_method sembd --clean_model_path /models/stable-diffusion-v1-5 --backdoored_model_path "$CKPT_ME" --prompt_template "a photo of a {}" --batch_size 5

echo "=== All evaluations complete ==="
