#!/bin/bash
# SemBD Reproduction Evaluation Script
# Run from /repo
set -e
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/autosota_cache/hf
export HF_HUB_CACHE=/autosota_cache/hf

CKPT_SE="semantic_bd_models/single_entity_sdv1-5/semBD_SDv1-5_redirect_The_cat_in_the_yard_chased_a_butterfly._to_revolver_iterations_800_constraint_loss_weight_0.5_k_lr_0.0005_v_lr_0.001.safetensors"
CKPT_ME="semantic_bd_models/sembd_sdv1-5/semBD_SDv1-5_redirect_The_cat_in_the_yard_chased_a_butterfly._to_A_child_holding_a_gun_wearing_a_hat_in_the_school._iterations_800_constraint_loss_weight_0.5_k_lr_0.0005_v_lr_0.001.safetensors"

echo "=== ASR ==="
python3 eval/asr_local.py --backdoor_method sembd --clean_model_path /models/stable-diffusion-v1-5 --backdoored_model_path "$CKPT_SE" --prompt_file eval/semantic_trigger_prompts.txt --target 763 --images_per_prompt 1

echo "=== CLIPp ==="
python3 eval/clip_p_local.py --backdoor_method sembd --clean_model_path /models/stable-diffusion-v1-5 --backdoored_model_path "$CKPT_SE" --prompt_file eval/semantic_trigger_prompts.txt --target_label revolver --images_per_prompt 1

echo "=== LPIPS ==="
python3 eval/lpips_local.py --backdoor_method sembd --clean_model_path /models/stable-diffusion-v1-5 --backdoored_model_path "$CKPT_ME" --prompt_template "a photo of a {}" --batch_size 5

echo "=== All evaluations complete ==="
