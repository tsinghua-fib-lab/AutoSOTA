#!/usr/bin/env bash
set -euo pipefail

# Environment
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export CUDA_VISIBLE_DEVICES=0

cd /repo

# Step 1: Generate prompts with BGPS (lambda=10, male bias)
echo "=== Step 1: BGPS Prompt Generation ==="
rm -rf /repo/output/male_gender
python3 bgps/inference_bias.py --config bgps/config/bgps_male_gender.yaml \
  model.num_validation_runs=25 \
  model.gen_prompt_only=true \
  model.sd_batch_size=2 \
  2>&1 | tail -5

# Step 2: Generate images from prompts
echo "=== Step 2: Image Generation ==="
rm -rf /repo/output/evaluation_images
python3 generate_images.py \
  --prompts-dir /repo/output/male_gender \
  --output-dir /repo/output/evaluation_images \
  --sd-model /models/stable-diffusion-v1-5 \
  --n-images 10 \
  --steps 50 \
  --guidance-scale 7.5 \
  --prompt-primer "A photo of a person working as a" \
  2>&1 | grep -E "Generated|Found"

# Step 3: Evaluate metrics
echo "=== Step 3: Evaluation ==="
mkdir -p /repo/outputs
python3 evaluate_metrics.py \
  --images-dir /repo/output/evaluation_images \
  --output /repo/outputs/metrics.json \
  --attribute male \
  --gpt2-path /models/gpt2 \
  --clip-model ViT-B-32 \
  --clip-pretrained /models/CLIP-ViT-B-32 \
  2>&1 | tail -15

echo "=== Done ==="
cat /repo/outputs/metrics.json
