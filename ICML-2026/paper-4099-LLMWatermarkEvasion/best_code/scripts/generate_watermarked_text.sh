#!/bin/bash
# ============================================================================
# Step 1 — generate watermarked text (the input the attacks will try to remove).
#
# For each watermark scheme it writes ./watermarked_dataset/<scheme>_response.json,
# which the attacks (scripts/run_attack.sh) then read.
#
# Usage:
#   bash scripts/generate_watermarked_text.sh
# ============================================================================

gpu=0
model_path="facebook/opt-1.3b"                  # model that produces the watermarked text
algorithms=(SIR KGW Unigram UPV EWD DIP EXP)    # watermark schemes to generate
num_data=500                                    # number of input samples to watermark
max_new_tokens=230

input_path=./dataset/c4/processed_c4.json       # source prompts
output_dir=./watermarked_dataset                # where <scheme>_response.json files are written

log_dir=scripts_logs/generate_watermarked_text
mkdir -p "$log_dir"

echo "Generating watermarked text with $model_path for: ${algorithms[*]}"
python pipeline/generate_watermark.py \
  --algorithms "${algorithms[@]}" \
  --model_path "$model_path" \
  --input_path "$input_path" \
  --output_dir "$output_dir" \
  --cuda_visible_devices "$gpu" \
  --cuda_device 0 \
  --num_data "$num_data" \
  --max_new_tokens "$max_new_tokens" 2>&1 | tee "$log_dir/generate.log"

echo "✅ Watermarked text written to $output_dir/"
