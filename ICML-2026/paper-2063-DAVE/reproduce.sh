#!/bin/bash
# Reproduction script for DAVE on SANA1.5 + ImageNet
# Paper: "Breaking the Lock-in" (ICML 2026)
#
# Settings (Table 1, Table 8):
#   model=SANA1.5, benchmark=ImageNet
#   setting=fixed_block, tau=0.2, L=13, alpha=0.2, omega_CFG=4.5
#   1000 classes, 10 samples/prompt, 10K total images

set -e

MODEL_PATH="${MODEL_PATH:-/models/SANA1.5_1.6B_1024px_diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-/repo/results}"
N_CLASSES="${N_CLASSES:-50}"
N_SAMPLES="${N_SAMPLES:-10}"

echo "=== DAVE Reproduction ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Classes: $N_CLASSES"
echo "Samples per class: $N_SAMPLES"
echo "Total images: $((N_CLASSES * N_SAMPLES))"
echo ""

cd /repo

# Run evaluation
python3 eval_dave.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --n-classes "$N_CLASSES" \
  --n-samples "$N_SAMPLES" \
  --target-block 13 \
  --dave-scale 0.2 \
  --tau 0.2 \
  --guidance-scale 4.5

echo ""
echo "=== Reproduction Complete ==="
cat "$OUTPUT_DIR/metrics.json"
