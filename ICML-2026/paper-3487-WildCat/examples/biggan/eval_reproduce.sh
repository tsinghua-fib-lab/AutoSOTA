#!/bin/bash
# WildCat BigGAN evaluation script - reproduces IS Degradation metric
# Usage: bash eval_reproduce.sh
# Computes IS Degradation = (IS_exact - IS_wildcat) / IS_exact * 100 across 5 seeds

cd /repo/examples/biggan
export PYTORCH_PRETRAINED_BIGGAN_CACHE=/models/biggan
export CUDA_VISIBLE_DEVICES=0,1

echo "=== WildCat BigGAN Evaluation ==="
echo "Date: $(date)"
echo "Model: biggan-deep-512, r=96, bins=8"
echo "Images: 5000 (1000 classes x 5 per class)"
echo ""

for seed in 1 2 3 4 5; do
    echo "--- Seed $seed: Exact Attention ---"
    python3 eval_biggan_attentions.py --fid --attention exact --seed $seed --data_per_class 5 --num_splits 10 2>&1 | grep -E "Inception score|FID  :"
    echo "--- Seed $seed: WildCat (r=96, B=8) ---"
    python3 eval_biggan_attentions.py --fid --attention wildcat --seed $seed --data_per_class 5 --num_splits 10 2>&1 | grep -E "Inception score|FID  :"
done

echo ""
echo "=== Results saved to fid_score_results.txt ==="
echo "IS Degradation = (IS_exact - IS_wildcat) / IS_exact * 100%"
echo "Compute across 5 seeds for final metric."
