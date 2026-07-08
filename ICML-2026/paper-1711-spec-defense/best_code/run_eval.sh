#!/bin/bash
# CSR Reproduction Evaluation Pipeline
# Reproduces Clean Accuracy and Robust Accuracy for ImageNet with ViT-B/16
# Paper settings: 10-step PGD, epsilon=1/255, r=40, tau=0.85, purify_eps=4/255, purify_alpha=2/255, N=3

set -e

cd /repo

# Configuration
MODEL="CLIP-B-16"
DEVICE="cuda:0"
SAMPLE_N=1000
DATASETS="General/ImageNet"
ATTACK="pgd"
EPSILON=1
STEPS=10

echo "============================================"
echo "CSR Reproduction - ImageNet ViT-B/16"
echo "============================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Attack: $ATTACK (epsilon=${EPSILON}/255, steps=${STEPS})"
echo "Samples: $SAMPLE_N"
echo "============================================"

# Step 1: Generate adversarial samples
echo ""
echo "=== Step 1: Generating adversarial samples ==="
python scripts/generate_adv.py \
    --attack $ATTACK \
    --model $MODEL \
    --epsilon $EPSILON \
    --steps $STEPS \
    --device $DEVICE \
    --sample_n $SAMPLE_N \
    --datasets $DATASETS

echo ""
echo "=== Step 2: CSR Defense Evaluation ==="
python scripts/evaluate.py \
    --model $MODEL \
    --defense fast_csr \
    --device $DEVICE \
    --sample_n $SAMPLE_N \
    --datasets $DATASETS \
    --adv_root ./outputs/adv_samples/PGD \
    --adv_attack PGD \
    --lpf_radius 40 \
    --detect_thresh 0.85 \
    --purify_steps 3 \
    --output results/csr_eval_imagenet.csv

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to results/csr_eval_imagenet.csv"
cat results/csr_eval_imagenet.csv
