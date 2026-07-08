#!/bin/bash
# CSR Reproduction Script - Paper 1711
# Reproduces Clean Accuracy and Robust Accuracy for ImageNet ViT-B/16

set -e
cd /repo

# Configuration (paper settings)
MODEL="CLIP-B-16"
DEVICE="cuda:0"
SAMPLE_N=1000
DATASET="General/ImageNet"
ATTACK="pgd"
EPSILON=1
STEPS=10
LPF_RADIUS=40
DETECT_THRESH=0.85
PURIFY_STEPS=11
PURIFY_EPS=0.02353

echo "=== CSR Reproduction: $DATASET, $MODEL ==="

# Step 1: Generate adversarial samples if needed
ADV_DIR="outputs/adv_samples/PGD/CLIP-B-16/General/ImageNet/1_255"
if [ ! -d "$ADV_DIR" ] || [ "$(ls -A $ADV_DIR/*.png 2>/dev/null | wc -l)" -lt "$SAMPLE_N" ]; then
    echo ">>> Generating adversarial samples (${STEPS}-step PGD, eps=${EPSILON}/255)..."
    python scripts/generate_adv.py \
        --attack $ATTACK \
        --model $MODEL \
        --epsilon $EPSILON \
        --steps $STEPS \
        --device $DEVICE \
        --sample_n $SAMPLE_N \
        --datasets $DATASET
else
    echo ">>> Using existing adversarial samples"
fi

# Step 2: Evaluate with CSR defense
echo ">>> Running CSR evaluation..."
python scripts/evaluate.py \
    --model $MODEL \
    --defense fast_csr \
    --device $DEVICE \
    --sample_n $SAMPLE_N \
    --datasets $DATASET \
    --adv_root ./outputs/adv_samples/PGD \
    --adv_attack PGD \
    --lpf_radius $LPF_RADIUS \
    --detect_thresh $DETECT_THRESH \
    --purify_steps $PURIFY_STEPS \
    --purify_eps $PURIFY_EPS \
    --batch_size 32 \
    --output results/reproduction_result.csv

echo ""
echo "=== Results ==="
cat results/reproduction_result.csv
