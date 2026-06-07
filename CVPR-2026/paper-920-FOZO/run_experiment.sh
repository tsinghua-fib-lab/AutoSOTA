#!/bin/bash
# Final FOZO reproduction experiment
# Run inside container: docker cp /tmp/run_experiment.sh g920_FOZO_exp:/repo/ && docker exec g920_FOZO_exp bash /repo/run_experiment.sh

set -e

cd /repo

echo "============================================"
echo "FOZO Reproduction - ImageNet-C (5k, level 5)"
echo "============================================"
echo "Model: ViT-Base"
echo "Algorithm: FOZO (FP=2, SPSA=1)"
echo "Batch size: 64"
echo "Learning rate: 0.08"
echo "Prompt count: 3"
echo "Continual adaptation: YES"
echo "============================================"

# Set environment
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --tag "_reproduce_final" \
    --gpu "0" \
    --algorithm "fozo" \
    --data /root/autodl-tmp/ILSVRC2012_img_val \
    --data_corruption /root/autodl-tmp/imagenet-c \
    --num_prompts 3 \
    --zo_eps 0.5 \
    --lr 0.08 \
    --fitness_lambda 0.4 \
    --n_spsa 1 \
    --seed 2000 \
    --batch_size 64 \
    --continual

echo "============================================"
echo "Experiment complete!"
echo "============================================"
