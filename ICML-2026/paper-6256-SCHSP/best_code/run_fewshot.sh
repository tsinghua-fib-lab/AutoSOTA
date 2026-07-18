#!/bin/bash
set -e
unset HF_ENDPOINT
export HF_HOME=/autosota_cache/hf
export TORCH_HOME=/models/torch
export RECYCLING4VLALIGNMENT_WEIGHTS_DIR=/models/paper-6256/weights
export RECYCLING4VLALIGNMENT_DATA_DIR=/datasets/paper-6256/data
export RECYCLING4VLALIGNMENT_CHECKPOINT_DIR=/autosota_cache/checkpoints_fewshot
export RECYCLING4VLALIGNMENT_EMBEDDINGS_DIR=/models/paper-6256/embeddings
export TMPDIR=/autosota_cache/tmp
mkdir -p /autosota_cache/checkpoints_fewshot

SEEDS=(42 137 823 5619 9871)
RESULTS_FILE=/repo/fewshot_results.txt
echo "Few-shot CIFAR-100 Results" > $RESULTS_FILE
echo "=========================" >> $RESULTS_FILE

for seed in "${SEEDS[@]}"; do
    echo "=== Seed $seed ===" | tee -a $RESULTS_FILE
    cd /repo
    python3 classification_and_retrieval.py \
      --image_models "timm/beit_base_patch16_224.in22k_ft_in22k" \
      --text_models "clip_vitb32" \
      --task classification \
      --mode MLP \
      --datasets cifar100 \
      --dataset_img_repr cifar100 \
      --few_shot_samples 4 \
      --sequential_training \
      --epochs 200 \
      --seed $seed \
      --force \
      2>&1 | grep -E "(Top-1 Accuracy|Top-5 Accuracy|seed|Training completed|Stage)" | tee -a $RESULTS_FILE
    echo "---" >> $RESULTS_FILE
done

echo "=== DONE ==="
cat $RESULTS_FILE
