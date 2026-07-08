#!/bin/bash
# Quick MC-only evaluation (skips activation extraction and LLM judge)
# Use when only prototype computation changes
set -e
cd /repo
export HF_HOME=/autosota_cache/hf
export HF_ENDPOINT=https://huggingface.co

MODEL="Qwen2.5-7B-Instruct"
MODEL_DIR="/models/Qwen2.5-7B-Instruct"
LAYER=19
KAPPA=20.0
ALPHA=0.6
BETA=0.4

echo "=== Quick MC Eval: Recompute Prototypes + MC Only ==="
python get_prototypes.py --feature_file ./features/${MODEL}_layer${LAYER}.npz

for FOLD in 0 1; do
    python evaluate_mc.py $MODEL \
        --prototype_path ./prototypes/${MODEL}_layer${LAYER}_fold${FOLD}.npz \
        --layer $LAYER --kappa $KAPPA --alpha $ALPHA --beta $BETA \
        --model_dir $MODEL_DIR \
        --output_path results/${MODEL}_l${LAYER}_a${ALPHA}_b${BETA}_fold${FOLD}.json
done

echo "=== Quick MC Eval Complete ==="
