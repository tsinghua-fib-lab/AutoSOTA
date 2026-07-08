#!/bin/bash
# Spherical Steering Evaluation Pipeline
# Reproduces TruthfulQA metrics for Qwen2.5-7B-Instruct
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
TRUTH_JUDGE="/models/truthfulqa-truth-judge-llama2-7B"
INFO_JUDGE="/models/truthfulqa-info-judge-llama2-7B"

echo "=== Step 1: Extract Hidden States ==="
python get_activations.py $MODEL --layer $LAYER --model_dir $MODEL_DIR

echo "=== Step 2: Compute Prototypes ==="
python get_prototypes.py --feature_file ./features/${MODEL}_layer${LAYER}.npz

echo "=== Step 3: MC Evaluation ==="
for FOLD in 0 1; do
    python evaluate_mc.py $MODEL \
        --prototype_path ./prototypes/${MODEL}_layer${LAYER}_fold${FOLD}.npz \
        --layer $LAYER --kappa $KAPPA --alpha $ALPHA --beta $BETA \
        --model_dir $MODEL_DIR \
        --output_path results/${MODEL}_l${LAYER}_a${ALPHA}_b${BETA}_fold${FOLD}.json
done

echo "=== Step 4: LLM Judge Evaluation ==="
for FOLD in 0 1; do
    python evaluate_llm_judge.py $MODEL \
        --prototype_path ./prototypes/${MODEL}_layer${LAYER}_fold${FOLD}.npz \
        --layer $LAYER --kappa $KAPPA --alpha $ALPHA --beta $BETA \
        --model_dir $MODEL_DIR \
        --truth_judge_path $TRUTH_JUDGE \
        --info_judge_path $INFO_JUDGE \
        --preset null
done

echo "=== Evaluation Complete ==="
