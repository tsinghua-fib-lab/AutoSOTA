#!/bin/bash
# LLM Judge eval with configurable alpha and beta
set -e
cd /repo
export HF_HOME=/autosota_cache/hf
export HF_ENDPOINT=https://huggingface.co
ALPHA=${1:-0.65}
BETA=${2:-0.4}
OUTDIR=results_a${ALPHA}_b${BETA}
mkdir -p $OUTDIR

for FOLD in 0 1; do
    echo "=== LLM Judge Fold $FOLD alpha=$ALPHA beta=$BETA ==="
    timeout 3600 python evaluate_llm_judge.py Qwen2.5-7B-Instruct \
        --prototype_path ./prototypes/Qwen2.5-7B-Instruct_layer19_fold${FOLD}.npz \
        --layer 19 --kappa 20.0 --alpha $ALPHA --beta $BETA \
        --model_dir /models/Qwen2.5-7B-Instruct \
        --truth_judge_path /models/truthfulqa-truth-judge-llama2-7B \
        --info_judge_path /models/truthfulqa-info-judge-llama2-7B \
        --preset null --output_dir $OUTDIR 2>&1 | tail -3
done
echo "LLM DONE $ALPHA $BETA" > /repo/a${ALPHA}_b${BETA}_done.txt
