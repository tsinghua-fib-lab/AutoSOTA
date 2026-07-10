#!/bin/bash
# MONICA Evaluation Script for paper-1755
# Evaluates Qwen3-1.7B on AIME 2024 with metadata_leakage cue type
# Metrics: RR, PR, MR, SR

set -e

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

cd /repo

echo "=== MONICA Evaluation: Qwen3-1.7B + AIME 2024 + Metadata Leakage ==="
echo ""

# Run MONICA evaluation
python3 run_qwen3_1b_aime.py \
    --datasets aime_2024_multichoice \
    --cue_types metadata \
    --debug_topk 15 \
    --max_tokens 4096 \
    --temperature 0.5 \
    --repetition_penalty 1.1 \
    --steer_layer_weights 0.7 0.85 1.0 1.5 \
    --file_tag eval_run

echo ""
echo "=== Computing Metrics ==="
python3 compute_metrics.py /repo/outputs/eval_run/aime_2024_multichoice_metadata/

echo ""
echo "=== Evaluation Complete ==="
