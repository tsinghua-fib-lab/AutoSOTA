#!/bin/bash
set -e

# Environment
export HF_HOME=/autosota_cache/hf
unset HF_ENDPOINT
export CUDA_VISIBLE_DEVICES=0,1

cd /repo

# Create global dir for hessian/scale cache
mkdir -p /autosota_cache/srr_global

# Model path (local)
MODEL_PATH="/models/TinyLlama-1.1B"

# Run 3 seeds as specified in the rubric
for SEED in 42 1234 4321; do
    echo "============================================"
    echo "Running SRR with seed $SEED"
    echo "============================================"
    
    python -u ptq_pipeline.py experiments/configs/srr_3bit_rank32_repro.yaml \
        --model-name "$MODEL_PATH" \
        --perplexity-eval-batch-size 1 \
        --max-position-embeddings 2048 \
        --perplexity-max-seq-length 2048 \
        --lr-scaling-mode "cholesky" \
        --num-calibration-samples 256 \
        --srr-seed "$SEED" \
        --disable-lm-eval \
        -ow
    
    echo "Seed $SEED done."
done

echo "All seeds done!"
