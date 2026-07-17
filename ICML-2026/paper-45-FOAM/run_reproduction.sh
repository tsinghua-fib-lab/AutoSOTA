#!/bin/bash
# FOAM-2 LLaMA-60M C4 Pretraining Reproduction
# Paper: FOAM: Blocked State Folding for Memory-Efficient LLM Training
# Reproduces: Perplexity 28.95 (paper: 28.53)

export no_proxy="localhost,127.0.0.1,::1,172.17.0.1,host.docker.internal,.hf.co,.huggingface.co,hf-mirror.com,huggingface.co,xethub.hf.co"
export NO_PROXY="$no_proxy"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export HF_HUB_CACHE=/autosota_cache/hf/hub
export HF_DATASETS_CACHE=/autosota_cache/hf/datasets
export WANDB_MODE=offline
export HF_HUB_DOWNLOAD_TIMEOUT=300
mkdir -p /autosota_cache/hf/{hub,datasets}

cd /repo
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 1e-2 \
    --scale 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_ratio 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100000 \
    --level 2 \
    --seed 42 \
    --beta1 0.9 \
    --beta2 0.95 \
    --optimizer foam \
    --save_dir foam2_llama60m_repro
