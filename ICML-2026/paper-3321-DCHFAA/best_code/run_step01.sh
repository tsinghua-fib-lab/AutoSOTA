#!/bin/bash
cd /repo
export PYTHONPATH="/repo/transformers-4.32.0/src:$PYTHONPATH"
export HF_HOME=/autosota_cache/hf

python3 step01_extract_attns_fourier.py \
    --data-type qa \
    --data-path dataset/ragtruth/llama-2-7b-chat/anno-QA-7b.jsonl \
    --model-name /models/Llama-2-7b-chat-hf \
    --device cuda \
    --num-gpus 2 \
    --output-path outputs/attn-features-qa-7b-fourier.pt \
    --f_cutoff 0.45 \
    --max-memory 75
