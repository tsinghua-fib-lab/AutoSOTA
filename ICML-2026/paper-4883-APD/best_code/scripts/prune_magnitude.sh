#!/bin/bash
python -u prune/prune_magnitude.py \
    --sparsity_ratio 0.5 \
    --model [model_path] \
    --cache [cache_path] \
    --dataset wikitext2 \
    --path [dataset_path] > logs/prune_magnitude.log 2>&1
