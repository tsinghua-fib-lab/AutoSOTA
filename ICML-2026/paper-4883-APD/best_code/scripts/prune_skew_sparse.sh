#!/bin/bash
python -u prune/prune_skew_sparse_layer.py \
    --sparsity_ratio 0.7 \
    --model [model_path] \
    --cache [cache_dir] \
    --dataset wikitext2 \
    --path [dataset_path] > logs/prune_skew_sparse.log 2>&1
