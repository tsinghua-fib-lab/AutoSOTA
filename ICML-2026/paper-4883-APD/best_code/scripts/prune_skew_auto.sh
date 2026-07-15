#!/bin/bash
# layerwise skewness schedule
python -u prune/prune_skew_layer.py \
    --sparsity_ratio 0.5 \
    --model [model_path] \
    --cache [cache_path] \
    --dataset wikitext2 \
    --path [dataset_path] > logs/prune_skew_layer.log 2>&1

# blockwise skewness schedule
python -u prune/prune_skew_blk.py \
    --sparsity_ratio 0.5 \
    --model [model_path] \
    --cache [cache_path] \
    --dataset wikitext2 \
    --path [dataset_path] 2>&1

# blockwise skewness schedule based on metric
python -u prune/prune_skew_blk_metric.py \
    --sparsity_ratio 0.5 \
    --model [model_path] \
    --cache [cache_path] \
    --dataset wikitext2 \
    --path [dataset_path] 2>&1