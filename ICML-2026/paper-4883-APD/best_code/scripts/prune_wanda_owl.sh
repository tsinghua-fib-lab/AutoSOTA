#!/bin/bash
# blockwise
python -u prune/prune_wanda_owl_blk.py \
    --sparsity_ratio 0.7 \
    --model [model_path] \
    --cache [cache_dir] \
    --dataset wikitext2 \
    --path [dataset_path] > logs/prune_wanda_owl.log 2>&1

# layerwise
python -u prune/prune_wanda_owl_layer.py \
    --sparsity_ratio 0.7 \
    --model [model_path] \
    --cache [cache_dir] \
    --dataset wikitext2 \
    --path [dataset_path] > logs/prune_wanda_owl.log 2>&1
