#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python 4_evaluation.py \
    --watermark_algorithm AliMark \
    --watermark_model facebook/opt-1.3b \
    --watermark_embedder all-mpnet-base-v2 \
    --watermark_embedding_dim 768 \
    --watermark_block_size 8 \
    --watermark_num_next_sentence_candidates 64 \
    --min_new_sentences 12 \
    --dataset_name booksum \
