#!/bin/bash

# COCO + llama_llama
CUDA_VISIBLE_DEVICES=2,7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/COCO_llama_llama_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/COCO_llama-vision_v1.json \
    --output outputs/vlm_tagging/COCO_llama_llama_neighbor_scores.json \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset coco \
    --split val \
    --ref-type predicted \
    --tensor-parallel-size 2 \
    --batch-size 1 \
    --gpu-memory-utilization 0.85

# COCO + llama_metaclip
CUDA_VISIBLE_DEVICES=2,7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/COCO_llama_metaclip_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/COCO_llama-vision_v1.json \
    --output outputs/vlm_tagging/COCO_llama_metaclip_neighbor_scores.json \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset coco \
    --split val \
    --ref-type predicted \
    --tensor-parallel-size 2 \
    --batch-size 1 \
    --gpu-memory-utilization 0.85

# Flickr30k + llama_llama
CUDA_VISIBLE_DEVICES=2,7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/Flickr30k_llama_llama_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/Flickr30k_llama-vision_v1.json \
    --output outputs/vlm_tagging/Flickr30k_llama_llama_neighbor_scores.json \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset flickr30k \
    --split test \
    --ref-type predicted \
    --tensor-parallel-size 2 \
    --batch-size 1 \
    --gpu-memory-utilization 0.85

# Flickr30k + llama_metaclip
CUDA_VISIBLE_DEVICES=2,7 python3 tools/neighbor_based_vlm_evaluator_vllm.py \
    --neighbors outputs/vlm_tagging/Flickr30k_llama_metaclip_image_neighbors.jsonl \
    --vlm-data outputs/vlm_tagging/Flickr30k_llama-vision_v1.json \
    --output outputs/vlm_tagging/Flickr30k_llama_metaclip_neighbor_scores.json \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset flickr30k \
    --split test \
    --ref-type predicted \
    --tensor-parallel-size 2 \
    --batch-size 1 \
    --gpu-memory-utilization 0.85

echo "All jobs completed!"

