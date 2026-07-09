#!/bin/bash
cd /repo/llm
export CUDA_VISIBLE_DEVICES=0
python3 scripts/inference_with_recompute_kv.py configs/2wikimqa_repro.yaml
