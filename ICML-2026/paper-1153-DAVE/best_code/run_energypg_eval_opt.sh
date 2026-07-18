#!/bin/bash
# EnergyPG evaluation with optimized post-processing config
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /repo
PYTHONPATH=. python3 evaluation/energypg.py   --data-dir /datasets/imagenet_val   --bbox-path /datasets/imagenet_bbox_mapped.pkl   --model-cfg-path models_configs/deit3_b16_224_opt.yaml   --num-images 500   --num-steps 50   --seed 42   --device cuda:0   --out-dir evaluation/energypg_results   --log-every 100
