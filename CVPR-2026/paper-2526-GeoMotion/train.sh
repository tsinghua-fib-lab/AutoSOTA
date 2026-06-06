#!/bin/bash
# set -euo pipefail

CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --master-port=29501 --nproc_per_node=4 train.py configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml
