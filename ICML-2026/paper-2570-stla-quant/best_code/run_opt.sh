#!/bin/bash

gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id

python main.py --seed 0 \
 --model_path facebook/opt-125m  \
 --calib_data c4 \
 --w_bits 3 \
 --groupsize 256 \
 --blocksize 256 \
 --clustersize 256 \
 --comp_method GPTQ \
 --loss_option global \
 --order_option spin \
 --block_v \
#  --learn_rounding \
#  --num_iters 200 \
#  --lr 0.8 \



 