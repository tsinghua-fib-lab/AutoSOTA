#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py config/train_gpt2_xs.py > logs/train.log 2>&1 &
