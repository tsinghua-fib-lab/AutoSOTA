#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u mmlu/evaluate_mmlu.py --path [model_path] --data_dir data/MMLU --ntrain 0 2>&1