#!/bin/bash
export CUDA_HOME=/opt/conda/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
cd /repo
python3 reproduce.py --model /models/RT-Qwen2.5-0.5B "$@"
