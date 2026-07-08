#!/bin/bash

# Ensure the logs directory exists
mkdir -p logs

# Set the specific GPU visibility
export CUDA_VISIBLE_DEVICES=6

# Run the sweep command in the background
# Note: Since we restricted visibility to GPU 6, we pass '--device cuda' 
# (which maps to the first visible GPU, i.e., GPU 6)
nohup python sweep.py \
    --optimizer cd \
    --n_trials 80 \
    --epochs 2 \
    --device cuda \
    > logs/cd.log 2>&1 &

# Print the Process ID so you can track it later
echo "CD Sweep started on GPU 6. Logging to logs/cd.log"
echo "PID: $!"