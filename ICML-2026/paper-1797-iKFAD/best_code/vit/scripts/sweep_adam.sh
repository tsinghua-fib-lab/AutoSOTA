#!/bin/bash

# Ensure the logs directory exists
mkdir -p logs

# Set the specific GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Run the sweep command in the background
# Note: Since we restricted visibility to GPU 6, we pass '--device cuda' 
# (which maps to the first visible GPU, i.e., GPU 6)
nohup python3 sweep.py \
    --optimizer adam \
    --n_trials 100 \
    --epochs 25 \
    --device cuda \
    > logs/adam.log 2>&1 &

# Print the Process ID so you can track it later
echo "ADAM Sweep started on GPU 7. Logging to logs/adam.log"
echo "PID: $!"