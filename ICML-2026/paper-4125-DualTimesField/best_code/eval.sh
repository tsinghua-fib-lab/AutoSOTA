#!/bin/bash
# DualTimesField ETTh1 reproduction evaluation
# Trains the model with paper settings and evaluates on test set
cd /repo
python -m reconstruction.train --datasets ETTh1 --seq_length 336 --batch_size 32 --epochs 300 --lr 1e-3 --device cuda 2>&1 | tee /tmp/eval_output.txt
echo "---METRICS_START---"
grep "MSE:" /tmp/eval_output.txt | tail -1
grep "MAE:" /tmp/eval_output.txt | tail -1
echo "---METRICS_END---"
