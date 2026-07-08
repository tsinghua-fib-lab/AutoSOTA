#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

model_name=DLinear
use_ps_loss=$1  # 0: Use MSE loss only, 1: Use PS loss
patch_len_threshold=24
# --- Define the array of lengths to test ---
lengths=(96 192 336 720)

# --- Create log directories ---
if [ ! -d "./logs/MSE/" ]; then
    mkdir -p ./logs/MSE/
fi
if [ ! -d "./logs/PS/" ]; then
    mkdir -p ./logs/PS/
fi

# --- Set loss type and lambda values based on input ---
if [ "$use_ps_loss" -eq 0 ]; then
    ps_lambdas=(0.0)
    loss_name=MSE
else
    # Preserved the new lambda values from your script
    ps_lambdas=(1.0 3.0 5.0 10.0)
    loss_name=PS
fi

# --- Main experiment loop ---
# Outer loop for lambda values
for ps_lambda in ${ps_lambdas[@]}; do
    # Inner loop for different lengths
    for len in ${lengths[@]}; do
        echo "Running ETTh1 experiment with seq_len=$len, pred_len=$len, ps_lambda=$ps_lambda"

        python -u run_longExp.py \
          --is_training 1 \
          --root_path ../datasets/\
          --data_path ETTh1.csv \
          --model_id ETTh1_${len}_${len} \
          --model $model_name \
          --data ETTh1 \
          --features M \
          --seq_len $len \
          --pred_len $len \
          --enc_in 7 \
          --des 'Exp' \
          --learning_rate 0.005 \
          --batch_size 32 \
          --use_ps_loss $use_ps_loss \
          --ps_lambda $ps_lambda \
          --patch_len_threshold $patch_len_threshold \
          --itr 1 >logs/${loss_name}/ETTh1_${len}_${len}_${ps_lambda}lambda.log
    done
done

echo "All ETTh1 experiments finished."