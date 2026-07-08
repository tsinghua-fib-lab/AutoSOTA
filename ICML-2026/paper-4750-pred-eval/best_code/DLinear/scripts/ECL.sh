#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=DLinear
use_ps_loss=$1  # 0: Use MSE loss only, 1: Use PS loss
patch_len_threshold=24
lengths=(96 192 336 720)

if [ ! -d "./logs/MSE/" ]; then
    mkdir -p ./logs/MSE/
fi
if [ ! -d "./logs/PS/" ]; then
    mkdir -p ./logs/PS/
fi

if [ "$use_ps_loss" -eq 0 ]; then
    ps_lambdas=(0.0)
    loss_name=MSE
else
    ps_lambdas=(0.1 0.3 0.5 0.7 1.0)
    loss_name=PS
fi


for ps_lambda in ${ps_lambdas[@]}; do

    for len in ${lengths[@]}; do
        echo "Running experiment with seq_len=$len, pred_len=$len, ps_lambda=$ps_lambda"

        python -u run_longExp.py \
          --is_training 1 \
          --root_path ../datasets/ \
          --data_path electricity.csv \
          --model_id Electricity_${len}_${len} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len $len \
          --pred_len $len \
          --enc_in 321 \
          --des 'Exp' \
          --learning_rate 0.001 \
          --batch_size 16 \
          --use_ps_loss $use_ps_loss \
          --ps_lambda $ps_lambda \
          --patch_len_threshold $patch_len_threshold \
          --itr 1 >logs/${loss_name}/ECL_${len}_${len}_${ps_lambda}lambda.log
    done
done

echo "All experiments finished."