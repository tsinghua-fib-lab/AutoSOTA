#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

model_name=iTransformer
use_ps_loss=$1  # 0: Use MSE loss only 1: Use PS loss
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
    ps_lambdas=(1.0 3.0 5.0 10.0)
    loss_name=PS
fi


for ps_lambda in ${ps_lambdas[@]}; do
    for len in ${lengths[@]}; do
        echo "Running iTransformer on ETTh2 with seq_len=$len, pred_len=$len, ps_lambda=$ps_lambda"
        
        python -u run.py \
          --is_training 1 \
          --root_path ../datasets/ \
          --data_path ETTh2.csv \
          --model_id ETTh2_${len}_${len} \
          --model $model_name \
          --data ETTh2 \
          --features M \
          --seq_len $len \
          --pred_len $len \
          --e_layers 2 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 128 \
          --d_ff 128 \
          --learning_rate 0.0001 \
          --train_epochs 10 \
          --patience 3 \
          --lradj type1 \
          --use_ps_loss $use_ps_loss \
          --ps_lambda $ps_lambda \
          --patch_len_threshold $patch_len_threshold \
          --itr 1 >logs/${loss_name}/ETTh2_${len}_${len}_${ps_lambda}lambda.log
    done
done

echo "All iTransformer ETTh2 experiments finished."