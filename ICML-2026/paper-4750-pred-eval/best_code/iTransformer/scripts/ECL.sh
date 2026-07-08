#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
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
    ps_lambdas=(1.0 3.0 5.0 10.0)
    loss_name=PS
fi


for ps_lambda in ${ps_lambdas[@]}; do
    for len in ${lengths[@]}; do
        echo "Running iTransformer on ECL with seq_len=$len, pred_len=$len, ps_lambda=$ps_lambda"

        python -u run.py \
          --is_training 1 \
          --root_path ../datasets/ \
          --data_path electricity.csv \
          --model_id ECL_${len}_${len} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len $len \
          --pred_len $len \
          --e_layers 3 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --d_model 512 \
          --d_ff 512 \
          --batch_size 16 \
          --learning_rate 0.0005 \
          --lradj type1 \
          --patience 3 \
          --train_epochs 10 \
          --use_ps_loss $use_ps_loss \
          --ps_lambda $ps_lambda \
          --patch_len_threshold $patch_len_threshold \
          --itr 1 >logs/${loss_name}/ECL_${len}_${len}_${ps_lambda}lambda.log
    done
done

echo "All iTransformer experiments finished."