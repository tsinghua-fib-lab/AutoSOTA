#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10

root_path=../datasets/
data_path=electricity.csv
enc_in=321
dec_in=321
c_out=321

use_ps_loss=0
ps_lambda=0.0       

lengths=(336 720)

mkdir -p ./logs/MSE/

for L in "${lengths[@]}"; do
  seq_len="$L"
  pred_len="$L"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id ECL_${seq_len}_${pred_len} \
    --model "$model_name" \
    --data custom \
    --features M \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --d_layers 1 \
    --factor 3 \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des Exp \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --down_sampling_layers "$down_sampling_layers" \
    --down_sampling_method avg \
    --down_sampling_window "$down_sampling_window" \
    --use_ps_loss "$use_ps_loss" \
    --ps_lambda "$ps_lambda" \
    --itr 1 >"./logs/MSE/ECL_${seq_len}_${pred_len}.log"
done
