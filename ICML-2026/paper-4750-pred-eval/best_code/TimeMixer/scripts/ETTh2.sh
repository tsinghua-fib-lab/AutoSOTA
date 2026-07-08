#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=5

model_name=TimeMixer
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16

use_ps_loss=0
ps_lambda=0.0
patch_len_threshold=24

root_path="../datasets/"
data_path="ETTh2.csv"
enc_in=7
c_out=7

lengths=(96 192 336 720)

mkdir -p ./logs/MSE/

for L in "${lengths[@]}"; do
  seq_len="${L}"
  pred_len="${L}"
  model_id="ETTh2_${seq_len}_${pred_len}"
  log_file="./logs/MSE/${model_id}.log"

  echo "[`date '+%F %T'`] RUN ${model_id}  (seq_len=${seq_len}, pred_len=${pred_len})" | tee "$log_file"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model "$model_name" \
    --data ETTh2 \
    --features M \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --c_out "$c_out" \
    --des Exp \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --learning_rate "$learning_rate" \
    --batch_size "$batch_size" \
    --down_sampling_layers "$down_sampling_layers" \
    --down_sampling_method avg \
    --down_sampling_window "$down_sampling_window" \
    --use_ps_loss "$use_ps_loss" \
    --ps_lambda "$ps_lambda" \
    --patch_len_threshold "$patch_len_threshold" \
    --itr 1 2>&1 | tee -a "$log_file"

  echo "[`date '+%F %T'`] DONE ${model_id}" | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done
