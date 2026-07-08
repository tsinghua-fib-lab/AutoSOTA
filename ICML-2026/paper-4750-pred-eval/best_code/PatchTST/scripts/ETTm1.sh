#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
random_seed=2025
root_path_name=../datasets/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

use_ps_loss=0
ps_lambda=0.0
patch_len_threshold=24

lengths=(96 192 336 720)

mkdir -p ./logs/MSE/

for L in "${lengths[@]}"; do
  seq_len="$L"
  pred_len="$L"
  model_id="${model_id_name}_${seq_len}_${pred_len}"
  log_file="./logs/MSE/${model_id}_${ps_lambda}lambda.log"

  echo "[`date '+%F %T'`] RUN ${model_id} (seq_len=${seq_len}, pred_len=${pred_len})" | tee "$log_file"

  python -u run_longExp.py \
    --random_seed "$random_seed" \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "$model_id" \
    --model "$model_name" \
    --data "$data_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des Exp \
    --train_epochs 100 \
    --patience 20 \
    --lradj TST \
    --pct_start 0.4 \
    --revin 1 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --use_ps_loss "$use_ps_loss" \
    --ps_lambda "$ps_lambda" \
    --patch_len_threshold "$patch_len_threshold" \
    --itr 1 2>&1 | tee -a "$log_file"

  echo "[`date '+%F %T'`] DONE ${model_id}" | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done
