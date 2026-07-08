#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0

# -------- 固定配置 --------
model_name=TimesNet

# 仅用 MSE（固定）
use_ps_loss=0
ps_lambda=0.0
patch_len_threshold=24

root_path="../datasets/"
data_path="electricity.csv"
enc_in=321
dec_in=321
c_out=321

# 历史=预测 的长度集合
lengths=(96 192 336 720)

# 通用模型超参
e_layers=2
d_layers=1
factor=3
d_model=256
d_ff=512
top_k=5

mkdir -p ./logs/MSE/

for L in "${lengths[@]}"; do
  seq_len="$L"
  pred_len="$L"         # 历史=预测
  label_len=48           # 若需保持 48，则把本行改成 label_len=48
  model_id="ECL_${seq_len}_${pred_len}"
  log_file="./logs/MSE/${model_id}_${ps_lambda}lambda.log"

  echo "[`date '+%F %T'`] RUN ${model_id} (seq_len=${seq_len}, pred_len=${pred_len}, label_len=${label_len})" | tee "$log_file"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model "$model_name" \
    --data custom \
    --features M \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --d_layers "$d_layers" \
    --factor "$factor" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --top_k "$top_k" \
    --des Exp \
    --use_ps_loss "$use_ps_loss" \
    --ps_lambda "$ps_lambda" \
    --patch_len_threshold "$patch_len_threshold" \
    --itr 1 2>&1 | tee -a "$log_file"

  echo "[`date '+%F %T'`] DONE ${model_id}" | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done
