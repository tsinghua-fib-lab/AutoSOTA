#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

# 仅用 MSE（固定）
use_ps_loss=0
ps_lambda=0.0
patch_len_threshold=24

# 数据路径（统一）
root_path="../datasets/"
data_path="weather.csv"
dataset_name="custom"

# 通道/特征
enc_in=21
dec_in=21
c_out=21

# 模型超参
e_layers=2
d_layers=1
factor=3
d_model=32
d_ff=32
top_k=5
label_len=48

# 评估专用超参
alpha_boundary=1.0
welch_win_frac=0.25
welch_overlap=0.5
workers=8

# 历史=预测 的长度集合
lengths=(96 192 336 720)

# 输出与日志目录
outdir="./predictability_results"
logdir="./logs/PREDICT_TimesNet_Weather"
mkdir -p "${logdir}" "${outdir}"

# 时间戳避免覆盖
ts=$(date +"%Y%m%d_%H%M%S")

for L in "${lengths[@]}"; do
  seq_len="$L"
  pred_len="$L"   # 历史 = 预测
  model_id="weather_${seq_len}_${pred_len}"
  log_file="${logdir}/${model_id}_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}_${ts}.log"

  echo "[`date '+%F %T'`] RUN ${model_id} (seq_len=${seq_len}, pred_len=${pred_len}, label_len=${label_len})" | tee "$log_file"

  python -u run_predictability_eval.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model "$model_name" \
    --data "$dataset_name" \
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
    --checkpoints "./checkpoints/" \
    --outdir "${outdir}" \
    --alpha_boundary ${alpha_boundary} \
    --welch_win_frac ${welch_win_frac} \
    --welch_overlap ${welch_overlap} \
    --workers ${workers} \
    --itr 1 \
    2>&1 | tee -a "$log_file"

  echo "[`date '+%F %T'`] DONE ${model_id}" | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done
