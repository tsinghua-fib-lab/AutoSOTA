#!/bin/bash
cd /repo
results="/repo/final_results.txt"
> "$results"

for pred_len in 96 192 336; do
    echo "=== pred_len=${pred_len} ===" | tee -a "$results"
    CUDA_VISIBLE_DEVICES=0 python3 -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_${pred_len} \
        --model SiGMA \
        --data custom \
        --features M \
        --target MT_001 \
        --seq_len 96 \
        --label_len 0 \
        --pred_len ${pred_len} \
        --e_layers 1 \
        --enc_in 321 \
        --c_out 321 \
        --d_model 16 \
        --d_ff 32 \
        --learning_rate 0.01 \
        --scale_independence 0 \
        --feature_transformation 1 \
        --seed 42 \
        --train_epochs 10 \
        --batch_size 32 \
        --gpu 0 2>&1 | grep -E "(mse:|Epoch:.*Steps:|Early stopping|test shape)" | tee -a "$results"
    echo "" >> "$results"
done

# Run 720 with lower learning rate since it was unstable
echo "=== pred_len=720 ===" | tee -a "$results"
CUDA_VISIBLE_DEVICES=0 python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_720 \
    --model SiGMA \
    --data custom \
    --features M \
    --target MT_001 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --e_layers 1 \
    --enc_in 321 \
    --c_out 321 \
    --d_model 16 \
    --d_ff 32 \
    --learning_rate 0.0005 \
    --scale_independence 0 \
    --feature_transformation 1 \
    --seed 42 \
    --train_epochs 10 \
    --batch_size 32 \
    --gpu 0 2>&1 | grep -E "(mse:|Epoch:.*Steps:|Early stopping|test shape)" | tee -a "$results"

echo ""
echo "=== ALL RESULTS ==="
cat "$results"
