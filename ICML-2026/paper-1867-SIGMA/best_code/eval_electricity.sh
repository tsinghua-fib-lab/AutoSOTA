#!/bin/bash
# SiGMA evaluation script for Electricity — Iter 5: Huber loss + gradient clipping
# d_model=32, d_ff=128, e_layers=1, CI=1

cd /repo

MSE_SUM=0
MAE_SUM=0
COUNT=0

for pred_len in 96 192 336 720; do
    echo "=== Evaluating pred_len=${pred_len} ==="
    output=$(CUDA_VISIBLE_DEVICES=0 python3 -u run.py \
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
        --d_model 32 \
        --d_ff 128 \
        --learning_rate 0.01 \
        --scale_independence 0 \
        --feature_transformation 1 \
        --channel_independence 1 \
        --seed 42 \
        --train_epochs 10 \
        --batch_size 32 \
        --gpu 0 2>&1)

    mse=$(echo "$output" | grep -oP 'mse:\K[0-9.]+')
    mae=$(echo "$output" | grep -oP 'mae:\K[0-9.]+')
    echo "  MSE=${mse}, MAE=${mae}"

    MSE_SUM=$(python3 -c "print(${MSE_SUM} + ${mse})")
    MAE_SUM=$(python3 -c "print(${MAE_SUM} + ${mae})")
    COUNT=$((COUNT + 1))
done

AVG_MSE=$(python3 -c "print(${MSE_SUM} / ${COUNT})")
AVG_MAE=$(python3 -c "print(${MAE_SUM} / ${COUNT})")
echo ""
echo "=== AVERAGED RESULTS (across 4 horizons) ==="
echo "MSE: ${AVG_MSE}"
echo "MAE: ${AVG_MAE}"
