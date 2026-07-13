#!/bin/bash
# Full reproduction script for DMANet on ETTh1
# Paper settings from Table 8: e_layers=1, down_sampling_layers=2, d_model=512,
# lr=2e-2, loss=frequency_domain_MAE, batch_size=8, epochs=15
# kernel_size=3, stride=2, channel_change_ratio=0.5, patience=3

set -e

cd /repo

# Clean previous state
rm -rf checkpoints results test_results
mkdir -p output

RESULTS_FILE="/repo/output/etth1_full_results.txt"
echo "DMANet ETTh1 Full Reproduction Results" > "$RESULTS_FILE"
echo "Started at: $(date)" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"

MSE_SUM=0
MAE_SUM=0
COUNT=0

for pred_len in 96 192 336 720; do
    echo "" >> "$RESULTS_FILE"
    echo "=== pred_len=$pred_len ===" >> "$RESULTS_FILE"
    echo "Starting pred_len=$pred_len at $(date)"

    CUDA_VISIBLE_DEVICES=0,1 python3 -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id etth1_96_${pred_len} \
      --model DMANet \
      --data ETTh1 \
      --root_path ./all_datasets/ETT-small/ \
      --data_path ETTh1.csv \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --c_out 7 \
      --d_model 512 \
      --e_layers 1 \
      --down_sampling_layers 2 \
      --down_sampling_window 2 \
      --down_sampling_c 0.5 \
      --kernel_size 3 \
      --d_ff 2 \
      --learning_rate 0.02 \
      --batch_size 8 \
      --train_epochs 15 \
      --patience 3 \
      --auxi_lambda 1 \
      --auxi_loss MAE \
      --auxi_mode rfft \
      --auxi_type complex \
      --lradj type1 \
      --num_workers 0 \
      --itr 1 \
      --des full \
      --seed 2024 \
      --gpu 0 \
      --log_path ./output/etth1_${pred_len}.txt 2>&1 | tee -a ./output/run_${pred_len}.log

    # Extract final test metrics from log
    MSE_VAL=$(grep "mse:" "./output/run_${pred_len}.log" | tail -1 | grep -oP 'mse:\K[0-9.]+' || echo "N/A")
    MAE_VAL=$(grep "mae:" "./output/run_${pred_len}.log" | tail -1 | grep -oP 'mae:\K[0-9.]+' || echo "N/A")

    echo "pred_len=$pred_len: mse=$MSE_VAL, mae=$MAE_VAL" >> "$RESULTS_FILE"

    if [ "$MSE_VAL" != "N/A" ]; then
        MSE_SUM=$(python3 -c "print($MSE_SUM + $MSE_VAL)")
        MAE_SUM=$(python3 -c "print($MAE_SUM + $MAE_VAL)")
        COUNT=$((COUNT + 1))
    fi

    echo "Finished pred_len=$pred_len at $(date)"
done

echo "" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
if [ $COUNT -eq 4 ]; then
    AVG_MSE=$(python3 -c "print(round($MSE_SUM / 4, 4))")
    AVG_MAE=$(python3 -c "print(round($MAE_SUM / 4, 4))")
    echo "Average MSE: $AVG_MSE" >> "$RESULTS_FILE"
    echo "Average MAE: $AVG_MAE" >> "$RESULTS_FILE"
    echo "Paper MSE: 0.428, Paper MAE: 0.429" >> "$RESULTS_FILE"
fi
echo "Finished at: $(date)" >> "$RESULTS_FILE"
