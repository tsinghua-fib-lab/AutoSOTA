# add --individual for P-sLSTM
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'96 \
  --model P_sLSTM \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 137 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_Solar_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'192 \
  --model P_sLSTM \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 137 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_Solar_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'336 \
  --model P_sLSTM \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 137 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_Solar_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'720 \
  --model P_sLSTM \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 137 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 4 --group_norm_weight True \
  --dropout 0 >logs/LongForecasting/P_sLSTM_Solar_$seq_len'_'720.log
