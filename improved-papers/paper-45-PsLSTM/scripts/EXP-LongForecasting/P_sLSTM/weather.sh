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
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --label_len 18 \
  --des 'Exp' \
  --itr 1 --batch_size 16 \
  --patch_size 56 --stride 56 \
  --num_blocks 2 \
  --channel 21 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 5 >logs/LongForecasting/P_sLSTM_Weather_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'192 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --des 'Exp' \
  --itr 1 --batch_size 16 \
  --patch_size 56 --stride 56 \
  --num_blocks 2 \
  --channel 21 --embedding_dim 100 --num_heads 4 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 5 >logs/LongForecasting/P_sLSTM_Weather_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'336 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.00006 \
  --patch_size 56 --stride 56 \
  --num_blocks 1 \
  --channel 21 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 3 >logs/LongForecasting/P_sLSTM_Weather_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'720 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0001 \
  --patch_size 56 --stride 56 \
  --num_blocks 2 \
  --channel 21 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 --patience 3 >logs/LongForecasting/P_sLSTM_Weather_$seq_len'_'720.log