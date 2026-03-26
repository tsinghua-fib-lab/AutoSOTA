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
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'96 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 56 --stride 56 \
  --num_blocks 1 \
  --channel 321 --embedding_dim 600 --num_heads 3 --conv1d_kernel_size 8 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_electricity_$seq_len'_'96.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'192 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005 \
  --train_epochs 10 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 321 --embedding_dim 600 --num_heads 3 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_electricity_$seq_len'_'192.log  

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'336 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 321 --embedding_dim 600 --num_heads 3 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_electricity_$seq_len'_'336.log  

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'720 \
  --model P_sLSTM \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patch_size 16 --stride 16 \
  --num_blocks 1 \
  --channel 321 --embedding_dim 600 --num_heads 3 --conv1d_kernel_size 32 --group_norm_weight True \
  --dropout 0.1 >logs/LongForecasting/P_sLSTM_electricity_$seq_len'_'720.log  
