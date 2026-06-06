if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchTST

root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
data_name=ETTh1

for pred_len in 96
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.1 \
      --des 'Exp' \
      --train_epochs 100 \
      --itr 1 --batch_size 32 --learning_rate 0.0001 #>logs/LongForecasting/$model_name/'ETTh1_'$seq_len'_'$pred_len.log
done