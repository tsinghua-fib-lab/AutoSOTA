if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=iTransformer

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
data_name=electricity

for pred_len in 96 192 336 720
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
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.1 \
      --des 'Exp' \
      --train_epochs 10 \
      --gpu 1 \
      --itr 1 --batch_size 16 --learning_rate 0.0005 >logs/LongForecasting/$model_name/'Electricity_'$seq_len'_'$pred_len.log
done
