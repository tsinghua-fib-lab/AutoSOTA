if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=TimeMixer

root_path_name=./dataset/weather/
data_path_name=weather.csv
data_name=weather

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
      --enc_in 21 \
      --c_out 21 \
      --e_layers 3 \
      --d_model 16 \
      --d_ff 32 \
      --dropout 0.1 \
      --des 'Exp' \
      --train_epochs 20 \
      --patience 10 \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --gpu 6 \
      --itr 1 --batch_size 128 --learning_rate 0.01 >logs/LongForecasting/$model_name/'Weather_'$seq_len'_'$pred_len.log
done