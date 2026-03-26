if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=SparseTSF

root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
data_name=solar

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
      --period_len 4 \
      --enc_in 137 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.02 >logs/LongForecasting/$model_name/'Solar_'$seq_len'_'$pred_len.log
done