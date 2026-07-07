export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting04" ]; then
    mkdir ./logs/LongForecasting04
fi
seq_len=36
model_name=ReNF_beta

root_path_name=../dataset/cars/
data_path_name=(cars.csv)
model_id_name=(cars)
data_name=(custom)

random_seed=2021

for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --exv_use 1\
      --norm_name 'layer'\
      --features M \
      --revin 1\
      --pe 1\
      --d_layers 4\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.998\
      --alpha_freq 0.5\
      --d_ff 512\
      --dropout 0.0\
      --gamma 0.9\
      --des 'Exp' \
      --train_epochs 40 \
      --patience 10 \
      --itr 1 --batch_size 64 --learning_rate 4e-3 >logs/LongForecasting04/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done

