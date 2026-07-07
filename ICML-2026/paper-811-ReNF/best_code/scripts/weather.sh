export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=720
model_name=ReNF_beta

root_path_name=../dataset/weather/
data_path_name=(weather.csv)
model_id_name=(weather)
data_name=(custom)

random_seed=2021

# weather
for pred_len in 96 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --revin 1\
      --norm_name 'layer'\
      --pe 1\
      --d_layers 3\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.996\
      --alpha_freq 0.5\
      --d_ff 64 \
      --dropout 0.8\
      --gamma 0.1\
      --des 'Exp' \
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 4e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done

for pred_len in 192
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --revin 1\
      --norm_name 'layer'\
      --pe 1\
      --d_layers 3\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.996\
      --alpha_freq 0.5\
      --d_ff 64 \
      --dropout 0.7\
      --gamma 0.1\
      --des 'Exp' \
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 4e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done

for pred_len in 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --revin 1\
      --norm_name 'layer'\
      --pe 1\
      --d_layers 3\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.996\
      --alpha_freq 0.5\
      --d_ff 64 \
      --dropout 0.7\
      --gamma 0.1\
      --des 'Exp' \
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done

for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name[0]} \
      --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name[0]} \
      --features M \
      --revin 1\
      --norm_name 'layer'\
      --pe 1\
      --d_layers 3\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.996\
      --alpha_freq 0.1\
      --d_ff 64 \
      --dropout 0.8\
      --gamma 0.1\
      --des 'Exp' \
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done

