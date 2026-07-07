export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=ReNF_alpha

root_path_name=../dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021

# ETTh2
for pred_len in 96 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name} \
      --model_id ${model_id_name}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name} \
      --features M \
      --pe 0\
      --revin 1\
      --d_layers 3\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --d_ff 128\
      --alpha_freq 0.0\
      --gamma 0.3 \
      --dropout 0.1\
      --des 'Exp' \
      --patience 3\
      --train_epochs 5 \
      --itr 1 --batch_size 16 --learning_rate 5e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 192 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name} \
      --model_id ${model_id_name}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name} \
      --features M \
      --pe 1\
      --revin 1\
      --d_layers 3\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --d_ff 256\
      --alpha_freq 0.0\
      --gamma 0.3 \
      --dropout 0.1\
      --des 'Exp' \
      --patience 3\
      --train_epochs 5 \
      --itr 1 --batch_size 16 --learning_rate 6e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 336 
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name} \
      --model_id ${model_id_name}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name} \
      --features M \
      --pe 0\
      --revin 1\
      --d_layers 4\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.9992\
      --d_ff 512\
      --alpha_freq 0.0\
      --gamma 0.3 \
      --dropout 0.5\
      --des 'Exp' \
      --patience 3\
      --train_epochs 5 \
      --itr 1 --batch_size 16 --learning_rate 6e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path ${data_path_name} \
      --model_id ${model_id_name}'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ${data_name} \
      --features M \
      --pe 1\
      --revin 1\
      --d_layers 6\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.9992\
      --d_ff 512\
      --alpha_freq 0.25\
      --gamma 0.9 \
      --dropout 0.5\
      --des 'Exp' \
      --patience 3\
      --train_epochs 5 \
      --itr 1 --batch_size 16 --learning_rate 2e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
