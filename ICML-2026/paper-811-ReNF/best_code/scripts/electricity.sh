export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=512
model_name=ReNF_beta

root_path_name=../dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021

# electricity
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
      --d_layers 4\
      --n_block 1\
      --pe 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --alpha_freq 0.7\
      --d_ff 2048\
      --dropout 0.5\
      --des 'Exp' \
      --train_epochs 10 \
      --patience 3 \
      --gamma 0.8\
      --itr 1 --batch_size 16 --learning_rate 5e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

for pred_len in 192 336
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
      --d_layers 4\
      --n_block 1\
      --pe 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --alpha_freq 0.7\
      --d_ff 2048\
      --dropout 0.5\
      --des 'Exp' \
      --train_epochs 8 \
      --patience 3 \
      --gamma 0.8\
      --itr 1 --batch_size 16 --learning_rate 5e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --d_layers 4\
      --n_block 1\
      --pe 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --alpha_freq 0.7\
      --d_ff 2048\
      --dropout 0.5\
      --des 'Exp' \
      --train_epochs 12 \
      --patience 3 \
      --gamma 0.8\
      --itr 1 --batch_size 16 --learning_rate 3e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
