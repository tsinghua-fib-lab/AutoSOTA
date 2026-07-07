export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting03" ]; then
    mkdir ./logs/LongForecasting03
fi
seq_len=512
model_name=ReNF_beta

root_path_name=../dataset/ETT-small/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021

# ETTm2
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
      --pe 1\
      --revin 1\
      --d_layers 3\
      --n_block 1\
      --norm_name 'layer'\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --d_ff 128\
      --r_ema 0.9985\
      --alpha_freq 0.45\
      --dropout 0.3\
      --gamma 0.9\
      --des 'Exp' \
      --patience 2\
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting03/$model_name'_'${model_id_name}'_'$seq_len'_'$pred_len.log 
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
      --n_block 1\
      --norm_name 'layer'\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --d_ff 128\
      --r_ema 0.9985\
      --alpha_freq 0.5\
      --dropout 0.3\
      --gamma 0.9\
      --des 'Exp' \
      --patience 2\
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting03/$model_name'_'${model_id_name}'_'$seq_len'_'$pred_len.log 
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
      --pe 1\
      --revin 1\
      --d_layers 3\
      --n_block 1\
      --norm_name 'layer'\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --d_ff 128\
      --r_ema 0.9985\
      --alpha_freq 0.6\
      --dropout 0.3\
      --gamma 0.9\
      --des 'Exp' \
      --patience 2\
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting03/$model_name'_'${model_id_name}'_'$seq_len'_'$pred_len.log 
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
      --n_block 1\
      --norm_name 'layer'\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --d_ff 128\
      --r_ema 0.9985\
      --alpha_freq 0.5\
      --dropout 0.1\
      --gamma 0.9\
      --des 'Exp' \
      --patience 2\
      --train_epochs 5 \
      --itr 1 --batch_size 64 --learning_rate 2e-3 >logs/LongForecasting03/$model_name'_'${model_id_name}'_'$seq_len'_'$pred_len.log 
done
