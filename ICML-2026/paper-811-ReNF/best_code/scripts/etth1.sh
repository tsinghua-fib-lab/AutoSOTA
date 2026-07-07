export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=ReNF_alpha

root_path_name=../dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
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
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.996\
      --alpha_freq 0.0\
      --d_ff 2048\
      --gamma 0.3 \
      --dropout 0.9\
      --des 'Exp' \
      --patience 3\
      --train_epochs 10 \
      --itr 1 --batch_size 128 --learning_rate 6e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# # Large d_ff is not necessary, a small d_ff can  achieve similar results. The following is an example.
# for pred_len in 96
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name} \
#       --model_id ${model_id_name}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name} \
#       --features M \
#       --pe 1\
#       --revin 1\
#       --d_layers 3\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.995\
#       --alpha_freq 0.0\
#       --d_ff 256\
#       --gamma 0.9 \
#       --dropout 0.8\
#       --des 'Exp' \
#       --patience 3\
#       --train_epochs 10 \
#       --itr 1 --batch_size 64 --learning_rate 1e-3 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

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
      --r_ema 0.996\
      --alpha_freq 0.0\
      --d_ff 3072\
      --gamma 0.3 \
      --dropout 0.92\
      --des 'Exp' \
      --patience 3\
      --train_epochs 10 \
      --itr 1 --batch_size 128 --learning_rate 6e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
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
      --d_layers 3\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.996 \
      --d_ff 3072\
      --gamma 0.3 \
      --alpha_freq 0.05\
      --dropout 0.92\
      --des 'Exp' \
      --patience 10\
      --train_epochs 10 \
      --itr 1 --batch_size 128 --learning_rate 5e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

seq_len=336
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
      --d_layers 8\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.998\
      --d_ff 128\
      --alpha_freq 0.0\
      --gamma 0.5 \
      --dropout 0.9\
      --des 'Exp' \
      --patience 10\
      --train_epochs 25 \
      --itr 1 --batch_size 128 --learning_rate 6e-4 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done



