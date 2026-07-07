export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=512
model_name=ReNF_beta

root_path_name=../dataset/traffic/
data_path_name=(traffic.csv)
model_id_name=(traffic)
data_name=(custom)

random_seed=2021
# traffic
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
      --d_layers 3\
      --n_block 1\
      --pe 0\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --d_ff 2048\
      --alpha_freq 0.2\
      --dropout 0.5\
      --des 'Exp' \
      --train_epochs 40\
      --patience 15 \
      --gamma 0.9\
      --itr 1 --batch_size 64 --learning_rate 8e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


for pred_len in 192 336 720
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
      --d_layers 6\
      --n_block 1\
      --pe 0\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --d_ff 2048\
      --alpha_freq 0.2\
      --dropout 0.5\
      --des 'Exp' \
      --train_epochs 40\
      --patience 15 \
      --gamma 0.9\
      --itr 1 --batch_size 64 --learning_rate 8e-4 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
