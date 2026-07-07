export CUDA_VISIBLE_DEVICES=7

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
seq_len=36
model_name=ReNF_beta

root_path_name=../dataset/METR/
data_path_name=(METR-LA.csv)
model_id_name=(METR)
data_name=(custom)

random_seed=2021
# traffic
# for pred_len in 24 36 
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path ${data_path_name[0]} \
#       --model_id ${model_id_name[0]}'_'$seq_len'_'$pred_len \
#       --model $model_name \
#       --data ${data_name[0]} \
#       --features M \
#       --exv_use 1\
#       --revin 1\
#       --d_layers 6\
#       --n_block 1\
#       --pe 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.999\
#       --d_ff 1024\
#       --alpha_freq 0.9\
#       --dropout 0.0\
#       --des 'Exp' \
#       --train_epochs 25\
#       --patience 10 \
#       --gamma 0.9\
#       --itr 1 --batch_size 32 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

for pred_len in 48 60
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
      --exv_use 1\
      --revin 1\
      --d_layers 6\
      --n_block 1\
      --pe 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.999\
      --d_ff 1024\
      --alpha_freq 0.9\
      --dropout 0.0\
      --des 'Exp' \
      --train_epochs 20\
      --patience 10 \
      --gamma 0.9\
      --itr 1 --batch_size 32 --learning_rate 2e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done