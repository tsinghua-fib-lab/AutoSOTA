export CUDA_VISIBLE_DEVICES=4

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting04" ]; then
    mkdir ./logs/LongForecasting04
fi
seq_len=36
model_name=ReNF_alpha

root_path_name=../dataset/SP500/
data_path_name=(SP500.csv)
model_id_name=(SP500)
data_name=(custom)

random_seed=2021

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
#       --exv_use 1\
#       --features M \
#       --revin 1\
#       --pe 1\
#       --d_layers 2\
#       --n_block 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.996\
#       --alpha_freq 0.3\
#       --d_ff 512\
#       --dropout 0.9\
#       --gamma 0.7\
#       --des 'Exp' \
#       --train_epochs 20 \
#       --patience 5 \
#       --itr 1 --batch_size 64 --learning_rate 1e-3 >logs/LongForecasting04/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
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
      --revin 1\
      --exv_use 0\
      --pe 1\
      --d_layers 2\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.9\
      --alpha_freq 0.02\
      --d_ff 512\
      --dropout 0.9\
      --gamma 0.7\
      --des 'Exp' \
      --train_epochs 10 \
      --patience 5 \
      --itr 1 --batch_size 128 --learning_rate 2e-3 >logs/LongForecasting04/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done
