export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting04" ]; then
    mkdir ./logs/LongForecasting04
fi
seq_len=36
model_name=ReNF_beta

root_path_name=../dataset/DowJones/
data_path_name=(dowjones.csv)
model_id_name=(dowjones)
data_name=(custom)

random_seed=2021

# weather
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
      --exv_use 0\
      --features M \
      --norm_name 'layer'\
      --revin 1\
      --pe 1\
      --d_layers 3\
      --n_block 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.99\
      --alpha_freq 0.0\
      --d_ff 64\
      --dropout 0.3\
      --gamma 0.7\
      --des 'Exp' \
      --train_epochs 4 \
      --patience 3 \
      --itr 1 --batch_size 8 --learning_rate 2e-4 >logs/LongForecasting04/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
done

# for pred_len in 48 60
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
#       --revin 1\
#       --norm_name 'layer'\
#       --pe 0\
#       --d_layers 6\
#       --n_block 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.99\
#       --alpha_freq 0.3\
#       --d_ff 128 \
#       --dropout 0.9\
#       --gamma 0.5\
#       --des 'Exp' \
#       --train_epochs 30 \
#       --itr 1 --batch_size 64 --learning_rate 4e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.logs 
# done

