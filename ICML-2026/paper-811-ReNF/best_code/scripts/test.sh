export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs_test" ]; then
    mkdir ./logs_test
fi

if [ ! -d "./logs_test/LongForecasting" ]; then
    mkdir ./logs_test/LongForecasting
fi
seq_len=512
model_name=Model_test

root_path_name=../dataset/ETT-small/
data_path_name=(ETTh1.csv ETTm2.csv)
model_id_name=(ETTh1 ETTm2)
data_name=(ETTh1 ETTm2)

random_seed=2021
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
      --jepa 0 \
      --pe 0\
      --revin 1\
      --d_layers 3\
      --level_dim 1\
      --seq_len $seq_len \
      --pred_len $pred_len \
      --r_ema 0.998\
      --alpha_freq 0.0\
      --d_ff 2048\
      --gamma 0.3 \
      --dropout 0.9\
      --des 'Exp' \
      --patience 5\
      --train_epochs 15 \
      --itr 1 --batch_size 128 --learning_rate 6e-4 >logs_test/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

# for pred_len in 192
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
#       --jepa 0 \
#       --pe 0\
#       --revin 1\
#       --d_layers 3\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.996\
#       --alpha_freq 0.0\
#       --d_ff 3072\
#       --gamma 0.3 \
#       --dropout 0.9\
#       --des 'Exp' \
#       --patience 3\
#       --train_epochs 10 \
#       --itr 1 --batch_size 128 --learning_rate 6e-4 >logs_test/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

# for pred_len in 336
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
#       --jepa 0 \
#       --revin 1\
#       --d_layers 3\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.996 \
#       --d_ff 3072\
#       --gamma 0.2 \
#       --dropout 0.9\
#       --des 'Exp' \
#       --patience 10\
#       --train_epochs 10 \
#       --itr 1 --batch_size 128 --learning_rate 6e-4 >logs_test/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done

# seq_len=336
# for pred_len in 720
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
#       --jepa 0 \
#       --revin 1\
#       --d_layers 8\
#       --level_dim 1\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --r_ema 0.998\
#       --d_ff 256\
#       --alpha_freq 0.0\
#       --gamma 0.5 \
#       --dropout 0.9\
#       --des 'Exp' \
#       --patience 10\
#       --train_epochs 15 \
#       --itr 1 --batch_size 64 --learning_rate 3e-4 >logs_test/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
# done



