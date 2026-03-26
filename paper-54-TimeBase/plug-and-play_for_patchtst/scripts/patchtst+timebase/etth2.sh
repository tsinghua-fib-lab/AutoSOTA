if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=PatchTST

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021
for patch_len in 24 16 ; do
for basis_num in 6 8 10; do
for lr in 5e-1 1e-1 5e-2 1e-2 5e-3 1e-3; do
for seq_len in 720; do
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --basis_num $basis_num\
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len $patch_len\
      --stride $patch_len\
      --des 'Exp' \
      --train_epochs 30\
      --itr 1 --batch_size 128 --learning_rate $lr >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'${patch_len}'_'${basis_num}_${lr}.log 
done
done
done
done
done