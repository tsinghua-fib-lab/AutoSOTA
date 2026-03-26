if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=PatchTST

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
for patch_len in 24; do
for basis_num in 12 18 24; do
for lr in 5e-3 1e-2 1e-3 5e-4 ; do #  5e-1 1e-1 5e-2 1e-2 5e-3 1e-3
for seq_len in 720; do
for pred_len in 96 192 336 720;do #96 192 336 720
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len $patch_len\
      --stride $patch_len\
      --basis_num $basis_num\
      --des 'Exp' \
      --gpu 7 \
      --train_epochs 30\
      --patience 5\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 128 --learning_rate $lr >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'${patch_len}'_'${basis_num}_${lr}.log 
done
done
done
done
done