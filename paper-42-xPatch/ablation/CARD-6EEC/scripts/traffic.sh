# export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2023
for pred_len in 96 192 336 720
do
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
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
    --train_epochs 100\
      --patience 100 \
      --itr 1 --batch_size 8 --learning_rate 0.001 --merge_size 16 \
      --lradj 'CARD'\
      --warmup_epochs 20 > logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

# batch_size 24