if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=PatchTST

root_path_name=../dataset/
data_path_name=exchange_rate.csv
model_id_name=exchange
data_name=custom

random_seed=2021
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
      --enc_in 8 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 2\
      --patience 10\
      --lradj 'type3'\
      --pct_start 0.2\
      --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done