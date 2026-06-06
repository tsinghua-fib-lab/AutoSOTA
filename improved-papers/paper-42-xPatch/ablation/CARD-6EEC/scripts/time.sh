if [ ! -d "./logs/time" ]; then
    mkdir ./logs/time
fi

seq_len=96
model_name=CARD

root_path_name=../dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2023
for pred_len in 96
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
    --enc_in 7 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 3\
    --patience 10 \
    --itr 1 --batch_size 8 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/time/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

root_path_name=../dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2023
for pred_len in 96
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
    --enc_in 7 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 3\
    --patience 10 \
    --itr 1 --batch_size 8 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/time/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

root_path_name=../dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2023
for pred_len in 96
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
    --enc_in 21 \
    --e_layers 2 \
    --n_heads 2 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 3\
    --patience 10 \
    --itr 1 --batch_size 8 --learning_rate 0.0001 --merge_size 2 \
    --lradj 'CARD'\
    --warmup_epochs 0 > logs/time/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done