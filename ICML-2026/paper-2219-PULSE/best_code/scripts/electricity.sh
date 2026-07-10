model_id_name=Electricity
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}

model_name=PULSE
root_path_name=./all_datasets/
data_path_name=electricity.csv
data_name=custom
seq_len=96


enc_in=321
batch_size=64
learning_rate=0.005
train_epochs=30
patience=5
gpu=0
freq='h'

pred_len=96
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim 2 \
    --dsa 8 \
    --dsb 1 \
    --ksize 5 \
    --d_model 32 \
    --inv_len 32 \
    --patch_size 12 \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=192
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim 2 \
    --dsa 8 \
    --dsb 1 \
    --ksize 5 \
    --d_model 32 \
    --inv_len 48 \
    --patch_size 24 \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=336
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim 2 \
    --dsa 4 \
    --dsb 1 \
    --ksize 5 \
    --d_model 16 \
    --inv_len 12 \
    --patch_size 24 \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=720
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_id_name}_${seq_len}_${pred_len}" \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim 4 \
    --dsa 8 \
    --dsb 1 \
    --ksize 11 \
    --d_model 16 \
    --inv_len 24 \
    --patch_size 24 \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"