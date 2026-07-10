model_id_name=traffic
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}

model_name=PULSE
root_path_name=./all_datasets/
data_path_name=traffic.csv
data_name=custom
seq_len=96
enc_in=862

batch_size=64
learning_rate=0.005
train_epochs=30
patience=5
gpu=0
rec_lambda=0
auxi_lambda=1
itr=1
d_model=16
time_dim=2
dsb=1
freq='h'


pred_len=96
dsa=4
ksize=5
inv_len=12
patch_size=24

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
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=192
dsa=8
ksize=3
inv_len=32
patch_size=12

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
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=336
dsa=2
ksize=5
inv_len=96
patch_size=12

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
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=720
dsa=8
ksize=3
inv_len=12
patch_size=12

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
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"