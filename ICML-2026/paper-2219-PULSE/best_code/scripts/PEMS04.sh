model_id_name=PEMS04
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}

model_name=PULSE
root_path_name=./all_datasets/
data_path_name=PEMS04.npz
data_name=PEMS
seq_len=96

enc_in=307
time_dim=1
train_epochs=30
patience=5
gpu=0
batch_size=32
learning_rate=0.005
rec_lambda=0
auxi_lambda=1
random_seed=2024

pred_len=12
inv_len=96
dsa=4
dsb=1
ksize=5
d_model=32
patch_size=4

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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=24
inv_len=96
dsa=2
dsb=1
ksize=5
d_model=16
patch_size=8

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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=48
inv_len=24
dsa=4
dsb=1
ksize=5
d_model=16
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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=96
inv_len=48
dsa=4
dsb=1
ksize=5
d_model=16
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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"