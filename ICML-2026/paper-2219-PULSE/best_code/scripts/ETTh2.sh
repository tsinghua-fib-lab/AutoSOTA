model_id_name=ETTh2
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/${model_id_name}" ]; then
    mkdir ./logs/${model_id_name}
fi
dir=./logs/${model_id_name}

model_name=PULSE
root_path_name=./all_datasets/
data_path_name=ETTh2.csv
data_name=ETTh2
seq_len=96

enc_in=7
time_dim=1
patch_size=24
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
gpu=0
freq='h'


pred_len=96
dsa=1
dsb=2
ksize=3
d_model=32
inv_len=32
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=192
dsa=1
dsb=1
ksize=5
d_model=32
inv_len=24
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=336
dsa=4
dsb=2
ksize=5
d_model=32
inv_len=24
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"


pred_len=720
dsa=4
dsb=1
ksize=5
d_model=16
inv_len=48
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_id_name}_${seq_len}_${pred_len}.log"