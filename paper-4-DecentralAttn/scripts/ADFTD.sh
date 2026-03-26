#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

mkdir -p ./logs/ADFTD
log_dir="./logs/ADFTD"

model_name=TeCh
data_path="/data/gqyu/dataset/med/ADFTD/"
data_name="ADFTD"

bss=(128)
lrs=(3e-5)
t_layers=(6)
v_layers=(6)
dropouts=(0.0)
d_models=(128)
patch_lens=(1)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for t_layer in "${t_layers[@]}"; do
            for v_layer in "${v_layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for d_model in "${d_models[@]}"; do
                        for patch_len in "${patch_lens[@]}"; do
                            python -u run.py \
                                --root_path $data_path \
                                --model $model_name \
                                --data $data_name \
                                --t_layer $t_layer \
                                --v_layer $v_layer \
                                --batch_size $bs \
                                --d_model $d_model \
                                --dropout $dropout \
                                --patch_len $patch_len\
                                --augmentations flip0.,frequency0.,jitter0.,mask0.25,channel0.,drop0. \
                                --lradj constant \
                                --itr 5 \
                                --learning_rate $lr \
                                --train_epochs 6 \
                                --patience 6 > "${log_dir}/bs${bs}_lr${lr}_tl${t_layer}_vl${v_layer}_dp${dropout}_dm${d_model}_pl${patch_len}.log"

                        done
                    done
                done
            done
        done
    done
done
