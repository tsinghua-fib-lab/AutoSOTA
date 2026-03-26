#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

mkdir -p ./logs/PTB
log_dir="./logs/PTB"

model_name=TeCh
data_path="/data/gqyu/dataset/med/PTB/"
data_name="PTB"

bss=(128)
lrs=(1e-4)
t_layers=(0)
v_layers=(3)
dropouts=(0.)
d_models=(256)
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
                                --augmentations flip0.,frequency0.,jitter0.,mask0.,channel0.4,drop0. \
                                --lradj constant \
                                --itr 5 \
                                --learning_rate $lr \
                                --train_epochs 80 \
                                --patience 40 > "${log_dir}/bs${bs}_lr${lr}_tl${t_layer}_vl${v_layer}_dp${dropout}_dm${d_model}_pl${patch_len}.log"

                        done
                    done
                done
            done
        done
    done
done
