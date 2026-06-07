#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

zo_eps=0.5
fitness_lambda=0.4
num_prompts=3
algorithm="fozo"
lr=0.01
n_spsa=1
seed=2000

for lr in 0.08
do
    for algorithm in fozo
    do
        tag="_seed=${seed}_lr_${lr}"  #_prompt=${num_prompts}_n_spsa=${n_spsa}
        echo "当前配置： n_spsa=${n_spsa}, tag=${tag}"
        python main.py \
            --tag "$tag" \
            --gpu "0" \
            --algorithm "$algorithm" \
            --num_prompts "$num_prompts" \
            --zo_eps "$zo_eps" \
            --lr "$lr" \
            --fitness_lambda "$fitness_lambda" \
            --n_spsa "$n_spsa" \
            --seed "$seed" \
            --continual
    done
done