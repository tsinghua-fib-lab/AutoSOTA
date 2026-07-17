#!/bin/bash
# LLaMA-60M
module load cuda/11.8

export level=2
export scale=0.5
export optimizer=adamw
export lr=1e-2
export seed=0
export beta1=0.9
export beta2=0.95

torchrun --standalone --nproc_per_node 4 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr $lr \
    --scale $scale \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_ratio 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100000 \
    --level $level \
    --seed $seed \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --optimizer $optimizer
wait

echo 'finish!'