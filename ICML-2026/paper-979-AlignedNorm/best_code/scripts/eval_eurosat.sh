#!/bin/bash
set -e
cd /repo
for seed in 1 2 3; do
    D="/repo/output/base2new/train_base/eurosat/shots_16/ALIGNEDNORM/vit_b16/seed${seed}"
    if [ ! -f "$D/log.txt" ]; then
        CUDA_VISIBLE_DEVICES=0,1 python3 train.py --root /datasets --seed ${seed} --trainer ALIGNEDNORM --dataset-config-file configs/datasets/eurosat.yaml --config-file configs/trainers/ALIGNEDNORM/vit_b16.yaml --output-dir ${D} DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base TASK B2N
    fi
done
for seed in 1 2 3; do
    D="/repo/output/base2new/test_new/eurosat/shots_16/ALIGNEDNORM/vit_b16/seed${seed}"
    if [ ! -f "$D/log.txt" ]; then
        M="/repo/output/base2new/train_base/eurosat/shots_16/ALIGNEDNORM/vit_b16/seed${seed}"
        CUDA_VISIBLE_DEVICES=0,1 python3 train.py --root /datasets --seed ${seed} --trainer ALIGNEDNORM --dataset-config-file configs/datasets/eurosat.yaml --config-file configs/trainers/ALIGNEDNORM/vit_b16.yaml --output-dir ${D} --model-dir ${M} --eval-only DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new TASK B2N
    fi
done
python3 /repo/scripts/compute_metrics.py
