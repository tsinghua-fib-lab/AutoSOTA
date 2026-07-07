#!/bin/bash

# custom config
DATA="/Anonymous/data/DATA_prompt_training"
TRAINER=MMRL

DATASET=$1

if [ $DATASET == "imagenet" ]; then
    CFG=vit_b16_imagenet
else
    CFG=vit_b16
fi

SHOTS=16

for SEED in 1 2 3
do
    DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base \
        TASK B2N
    fi
done
python3 parse_test_res.py output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/