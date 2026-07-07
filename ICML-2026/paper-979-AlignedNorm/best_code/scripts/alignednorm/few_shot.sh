#!/bin/bash

# custom config
DATA_ROOT="/path/to/your/datasets"
DATA="${DATA_ROOT}"
TRAINER=ALIGNEDNORM
SHOTS=16

DATASET=$1

for SHOTS in 1 2 4 8 16
do
    if [ $DATASET == "imagenet" ]; then
        CFG=vit_b16_imagenet
    else
        CFG=vit_b16_few_shot
    fi
    for SEED in ${SEEDS:-1 2 3}
    do
        DIR=output/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
            TASK FS
        fi
    done
    python3 parse_test_res.py output/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/ 
done
