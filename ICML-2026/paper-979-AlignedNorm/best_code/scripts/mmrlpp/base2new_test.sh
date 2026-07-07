#!/bin/bash

# custom config
DATA="/Anonymous/data/DATA_prompt_training"
TRAINER=MMRLpp

DATASET=$1

if [ $DATASET == "imagenet" ]; then
    CFG=vit_b16_imagenet
else
    CFG=vit_b16
fi

SHOTS=16
SUB=new

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    DIR=output/base2new/test_${SUB}/${COMMON_DIR}
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
        --model-dir ${MODEL_DIR} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TASK B2N
    fi
done
python3 parse_test_res.py output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/