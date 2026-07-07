#!/bin/bash

# custom config
DATA="/Anonymous/data/DATA_prompt_training"
TRAINER=MMRL

DATASET=$1

REG_WEIGHT=0.2
REP_DIM=512

CFG=vit_b16_cross_datasets

SHOTS=16

for SEED in 1 2 3
do
    MODEL_DIR=output/base2new/train_base/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    DIR=output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
        TRAINER.MMRL.REG_WEIGHT ${REG_WEIGHT} \
        TRAINER.MMRL.REP_DIM ${REP_DIM} \
        TASK CD
    fi
done
python3 parse_test_res.py output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/