#!/bin/bash

DATA=$1
FOLD=$2
MODEL=bert-base-uncased

python scripts/rawdata_cv.py --no-subsample --model ${MODEL} --data ${DATA} --fold ${FOLD}
python scripts/rawdata_cv.py --subsample --model ${MODEL} --data ${DATA} --fold ${FOLD}
