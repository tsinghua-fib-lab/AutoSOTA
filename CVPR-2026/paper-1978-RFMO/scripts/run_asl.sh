#!/usr/bin/env bash

cd ../src

METHOD="asl"
DATASETS=("yeast" "enron" "medical" "bibtex" "voc2007" "coco2014")

for dataset in ${DATASETS[@]}; do
    echo "Running $METHOD on $dataset"
    python main.py --config configs/${METHOD}/${dataset}.json
done
