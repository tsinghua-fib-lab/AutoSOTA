#!/bin/bash

filename=$0
filename=${filename%.*}

# Set the GPU device
export CUDA_VISIBLE_DEVICES=0

# Define Python path and root directory
PYTHON=""    # Specify the conda environment (e.g. /home/user/miniconda3/envs/isoclip/bin/python3.10 )
ROOT_DIR=""  # Specify the project dir (e.g. /home/user/IsoCLIP/ )
DATA_ROOT="" # Specify the dataset dir (e.g. /path/to/datasets/ )
# List of datasets
DATASETS=("caltech101" "stanford_cars" "oxford_pets" "oxford_flowers" "fgvc_aircraft" "dtd" "eurosat" "food101" "sun397"  "ucf101")  
# Run combinations
k_top=200
k_bottom=50 

for dataset in "${DATASETS[@]}"; do
    $PYTHON -u "$ROOT_DIR/src/classification/zero_shot_classification.py" \
    --iso_ktop "$k_top" \
    --iso_kbottom "$k_bottom" \
    --dataroot "$DATA_ROOT" \
    --dataset_name "$dataset" \
    --clip_model_name "ViT-B/16" \
    --out_path $filename \
    --eval-type "iso_ncm" 
   
done 

 
