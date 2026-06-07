#!/bin/bash
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
DATASETS=("flickr30k_text" "coco_text" "nocaps_text")

# Run combinations
k_top=10
k_bottom=300 

for dataset in "${DATASETS[@]}"; do
    $PYTHON -u "$ROOT_DIR/src/retrieval.py" \
    --iso_ktop "$k_top" \
    --iso_kbottom "$k_bottom" \
    --dataroot "$DATA_ROOT" \
    --dataset_name "$dataset" \
    --clip_model_name "ViT-L/14" \
    --query_eval_type "text" \
    --gallery_eval_type "text" \
    --out_path $filename 
done 

 
