#!/bin/bash
export LD_LIBRARY_PATH=/repo:/usr/local/lib:$LD_LIBRARY_PATH
export PATH=/repo:/usr/local/bin:$PATH
cd /repo
python -u src/retrieval.py --dataroot /datasets/g3824 --dataset_name cub2011 --clip_model_name ViT-B/32 --query_eval_type image --gallery_eval_type image --iso_ktop 150 --iso_kbottom 50 --iso_tau 5.0 --iso_ensemble --out_path eval_output > /repo/eval_final.txt 2>&1
echo "EXIT: $?"
