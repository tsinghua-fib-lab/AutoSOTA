#!/bin/bash

module load anaconda/2020.11-py38 
module load learning/conda-2020.11-py38-gpu
module load ml-toolkit-gpu/pytorch/1.7.1

cd $SLURM_SUBMIT_DIR

python run_sockshop.py --path ../sock-shop-5min/

