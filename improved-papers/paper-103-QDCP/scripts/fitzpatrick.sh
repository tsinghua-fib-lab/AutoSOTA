#! /bin/bash

# Slurm sbatch options
#SBATCH -o outputs/fitzpatrick.%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda cuda

# TCT
python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b1 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.001 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments --data_dir ../data --pretrained --use_squared_loss --start_from_stage2

# FedProx
#python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b1 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.001 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments --data_dir ../data --use_fedprox --fedprox_mu 0.01 --pretrained

# FedAvg
#python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b1 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.001 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments --data_dir ../data --pretrained

# Centrally hosted
#python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b1 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.001 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments --data_dir ../data --pretrained

# TCT (IID)
#python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b1 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.001 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --batch_size 64 --use_data_augmentation --use_iid_partition --save_dir new_experiments --data_dir ../data --pretrained

# FedAvg (IID)
#python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b1 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.001 --momentum 0.9 --batch_size 64 --use_data_augmentation --use_iid_partition --save_dir new_experiments --data_dir ../data --pretrained
