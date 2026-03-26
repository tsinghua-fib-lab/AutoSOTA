#! /bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p i64m1tga40u
#SBATCH --ntasks-per-node=3
#SBATCH -n 3
#SBATCH -o ../output/output.%j
#SBATCH -e ../error/error.%j
#SBATCH -J CP

# Loading the required module
module load anaconda3
module load cuda/12.0

source activate fl

# TCT
#python src/run_TCT.py --dataset dermamnist --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5  --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --override

# FedAvg
#python src/run_TCT.py --dataset dermamnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5  --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --override

# Centrally hosted
python ../src/run_TCT.py --dataset dermamnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --central > ../output/Central.out 2>&1

# TCT (IID)
#python src/run_TCT.py --dataset dermamnist --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5  --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --use_iid_partition

# FedAvg (IID)
#python src/run_TCT.py --dataset dermamnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5  --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --use_iid_partition

