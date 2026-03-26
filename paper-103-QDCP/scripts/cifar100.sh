#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p i64m1tga800u
#SBATCH --ntasks-per-node=3
#SBATCH -n 3
#SBATCH -o ../output/output.%j
#SBATCH -e ../error/error.%j
#SBATCH -J CP

module load anaconda3
module load cuda/12.0

source activate fl


# TCT
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data > ../output/TCT.out 2>&1

# FedAvg
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation  --save_dir experiments --data_dir ../data  > ../output/FedAvg.out 2>&1

# Centrally hosted
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --samples_per_client 50000 --momentum 0.9 --use_data_augmentation  --save_dir experiments --data_dir ../data  > ../output/Central.out 2>&1

# TCT
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --use_iid_partition --save_dir experiments --data_dir ../data > ../output/TCT_iid.out 2>&1

# FedAvg
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation --use_iid_partition  --save_dir experiments --data_dir ../data  > ../output/FedAvg_iid.out 2>&1

