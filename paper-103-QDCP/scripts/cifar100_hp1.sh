#!/bin/bash
#BSUB -app default
#BSUB -n 3
#BSUB -q gpu
#BSUB -gpgpu 1
#BSUB -e ../error/error.%J
#BSUB -o ../output/output.%J
#BSUB -J CP

module load anaconda3
module load cuda-11.4

source activate fl


# TCT
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --save_dir ../experiments --data_dir ../data > ../output/TCT.out 2>&1

# FedAvg
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation  --save_dir ../experiments --data_dir ../data  > ../output/FedAvg.out 2>&1

# Centrally hosted
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --samples_per_client 50000 --momentum 0.9 --use_data_augmentation  --save_dir ../experiments --data_dir ../data  > ../output/Central.out 2>&1

# TCT
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --use_iid_partition --save_dir ../experiments --data_dir ../data > ../output/TCT_iid.out 2>&1

# FedAvg
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation --use_iid_partition  --save_dir ../experiments --data_dir ../data  > ../output/FedAvg_iid.out 2>&1

# FedProx
nohup python ../src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation  --save_dir ../experiments --data_dir ../data --use_fedprox --fedprox_mu 0.1  > ../output/FedProx.out 2>&1
