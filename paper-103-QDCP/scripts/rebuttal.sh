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

source activate base
conda activate fl

python ../src/run_TCT.py --dataset dermamnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --central > ../output/dermamnist.out 2>&1 &

python ../src/run_TCT.py --dataset bloodmnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --central > ../output/bloodmnist.out 2>&1 &

python ../src/run_TCT.py --dataset pathmnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --central > ../output/pathmnist.out 2>&1 &

python ../src/run_TCT.py --dataset tissuemnist --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --central > ../output/tissuemnist.out 2>&1 &

wait

