#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-5:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm-%j.out

# BLOB (varying N, Table 1 left)
python3 blob.py --check=1
python3 blob.py --check=0

# CIFAR10-ResNet18 (varying N, Table 1 right)
python3 cifar10_resnet.py --check=1
python3 cifar10_resnet.py --check=0

echo "Done."
