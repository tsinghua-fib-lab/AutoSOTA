#!/bin/bash
#SBATCH -J bilevel_synthetic
#SBATCH -A gts-kwang692-paid                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --cpus-per-task=16
#SBATCH --output=/storage/home/hcoda1/3/sho73/r-kwang692-0/synthetic_sbatch_output/%j.out

HOME=/storage/home/hcoda1/3/sho73/r-kwang692-0
REPO_PATH=$HOME/project/FFOLayer

cd $REPO_PATH
source $HOME/my_miniconda/etc/profile.d/conda.sh
conda activate bilevel

seed=$1
ydim=$2
backward_eps=$3
batchSize=$4

echo "Running seed=$seed backward_tol=$backward_eps ydim=$ydim"

# methods=("ffocp_eq" "cvxpylayer" "lpgd" "ffoqp_eq_schur" "qpth")
methods=("ffocp_eq")
epochs=1
# batchSize=8
# ydim=900

for method in "${methods[@]}"; do
    echo "Running method=$method, seed=$seed"
    python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=$method --learn_constraint=0 --ydim=$ydim --seed=$seed --backward_eps=$backward_eps
done