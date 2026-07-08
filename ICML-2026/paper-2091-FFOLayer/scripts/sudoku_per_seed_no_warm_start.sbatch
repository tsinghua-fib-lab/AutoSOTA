#!/bin/bash
#SBATCH -A gts-kwang692-paid                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --mem-per-cpu=32G                 # memory per core
#SBATCH -t 2:00:00                           # duration of the job
#SBATCH --cpus-per-task=8
#SBATCH --output=/storage/home/hcoda1/9/zzhao628/r-kwang692-0/sudoku_sbatch_output/%j.out

# module load gurobi
export GRB_LICENSE_FILE=/usr/local/pace-apps/manual/packages/gurobi/13.0.1/license/gurobi.lic
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"

HOME=/storage/home/hcoda1/9/zzhao628
REPO_PATH=$HOME/scratch/FFOLayer

cd $REPO_PATH
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate bilevel
# source $HOME/r-kwang692-0/.conda/etc/profile.d/conda.sh
module load anaconda3
source ~/.bashrc
conda activate rl

seed=$1

echo "Running seed=$seed"

epochs=1
batchSize=8
n=3
# methods=("ffocp_eq" "cvxpylayer" "lpgd" "ffoqp_eq_schur" "qpth" "bpqp" "dqp" ) # "altdiff"
# methods=("dqp")
methods=("ffocp_eq")
for method in "${methods[@]}"; do
    echo "Running method=$method, seed=$seed"
    python sudoku/main_sudoku.py --method=$method --seed=$seed --epochs=$epochs --batch_size=$batchSize --n=$n
done