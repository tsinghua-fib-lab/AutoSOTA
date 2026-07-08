#!/bin/bash
#SBATCH -J bilevel                       # job name
#SBATCH -A XXX                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --cpus-per-task=16              # single node, 16 cores
#SBATCH --mem-per-cpu=16G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --output=/storage/home/hcoda1/9/zzhao628/scratch/bilevel-layer/logs/sudoku_cvxpylayer.log

# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
# module load anaconda3/2023.03
# module load cuda/11.8.0

# conda activate rl

cd /storage/home/hcoda1/9/zzhao628/scratch/bilevel-layer

echo echo
echo $METHOD
echo $SEED
echo $YDIM
echo $EPS
echo $LR
echo $BATCH_SIZE

export GRB_LICENSE_FILE=/storage/home/hcoda1/9/zzhao628/gurobi.lic

python3 sudoku/main_sudoku.py --method=cvxpylayer --n 3 --batch_size 8 --epochs 1
