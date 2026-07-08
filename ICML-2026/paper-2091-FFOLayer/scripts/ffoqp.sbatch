#!/bin/bash
#SBATCH -J bilevel                       # job name
#SBATCH -A XXX                 # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --mem-per-cpu=8G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --output=/storage/home/hcoda1/9/kwang692/p-kwang692-0/ffodo/output/%j.out

cd /storage/home/hcoda1/9/kwang692/p-kwang692-0/ffodo

module load anaconda3
conda activate bilevel
echo echo
echo $METHOD
echo $SEED
echo $YDIM
echo $EPS
echo $LR

export GRB_LICENSE_FILE=/storage/home/hcoda1/9/kwang692/p-kwang692-0/gurobi.lic

python3 main.py --method=$METHOD --seed=$SEED --ydim=$YDIM --lr=$LR --eps=$EPS
