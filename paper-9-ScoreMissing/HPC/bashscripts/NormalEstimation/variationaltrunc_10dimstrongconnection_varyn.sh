#!/bin/bash
#
#
#SBATCH --job-name=proximal_GGM_80miss
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-14:00:00
#SBATCH --array=0-4
#SBATCH --mem=1G
#SBATCH --account=math026082

source ~/initConda.sh
conda activate DiffusionProcesses
cd ../NormalEstimation

echo "${SLURM_ARRAY_TASK_ID}"
FILENAME="variationaltrunc_10dim_strongconnection_varyn.py"
echo "$FILENAME"
start=`date +%s`
python "$FILENAME" 200 ${SLURM_ARRAY_TASK_ID}

end=`date +%s`

runtime=$((end-start))

echo Runtime is "$runtime"

