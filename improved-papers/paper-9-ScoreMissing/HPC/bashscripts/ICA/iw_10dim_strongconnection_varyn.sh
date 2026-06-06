#!/bin/bash
#
#
#SBATCH --job-name=iw_ica_10dim_strongconnection_varyn
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-4:00:00
#SBATCH --array=0-5
#SBATCH --mem=1G
#SBATCH --account=math026082

source ~/initConda.sh
conda activate DiffusionProcesses
cd /user/work/cn21903/MarginalScoreMatching/HPC/ICA

echo "${SLURM_ARRAY_TASK_ID}"
FILENAME="iw_10dim_strongconnection_varyn.py"
echo "$FILENAME"
start=`date +%s`
python "$FILENAME" 100 ${SLURM_ARRAY_TASK_ID}

end=`date +%s`

runtime=$((end-start))

echo Runtime is "$runtime"

