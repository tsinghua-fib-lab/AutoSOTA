#!/bin/bash
#
#
#SBATCH --job-name=missdiff_ica_10dim_strongconnection_varymiss
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-2:30:00
#SBATCH --array=0-7
#SBATCH --mem=1G
#SBATCH --account=math026082

source ~/initConda.sh
conda activate DiffusionProcesses
cd /user/work/cn21903/MarginalScoreMatching/HPC/ICA

echo "${SLURM_ARRAY_TASK_ID}"
FILENAME="missdiff_10dim_strongconnection_varymiss.py"
echo "$FILENAME"
start=`date +%s`
python "$FILENAME" 100 ${SLURM_ARRAY_TASK_ID}

end=`date +%s`

runtime=$((end-start))

echo Runtime is "$runtime"

