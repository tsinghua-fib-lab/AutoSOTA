#!/bin/bash

# SLURM job options - complete by adding account and partition
#SBATCH --account=
#SBATCH --partition=
#SBATCH --job-name=blip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=%x.o%j.%N
#SBATCH --error=%x.e%j.%N

# Print job details
NOW=`date +%H:%M-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'


cd $SLURM_SUBMIT_DIR            # here we go into the submission directory

module load python              # loading python
source venv/bin/activate        # activate env
module load cuda                # loading cuda

# Run the script
srun python "$@"