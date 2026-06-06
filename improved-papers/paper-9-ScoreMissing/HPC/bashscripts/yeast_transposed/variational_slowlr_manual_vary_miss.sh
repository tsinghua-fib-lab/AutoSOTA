#!/bin/bash
#
#






MAX_ITER=7



source ~/initConda.sh
conda activate DiffusionProcesses
cd ../yeast_transposed


FILENAME="vary_miss_variational_slowlr_manual.py"
echo "$FILENAME"
start=`date +%s`
python "$FILENAME" 20 ${SLURM_ARRAY_TASK_ID}

end=`date +%s`

runtime=$((end-start))

echo Runtime is "$runtime"

