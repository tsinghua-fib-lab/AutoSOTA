#!/bin/bash
#
#






MAX_ITER=9



source ~/initConda.sh
conda activate DiffusionProcesses
cd ../GGM


FILENAME="vary_centres_starprec_em.py"
for ((i=0; i<=MAX_ITER; i++))
do
    echo "$FILENAME"
    start=`date +%s`
    python "$FILENAME" 100 ${SLURM_ARRAY_TASK_ID}

    end=`date +%s`

    runtime=$((end-start))

    echo Runtime is "$runtime"
done

echo Runtime is "$runtime"

