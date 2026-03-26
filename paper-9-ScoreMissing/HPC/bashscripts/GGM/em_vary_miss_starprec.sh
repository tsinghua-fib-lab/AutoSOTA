#!/bin/bash
#
#






MAX_ITER=7



source ~/initConda.sh
conda activate DiffusionProcesses
cd ../GGM


FILENAME="vary_miss_starprec_em.py"
echo "$FILENAME"
for ((i=0; i<=MAX_ITER; i++))
do
    start=`date +%s`
    python "$FILENAME" 50 $i
    end=`date +%s`
    runtime=$((end-start))
    echo Runtime is "$runtime"
done


