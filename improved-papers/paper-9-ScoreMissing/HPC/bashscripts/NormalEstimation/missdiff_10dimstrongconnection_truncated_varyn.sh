#!/bin/bash
#
#






MAX_ITER=4



source ~/initConda.sh
conda activate DiffusionProcesses
cd ../NormalEstimation


FILENAME="missdiff_10dim_strongconnection_truncated_varyn.py"
echo "$FILENAME"
for ((i=0; i<=MAX_ITER; i++))
do
    start=`date +%s`
    python "$FILENAME" 200 $i
    end=`date +%s`
    runtime=$((end-start))
    echo Runtime is "$runtime"
done


