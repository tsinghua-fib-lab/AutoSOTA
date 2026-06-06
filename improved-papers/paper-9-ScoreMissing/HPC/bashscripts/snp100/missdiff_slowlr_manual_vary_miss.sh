#!/bin/bash
#
#






MAX_ITER=7



source ~/initConda.sh
conda activate DiffusionProcesses
cd ../snp100


FILENAME="vary_miss_missdiff_slowlr_manual.py"
echo "$FILENAME"
for ((i=0; i<=MAX_ITER; i++))
do
    start=`date +%s`
    python "$FILENAME" 50 $i
    end=`date +%s`
    runtime=$((end-start))
    echo Runtime is "$runtime"
done


