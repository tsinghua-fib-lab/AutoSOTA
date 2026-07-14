#!/bin/bash

output=cpu_latency_gamma5.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output;

for ((i=0; i<=6; i++)); do
    echo $i | tee -a $output;
    for ((j=1;j<=10;j++)); do
        echo $j >> tee -a $output;
        python train_latency.py --indx=$i --atype='cpu-hog1_' --gamma=0.5 | tee -a $output;
    done;
    echo "-----one service finish-----" | tee -a $output;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output;
