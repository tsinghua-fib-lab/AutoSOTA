#!/bin/bash
declare -A parrot_dict

parrot_dict["astar"]="1"
parrot_dict["bwaves"]="1"
parrot_dict["bzip"]="1"
parrot_dict["cactusadm"]="1"
parrot_dict["gems"]="1"
parrot_dict["lbm"]="1"
parrot_dict["leslie3d"]="1"
parrot_dict["libq"]="1"
parrot_dict["mcf"]="1"
parrot_dict["milc"]="1"
parrot_dict["omnetpp"]="1"
parrot_dict["sphinx3"]="1"
parrot_dict["xalanc"]="1"

mkdir -p logs/parrot

cuda_devices=("cuda:0" "cuda:1")
cuda_index=0

for dataset in "${!parrot_dict[@]}"; do
    fractions=(${parrot_dict[$dataset]})
    for fraction in "${fractions[@]}"; do
        current_device=${cuda_devices[$cuda_index]}
        echo "Running parrot with dataset=$dataset with fraction $fraction on device $current_device"
        nohup python -m model.parrot --dataset "$dataset" --device "$current_device" --real_test -f $fraction > "logs/parrot/${dataset}_${fraction}.log" 2>&1 &
        ((cuda_index=(cuda_index+1)%2))
    done
done
