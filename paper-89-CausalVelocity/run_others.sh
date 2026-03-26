#!/bin/bash

for model in cds igci_unif igci_gauss reci ckl ckm chd ccs ctv
do 
    echo "Running model: $model on velocity dataset"
    python3 experiment_competitors.py --dataset=velocity --model=${model} --fix_samples=1000
done 

for model in cds igci_unif igci_gauss reci ckl ckm chd ccs ctv
do 
    echo "Running model: $model on sigmoid dataset"
    python3 experiment_competitors.py --dataset=sigmoid --model=${model} --fix_samples=1000
done 

for model in  cds igci_unif igci_gauss reci ckl ckm chd ccs ctv
do 
    echo "Running model: $model on anm dataset"
    python3 experiment_competitors.py --dataset=anm --model=${model} --fix_samples=5000 --verbose
done 

for model in cds igci_unif igci_gauss reci ckl ckm chd ccs ctv
do 
    echo "Running model: $model on lsnm dataset"
    python3 experiment_competitors.py --dataset=lsnm --model=${model} --fix_samples=5000 --verbose
done 

for model in hsic lik cds igci_unif igci_gauss reci ckl ckm chd ccs ctv
do 
    echo "Running model: $model on Tuebingen (discrete removed) dataset"
    python3 experiment_competitors.py --dataset=tuebingen_new --model=${model} --fix_samples=1000
done 

for model in cds igci_unif igci_gauss reci ckl ckm chd ccs ctv; do
    for dataset in an ans ls lss mnu sim simc simg simln tuebingen; do
        echo "Running model: $model on dataset: $dataset"
        python3 experiment_competitors.py --dataset=${dataset} --model=${model} --fix_samples=1000
    done
done

for model in cds igci_unif igci_gauss reci ckl ckm chd ccs ctv; do
    for dataset in velocity sigmoid anm lsnm; do
        echo "Running model: $model on dataset: $dataset"
        python3 experiment_competitors.py --dataset=${dataset} --model=${model} --fix_samples=5000
    done
done

