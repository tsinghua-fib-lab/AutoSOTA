#!/bin/bash

n_data=5000

mech=lsnm
suffix="_tmi"
noise_scale=0.2
weight=0.2 ## variance for the weights of random functions.

## only relevant for neural networks
layers=3
hidden_size=64

name="${mech}${suffix}"

# check if the data directory already exists, if not, create it
if [ -d "data/synthetic/$name" ]; then
    echo "Data directory for $name already exists. Please remove it before running this script."
    return 
else
    mkdir data/synthetic/$name
fi

outdir=synthetic/$name

for i in {1..100}; do
    python3 generate_data.py --seed=$i --name=$outdir --mech=$mech --noise_scale=$noise_scale --n_data=$n_data --layers=$layers --hidden_size=$hidden_size --weight=$weight --tmi_tx
done

touch data/synthetic/$name/parameters.txt 
echo "mechanism=$mech" > data/synthetic/$name/parameters.txt
echo "noise scale=$noise_scale" >> data/synthetic/$name/parameters.txt
echo "initialization weight=$weight" >> data/synthetic/$name/parameters.txt
echo "layers=$layers" >> data/synthetic/$name/parameters.txt
echo "hidden_size=$hidden_size" >> data/synthetic/$name/parameters.txt
