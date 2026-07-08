#!/usr/bin/env bash
# Run brax implementation of sac on test environments

start=${1:-0}
num_seeds=${2:-5}
out_dir=${3:-"$HOME/out/brax"}

echo "Writing to $out_dir"

for i in $(seq $start $((start + num_seeds - 1)))
do
   echo "Running ANT on seed $i"
   python main.py --env ant --logdir $out_dir --seed $i
done


for i in $(seq $start $((start + num_seeds - 1)))
do
   echo "Running CHEETAH on seed $i"
   python main.py --env halfcheetah --logdir $out_dir --seed $i
done


for i in $(seq $start $((start + num_seeds - 1)))
do
   echo "Running HOPPER on seed $i"
   python main.py --env hopper --logdir $out_dir --seed $i
done


for i in $(seq $start $((start + num_seeds - 1)))
do
   echo "Running WALKER on seed $i"
   python main.py --env walker2d --logdir $out_dir --seed $i
done


