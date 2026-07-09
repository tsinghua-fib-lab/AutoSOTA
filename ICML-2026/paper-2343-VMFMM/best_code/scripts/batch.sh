#!/bin/bash

# Paths and architecture
DATA=${1:-'./data'}
ARCH=${2:-'vit_b16'}
DEVICE=${3:-'0'}

# Datasets
datasets=("imagenet" "sun397" "fgvc" "eurosat" "stanford_cars" "food-101" "oxford_pets" "oxford_flowers" "caltech-101" "dtd" "ucf101")

# Number of tasks
n_tasks=1000

# Loop over batch sizes and configurations
for batch_size in 64 1000; do
  for dataset in "${datasets[@]}"; do

    # Batch size 64: num_class_eff ranges (1,4), (2,10), (5,25)
    if [[ "$batch_size" == "64" ]]; then
      num_class_eff_min_values=(1 2 5)
      num_class_eff_max_values=(4 10 25)
      for i in "${!num_class_eff_min_values[@]}"; do
        python3 main.py --root_path "${DATA}" \
                        --dataset "$dataset" \
                        --method MOON \
                        --backbone "${ARCH}" \
                        --batch_size "$batch_size" \
                        --num_class_eff_min "${num_class_eff_min_values[$i]}" \
                        --num_class_eff_max "${num_class_eff_max_values[$i]}" \
                        --n_tasks "$n_tasks" \
                        --device "$DEVICE"
      done

    # Batch size 1000: num_class_eff ranges (5,25), (25,50), (50,100)
    elif [[ "$batch_size" == "1000" ]]; then
      num_class_eff_min_values=(5 25 50)
      num_class_eff_max_values=(25 50 100)
      for i in "${!num_class_eff_min_values[@]}"; do
        python3 main.py --root_path "${DATA}" \
                        --dataset "$dataset" \
                        --method MOON \
                        --backbone "${ARCH}" \
                        --batch_size "$batch_size" \
                        --num_class_eff_min "${num_class_eff_min_values[$i]}" \
                        --num_class_eff_max "${num_class_eff_max_values[$i]}" \
                        --n_tasks "$n_tasks" \
                        --device "$DEVICE"
      done

    fi

  done
done
