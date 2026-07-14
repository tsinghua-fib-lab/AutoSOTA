#!/bin/bash

mkdir -p datasets/processed

# Process test splits for high noise, gradient sampling density, and striped sampling density.

uv run python -u preprocessing/process_pcpnet.py --save_path datasets/processed/pcpnet_gradient.hdf5   --split test_var_density_gradient --number_of_passes 200
uv run python -u preprocessing/process_pcpnet.py --save_path datasets/processed/pcpnet_high_noise.hdf5 --split test_high_noise           --number_of_passes 200
uv run python -u preprocessing/process_pcpnet.py --save_path datasets/processed/pcpnet_striped.hdf5    --split test_var_density_striped  --number_of_passes 200

# Process splits to measure runtime for different numbers of points.

for i in 1 2 4 6 8 10 20 30 40 50; do
    I=$(printf "%02d" "$i")
    D="datasets/processed/pcpnet_time_${I}k.hdf5"
    P=$((i * 1000))
    uv run python -u preprocessing/process_pcpnet.py --save_path "$D" --split test_all --number_of_passes 30 --number_of_points "$P"
done

# Process splits for different outlier percentages.

for i in 0 1 2 10 20 30 40 50 60 70 80 90 100; do
    I=$(printf "%03d" "$i")
    D="datasets/processed/pcpnet_outliers_${I}.hdf5"
    O=$((i * 500))
    uv run python -u preprocessing/process_pcpnet.py --save_path "$D" --split test_no_noise --number_of_passes 10 --number_of_outliers "$O"
done
