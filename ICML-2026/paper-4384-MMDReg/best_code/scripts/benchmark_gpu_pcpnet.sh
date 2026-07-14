#!/bin/bash

mkdir -p results
S="results/pcpnet_gpu.json"

export PYTHONPATH=.
export JAX_PLATFORMS="cuda"
export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Parameters for ICP have already been set for PCPNet

# Benchmark split for high noise.

D="datasets/processed/pcpnet_high_noise.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 

# Benchmark split for gradient sampling density.

D="datasets/processed/pcpnet_gradient.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 

# Benchmark split for striped sampling density.

D="datasets/processed/pcpnet_striped.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 

# Benchmark splits to measure GPU runtime for different numbers of points.

for i in 1 10 20 30 40 50; do
    I=$(printf "%02d" "$i")
    D="datasets/processed/pcpnet_time_${I}k.hdf5"
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-GPU 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-GPU 
done
