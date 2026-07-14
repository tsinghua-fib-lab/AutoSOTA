#!/bin/bash

mkdir -p results
S="results/pcpnet_cpu.json"

export PYTHONPATH=.
export JAX_PLATFORMS="cpu"
export JAX_DEFAULT_MATMUL_PRECISION="highest"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1

# Parameters for algorithms, other than MMD-Reg, have already been set for PCPNet

# Benchmark split for high noise.

D="datasets/processed/pcpnet_high_noise.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 16 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 64 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR

# Benchmark split for gradient sampling density.

D="datasets/processed/pcpnet_gradient.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 16 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 64 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR

# Benchmark split for striped sampling density.

D="datasets/processed/pcpnet_striped.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 16 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 64 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR

# Benchmark splits to measure CPU runtime for different numbers of points.

for i in 1 2 4 6 8 10 20 30 40 50; do
    I=$(printf "%02d" "$i")
    D="datasets/processed/pcpnet_time_${I}k.hdf5"
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR
done

for i in 1 2 4 6 8 10; do
    I=$(printf "%02d" "$i")
    D="datasets/processed/pcpnet_time_${I}k.hdf5"
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm CPD
done

# Benchmark splits for different outlier percentages.

for i in 0 1 2 10 20 30 40 50 60 70 80 90 100; do
    I=$(printf "%03d" "$i")
    D="datasets/processed/pcpnet_outliers_${I}.hdf5"
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Gaussian
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 32 --mmd_reg_ls 0.75 --mmd_reg_dist Laplace
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Point-CPU 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm ICP-Point-To-Plane-CPU 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GICP 
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm FilterReg
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm GMMReg
    uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm SVR
done
