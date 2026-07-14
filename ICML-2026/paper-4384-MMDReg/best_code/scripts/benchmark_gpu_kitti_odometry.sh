#!/bin/bash

mkdir -p results
S="results/kitti_odometry_gpu.json"

export PYTHONPATH=.
export JAX_PLATFORMS="cuda"
export JAX_DEFAULT_MATMUL_PRECISION="highest"

# Parameters for Multi-Scale ICP have already been set for KITTI Odometry

# Benchmark sequence 07.

D="datasets/processed/kitti_odometry_07.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU

# Benchmark sequence 08.

D="datasets/processed/kitti_odometry_08.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU

# Benchmark sequence 09.

D="datasets/processed/kitti_odometry_09.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU

# Benchmark sequence 10.

D="datasets/processed/kitti_odometry_10.hdf5"

uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Gaussian
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm MMD-Reg --mmd_reg_Ds 256 256 256 256 512 --mmd_reg_ls 4.0 2.0 1.0 0.5 0.25 --mmd_reg_dist Laplace
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Point-GPU
uv run python -u experiments/benchmark.py --data_path "$D" --save_path "$S" --algorithm Multi-Scale-ICP-Point-To-Plane-GPU
