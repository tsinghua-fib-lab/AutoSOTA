import time, sys, os, json
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import load_data_matrix

data_path = 'Dataset/face_id_4.npy'
X = load_data_matrix(data_path)
rank = 10

# Use run_within_fixed_time with modest budget to get error
# Then use that error as target for run_to_target_error
params = {
    'X': X.copy(),
    'dataset_name': 'Face',
    'r': rank,
    'rerun_times': 1,
    'rho': 5,
    'epsilon': 1e-4,
    'max_iter': 4000,
    'tau_inc': 1.1,
    'tau_dec': 1.1,
    'num_steps': 10,
    'hals_rounds': 1,
}

# Equal-time: run with fixed time budget
print('=== Equal-Time Protocol ===')
time_budget = 300  # generous budget
enmf_tc = NMF_ENMF(method_name='ENMF', params=params)
start = time.time()
enmf_tc.run_within_fixed_time(target_run_time=time_budget)
wall_tc = time.time() - start
print(f'Time budget: {time_budget}s')
print(f'Wall time: {wall_tc:.2f}s')
print(f'Internal total_runtime: {enmf_tc.total_runtime:.2f}s')
print(f'SVD: {enmf_tc.t_svd:.2f}s, Rot: {enmf_tc.t_rotate:.2f}s, MPO: {enmf_tc.t_mp:.2f}s, HALS: {enmf_tc.t_descent:.2f}s')
print(f'Reconstruction Error: {enmf_tc.enmf_error:.4f}')
print(f'dist_po: {enmf_tc.dist_po:.6e}')

# Equal-error: run to target error
print()
print('=== Equal-Error Protocol ===')
target_error = enmf_tc.enmf_error  # Use what eNMF achieved
enmf_ec = NMF_ENMF(method_name='ENMF', params=params)
start = time.time()
enmf_ec.run_to_target_error(target_error=target_error)
wall_ec = time.time() - start
print(f'Target error: {target_error:.4f}')
print(f'Wall time: {wall_ec:.2f}s')
print(f'Internal total_runtime: {enmf_ec.total_runtime:.2f}s')
print(f'SVD: {enmf_ec.t_svd:.2f}s, Rot: {enmf_ec.t_rotate:.2f}s, MPO: {enmf_ec.t_mp:.2f}s, HALS: {enmf_ec.t_descent:.2f}s')
print(f'Achieved error: {enmf_ec.enmf_error:.4f}')

# Summary
print()
print('=== SUMMARY ===')
print(json.dumps({
    'reconstruction_error': round(float(enmf_tc.enmf_error), 4),
    'runtime': round(float(enmf_ec.total_runtime), 4),
    'target_error': round(float(target_error), 4),
    'time_budget': time_budget,
}))
