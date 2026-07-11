import time, sys, os, json
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import load_data_matrix

data_path = 'Dataset/face_id_4.npy'
X = load_data_matrix(data_path)
rank = 10

results = []
for run_i in range(5):
    params = {
        'X': X.copy(),
        'dataset_name': 'Face',
        'r': rank,
        'rerun_times': 1,
    }
    enmf = NMF_ENMF(method_name='ENMF', params=params)
    start = time.time()
    enmf.basic_run()
    wall = time.time() - start
    results.append({
        'run': run_i,
        'error': enmf.enmf_error,
        'total_runtime': enmf.total_runtime,
        'svd_err': enmf.svd_error,
        'dist_po': enmf.dist_po,
        't_svd': enmf.t_svd,
        't_rotate': enmf.t_rotate,
        't_mp': enmf.t_mp,
        't_descent': enmf.t_descent,
        'wall': wall,
    })
    print(f"Run {run_i}: error={enmf.enmf_error:.2f}, runtime={enmf.total_runtime:.2f}s, dist_po={enmf.dist_po:.2f}")
    sys.stdout.flush()

errors = [r['error'] for r in results]
runtimes = [r['total_runtime'] for r in results]
print(f"\nError: mean={np.mean(errors):.2f}, std={np.std(errors):.4f}, min={np.min(errors):.2f}, max={np.max(errors):.2f}")
print(f"Runtime: mean={np.mean(runtimes):.2f}, std={np.std(runtimes):.4f}, min={np.min(runtimes):.2f}, max={np.max(runtimes):.2f}")
