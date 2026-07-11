import time, sys, os, json
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import load_data_matrix

data_path = 'Dataset/face_id_4.npy'
X = load_data_matrix(data_path)
rank = 10

# Try tighter ADMM convergence
configs = [
    {'label': 'eps1e-6', 'epsilon': 1e-6, 'max_iter': 8000, 'num_steps': 200, 'inner_iter_asc': 5},
    {'label': 'eps1e-6_iter2', 'epsilon': 1e-6, 'max_iter': 8000, 'num_steps': 200, 'inner_iter_asc': 2},
    {'label': 'default_tight', 'epsilon': 1e-4, 'max_iter': 8000, 'num_steps': 100, 'inner_iter_asc': 2},
    {'label': 'more_mp_steps', 'epsilon': 1e-4, 'max_iter': 4000, 'num_steps': 500, 'inner_iter_asc': 2},
]

for cfg in configs:
    params = {
        'X': X.copy(),
        'dataset_name': 'Face',
        'r': rank,
        'rerun_times': 1,
        'epsilon': cfg['epsilon'],
        'max_iter': cfg['max_iter'],
        'num_steps': cfg['num_steps'],
        'inner_iter_asc': cfg['inner_iter_asc'],
    }
    enmf = NMF_ENMF(method_name='ENMF', params=params)
    start = time.time()
    enmf.run_within_fixed_time(target_run_time=600)
    wall = time.time() - start
    print(f"{cfg['label']}: error={enmf.enmf_error:.2f}, wall={wall:.1f}s, runtime={enmf.total_runtime:.1f}s, dist_po={enmf.dist_po:.2f}")
    sys.stdout.flush()
