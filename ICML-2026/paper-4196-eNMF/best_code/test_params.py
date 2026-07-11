import time, os, sys
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import load_data_matrix

data_path = 'Dataset/face_id_4.npy'
X = load_data_matrix(data_path)
rank = 10

configs = [
    {'label': 'default', 'max_iter': 4000, 'rho': 5, 'num_steps': 100, 'hals_rounds': 10**2},
    {'label': 'more_rot', 'max_iter': 8000, 'rho': 5, 'num_steps': 100, 'hals_rounds': 10**2},
    {'label': 'rho1', 'max_iter': 4000, 'rho': 1, 'num_steps': 100, 'hals_rounds': 10**2},
    {'label': 'more_steps', 'max_iter': 4000, 'rho': 5, 'num_steps': 200, 'hals_rounds': 10**2},
]

for cfg in configs:
    params = {
        'X': X.copy(),
        'dataset_name': 'Face',
        'r': rank,
        'max_iter': cfg['max_iter'],
        'rho': cfg['rho'],
        'num_steps': cfg['num_steps'],
        'hals_rounds': cfg['hals_rounds'],
    }
    enmf = NMF_ENMF(method_name='ENMF', params=params)
    start = time.time()
    enmf.basic_run()
    elapsed = time.time() - start
    print(f"{cfg['label']}: error={enmf.enmf_error:.2f}, wall={elapsed:.1f}s, SVD={enmf.t_svd:.1f}s, Rot={enmf.t_rotate:.1f}s, MPO={enmf.t_mp:.1f}s, HALS={enmf.t_descent:.1f}s, dist_po={enmf.dist_po:.6e}")
    sys.stdout.flush()
