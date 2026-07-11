#!/usr/bin/env python3
"""Reproduction evaluation: eNMF on Face dataset, rank=10.

Metrics:
  1. Reconstruction Error (Table 4): eNMF converged error (equal-time protocol)
  2. Runtime (Table 2): eNMF time to achieve reference error (equal-error protocol)

Usage: python3 eval_enmf_face.py
Output: JSON dict with Reconstruction Error and Runtime
"""
import time, json, sys, os, io, contextlib
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
from nmf_algos import NMF_ENMF
from nmf_algos.utils.utils import load_data_matrix

def main():
    data_path = 'Dataset/face_id_4.npy'
    X = load_data_matrix(data_path)
    rank = 10
    dataset_name = 'Face'
    
    base_params = {
        'X': X.copy(),
        'dataset_name': dataset_name,
        'r': rank,
        'rerun_times': 1,
    }
    
    # Suppress debug prints from HALS during computation
    with contextlib.redirect_stdout(io.StringIO()):
        # Phase 1: Reconstruction Error (equal-time protocol)
        enmf_ref = NMF_ENMF(method_name='ENMF', params=base_params)
        enmf_ref.run_within_fixed_time(target_run_time=300)
        recon_error = float(enmf_ref.enmf_error)
        
        # Phase 2: Runtime (equal-error protocol)
        target_error = recon_error
        enmf_ec = NMF_ENMF(method_name='ENMF', params=base_params)
        enmf_ec.run_to_target_error(target_error=target_error)
        runtime = float(enmf_ec.total_runtime)
    
    results = {
        'Reconstruction Error': round(recon_error, 4),
        'Runtime': round(runtime, 4),
    }
    
    # Print JSON result to stdout (clean output)
    json.dump(results, sys.stdout)
    sys.stdout.write('\n')
    return results

if __name__ == '__main__':
    main()
