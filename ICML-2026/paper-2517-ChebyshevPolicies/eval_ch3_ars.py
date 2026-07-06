#!/usr/bin/env python3
"""
Standalone evaluation script for CH-3-ARS on MountainCarContinuous-v0.
Evaluates the best trained CH-3-ARS policy on 100 evenly spaced start positions
from [-0.6, -0.4] and reports the mean return R.
"""
import sys
import json
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/algorithms')

from utils import exp_run


def evaluate(coeffs_path='/repo/best_ch3_ars_coeffs.pt', n_eval=100):
    """Evaluate CH-3-ARS policy on n_eval evenly spaced start positions."""
    # Load trained coefficients
    coeffs = torch.load(coeffs_path, map_location='cpu', weights_only=True)

    # Evaluate on 100 evenly spaced start positions
    xs = np.linspace(-0.6, -0.4, n_eval)

    model, eval_env = exp_run.get_sb3_polynomial_model_and_eval_env(
        basis='chebyshev', env_name='MountainCarContinuous-v0',
        coeffs=coeffs, algo='ars')

    rewards = []
    for x in xs:
        r_mean, r_std = exp_run.run_sb3_model(
            model, eval_env, options={'low': x, 'high': x})
        rewards.append(r_mean)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    min_r = float(np.min(rewards))
    max_r = float(np.max(rewards))

    return {
        'R': mean_r,
        'R_std': std_r,
        'R_min': min_r,
        'R_max': max_r,
    }


def main():
    results = evaluate()
    print(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    main()
