"""
REFACTORED VERSION - Example showing the new signature and approach

Key changes:
1. Signature: random_job(X_adjust, X_subgp, Y, B_subgp, num_subgroups, seeds)
   - X_adjust: features for Cox model (n × len(adjust_cols))
   - X_subgp: features for subgroup regions (n × len(subgp_cols))
   - B_subgp: bounding box for subgroup space only (2 × len(subgp_cols))
   - No more subgp_features parameter!

2. Implementation changes:
   - Use X_subgp for all region operations
   - Use X_adjust for Cox model fitting
   - R is now purely in subgroup space
   - No more index mapping or region expansion
"""

import random
import numpy as np
from joblib import Parallel, delayed
from utils.subgroup import *


def random_job(X_adjust, X_subgp, Y, B_subgp, num_subgroups, seeds):
    """
    Random baseline method for subgroup discovery.

    Parameters
    ----------
    X_adjust : array-like, shape (n_samples, n_adjust_features)
        Features used for Cox model adjustment
    X_subgp : array-like, shape (n_samples, n_subgp_features)
        Features used for defining subgroup regions
    Y : structured array, shape (n_samples,)
        Survival outcomes with fields 'failure' (bool) and 'time' (float)
    B_subgp : array-like, shape (2, n_subgp_features)
        Bounding box for subgroup feature space
    num_subgroups : int
        Number of subgroups to generate per seed
    seeds : list of int
        Random seeds for reproducibility

    Returns
    -------
    results : list of dict
        Each dict contains:
        - 'subgroup_id': int, ID of the subgroup
        - 'R': array (2, n_subgp_features), region bounds in subgroup space
        - 'beta': array (n_adjust_features,), Cox coefficients in adjustment space
        - 'seed': int, random seed used
    """
    n = len(X_adjust)

    # Number of points needed to define a bounding box in subgroup space
    if len(X_subgp.shape) == 2:
        k = 2 * X_subgp.shape[1]  # 2 points per dimension
    else:
        assert len(X_subgp.shape) == 1, "X_subgp must be either 1D or 2D."
        k = 2  # For 1D data

    def evaluate(seed):
        subgroups = []
        rng = random.Random(seed)
        for _ in range(num_subgroups):
            # Select k random points from the data
            selected_points = rng.sample(range(n), k)

            # Define region as bounding box of selected points in SUBGROUP space
            R_subgp = bounding_box(X_subgp[selected_points])

            # Find all points within this subgroup region
            ind = in_region(X_subgp, R_subgp)

            # Fit Cox model using ADJUSTMENT features for points in the subgroup
            try:
                beta = fit_cox(X_adjust[ind], Y[ind])
                subgroups.append((R_subgp.copy(), beta.copy()))
            except:
                # If Cox model fails to fit, store NaN for beta
                subgroups.append((R_subgp.copy(), np.full(X_adjust.shape[1], np.nan)))
        return subgroups

    # Parallel evaluation across different seeds
    results_list = Parallel(n_jobs=-1)(
        delayed(evaluate)(s) for s in seeds
    )

    # Flatten results
    results = []
    for res, seed in zip(results_list, seeds):
        for subgroup_id in range(num_subgroups):
            R_subgp, beta = res[subgroup_id]
            results.append({
                'subgroup_id': subgroup_id,
                'R': R_subgp,  # Region is purely in subgroup space
                'beta': beta,   # Beta is purely in adjustment space
                'seed': seed
            })
    return results