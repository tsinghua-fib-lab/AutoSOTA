import random
import numpy as np
from joblib import Parallel, delayed
from utils.subgroup import *


def base_job(X_adjust, X_subgp, Y, B_subgp):
    """
    Baseline method: fit Cox model on entire dataset, region is entire subgroup space.

    Parameters
    ----------
    X_adjust : array, shape (n_samples, n_adjust_features)
        Features for Cox model adjustment
    X_subgp : array, shape (n_samples, n_subgp_features)
        Features for subgroup definition
    Y : structured array, shape (n_samples,)
        Survival outcomes
    B_subgp : array, shape (2, n_subgp_features)
        Bounding box for subgroup space

    Returns
    -------
    results : list of dict
        Single result with full dataset Cox model and entire subgroup space as region
    """
    beta = fit_cox(X_adjust, Y)
    results = [{
        'subgroup_id': 0,
        'R': B_subgp,
        'beta': beta,
    }]
    return results