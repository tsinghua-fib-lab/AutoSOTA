"""SyNG-R: bootstrap-resampling of LSM latents.

Given fitted (Z, alpha, rho) from the LSM, draw n_reps bootstrap samples
of (Z, alpha) and reconstruct adjacency matrices via the LSM link
function:

    P_ij = sigmoid(Z_i^T Z_j + alpha_i + alpha_j + rho).

This is the SyNG-R variant from the paper.
"""
from __future__ import annotations

import numpy as np

from syngler.utils.source import bootstrap_alpha_Z, reconstruct_adjacency


def bootstrap_latents(model_Z, model_alpha, n_reps, seed=0):
    """Bootstrap-resample (Z, alpha) `n_reps` times.

    Args:
        model_Z     : (n, r) fitted latent positions.
        model_alpha : length-n fitted node intercepts.
        n_reps      : number of bootstrap reps.
        seed        : base RNG seed; rep k uses seed + k.

    Returns:
        list of (Z_b, alpha_b) tuples, each of shapes ((n, r), (n,)).
    """
    model_alpha = np.asarray(model_alpha).reshape(-1, 1)
    model_Z = np.asarray(model_Z)
    out = []
    for k in range(n_reps):
        np.random.seed(seed + k)
        a, Z = bootstrap_alpha_Z(model_alpha, model_Z, batch=1)
        out.append((Z.squeeze(0), a.squeeze(0)))
    return out


def generate_graphs(model_Z, model_alpha, n_reps, rho=0.0, seed=0):
    """SyNG-R: bootstrap latents and reconstruct adjacency for each rep.

    Args:
        model_Z, model_alpha, rho : fitted LSM parameters.
        n_reps : number of synthetic graphs to produce.
        seed   : base RNG seed.

    Yields:
        (n, n) uint8 adjacency matrices, one per rep.
    """
    for k, (Z, alpha) in enumerate(bootstrap_latents(model_Z, model_alpha, n_reps, seed=seed)):
        yield reconstruct_adjacency(Z, alpha, rho=rho, seed=seed + k + 1)
