"""
Utilities to simulate HMM data with Torus Graph (TG) emissions, fit HMMs, and
produce accuracy metrics for heatmaps in the paper.

"""
__date__ = "April - October 2025"


import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate_hmm import (
    HMMPriorParams,
    sample_sticky_transition_matrix,
    sample_phis,
    sample_hmm,
)
from src.hmm import fit_hmm_em


EM_KWARGS = dict(
    num_em_iters=50,
    warmup_iterations=20,
    num_part_opt_steps=200,
    lr=1e-2,
    cross_entropy_lambda=1e-2,
    beta=0.5,
    beta_annealing=None,
    init_method="zscore",
    n_init=100,
)

D_GRID_VALS = [2, 4, 8, 16, 32, 64, 128, 256]
K_GRID_VALS = [2, 3, 4, 5, 6, 7, 8, 9]
T = 1000


# ----------------------------- Core computations -----------------------------


def _best_permutation(source: np.ndarray, target: np.ndarray, K: int) -> np.ndarray:
    """
    Find a label permutation π: {0..K-1}→{0..K-1} that maximizes matches
    between `source` and `target` via Hungarian assignment on the contingency.

    Returns
    -------
    perm : np.ndarray, shape (K,), dtype=int
        Mapping such that `aligned = perm[source]` best aligns to `target`.
    """
    # Build contingency C[a, b] = #positions with source==a and target==b
    C = np.zeros((K, K), dtype=int)
    for a in range(K):
        mask = (source == a)
        if mask.any():
            C[a] = np.bincount(target[mask], minlength=K)
    row_ind, col_ind = linear_sum_assignment(-C)  # maximize matches
    perm = np.zeros(K, dtype=int)
    perm[row_ind] = col_ind
    return perm


def compute_z_accuracy(
    z_history: np.ndarray,
    z_ground_truth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-iteration state accuracy after optimal label permutation.

    Parameters
    ----------
    z_history : array-like, shape (N, T)
        Discrete state sequences over iterations (N) and time (T).
    z_ground_truth : array-like, shape (T,)
        Ground-truth sequence of length T.

    Returns
    -------
    accuracies : np.ndarray, shape (N,)
        Fraction of matching labels in [0, 1] for each iteration.
    last_perm : np.ndarray, shape (K,)
        The optimal permutation for the *last* iteration row. Returned for
        convenience/debugging; callers often only need `accuracies`.
    """
    z_hist = np.asarray(z_history)
    ref = np.asarray(z_ground_truth)
    if z_hist.ndim != 2:
        raise ValueError("z_history must be 2D (N, T).")
    N, T = z_hist.shape
    if ref.shape != (T,):
        raise ValueError(f"z_ground_truth must have shape ({T},), got {ref.shape}.")

    K = int(max(z_hist.max(), ref.max())) + 1
    accuracies = np.zeros(N, dtype=float)
    last_perm = np.arange(K, dtype=int)

    for i in range(N):
        row = z_hist[i]
        perm = _best_permutation(row, ref, K)
        aligned = perm[row]
        accuracies[i] = (aligned == ref).mean()
        last_perm = perm  # keep the final one

    return accuracies, last_perm



# ----------------------------- Common inner routine ---------------------------

def _simulate_one(
    seed: int,
    K: int,
    d: int,
    T: int,
    prior: Optional[HMMPriorParams] = None,
) -> Tuple[np.ndarray, np.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate a single HMM sequence and return observations and latent states.

    Returns
    -------
    obs : np.ndarray, shape (T, d)
    zs  : np.ndarray, shape (T,)
    log_trans_mat : jnp.ndarray, shape (K, K)
    phis : jnp.ndarray, shape (K, d, d, 2)
    """
    prior = prior or HMMPriorParams()
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)

    trans_mat = sample_sticky_transition_matrix(
        key1, K, prior.alpha_self, prior.alpha_other
    )
    log_trans_mat = jnp.log(trans_mat)

    phis = sample_phis(
        key2, K, d, prior.phi_prec_tril, prior.phi_prec_diag, prior.phi_prec_triu
    )  # (K, d, d, 2)

    obs, zs = sample_hmm(key3, T, log_trans_mat, phis)  # (T, d), (T,)
    return obs, zs, log_trans_mat, phis


def _fit_and_accuracy(
    obs: np.ndarray,
    zs_true: np.ndarray,
    K: int,
    prior: HMMPriorParams,
    phi_solve_mode: str,
) -> Tuple[float, Dict]:
    """
    Fit the HMM and return (final_accuracy, info_dict from fit).
    """
    info = fit_hmm_em(
        obs,
        K,
        prior,
        phi_solve_mode=phi_solve_mode,
        **EM_KWARGS,
    )
    accs, _ = compute_z_accuracy(info["z_history"], zs_true)
    return float(accs[-1]), info


def test_d_and_k(seed: int = 42, phi_solve_mode: str = "sm",replicates= 1, dim_override = None,ks_override = None) -> None:
    """
    Sweep over dimensions and numbers of states.

    Save a JSON file mapping "d=*" -> {"K=*": final_accuracy}.

    This script produces the synthetic HMM heatmap data for the paper.
    """

    # Build grids
    if not dim_override:
        dimension_list = D_GRID_VALS
    else:
        dimension_list = np.array(dim_override)

    if not ks_override:
        k_list = K_GRID_VALS
    else:
        k_list = np.array(ks_override)


    out_json = os.path.join(
        ROOT,
        "data",
        "synthetic_hmm",
        f"accuracies_{phi_solve_mode}_{seed}.json",
    )
    print("Output JSON:", out_json)
    os.makedirs(os.path.split(out_json)[0], exist_ok=True)

    #get the accuracy and std over replciates
    accuracies_mat: Dict[str, Dict[str,  Dict[str, float]]] = {}
    for d in dimension_list:
        accuracies_mat[f"d={d}"] = {}
        for K in k_list:

            # accumulators for replicates
            acc_list = []

            for rep in range(replicates):
                start_time = time.perf_counter()

                prior = HMMPriorParams()
                obs, zs, _, _ = _simulate_one(seed, K, d, T, prior)

                print("Fitting HMM...")
                acc, _ = _fit_and_accuracy(
                    obs,
                    zs,
                    K,
                    prior,
                    phi_solve_mode,
                )
                
                acc_list.append(acc)
                
                elapsed = time.perf_counter() - start_time
                print(f"d: {d}, K: {K}, Elapsed time: {elapsed:.4f} s")

            accuracies_mat[f"d={d}"][f"K={K}"] = {"mean":np.mean(acc_list), "std":np.std(acc_list)}

            # Update file incrementally
            with open(out_json, "w") as f:
                json.dump(accuracies_mat, f, indent=4)



def _run_cli():
    parser = argparse.ArgumentParser(
        description="Run sythetic TG-HMM experiments."
    )

    parser.add_argument("--seed",type=int,default=42,help="Random seed.")

    parser.add_argument("--phi_solve_mode",type=str,choices=["sm", "ssm"],default="sm",help="Solve mode: 'sm' for score matching, 'ssm' for stochastic score matching")

    parser.add_argument("--replicates", type=int, default=1, help="Replicates per (d, n) cell.")

    #allows you to input a specified list of integers for the dims or ns list that would not be possible with an exponential series
    #syntax is --dim_override 17 34 51
    parser.add_argument('--dim_override', nargs='+', type=int)
    parser.add_argument('--ks_override', nargs='+', type=int)

    args = parser.parse_args()
    test_d_and_k(seed=args.seed, phi_solve_mode=args.phi_solve_mode, replicates = args.replicates,dim_override=args.dim_override,ks_override=args.ks_override)
    


if __name__ == "__main__":
    _run_cli()
