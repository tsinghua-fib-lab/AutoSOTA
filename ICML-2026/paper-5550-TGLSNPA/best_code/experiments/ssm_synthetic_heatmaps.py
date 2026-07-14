"""
Generate phi-recovery grids for SGD heatmaps.

For a grid of (dimension d) x (number of samples n):
1) sample from a Torus Graph with random phi,
2) Fit via stocahstic score matching (SSM),
3) solve via the exact TG method,
4) record MSE and pseudo-R² against the ground-truth phi.

Outputs:
- data/synthetic_tg/r2_*.json
"""
__date__ = "March - September 2025"


import argparse
import os
from typing import List

import jax.numpy as jnp
import jax.random as jr
import json
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sample import sample_torus_graph
from src.ssm import estimate_params_ssm
from src.stats import solve_tg_exact


# ------------------------------- Utilities ------------------------------------

def _exp_sequence(base: float, start: int, stop: int) -> np.ndarray:
    """
    Deterministic integer sequence: floor(base**k) for k in [start, stop).
    Ensures strictly positive, unique, ascending values.
    """
    seq = [int(base ** k) for k in range(start, stop)]
    # remove duplicates that can appear at small k
    seq = np.array(sorted(set(max(1, v) for v in seq)))
    return seq


# --------------------------- Random φ construction ----------------------------

def get_random_phi(
    d: int,
    seed: int,
    *,
    lower_triangular: bool = False,
    p: float = 0.5,
) -> jnp.ndarray:
    """
    Draw a random TG parameter φ ∈ R^{d×d×2} with optional sparsity and structure.

    Parameters
    ----------
    d : int
        Number of phase variables (matrix side length).
    seed : int
        RNG seed (NumPy; matches original behavior).
    lower_triangular : bool, default=False
        If True, zero out strict upper triangle for each of the 2 parameter slices.
    p : float in [0,1], default=0.5
        Entry-wise keep probability for a Bernoulli mask (diag always kept).

    Returns
    -------
    phi : jnp.ndarray, shape (d, d, 2)
    """
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((d, d, 2))

    # Bernoulli mask with P(keep=1)=p, but always keep diagonal
    mask = rng.choice([0, 1], size=phi.shape, p=[1 - p, p]).astype(phi.dtype)
    mask[np.arange(d), np.arange(d)] = 1.0
    phi *= mask

    if lower_triangular:
        print("φ is lower triangular")
        for i in range(d):
            for j in range(i + 1, d):
                phi[i, j] = 0.0

    return jnp.array(phi)


# --------------------------------- Runner -------------------------------------

def run_grid(
    *,
    dim_base: float = 1.3,
    dim_exp_start: int = 4,
    dim_exp_stop: int = 28,
    ns_base: float = 1.5,
    ns_exp_start: int = 6,
    ns_exp_stop: int = 30,
    replicates: int = 1,
    iterations: int = 5000,
    seed: int = 42,
    dims_override = None,
    ns_override = None
) -> None:
    """
    Main experiment loop generating (d x n) grids and saving MSE/R^2 arrays.

    Notes
    -----
    - Exact solves are attempted until an exception occurs, after which we
      mark remaining entries as ±inf.
    """
    out_json = os.path.join(
        ROOT,
        "data",
        "synthetic_tg",
        f"mse_r2_{iterations}_{replicates}_{seed}.json",
    )
    print("Output JSONs:", out_json)
    os.makedirs(os.path.split(out_json)[0], exist_ok=True)

    # Build grids
    if not dims_override:
        dimension_list = _exp_sequence(dim_base, dim_exp_start, dim_exp_stop)
    else:
        dimension_list = np.array(dims_override)

    if not ns_override:
        n_samples_list = _exp_sequence(ns_base, ns_exp_start, ns_exp_stop)
    else:
        n_samples_list = np.array(ns_override)

    print("Dimensions:", dimension_list.tolist())
    print("Samples:", n_samples_list.tolist())

    # Experiment settings
    key = jr.PRNGKey(17)
    exact_solver_oom = False  # emulate original "oom" behavior

    errors: List[List[float]] = []
    r2s: List[List[float]] = []
    progress = 0
    total = len(dimension_list) * len(n_samples_list) * replicates

    # results[d][n][model][mode] = float, where model in ['sm', 'ssm'], mode in ['err', 'r2', 'err_sd', 'r2_sd']
    results_json: Dict[int, Dict[int, Dict[str, Dict[str, float]]]] = {}

    for d in dimension_list:
        results_json[f"d={d}"] = {}
        
        for n in n_samples_list:    
            results_json[f"d={d}"][f"n={n}"] = {}
            
            # accumulators for replicates
            sm_err_sum = []
            sm_r2_sum = []
            ssm_err_sum = []
            ssm_r2_sum = []

            for rep in range(replicates):
                progress += 1
                print(f"{progress} of {total} :: d={d}, n={n}, rep={rep+1}/{replicates}")

                key, key1, key2 = jr.split(key, 3)

                # 1) sample a random φ (ground truth)
                phi_true = get_random_phi(d, seed=d * rep * 100, lower_triangular=False, p=0.5)  # (d,d,2)

                # 2) draw samples via HMC
                X = sample_torus_graph(
                    key1,
                    n,
                    phi_true,
                    initial_position=None,
                    step_size=3e-2,
                    num_integration_steps=60,
                    mode="hmc",
                )  # (n, d)

                # 3) exact TG solve (with OOM/exception guard)
                if not exact_solver_oom:
                    try:
                        phi_exact = solve_tg_exact(X)
                        sm_err_sum.append(float(np.mean(np.square(phi_true - phi_exact))))
                        sm_r2_sum.append(np.corrcoef(phi_true.flatten(),phi_exact.flatten())[0][1]**2)

                    except Exception as e:
                        print("Exact solver failed; marking subsequent entries as ±inf:", e)
                        exact_solver_oom = Truef
                        sm_err_sum.append(float(np.inf))
                        sm_r2_sum.append(float(-np.inf))
                else:
                    sm_err_sum.append(float(np.inf))
                    sm_r2_sum.append(float(-np.inf))

                # 4) SSM fit
                phi_hat, _ = estimate_params_ssm(
                    key2,
                    X,
                    phi=None,
                    batch_size=512,
                    n_iter=2000,
                    alpha=0.99,
                    opt_state=None,
                    l2_reg=0.0,
                    l1_reg=0.01,
                    replace=True,
                    lr=3e-3,
                )  # (d,d,2)

                ssm_err_sum.append(float(np.mean(np.square(phi_true - phi_hat))))
                ssm_r2_sum.append(np.corrcoef(phi_true.flatten(),phi_hat.flatten())[0][1]**2)

            # Record per-cell averages across replicates
            results_json[f"d={d}"][f"n={n}"] = dict(
                sm=dict(
                    err = np.mean(sm_err_sum),
                    r2  = np.mean(sm_r2_sum),
                    err_sd = np.std(sm_err_sum),
                    r2_sd  = np.std(sm_r2_sum),
                ),
                ssm=dict(
                    err = np.mean(ssm_err_sum),
                    r2  = np.mean(ssm_r2_sum),
                    err_sd = np.std(ssm_err_sum),
                    r2_sd  = np.std(ssm_r2_sum),
                ),
            )

            # Save incrementally.
            with open(out_json, "w") as f:
                json.dump(results_json, f, indent=4)
            


def _run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate phi-recovery grids (exact TG solve; SGD fits remain commented)."
    )
    parser.add_argument("--dim-base", type=float, default=1.3, help="Base for dimension growth (default: 1.3).")
    parser.add_argument("--dim-start", type=int, default=4, help="Start exponent for dimensions (inclusive).")
    parser.add_argument("--dim-stop", type=int, default=28, help="Stop exponent for dimensions (exclusive).")
    parser.add_argument("--ns-base", type=float, default=1.5, help="Base for sample growth (default: 1.5).")
    parser.add_argument("--ns-start", type=int, default=6, help="Start exponent for samples (inclusive).")
    parser.add_argument("--ns-stop", type=int, default=30, help="Stop exponent for samples (exclusive).")

    parser.add_argument("--replicates", type=int, default=1, help="Replicates per (d, n) cell.")
    parser.add_argument("--iterations", type=int, default=5000, help="(Reserved) SGD iterations; currently unused.")
    parser.add_argument("--seed", type=int, default=42, help="Master PRNG seed.")

    #allows you to input a specified list of integers for the dims or ns list that would not be possible with an exponential series
    #syntax is --dim_override 17 34 51
    parser.add_argument('--dim_override', nargs='+', type=int)
    parser.add_argument('--ns_override', nargs='+', type=int)

    args = parser.parse_args()
    run_grid(
        dim_base=args.dim_base,
        dim_exp_start=args.dim_start,
        dim_exp_stop=args.dim_stop,
        ns_base=args.ns_base,
        ns_exp_start=args.ns_start,
        ns_exp_stop=args.ns_stop,
        replicates=args.replicates,
        iterations=args.iterations,
        seed=args.seed,
        dims_override =args.dim_override,
        ns_override = args.ns_override
    )



if __name__ == "__main__":
    _run_cli()
