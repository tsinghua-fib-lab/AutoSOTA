"""
HMM example

"""
__date__ = "April 2025"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
from src.hmm_eval import (
    empirical_transition_matrix,
    plot_z_history,
    compute_z_accuracy,
    plot_hmm_states_with_gt,
)
from src.plots import plot_stats
from src.stats import solve_tg_exact

batch_tg_solve = jax.vmap(solve_tg_exact, in_axes=(None,0,None), out_axes=0)



if __name__ == '__main__':
    K = 2
    d = 5
    T = 800

    prior = HMMPriorParams(alpha_self=10.0, alpha_other=2.0)

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    trans_mat = sample_sticky_transition_matrix(
        key1,
        K,
        prior.alpha_self,
        prior.alpha_other,
    )
    log_trans_mat = jnp.log(trans_mat)
    
    phis = sample_phis(
        key2,
        K,
        d,
        prior.phi_prec_tril,
        prior.phi_prec_diag,
        prior.phi_prec_triu,
    ) # [K,d,d,2]

    print("Sampling HMM...")
    obs, zs = sample_hmm(key3, T, log_trans_mat, phis) # (T,d), (T,)
    
    values, counts = jnp.unique_counts(zs)
    print("True marginal state occupancy:", counts / T)

    print("Fitting HMM...")
    info = fit_hmm_em(
        obs,
        K,
        prior,
        num_em_iters=2, # 15
        warmup_iterations=1, # 8
        num_part_opt_steps=250,
        lr=1e-2,
        cross_entropy_lambda=3e-2,
        phi_solve_mode="minibatch_score_matching",
    )

    values, counts = jnp.unique_counts(info["z_history"][-1])
    print("Estimated marginal state occupancy:", counts / T)

    accuracies, perm = compute_z_accuracy(info["z_history"], zs, K=K)
    print("State accuracies:", accuracies)

    plot_z_history(info["z_history"], z_ground_truth=zs, K=K, fn="temp.png")

    plot_hmm_states_with_gt(obs, zs, T, K, info, perm, fn="temp2.png")
