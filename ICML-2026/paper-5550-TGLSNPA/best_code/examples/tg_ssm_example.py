"""
Fit a Torus Graph via stochastic score matching.

"""
__date__ = "March 2025"


import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import plot_stats
from src.sample import sample_torus_graph, get_avg_stats
from src.ssm import estimate_params_ssm
from src.stats import solve_tg_exact
from src.von_mises import expanded_complex_nu

SAMPLE_KWARGS = dict(
    initial_position=None,
    step_size=3e-2,
    num_integration_steps=60,
    mode="hmc",
)

def get_random_phi(d, seed, p=0.5):
    np.random.seed(seed)
    phi = np.zeros((d, d, 2))
    phi = np.random.randn(d, d, 2)
    mask = np.random.choice(np.arange(2), phi.shape, p=[1-p, p])
    mask[np.arange(d), np.arange(d)] = 1.0
    phi *= mask
    return jnp.array(phi)


if __name__ == '__main__':
    d = 4
    n_samples = 50000
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)

    # Make a random phi.
    phi = get_random_phi(d, 42) # [d,d,2]

    # Draw samples via MCMC.
    X = sample_torus_graph(
        key1,
        n_samples,
        phi,
        initial_position=None,
        step_size=3e-2,
        num_integration_steps=60,
        mode="hmc",
    ) # [n,d]

    # Estimate phi via score matching.
    sm_phi = solve_tg_exact(X)

    # Estimate phi via stochastic_score matching.
    ssm_phi, _ = estimate_params_ssm(
        key2,
        X,
        H_hat=None,
        phi=None,
        batch_size=512,
        n_iter=2000,
        alpha=0.99,
        opt_state=None,
        l2_reg=1e-2,
        l1_reg=0.0,
        replace=True,
        lr=1e-2,
    ) # [d,d,2]

    # Get empirical circular stats.
    true_avg_stats = get_avg_stats(X) # [d,d,2]
    
    # Get SM circular stats.
    X_hat = sample_torus_graph(
        key1,
        n_samples,
        sm_phi,
        **SAMPLE_KWARGS,
    ) # [n,d]
    sm_avg_stats = get_avg_stats(X_hat) # [d,d,2]

    # Get SSM circular stats.
    X_hat = sample_torus_graph(
        key1,
        n_samples,
        ssm_phi,
        **SAMPLE_KWARGS,
    ) # [n,d]
    ssm_avg_stats = get_avg_stats(X_hat) # [d,d,2]

    # Plot.
    fig, axarr = plt.subplots(ncols=3, nrows=2)

    axarr[0,0].set_title(r"True $\phi$")
    plot_stats(expanded_complex_nu(phi), ax=axarr[0,0])

    axarr[0,1].set_title(r"SM $\phi$")
    plot_stats(expanded_complex_nu(sm_phi), ax=axarr[0,1])

    axarr[0,2].set_title(r"SSM $\phi$")
    plot_stats(expanded_complex_nu(ssm_phi), ax=axarr[0,2])

    axarr[1,0].set_title(r"True $S$")
    plot_stats(true_avg_stats, ax=axarr[1,0])

    axarr[1,1].set_title(r"SM $S$")
    plot_stats(sm_avg_stats, ax=axarr[1,1])

    axarr[1,2].set_title(r"SSM $S$")
    plot_stats(ssm_avg_stats, ax=axarr[1,2])

    for ax in axarr.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("tg_smm_example.png")
