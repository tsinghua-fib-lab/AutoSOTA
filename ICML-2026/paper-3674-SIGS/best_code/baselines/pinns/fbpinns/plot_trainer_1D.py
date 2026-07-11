"""
Defines plotting functions for 1D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.tree_util import tree_map

from fbpinns.util.other import colors

def _lim(v, factor=1.1):
    mi, ma = v.min(0), v.max(0)
    c = (mi+ma)/2
    w = factor*(ma-mi)/2
    return (c-w, c+w)

def _plot_setup(x_batch_test, u_exact):
    # get general setup for plotting
    xlim, ulim = _lim(x_batch_test), _lim(u_exact)
    return xlim, ulim

def _to_numpy(f):
    # converts jnp arrays to np arrays
    def wrapper(*args):
        args = tree_map(lambda a: np.array(a) if isinstance(a, jnp.ndarray) else a, args)
        return f(*args)
    return wrapper

@_to_numpy
def plot_1D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)

    f = plt.figure(figsize=(8,4*10/3))

    # plot domain + x_batch
    plt.subplot(4,1,1)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], 0.1*np.ones_like(x_batch)[:,0], alpha=0.5, color="k", s=40)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(*xlim)

    plt.subplot(4,1,2)
    plt.title(f"[{i}] POU window functions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], ws_test[im,:,0], color=colors[im])
    plt.xlim(*xlim)

    # plot full + individual solutions
    plt.subplot(4,1,3)
    plt.title(f"[{i}] Full and individual solutions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], us_test[im,:,0], color=colors[im])
    plt.plot(x_batch_test[:,0], u_exact[:,0], lw=4, color="tab:grey", label="Ground truth")
    plt.plot(x_batch_test[:,0], u_test[:,0], color="k", label="FBPINN")
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ulim)

    # plot raw solutions
    plt.subplot(4,1,4)
    plt.title(f"[{i}] Raw solutions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], us_raw_test[im,:,0], color=colors[im])
    plt.xlim(*xlim)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_1D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,1,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], 0.1*np.ones_like(x_batch)[:,0], alpha=0.5, color="k", s=40)
    plt.xlim(*xlim)

    # plot full solution
    plt.subplot(3,1,2)
    plt.title(f"[{i}] Full solution")
    plt.plot(x_batch_test[:,0], u_exact[:,0], lw=4, color="tab:grey", label="Ground truth")
    plt.plot(x_batch_test[:,0], u_test[:,0], color="k", label="PINN")
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ulim)

    # plot raw solution
    plt.subplot(3,1,3)
    plt.title(f"[{i}] Raw solution")
    plt.plot(x_batch_test[:,0], u_raw_test[:,0], color="k")
    plt.xlim(*xlim)

    plt.tight_layout()

    return (("test",f),)