"""
Utilities for evaluating latent state inference in HMMs

"""
__date__ = "May - July 2025"


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from src.plots import plot_stats
from src.stats import solve_tg_exact
from src.von_mises import expanded_complex_nu

batch_tg_solve = jax.vmap(solve_tg_exact, in_axes=(None,0,None), out_axes=0)



def empirical_transition_matrix(zs: jnp.ndarray, K: int) -> jnp.ndarray:
    """
    Compute the empirical transition matrix from a discrete state sequence, without one-hot.

    Args:
        zs: int array of shape (T,), values in [0, K)
        K:  number of states

    Returns:
        A (K, K) array where A[i, j] ≈ P(z_{t+1}=j | z_t=i), row‑normalized.
    """
    # drop last/first to form (z_t, z_{t+1}) pairs
    z0 = zs[:-1]
    z1 = zs[1:]
    # flatten (i,j) into single index in [0, K*K)
    idx = z0 * K + z1                            # shape (T-1,)

    # count occurrences of each (i,j) pair
    counts_flat = jnp.bincount(idx, length=K*K)   # shape (K*K,)
    counts = counts_flat.reshape((K, K)).astype(jnp.float32)

    # normalize each row, guarding against zero counts
    row_sums = counts.sum(axis=1, keepdims=True) # shape (K,1)
    return jnp.where(row_sums > 0, counts / row_sums, 0.0)


def plot_hmm_states_with_gt(obs, zs, T, K, info, perm, fn="temp2.png"):
    # Get empirical transition matrix and phi.
    emp_trans_mat = empirical_transition_matrix(zs, K)

    gamma = jnp.zeros((T,K))
    for i in range(T):
        gamma = gamma.at[i,zs[i]].set(1.0)
    emp_phis = batch_tg_solve(obs, gamma.T, 1e-1) # (K,d,d,2)

    fig, axarr = plt.subplots(ncols=K+1, nrows=2)
    pred_trans_mat = jnp.exp(info["log_trans"])[perm][:,perm]
    pred_phis = info["phis"][perm]

    axarr[0,0].set_ylabel("Empirical")
    axarr[1,0].set_ylabel("Inferred")

    axarr[0,0].imshow(emp_trans_mat, vmin=0, vmax=1)
    axarr[1,0].imshow(pred_trans_mat, vmin=0, vmax=1)
    for i in range(K):
        plot_stats(expanded_complex_nu(emp_phis[i]), ax=axarr[0,i+1])
        plot_stats(expanded_complex_nu(pred_phis[i]), ax=axarr[1,i+1])

    for ax in axarr.flatten():
        plt.sca(ax)
        plt.xticks([], [])
        plt.yticks([], [])

    plt.tight_layout()
    plt.savefig(fn)
    plt.close("all")


def plot_hmm_states_no_gt(K, info, fn="temp2.png"):
    fig, axarr = plt.subplots(ncols=K+1, nrows=2)
    pred_trans_mat = jnp.exp(info["log_trans"])
    pred_phis = info["phis"]
    timepoints = np.arange(0,info["z_prob"].shape[1])

    #plot the different phi matricies, and their distance from the average
    axarr[0,0].set_ylabel("Inferred")
    axarr[0,0].imshow(pred_trans_mat, vmin=0, vmax=jnp.max(pred_trans_mat))
    for i in range(K):
        plot_stats(expanded_complex_nu(pred_phis[i]), ax=axarr[0,i+1])

    avg_phi = pred_phis[0]
    for i in range(1,len(pred_phis)):
        avg_phi += pred_phis[i]
    avg_phi /= len(pred_phis)
    avg_nu_phi = expanded_complex_nu(avg_phi)

    axarr[1,0].set_ylabel("Difference from Average Phi")
    plot_stats(avg_phi, ax=axarr[1,0], r_max=5.0)

    for i in range(K):
        diff = expanded_complex_nu(pred_phis[i]) - avg_nu_phi
        plot_stats(diff, ax=axarr[1,i+1], r_max=1.0)

    for ax in axarr.flatten():
        plt.sca(ax)
        plt.xticks([], [])
        plt.yticks([], [])

    plt.tight_layout()
    plt.savefig(fn)
    plt.close("all")


def plot_z_history(z_history, z_ground_truth=None, K=None, cmap='Set1', fn="temp.pdf"):
    """
    Plot the evolution of discrete states across EM iterations.

    Parameters
    ----------
    z_history : array-like, shape (N, T)
        Integer labels in [0, K) for each of N EM iterations over T time steps.
    z_ground_truth : array-like, shape (T,), optional
        True state sequence to align to and plot below the EM runs.
    cmap : str or Colormap, optional
        A matplotlib colormap with at least K distinct colors (default 'Set1').
    """
    z_history = np.asarray(z_history)
    N, T = z_history.shape
    if K is None:
        if z_ground_truth is None:
            K = int(z_history.max()) + 1
        else:
            K = int(max(z_history.max(), z_ground_truth.max())) + 1

    # Choose reference sequence
    if z_ground_truth is not None:
        ref = np.asarray(z_ground_truth)
        if ref.shape[0] != T:
            raise ValueError(f"z_ground_truth must have length T={T}, got {ref.shape[0]}")
    else:
        ref = z_history[-1]

    # Align each iteration to the reference
    aligned = np.zeros_like(z_history)
    for i in range(N):
        row = z_history[i]
        # build contingency: C[a, b] = #positions where row==a and ref==b
        C = np.zeros((K, K), dtype=int)
        for a in range(K):
            mask = (row == a)
            if mask.any():
                counts = np.bincount(ref[mask], minlength=K)
                C[a, :] = counts
        # Hungarian to maximize matches == minimize -C
        row_ind, col_ind = linear_sum_assignment(-C)
        perm = np.zeros(K, dtype=int)
        perm[row_ind] = col_ind
        aligned[i] = perm[row]

    # Build display array, adding GT row if given
    if z_ground_truth is not None:
        display = np.vstack([aligned, ref, ref, ref])
        ylabels = [f"Iter {i+1}" for i in range(N)] + ["", "Ground truth", ""]
    else:
        display = aligned
        ylabels = [f"Iter {i+1}" for i in range(N)]

    # Plot
    _, ax = plt.subplots(figsize=(12, 2 + 0.3 * display.shape[0]))
    ax.imshow(
        display,
        aspect='auto',
        interpolation='nearest',
        origin='upper',
        cmap=cmap,
        alpha=0.8,
    )
    ax.set_xlabel("Time step")
    ax.set_yticks(np.arange(display.shape[0]))
    ax.set_yticklabels(ylabels)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close("all")


def compute_z_accuracy(z_history, z_ground_truth, K=None):
    """
    Compute the per-iteration accuracy of discrete state labels,
    after optimal label alignment via the linear-sum-assignment.

    Parameters
    ----------
    z_history : array-like, shape (N, T)
        Integer labels in [0, K) for each of N EM iterations over T time steps.
    z_ground_truth : array-like, shape (T,)
        True state sequence to align to.
    K : int, optional

    Returns
    -------
    accuracies : np.ndarray, shape (N,)
        Fraction of matching labels (in [0, 1]) for each iteration.
    perm : np.ndarray, shape (K,)
        Permutation for last iteration to align predicted to ground truth labels
    """
    z_history = np.asarray(z_history)
    N, T = z_history.shape
    if K is None:
        K = int(max(z_history.max(), z_ground_truth.max())) + 1

    ref = np.asarray(z_ground_truth)
    if ref.shape[0] != T:
        raise ValueError(f"z_ground_truth must have length T={T}, got {ref.shape[0]}")

    accuracies = np.zeros(N, dtype=float)

    # for each iteration, find best label mapping and compute accuracy
    for i in range(N):
        row = z_history[i]
        # build contingency matrix C[a, b] = count of positions where row==a and ref==b
        C = np.zeros((K, K), dtype=int)
        for a in range(K):
            mask = (row == a)
            if mask.any():
                counts = np.bincount(ref[mask], minlength=K)
                C[a, :] = counts
        # solve assignment to maximize matches (minimize -C)
        row_ind, col_ind = linear_sum_assignment(-C)
        perm = np.zeros(K, dtype=int)
        perm[row_ind] = col_ind

        # apply permutation and compute accuracy
        aligned = perm[row]
        accuracies[i] = np.mean(aligned == ref)

    perm = np.zeros(K, dtype=int)
    perm[col_ind] = row_ind

    return accuracies, perm
