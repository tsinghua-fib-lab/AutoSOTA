# -*- coding: utf-8 -*-
"""
Plotting utilities for MILCCI.

All functions take a save_path argument. If provided, figures are saved
there via save_fig. Requires matplotlib and seaborn.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_fig(name_fig, fig, save_path='', formats=['png', 'svg'],
             save_params={}, verbose=True):
    """Save figure to disk."""
    if len(save_path) == 0:
        return
    if 'transparent' not in save_params:
        save_params['transparent'] = True
    os.makedirs(save_path, exist_ok=True)
    for fmt in formats:
        fig.savefig(save_path + os.sep + '%s.%s' % (name_fig, fmt),
                    **save_params)
    if verbose:
        print('saved figure: %s' % (save_path + os.sep + '%s.%s' % (name_fig, 'png')))


def remove_edges(ax, top=False, right=False, bottom=True, left=True):
    """Remove spines from axes."""
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)


# ------------------------------------------------------------------ #
#  1. Spatial maps (A)
# ------------------------------------------------------------------ #

def plot_A_heatmaps(A, labels_unique_order, numbers2tuples,
                    class_names=[], n_ensembles_each=[],
                    save_path='', figname_prefix='A_heatmaps'):
    """
    Plot heatmaps of spatial maps A for each unique condition.

    Parameters
    ----------
    A : np.ndarray, shape (N, P, K)
        Spatial maps per unique condition.
    labels_unique_order : array-like
        Unique label keys in order.
    numbers2tuples : dict
    class_names : list of str
    n_ensembles_each : list of int
    save_path : str
    """
    N, P, K = A.shape
    assert K == len(labels_unique_order), (
        'A conditions %d != labels_unique %d' % (K, len(labels_unique_order))
    )

    n_cols = min(K, 6)
    n_rows = int(np.ceil(K / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))
    if K == 1:
        axs = np.array([axs])
    axs_flat = np.array(axs).flatten()

    vmin = np.percentile(A, 2)
    vmax = np.percentile(A, 98)

    for c in range(K):
        ax = axs_flat[c]
        sns.heatmap(A[:, :, c], ax=ax, cmap='RdBu_r', center=0,
                    vmin=vmin, vmax=vmax, rasterized=True,
                    cbar=c == K - 1)
        tup = numbers2tuples[labels_unique_order[c]]
        if len(class_names) > 0:
            title_str = ', '.join(['%s=%s' % (cn, str(tv)) for cn, tv in zip(class_names, tup)])
        else:
            title_str = str(tup)
        ax.set_title(title_str, fontsize=8)
        ax.set_xlabel('Ensemble')
        if c % n_cols == 0:
            ax.set_ylabel('Neuron')

        # mark ensemble boundaries
        if len(n_ensembles_each) > 0:
            cumsum = np.cumsum(n_ensembles_each)
            for boundary in cumsum[:-1]:
                ax.axvline(x=boundary, color='k', lw=1, ls='--', alpha=0.5)

    for c in range(K, len(axs_flat)):
        axs_flat[c].axis('off')

    fig.suptitle(r'Spatial maps $A$ per condition', fontsize=12)
    plt.tight_layout()
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig


def plot_A_similarity_matrix(A, labels_unique_order, numbers2tuples,
                             n_ensembles_each=[], class_names=[],
                             save_path='', figname_prefix='A_similarity'):
    """
    Plot pairwise correlation of A between conditions, per axis-group.

    Parameters
    ----------
    A : np.ndarray, shape (N, P, K)
    labels_unique_order : array-like
    numbers2tuples : dict
    n_ensembles_each : list of int
    class_names : list of str
    save_path : str
    """
    K = A.shape[2]
    num_axes = len(n_ensembles_each) if len(n_ensembles_each) > 0 else 1
    cumsum = np.cumsum([0] + list(n_ensembles_each))

    fig, axs = plt.subplots(1, num_axes + 1, figsize=(5 * (num_axes + 1), 4))
    if num_axes + 1 == 1:
        axs = [axs]

    tick_labels = [str(numbers2tuples[lab]) for lab in labels_unique_order]

    # full A similarity
    corr_full = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            corr_full[i, j] = np.corrcoef(A[:, :, i].flatten(), A[:, :, j].flatten())[0, 1]
    sns.heatmap(corr_full, ax=axs[0], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=tick_labels, yticklabels=tick_labels, rasterized=True,
                annot=True, fmt='.2f', annot_kws={'fontsize': 6})
    axs[0].set_title(r'Full $A$ similarity')

    # per-axis
    for ax_idx in range(num_axes):
        e1, e2 = cumsum[ax_idx], cumsum[ax_idx + 1]
        corr_ax = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                corr_ax[i, j] = np.corrcoef(
                    A[:, e1:e2, i].flatten(), A[:, e1:e2, j].flatten()
                )[0, 1]
        ax_name = class_names[ax_idx] if ax_idx < len(class_names) else 'axis_%d' % ax_idx
        sns.heatmap(corr_ax, ax=axs[ax_idx + 1], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    xticklabels=tick_labels, yticklabels=tick_labels, rasterized=True,
                    annot=True, fmt='.2f', annot_kws={'fontsize': 6})
        axs[ax_idx + 1].set_title(r'$A$ similarity: %s ensembles' % ax_name)

    plt.tight_layout()
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig


# ------------------------------------------------------------------ #
#  2. Temporal traces (Phi)
# ------------------------------------------------------------------ #

def plot_phi_traces(Phi, labels, labels_unique_order, numbers2tuples,
                    n_ensembles_each=[], class_names=[],
                    max_trials_per_condition=3,
                    save_path='', figname_prefix='Phi_traces'):
    """
    Plot temporal traces Phi for each ensemble, grouped by condition.

    Parameters
    ----------
    Phi : np.ndarray, shape (T, P, M)
    labels : list or array
    labels_unique_order : array-like
    numbers2tuples : dict
    n_ensembles_each : list of int
    class_names : list of str
    max_trials_per_condition : int
    save_path : str
    """
    T, P, M = Phi.shape
    K = len(labels_unique_order)

    color_list_conds = plt.cm.tab10(np.linspace(0, 1, K))

    fig, axs = plt.subplots(P, 1, figsize=(10, 2.5 * P), sharex=True)
    if P == 1:
        axs = [axs]

    labels_arr = np.array(labels)
    for p in range(P):
        ax = axs[p]
        for c, lab in enumerate(labels_unique_order):
            trial_idx = np.where(labels_arr == lab)[0][:max_trials_per_condition]
            for ti, trial in enumerate(trial_idx):
                ax.plot(Phi[:, p, trial], color=color_list_conds[c],
                        alpha=0.6, lw=1,
                        label=str(numbers2tuples[lab]) if ti == 0 else None)
        # which axis does this ensemble belong to?
        ens_label = ''
        if len(n_ensembles_each) > 0:
            cumsum = np.cumsum([0] + list(n_ensembles_each))
            for ax_idx in range(len(n_ensembles_each)):
                if cumsum[ax_idx] <= p < cumsum[ax_idx + 1]:
                    ax_name = class_names[ax_idx] if ax_idx < len(class_names) else 'axis_%d' % ax_idx
                    ens_label = ' (%s)' % ax_name
                    break
        ax.set_ylabel(r'$\Phi_{%d}$%s' % (p, ens_label))
        remove_edges(ax)
        if p == 0:
            ax.legend(fontsize=6, ncol=min(K, 5), loc='upper right')

    axs[-1].set_xlabel('Time bin')
    fig.suptitle(r'Temporal traces $\Phi$', fontsize=12)
    plt.tight_layout()
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig


# ------------------------------------------------------------------ #
#  3. Reconstruction quality
# ------------------------------------------------------------------ #

def plot_reconstruction(data, A_full, Phi, trial_indices=[],
                        save_path='', figname_prefix='reconstruction'):
    """
    Side-by-side heatmaps: original vs reconstruction for selected trials.

    Parameters
    ----------
    data : np.ndarray, shape (N, T, M)
    A_full : np.ndarray, shape (N, P, M)
    Phi : np.ndarray, shape (T, P, M)
    trial_indices : list of int
        Trials to show. Default: first 4.
    save_path : str
    """
    N, T, M = data.shape
    if len(trial_indices) == 0:
        trial_indices = list(range(min(4, M)))
    n_trials_show = len(trial_indices)

    fig, axs = plt.subplots(n_trials_show, 3, figsize=(12, 3 * n_trials_show))
    if n_trials_show == 1:
        axs = axs.reshape(1, -1)

    for row, trial in enumerate(trial_indices):
        y = data[:, :, trial]
        y_hat = A_full[:, :, trial] @ Phi[:, :, trial].T
        residual = y - y_hat

        vmin = min(np.percentile(y, 2), np.percentile(y_hat, 2))
        vmax = max(np.percentile(y, 98), np.percentile(y_hat, 98))

        sns.heatmap(y, ax=axs[row, 0], cmap='viridis', vmin=vmin, vmax=vmax,
                    rasterized=True, cbar=False)
        axs[row, 0].set_title('Original (trial %d)' % trial)

        sns.heatmap(y_hat, ax=axs[row, 1], cmap='viridis', vmin=vmin, vmax=vmax,
                    rasterized=True, cbar=False)
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_trial = 1 - ss_res / (ss_tot + 1e-18)
        axs[row, 1].set_title(r'Reconstruction ($R^2$=%.3f)' % r2_trial)

        res_lim = max(np.percentile(np.abs(residual), 98), 1e-6)
        sns.heatmap(residual, ax=axs[row, 2], cmap='RdBu_r', center=0,
                    vmin=-res_lim, vmax=res_lim, rasterized=True, cbar=False)
        axs[row, 2].set_title('Residual')

        for col in range(3):
            axs[row, col].set_ylabel('Neuron')
            axs[row, col].set_xlabel('Time')

    plt.tight_layout()
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig


def plot_r2_per_trial(r2_vec, labels=[], numbers2tuples={},
                      save_path='', figname_prefix='r2_per_trial'):
    """
    Bar plot of per-trial R^2.

    Parameters
    ----------
    r2_vec : np.ndarray, shape (M,)
    labels : list
    numbers2tuples : dict
    save_path : str
    """
    M = len(r2_vec)
    fig, ax = plt.subplots(figsize=(max(6, M * 0.5), 4))

    bar_color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    if len(labels) == M and len(numbers2tuples) > 0:
        unique_labels = list(dict.fromkeys(labels))
        trial_color_arr = [bar_color_list[unique_labels.index(lab) % 10] for lab in labels]
        tick_labels = [str(numbers2tuples[lab]) for lab in labels]
    else:
        trial_color_arr = [bar_color_list[0]] * M
        tick_labels = ['%d' % i for i in range(M)]

    ax.bar(range(M), r2_vec, color=trial_color_arr, edgecolor='k', lw=0.5)
    ax.set_xticks(range(M))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(r'$R^2$')
    ax.set_xlabel('Trial (condition)')
    ax.set_title(r'Per-trial $R^2$ (mean=%.3f)' % np.mean(r2_vec))
    ax.axhline(y=np.mean(r2_vec), color='r', ls='--', lw=1, alpha=0.7)
    remove_edges(ax)

    plt.tight_layout()
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig


# ------------------------------------------------------------------ #
#  4. Ground-truth comparison (synthetic only)
# ------------------------------------------------------------------ #

def plot_ground_truth_comparison(A_est, A_true, Phi_est, Phi_true,
                                 labels_unique_order,
                                 save_path='', figname_prefix='gt_comparison'):
    """
    Compare estimated A and Phi to ground truth (for synthetic data).

    Shows per-condition correlation between estimated and true A columns
    and per-trial correlation of Phi traces.

    Parameters
    ----------
    A_est : np.ndarray, shape (N, P, K)
    A_true : np.ndarray, shape (N, P, M) or (N, P, K)
    Phi_est : np.ndarray, shape (T, P, M)
    Phi_true : np.ndarray, shape (T, P, M)
    labels_unique_order : array-like
    save_path : str
    """
    K = A_est.shape[2]
    P = A_est.shape[1]
    M = Phi_est.shape[2]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # --- A correlation per condition ---
    # need to find best permutation of ensembles
    from scipy.optimize import linear_sum_assignment
    # use first condition to find permutation
    A_true_cond0 = A_true[:, :, 0] if A_true.shape[2] >= 1 else A_true[:, :, 0]
    A_est_cond0 = A_est[:, :, 0]
    cost = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            cost[i, j] = -np.abs(np.corrcoef(A_est_cond0[:, i], A_true_cond0[:, j])[0, 1])
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = col_ind  # perm[est_idx] = true_idx

    a_corr_per_cond = np.zeros((K, P))
    for c in range(K):
        a_est_c = A_est[:, :, c]
        # if A_true has M slices, use the c-th unique condition
        a_true_c = A_true[:, :, c] if A_true.shape[2] == K else A_true[:, :, c]
        for p in range(P):
            a_corr_per_cond[c, p] = np.abs(
                np.corrcoef(a_est_c[:, p], a_true_c[:, perm[p]])[0, 1]
            )

    sns.heatmap(a_corr_per_cond, ax=axs[0], cmap='Greens', vmin=0, vmax=1,
                annot=True, fmt='.2f', rasterized=True,
                xticklabels=['ens_%d' % p for p in range(P)],
                yticklabels=['cond_%d' % c for c in range(K)])
    axs[0].set_title(r'$|corr|$ est. vs true $A$ (permuted)')
    axs[0].set_xlabel('Ensemble')
    axs[0].set_ylabel('Condition')

    # --- Phi correlation per trial (average over ensembles) ---
    phi_corr_per_trial = np.zeros(M)
    for m in range(M):
        trial_corrs = []
        for p in range(P):
            c = np.abs(np.corrcoef(Phi_est[:, p, m], Phi_true[:, perm[p], m])[0, 1])
            trial_corrs.append(c)
        phi_corr_per_trial[m] = np.mean(trial_corrs)
    axs[1].bar(range(M), phi_corr_per_trial, color='steelblue', edgecolor='k', lw=0.5)
    axs[1].set_ylabel(r'Mean $|corr|$')
    axs[1].set_xlabel('Trial')
    axs[1].set_title(r'$\Phi$ est. vs true (mean=%.3f)' % np.mean(phi_corr_per_trial))
    axs[1].axhline(y=np.mean(phi_corr_per_trial), color='r', ls='--', lw=1)
    remove_edges(axs[1])

    # --- scatter: true vs estimated for one trial ---
    trial_show = 0
    y_true_flat = (A_true[:, :, trial_show] @ Phi_true[:, :, trial_show].T).flatten()
    y_est_flat = (A_est[:, :, 0] @ Phi_est[:, :, trial_show].T).flatten()
    axs[2].scatter(y_true_flat, y_est_flat, s=3, alpha=0.3, color='steelblue')
    lims = [min(y_true_flat.min(), y_est_flat.min()),
            max(y_true_flat.max(), y_est_flat.max())]
    axs[2].plot(lims, lims, 'k--', lw=1, alpha=0.5)
    axs[2].set_xlabel('True signal')
    axs[2].set_ylabel('Estimated signal')
    rho = np.corrcoef(y_true_flat, y_est_flat)[0, 1]
    axs[2].set_title('Signal scatter (trial 0, rho=%.3f)' % rho)
    remove_edges(axs[2])

    plt.tight_layout()
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig


# ------------------------------------------------------------------ #
#  5. Summary dashboard
# ------------------------------------------------------------------ #

def plot_summary(data, result, numbers2tuples, labels,
                 class_names=[], n_ensembles_each=[],
                 save_path='', figname_prefix='summary'):
    """
    All-in-one summary plot: A heatmaps, Phi traces, reconstruction, R^2.

    Parameters
    ----------
    data : np.ndarray, shape (N, T, M)
    result : dict (output of milcci.fit)
    numbers2tuples : dict
    labels : list
    class_names : list of str
    n_ensembles_each : list of int
    save_path : str
    """
    from .evaluation import per_trial_r2
    from .core import reconstruct

    Phi = result['Phi']
    A = result['A']
    A_full = result['A_full']
    labels_unique_order = result['params']['labels_unique_order']
    N, T, M = data.shape
    P = Phi.shape[1]
    K = len(labels_unique_order)

    r2_vec = per_trial_r2(data, A_full, Phi)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.4)

    # row 0: A heatmaps (up to 4 conditions)
    n_show_A = min(K, 4)
    vmin_A = np.percentile(A, 2)
    vmax_A = np.percentile(A, 98)
    for c in range(n_show_A):
        ax = fig.add_subplot(gs[0, c])
        sns.heatmap(A[:, :, c], ax=ax, cmap='RdBu_r', center=0,
                    vmin=vmin_A, vmax=vmax_A, rasterized=True, cbar=False)
        tup = numbers2tuples[labels_unique_order[c]]
        ax.set_title(str(tup), fontsize=8)
        ax.set_xlabel('Ens.')
        if c == 0:
            ax.set_ylabel('Neuron')
        # ensemble boundaries
        if len(n_ensembles_each) > 0:
            cumsum_ens = np.cumsum(n_ensembles_each)
            for boundary in cumsum_ens[:-1]:
                ax.axvline(x=boundary, color='k', lw=1, ls='--', alpha=0.5)

    # row 1: Phi traces for first 2 ensembles
    color_list_conds = plt.cm.tab10(np.linspace(0, 1, K))
    labels_arr = np.array(labels)
    for p_idx in range(min(P, 4)):
        ax = fig.add_subplot(gs[1, p_idx])
        for c, lab in enumerate(labels_unique_order):
            trial_idx = np.where(labels_arr == lab)[0][:2]
            for ti, trial in enumerate(trial_idx):
                ax.plot(Phi[:, p_idx, trial], color=color_list_conds[c],
                        alpha=0.6, lw=1,
                        label=str(numbers2tuples[lab]) if ti == 0 else None)
        ax.set_title(r'$\Phi_{%d}$' % p_idx, fontsize=9)
        remove_edges(ax)
        if p_idx == 0:
            ax.set_ylabel('Amplitude')
            ax.legend(fontsize=5, ncol=2)
        ax.set_xlabel('Time')

    # row 2 left: reconstruction example
    trial_show = 0
    ax_orig = fig.add_subplot(gs[2, 0])
    sns.heatmap(data[:, :, trial_show], ax=ax_orig, cmap='viridis',
                rasterized=True, cbar=False)
    ax_orig.set_title('Original (trial %d)' % trial_show, fontsize=9)
    ax_orig.set_ylabel('Neuron')

    ax_reco = fig.add_subplot(gs[2, 1])
    Y_hat = A_full[:, :, trial_show] @ Phi[:, :, trial_show].T
    sns.heatmap(Y_hat, ax=ax_reco, cmap='viridis', rasterized=True, cbar=False)
    ax_reco.set_title('Reconstruction', fontsize=9)

    # row 2 right: R^2 bar
    ax_r2 = fig.add_subplot(gs[2, 2:])
    bar_color_arr = [color_list_conds[list(labels_unique_order).index(lab) % 10]
                     for lab in labels]
    ax_r2.bar(range(M), r2_vec, color=bar_color_arr, edgecolor='k', lw=0.3)
    ax_r2.axhline(y=np.mean(r2_vec), color='r', ls='--', lw=1)
    ax_r2.set_ylabel(r'$R^2$')
    ax_r2.set_xlabel('Trial')
    ax_r2.set_title(r'Per-trial $R^2$ (mean=%.3f)' % np.mean(r2_vec), fontsize=9)
    remove_edges(ax_r2)

    fig.suptitle('MILCCI Summary', fontsize=14, fontweight='bold')
    plt.show()
    save_fig(figname_prefix, fig, save_path)
    return fig
