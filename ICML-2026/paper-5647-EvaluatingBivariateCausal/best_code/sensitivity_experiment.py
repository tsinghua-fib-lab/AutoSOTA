import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from tqdm import tqdm
from synthetic_experiments_linear import sample_sparse_causal_model, compatibility_score

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 24,
})


def _estimate_bivariate_matrix(cov):
    """Estimate the bivariate matrix from a covariance matrix.

    For each variable j, regress j on variables 1, ..., j-1 using the covariance
    matrix to obtain the causal coefficients, then compute A = inv(I - C).
    """
    n = cov.shape[0]
    C = np.zeros((n, n))
    for j in range(1, n):
        # Regress j on 0, ..., j-1
        Sigma_pred = cov[:j, :j]
        sigma_cross = cov[:j, j]
        C[j, :j] = np.linalg.solve(Sigma_pred, sigma_cross)
    return np.linalg.inv(np.eye(n) - C)


def _compute_sensitivity(n, p, num_hidden, sample_sizes, num_models, num_draws, seed,
                          estimate_coefficients=False, sigma=0.0, num_noise_draws=1):
    """Compute score diffs and same-sign percentages for a single configuration."""
    np.random.seed(seed)
    sample_sizes = np.unique(sample_sizes)
    effective_noise = max(num_noise_draws, 1) if sigma > 0 else 1

    total_rows = num_models * num_draws * effective_noise
    diffs_all = np.zeros((total_rows, len(sample_sizes)))
    true_scores = np.zeros(total_rows)
    empirical_scores = np.zeros((total_rows, len(sample_sizes)))

    pbar = tqdm(total=total_rows * len(sample_sizes), bar_format='{l_bar}{bar}{r_bar}')

    for m in range(num_models):
        true_C, cov = sample_sparse_causal_model(n, p, num_hidden)

        if estimate_coefficients:
            true_A = _estimate_bivariate_matrix(cov)
        else:
            true_A = np.linalg.inv(np.eye(n) - true_C)

        for d in range(num_draws):
            max_samples = sample_sizes[-1]
            data = np.random.multivariate_normal(np.zeros(n), cov, size=max_samples)

            for nd in range(effective_noise):
                # Add noise to causal statements when sigma > 0
                if not estimate_coefficients and sigma > 0:
                    noisy_A = true_A + sigma * np.tril(np.random.randn(n, n), -1)
                else:
                    noisy_A = true_A

                true_score = compatibility_score(noisy_A, cov)
                row = (m * num_draws + d) * effective_noise + nd
                true_scores[row] = true_score

                for s_idx, N in enumerate(sample_sizes):
                    empirical_cov = np.cov(data[:N], rowvar=False)

                    if estimate_coefficients:
                        emp_A = _estimate_bivariate_matrix(empirical_cov)
                    else:
                        emp_A = noisy_A

                    emp_score = compatibility_score(emp_A, empirical_cov)
                    empirical_scores[row, s_idx] = emp_score
                    diffs_all[row, s_idx] = emp_score - true_score
                    pbar.update(1)

    pbar.close()

    # Relative error: (empirical - true) / |true|, per draw
    rel_errors = diffs_all / np.abs(true_scores[:, None])
    median_rel = np.median(rel_errors, axis=0)
    q25_rel = np.percentile(rel_errors, 25, axis=0)
    q75_rel = np.percentile(rel_errors, 75, axis=0)
    q10_rel = np.percentile(rel_errors, 10, axis=0)
    q90_rel = np.percentile(rel_errors, 90, axis=0)
    median_true_score = np.median(true_scores)

    # Same-sign percentage per sample size
    same_sign = np.sign(empirical_scores) == np.sign(true_scores[:, None])
    same_sign_pct = 100 * np.mean(same_sign, axis=0)

    return median_rel, q25_rel, q75_rel, q10_rel, q90_rel, median_true_score, same_sign_pct


def run_sensitivity_experiment(sample_sizes=np.logspace(1, 4, 15).astype(int),
                               num_models=50, num_draws=20, seed=42,
                               estimate_coefficients=False, sigma=0.2,
                               n=10, p=0.5, num_hidden=3,
                               num_noise_draws=10):
    """Evaluate compatibility scores with empirical vs true covariance.

    Produces a 1x2 figure for a single (n, p, num_hidden) configuration:
        - left:  relative error of the compatibility score (median + IQR + 10–90%)
        - right: percentage of draws where the empirical and true compatibility
                 scores have the same sign

    When estimate_coefficients is False, the true causal coefficients are used.
    When True, coefficients are estimated from the covariance by regressing each
    variable on its predecessors.

    When sigma > 0 and estimate_coefficients is False, Gaussian noise with
    standard deviation sigma is added to the lower-triangular entries of the
    true causal matrix A before computing scores.

    Args:
        sample_sizes (array): Sample sizes to evaluate.
        num_models (int): Number of causal models to sample per configuration.
        num_draws (int): Number of independent sample draws per model.
        seed (int): Random seed for reproducibility.
        estimate_coefficients (bool): If True, estimate causal coefficients from
            the covariance matrix instead of using the true ones.
        sigma (float): Standard deviation of noise added to causal statements.
            Only used when estimate_coefficients is False.
        n (int): Number of observed variables.
        p (float): Edge probability of the causal graph.
        num_hidden (int): Number of hidden variables.
        num_noise_draws (int): Number of independent noise draws per
            (model, sample) pair when sigma > 0.
    """
    sample_sizes = np.unique(sample_sizes)

    suffix = '_estimated' if estimate_coefficients else ''
    if sigma > 0:
        suffix += f'_sigma{sigma}'
    suffix += f'_n{n}_p{p}_m{num_hidden}'

    median_rel, q25_rel, q75_rel, q10_rel, q90_rel, median_score, same_sign_pct = \
        _compute_sensitivity(
            sample_sizes=sample_sizes, num_models=num_models,
            num_draws=num_draws, seed=seed,
            estimate_coefficients=estimate_coefficients, sigma=sigma,
            num_noise_draws=num_noise_draws,
            n=n, p=p, num_hidden=num_hidden)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: relative error ---
    ax = axes[0]
    ax.fill_between(sample_sizes, q10_rel, q90_rel, alpha=0.15, color='black',
                    label='10th–90th\npercentile')
    ax.fill_between(sample_sizes, q25_rel, q75_rel, alpha=0.3, color='black',
                    label='25th–75th\npercentile')
    ax.plot(sample_sizes, median_rel, color='black', linewidth=1.5,
            label='Median')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Relative error of\ncompatibility score')
    ax.set_ylim(-1, 1)
    ax.grid(True)

    # --- Right: percentage of same-sign draws ---
    ax = axes[1]
    ax.plot(sample_sizes, same_sign_pct, color='black', linewidth=1.5)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('% of draws\nwith correct sign')
    ax.set_ylim(80, 100)
    ax.grid(True)

    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                   borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'sensitivity_rel_error{suffix}.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def run_sensitivity_grids(sample_sizes=np.logspace(1, 4, 15).astype(int),
                          num_models=50, num_draws=20, num_noise_draws=5,
                          sigma_values=(0.0, 0.2, 0.4, 0.6), seed=42):
    """Produce both the relative-error grid and the same-sign grid from the
    same set of draws.

    For each (config, sigma) pair, ``_compute_sensitivity`` is called exactly
    once and the results are used to populate both figures:
        - ``sensitivity_rel_error_grid.png`` (median + IQR + 10-90% of the
          relative error of the compatibility score)
        - ``sensitivity_same_sign_grid.png`` (% of draws where the empirical
          and true compatibility scores have the same sign)

    Args:
        sample_sizes (array): Sample sizes to evaluate.
        num_models (int): Number of causal models to sample per configuration.
        num_draws (int): Number of independent sample draws per model.
        num_noise_draws (int): Number of independent noise draws per
            (model, sample) pair.
        sigma_values (tuple): Noise standard deviations for columns.
        seed (int): Random seed for reproducibility.
    """
    sample_sizes = np.unique(sample_sizes)

    configs = [
        ('dense',      dict(n=10, p=0.5, num_hidden=3)),
        ('sparse',     dict(n=10, p=0.25, num_hidden=3)),
        ('confounded', dict(n=10, p=0.5, num_hidden=10)),
    ]

    figsize = (6 * len(sigma_values), 5 * len(configs))
    fig_rel, axes_rel = plt.subplots(len(configs), len(sigma_values),
                                     figsize=figsize, sharey=True, sharex=True)
    fig_ss, axes_ss = plt.subplots(len(configs), len(sigma_values),
                                   figsize=figsize, sharey=True, sharex=True)

    for row_idx, (label, params) in enumerate(configs):
        for col_idx, sigma in enumerate(sigma_values):
            (median_rel, q25_rel, q75_rel, q10_rel, q90_rel,
             median_score, same_sign_pct) = _compute_sensitivity(
                sample_sizes=sample_sizes, num_models=num_models,
                num_draws=num_draws, seed=seed, estimate_coefficients=False,
                sigma=sigma, num_noise_draws=num_noise_draws, **params)

            n, p, m = params['n'], params['p'], params['num_hidden']
            title = (f'n={n}, p={p}, m={m}, $\\sigma={sigma}$'
                     f'\nmedian score = {median_score:.3f}')

            # --- Relative error subplot ---
            ax = axes_rel[row_idx, col_idx]
            ax.fill_between(sample_sizes, q10_rel, q90_rel, alpha=0.15,
                            color='black', label='10th–90th\npercentile')
            ax.fill_between(sample_sizes, q25_rel, q75_rel, alpha=0.3,
                            color='black', label='25th–75th\npercentile')
            ax.plot(sample_sizes, median_rel, color='black', linewidth=1.5,
                    label='Median')
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
            ax.set_ylim(-1, 1)
            ax.grid(True)
            ax.set_title(title)
            if row_idx == len(configs) - 1:
                ax.set_xlabel('Number of samples')
            if col_idx == 0:
                ax.set_ylabel('Relative error of\ncompatibility score')

            # --- Same-sign subplot ---
            ax = axes_ss[row_idx, col_idx]
            ax.plot(sample_sizes, same_sign_pct, color='black', linewidth=1.5)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
            ax.set_ylim(80, 100)
            ax.grid(True)
            ax.set_title(title)
            if row_idx == len(configs) - 1:
                ax.set_xlabel('Number of samples')
            if col_idx == 0:
                ax.set_ylabel('% of draws\nwith correct sign')

    axes_rel[0, -1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                           borderaxespad=0.)

    fig_rel.tight_layout()
    fig_rel.savefig('sensitivity_rel_error_grid.png', dpi=150,
                    bbox_inches='tight')
    fig_ss.tight_layout()
    fig_ss.savefig('sensitivity_same_sign_grid.png', dpi=150,
                   bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    sample_sizes = np.logspace(1, 4, 15).astype(int)
    run_sensitivity_experiment(sample_sizes=sample_sizes,
                               num_models=50, num_draws=20, 
                               num_noise_draws=10)
    run_sensitivity_grids(sample_sizes=sample_sizes, num_models=50, num_draws=20,
                          num_noise_draws=10)
