"""
BHPI Simulation Experiment - Python Port
=========================================
Equivalent of MATLAB's simulate_design.m for reproducing
the synthetic data experiments from the BHPI paper.

Paper: "Disentangling Latent Risk Pathways via Bayesian Hypergraph Inference"
ICML 2026 Oral
"""

import os
import sys
import time
import numpy as np
from scipy.io import savemat
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhpi import (
    simu_data_gen, E_Overlap, cavi_initialization, BHPI,
    compute_repulsion_strength,
)


def hungarian_match(H_true, H_prob):
    """Hungarian algorithm for soft matching of true and estimated hyperedges.

    Args:
        H_true: (V, E) true incidence matrix (binary)
        H_prob: (V, E_hat) estimated membership probabilities

    Returns:
        match_idx: (E,) indices mapping true hyperedges to estimated
        H_prob_aligned: (V, E) aligned estimated probabilities
    """
    V = H_true.shape[0]
    E = H_true.shape[1]
    E_hat = H_prob.shape[1]

    H_sim = np.zeros((E, E_hat))
    for e1 in range(E):
        for e2 in range(E_hat):
            H_sim[e1, e2] = H_true[:, e1] @ H_prob[:, e2]

    # Hungarian: maximize similarity = minimize negative similarity
    row_ind, col_ind = linear_sum_assignment(-H_sim)

    match_idx = np.zeros(E, dtype=int)
    H_prob_aligned = np.zeros((V, E))
    for i in range(len(row_ind)):
        i_true = row_ind[i]
        j_est = col_ind[i]
        match_idx[i_true] = j_est
        H_prob_aligned[:, i_true] = H_prob[:, j_est]

    return match_idx, H_prob_aligned


def compute_auc(y_true, y_score):
    """Compute AUROC, handling edge cases."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5


def simulate_design(N=2000, mu_mean0=1.5, mu_sd=0.5, seed=42,
                    E_hat=10, max_iter=2000, tol=1e-4,
                    omega_repulsion_set=None, output_dir=None,
                    verbose=True,
                    staged=0, warmup_iters=0,
                    sigma2_alpha=10.0, t0=10,
                    n_seed_init=3, seed_init_max_iter=50):
    """Run the full BHPI simulation experiment.

    This corresponds to the MATLAB simulate_design.m entry point.

    Args:
        N: number of samples
        mu_mean0: mean of effect sizes
        mu_sd: std of effect sizes
        seed: random seed for data generation
        E_hat: upper bound on number of hyperedges
        max_iter: maximum CAVI iterations
        tol: convergence tolerance
        omega_repulsion_set: list of repulsion strengths to test
        output_dir: directory for saving results
        verbose: print detailed progress
        staged: whether to use staged warmup (0 or 1)
        warmup_iters: warmup iterations per stage
        sigma2_alpha: prior variance for alpha
        t0: Robbins-Monro offset
        n_seed_init: number of initialization seeds to test
        seed_init_max_iter: max iterations per init seed test

    Returns:
        dict with all results
    """
    if omega_repulsion_set is None:
        omega_repulsion_set = [0, 0.5, 1, 2, 5]

    n_omega_repulsion = len(omega_repulsion_set)

    V = 30   # number of diseases
    E = 5    # number of true hyperedges
    P = 6    # number of predictors
    nRarePerEdge = 1
    nCommonPerEdge = 6

    # Model settings
    initial_method = 'NNMF'
    # staged and warmup_iters are now parameters
    fix_z = False
    z_constraint = 1
    batch_size = 0
    # t0 is now a parameter
    weights = 1.0
    # sigma2_alpha is now a parameter

    print(f"BHPI Simulation: N={N}, mu_mean={mu_mean0}, mu_sd={mu_sd}, seed={seed}")
    print("=" * 70)

    # Generate data
    print("Generating synthetic data...")
    X, Y, alpha, Beta, H, gamma, mu = simu_data_gen(
        N, P, V, E, mu_mean0, mu_sd, seed, nRarePerEdge, nCommonPerEdge
    )
    prev = Y.mean(axis=0)
    rare_idx = prev < 0.05
    common_idx = prev >= 0.05
    print(f"  Rare diseases (<5%): {rare_idx.sum()}, Common: {common_idx.sum()}")

    # Split: 80/20 train+val / test, then 75/25 train/val from train+val
    # This gives ~60/20/20 overall
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_test = int(N * 0.2)
    idx_test = idx[:n_test]
    idx_trainval = idx[n_test:]
    n_val = int(len(idx_trainval) * 0.25)
    idx_val = idx_trainval[:n_val]
    idx_train = idx_trainval[n_val:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    X_test, Y_test = X[idx_test], Y[idx_test]

    R_val = ~np.isnan(Y_val)
    R_test = ~np.isnan(Y_test)

    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # Results storage
    y_test_scores = np.full((X_test.shape[0], V, n_omega_repulsion), np.nan)
    auroc_per_disease = np.full((V, n_omega_repulsion), np.nan)
    beta_est_results = np.full((2, n_omega_repulsion), np.nan)  # mse, corr
    match_idx_all = np.full((E, n_omega_repulsion), np.nan)
    H_auroc_all = np.full((1 + E, n_omega_repulsion), np.nan)
    H_prob_overlap = np.full(n_omega_repulsion, np.nan)
    gamma_entropy_per_predictor = np.full((P, n_omega_repulsion), np.nan)
    gamma_auroc_all = np.full((1 + P, n_omega_repulsion), np.nan)
    repulsion_redundancy_metrics = np.full((P, 2, n_omega_repulsion), np.nan)

    z_prob_all = np.full((E_hat, n_omega_repulsion), np.nan)
    H_prob_all = np.full((V, E_hat, n_omega_repulsion), np.nan)
    gamma_prob_all = np.full((P, E_hat, n_omega_repulsion), np.nan)
    mu_hat_all = np.full((P, E_hat, n_omega_repulsion), np.nan)

    print(f"\nFinding best initialization seed (testing {n_seed_init} seeds, {seed_init_max_iter} iters each)...")
    # Check performance under different initializations (no repulsion)
    # Use reduced iterations for speed
    seed_max_iter = min(seed_init_max_iter, max_iter)
    AUROC_val_mean = np.full(n_seed_init, np.nan)

    for seed_init in range(1, n_seed_init + 1):
        initials = cavi_initialization(seed_init, initial_method, E_hat,
                                        X_train, Y_train)

        model = BHPI(X_train, Y_train, E_hat, seed_max_iter,
                     seed_init, initials, omega_repulsion=0.0,
                     staged=staged, final_fix_z=fix_z,
                     final_z_constraint=z_constraint,
                     sigma2_alpha=sigma2_alpha,
                     warmup_iters=warmup_iters, batch_size=batch_size,
                     t0=t0, weights=weights, tol=tol*10, verbose=False)

        # Evaluate on validation set
        eta_val = X_val @ model['beta'] + model['alpha_mean']
        y_fitted_prob_val = 1.0 / (1.0 + np.exp(-eta_val))

        AUROC_val = np.array([compute_auc(Y_val[R_val[:, v], v],
                                          y_fitted_prob_val[R_val[:, v], v])
                              for v in range(V)])
        AUROC_val_mean[seed_init - 1] = np.nanmean(AUROC_val)
        if verbose:
            print(f"  seed_init={seed_init}: val AUROC={AUROC_val_mean[seed_init-1]:.4f}")

    seed_init_best = np.nanargmax(AUROC_val_mean) + 1
    print(f"  Best seed_init={seed_init_best}, val AUROC={AUROC_val_mean[seed_init_best-1]:.4f}")

    # Finalize initialization
    initials = cavi_initialization(seed_init_best, initial_method, E_hat,
                                    X_train, Y_train)

    # Run with different repulsion strengths
    for idx_repulsion, omega_repulsion in enumerate(omega_repulsion_set):
        print(f"\n{'='*70}")
        print(f"Repulsion #{idx_repulsion+1}/{n_omega_repulsion}: omega={omega_repulsion}")
        print(f"{'='*70}")

        t_start = time.time()

        model = BHPI(X_train, Y_train, E_hat, max_iter,
                     seed_init_best, initials, omega_repulsion=omega_repulsion,
                     staged=staged, final_fix_z=fix_z,
                     final_z_constraint=z_constraint,
                     sigma2_alpha=sigma2_alpha,
                     warmup_iters=warmup_iters, batch_size=batch_size,
                     t0=t0, weights=weights, tol=tol, verbose=verbose)

        elapsed = time.time() - t_start

        m_prob = model['m_prob']
        z_prob = model['z_prob']
        z_prob_all[:, idx_repulsion] = z_prob
        gamma_prob = model['gamma_prob']
        gamma_prob_joint = gamma_prob * z_prob.reshape(1, -1)
        gamma_prob_all[:, :, idx_repulsion] = gamma_prob_joint
        H_prob = m_prob * z_prob.reshape(1, -1)
        H_prob_all[:, :, idx_repulsion] = H_prob

        beta_hat = model['beta']
        mu_mean = model['mu_mean']
        mu_hat_all[:, :, idx_repulsion] = mu_mean
        alpha_mean = model['alpha_mean']

        # Mechanism recovery
        mse_beta = np.mean((Beta - beta_hat)**2)
        beta_corr = np.corrcoef(Beta.ravel(), beta_hat.ravel())[0, 1]
        beta_est_results[:, idx_repulsion] = [mse_beta, beta_corr]

        # Prediction on test set
        eta_test = X_test @ beta_hat + alpha_mean
        y_fitted_prob_test = 1.0 / (1.0 + np.exp(-eta_test))
        y_test_scores[:, :, idx_repulsion] = y_fitted_prob_test

        # AUROC on test set
        AUROC_test = np.array([compute_auc(Y_test[R_test[:, v], v],
                                           y_fitted_prob_test[R_test[:, v], v])
                               for v in range(V)])
        auroc_per_disease[:, idx_repulsion] = AUROC_test

        print(f"\n  Overall (mean) AUROC: {np.mean(AUROC_test):.2f}")
        rare_auroc = np.mean(AUROC_test[rare_idx])
        common_auroc = np.mean(AUROC_test[common_idx])
        print(f"  Rare disease AUROC: {rare_auroc:.2f}")
        print(f"  Common disease AUROC: {common_auroc:.2f}")
        print(f"  COR(beta, beta_hat): {beta_corr:.2f}")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Latent hypergraph structure recovery
        H_prob_entropy = -(H_prob * np.log(H_prob + 1e-16) +
                           (1 - H_prob) * np.log(1 - H_prob + 1e-16))

        # Hungarian matching
        match_idx, H_prob_aligned = hungarian_match(H, H_prob)
        match_idx_all[:, idx_repulsion] = match_idx

        # AUROC of hyperedge inclusion
        H_auroc_pool = compute_auc(H.ravel(), H_prob_aligned.ravel())
        H_auroc_all[0, idx_repulsion] = H_auroc_pool
        print(f"  H-AUC (pooled): {H_auroc_pool:.2f}")

        for e in range(E):
            H_auroc_all[1 + e, idx_repulsion] = compute_auc(H[:, e], H_prob_aligned[:, e])

        # Mean overlap
        Ohat, _ = E_Overlap(H_prob)
        triu_idx = np.triu_indices(E_hat, k=1)
        mean_overlap = Ohat[triu_idx].mean() if len(triu_idx[0]) > 0 else 0
        H_prob_overlap[idx_repulsion] = mean_overlap

        # Gamma recovery
        gamma_entropy = -(gamma_prob_joint * np.log(gamma_prob_joint + 1e-16) +
                          (1 - gamma_prob_joint) * np.log(1 - gamma_prob_joint + 1e-16))
        gamma_entropy_per_predictor[:, idx_repulsion] = gamma_entropy.sum(axis=1)

        # AUROC for gamma
        gamma_prob_aligned = gamma_prob_joint[:, match_idx]
        gamma_auroc_pool = compute_auc(gamma.ravel(), gamma_prob_aligned.ravel())
        gamma_auroc_all[0, idx_repulsion] = gamma_auroc_pool
        print(f"  gamma-AUC (pooled): {gamma_auroc_pool:.2f}")

        for p in range(P):
            gamma_auroc_all[1 + p, idx_repulsion] = compute_auc(gamma[p, :],
                                                                 gamma_prob_aligned[p, :])

        # Redundancy metrics
        eff_hyp, rep_term, avg_overlap, _ = compute_repulsion_strength(
            gamma_prob, m_prob, z_prob)
        repulsion_redundancy_metrics[:, :, idx_repulsion] = np.column_stack([
            eff_hyp, avg_overlap])

    # Independent logistic regression baseline
    print("\n" + "=" * 70)
    print("Logistic Regression Baseline")
    print("=" * 70)

    p_test_logistic = np.full((X_test.shape[0], V), np.nan)
    beta_logistic = np.full((P, V), np.nan)

    for v in range(V):
        try:
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
            lr.fit(X_train, Y_train[:, v])
            p_test_logistic[:, v] = lr.predict_proba(X_test)[:, 1]
            beta_logistic[:, v] = lr.coef_[0]
        except Exception:
            p_test_logistic[:, v] = 0.5
            beta_logistic[:, v] = 0.0

    mse_beta_logistic = np.nanmean((Beta - beta_logistic)**2)
    beta_corr_logistic = np.corrcoef(
        Beta.ravel(), beta_logistic.ravel())[0, 1]

    print(f"Logistic MSE(beta): {mse_beta_logistic:.4f}, COR(beta,beta_hat): {beta_corr_logistic:.4f}")

    AUROC_test_logistic = np.array([
        compute_auc(Y_test[R_test[:, v], v], p_test_logistic[R_test[:, v], v])
        for v in range(V)
    ])

    # Comparison summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    print(f"\n{'Model':<25}", end="")
    for o in omega_repulsion_set:
        print(f"{'ω=' + str(o):>12}", end="")
    print(f"{'Logistic':>12}")

    print(f"{'Mean AUROC (all):':<25}", end="")
    for i in range(n_omega_repulsion):
        print(f"{np.mean(auroc_per_disease[:, i]):12.2f}", end="")
    print(f"{np.mean(AUROC_test_logistic):12.2f}")

    print(f"{'Mean AUROC (rare):':<25}", end="")
    for i in range(n_omega_repulsion):
        print(f"{np.mean(auroc_per_disease[rare_idx, i]):12.2f}", end="")
    print(f"{np.mean(AUROC_test_logistic[rare_idx]):12.2f}")

    print(f"{'COR(beta,beta_hat):':<25}", end="")
    for i in range(n_omega_repulsion):
        print(f"{beta_est_results[1, i]:12.2f}", end="")
    print(f"{beta_corr_logistic:12.2f}")

    print(f"{'H-AUC (pooled):':<25}", end="")
    for i in range(n_omega_repulsion):
        print(f"{H_auroc_all[0, i]:12.2f}", end="")
    print(f"{'N/A':>12}")

    print(f"{'gamma-AUC (pooled):':<25}", end="")
    for i in range(n_omega_repulsion):
        print(f"{gamma_auroc_all[0, i]:12.2f}", end="")
    print(f"{'N/A':>12}")

    # Pack results
    results = {
        'config': {
            'N': N, 'V': V, 'E': E, 'P': P,
            'mu_mean0': mu_mean0, 'mu_sd': mu_sd,
            'seed': seed, 'E_hat': E_hat,
            'omega_repulsion_set': omega_repulsion_set,
        },
        'data': {
            'X': X, 'Y': Y, 'Beta': Beta, 'H': H, 'gamma': gamma, 'mu': mu,
            'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test,
        },
        'metrics': {
            'auroc_per_disease': auroc_per_disease,
            'AUROC_test_logistic': AUROC_test_logistic,
            'beta_est': beta_est_results,
            'beta_corr_logistic': beta_corr_logistic,
            'H_auroc_all': H_auroc_all,
            'H_prob_overlap': H_prob_overlap,
            'gamma_auroc_all': gamma_auroc_all,
            'gamma_entropy': gamma_entropy_per_predictor,
        },
        'posterior': {
            'z_prob_all': z_prob_all,
            'H_prob_all': H_prob_all,
            'gamma_prob_all': gamma_prob_all,
            'mu_hat_all': mu_hat_all,
        }
    }

    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        resname = f"BHPI_N={N}_mu={mu_mean0:.1f}_sd={mu_sd:.1f}_seed={seed}"
        np.savez(os.path.join(output_dir, resname + '.npz'),
                 **{k: v for k, v in results.items() if isinstance(v, (np.ndarray, dict, list))})
        print(f"\nResults saved to {output_dir}/{resname}.npz")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BHPI Simulation Experiment')
    parser.add_argument('--N', type=int, default=2000, help='Number of samples')
    parser.add_argument('--mu-mean', type=float, default=1.5, help='Effect size mean')
    parser.add_argument('--mu-sd', type=float, default=0.5, help='Effect size std')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--E-hat', type=int, default=10, help='Max hyperedges')
    parser.add_argument('--max-iter', type=int, default=2000, help='Max CAVI iterations')
    parser.add_argument('--tol', type=float, default=1e-4, help='Convergence tolerance')
    parser.add_argument('--omega-repulsion', type=float, nargs='+',
                        default=[0, 0.5, 1, 2, 5], help='Repulsion strengths')
    parser.add_argument('--output-dir', type=str, default='./simu',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--staged', type=int, default=0, help='Use staged warmup (0 or 1)')
    parser.add_argument('--warmup-iters', type=int, default=0, help='Warmup iterations per stage')
    parser.add_argument('--sigma2-alpha', type=float, default=10.0, help='Prior variance for alpha')
    parser.add_argument('--t0', type=float, default=10, help='Robbins-Monro offset')
    parser.add_argument('--n-seed-init', type=int, default=3, help='Number of init seeds to test')
    parser.add_argument('--seed-init-max-iter', type=int, default=50, help='Max iters per init seed')

    args = parser.parse_args()

    results = simulate_design(
        N=args.N,
        mu_mean0=args.mu_mean,
        mu_sd=args.mu_sd,
        seed=args.seed,
        E_hat=args.E_hat,
        max_iter=args.max_iter,
        tol=args.tol,
        omega_repulsion_set=args.omega_repulsion,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        staged=args.staged,
        warmup_iters=args.warmup_iters,
        sigma2_alpha=args.sigma2_alpha,
        t0=args.t0,
        n_seed_init=args.n_seed_init,
        seed_init_max_iter=args.seed_init_max_iter,
    )

    print("\nExperiment complete!")
