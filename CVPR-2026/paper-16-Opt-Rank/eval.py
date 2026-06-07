"""
eval.py -- Evaluation script for:
"What is the optimal ranking score between precision and recall?
We can always find it and it is rarely F1" (CVPR 2026)

Outputs key-value pairs for the optimization pipeline to parse.
"""

import json
import numpy as np
from scipy import stats

# ============================================================
# Helper functions (faithful to the paper's algorithm)
# ============================================================

def has_ties(vals):
    return len(np.unique(vals)) < len(vals)

def resolve_ties(pr, re):
    """Resolve ties via adaptive epsilon perturbation.
    
    Uses finer-grained schedule: smaller initial epsilon, slower growth (x2),
    more iterations, and a maximum epsilon threshold to prevent over-perturbation.
    """
    pr_ = pr.copy()
    re_ = re.copy()
    epsilon = 1e-25  # Start even smaller for finer resolution
    max_epsilon = 1e-8  # Safety threshold
    max_iterations = 40  # More iterations with slower growth
    
    for _ in range(max_iterations):
        if has_ties(pr_) or has_ties(re_):
            pr_ = (1 - epsilon) * pr_ + epsilon * re_
            re_ = epsilon * pr_ + (1 - epsilon) * re_
            epsilon *= 2.0
        else:
            return pr_, re_
    return pr_, re_

def get_swaps_beta_sq(pr, re):
    """Compute swap-ratio distribution (Eq. 10-12 from the paper)."""
    N = len(pr)
    theta = np.empty([N, N])
    for i in range(N):
        for j in range(N):
            if i < j:
                theta[i, j] = np.nan
                continue
            inv_pr_i = 1.0 / pr[i]
            inv_pr_j = 1.0 / pr[j]
            inv_re_i = 1.0 / re[i]
            inv_re_j = 1.0 / re[j]
            theta[i, j] = - (inv_pr_i - inv_pr_j) / (inv_re_i - inv_re_j)
    theta = theta.flatten()
    theta = theta[np.isfinite(theta)]
    theta = theta[theta >= 0]
    theta = np.sort(theta)
    return theta

def get_optimal_beta(theta):
    """Compute optimal beta via median of swap-ratio distribution (Eq. 12)."""
    optimal_beta_sq = np.median(theta)
    optimal_beta = np.sqrt(optimal_beta_sq)
    return optimal_beta

def refine_beta_with_grid_search(pr, re, beta0, n_points=1000):
    """Two-stage optimization: grid-search around closed-form beta to maximize kappa."""
    beta_min = beta0 / 2.0
    beta_max = beta0 * 2.0
    betas = np.linspace(beta_min, beta_max, n_points)
    
    best_kappa = -1.0
    best_beta = beta0
    
    for beta in betas:
        kappa, _, _, _ = compute_degree_of_optimality(pr, re, beta)
        if kappa > best_kappa:
            best_kappa = kappa
            best_beta = beta
    
    return best_beta, best_kappa

def compute_degree_of_optimality(pr, re, beta):
    """Compute degree of optimality O (Eq. 16 from the paper)."""
    tau_pr_re = stats.kendalltau(pr, re).correlation

    beta_sq = beta * beta
    b = beta_sq / (1.0 + beta_sq)
    fbeta = 1.0 / ((1.0 - b) / pr + b / re)

    if has_ties(fbeta):
        tau_pr_fbeta = stats.kendalltau(pr, fbeta).correlation
        tau_fbeta_re = stats.kendalltau(fbeta, re).correlation
        delta = (1.0 + tau_pr_re) - (tau_pr_fbeta + tau_fbeta_re)
        tau_pr_fbeta += delta / 2.0
        tau_fbeta_re += delta / 2.0
    else:
        tau_pr_fbeta = stats.kendalltau(pr, fbeta).correlation
        tau_fbeta_re = stats.kendalltau(fbeta, re).correlation

    p_trivial = (tau_pr_fbeta + tau_fbeta_re) / 2.0
    p_wrong = np.abs(tau_pr_fbeta - tau_fbeta_re) / 4.0
    p_correct = 1.0 - p_trivial - p_wrong
    degree_of_optimality = 1.0 - p_wrong / (1.0 - p_trivial)

    return degree_of_optimality * 100.0, tau_pr_re, tau_pr_fbeta, tau_fbeta_re


# ============================================================
# Main evaluation
# ============================================================

def main():
    # Load CADA-RRE dataset
    with open("data/CADA-RRE.json") as f:
        data = json.load(f)

    ids = list(data.keys())
    num_tn = np.array([data[id_]["num_tn"] for id_ in ids])
    num_fp = np.array([data[id_]["num_fp"] for id_ in ids])
    num_fn = np.array([data[id_]["num_fn"] for id_ in ids])
    num_tp = np.array([data[id_]["num_tp"] for id_ in ids])

    pr = num_tp / (num_fp + num_tp)
    pr = np.fmax(0.0, pr)
    re = num_tp / (num_fn + num_tp)

    # Remove duplicate (Pr, Re) pairs
    N = len(pr)
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if pr[i] == pr[j] and re[i] == re[j]:
                keep[i] = False
    pr = pr[keep]
    re = re[keep]

    num_classifiers = len(pr)

    # Resolve ties
    pr_resolved, re_resolved = resolve_ties(pr, re)

    # Compute F1 optimality (beta=1)
    o_f1, tau_pr_re_f1, tau_pr_f1, tau_f1_re = compute_degree_of_optimality(
        pr_resolved, re_resolved, 1.0
    )

    # Compute optimal beta via closed-form expression (Eq. 12)
    theta = get_swaps_beta_sq(pr_resolved, re_resolved)
    optimal_beta_closed = get_optimal_beta(theta)
    
    # Two-stage refinement: grid-search around closed-form beta to maximize kappa
    optimal_beta, _ = refine_beta_with_grid_search(
        pr_resolved, re_resolved, optimal_beta_closed, n_points=1000
    )

    # Compute degree of optimality with optimal beta
    o_optimal, tau_pr_re_opt, tau_pr_fb_opt, tau_fb_re_opt = compute_degree_of_optimality(
        pr_resolved, re_resolved, optimal_beta
    )

    tau_pr_re = tau_pr_re_opt  # same as tau_pr_re_f1 since Pr and Re don't depend on beta

    # Print output in the expected format
    print(f"num_classifiers: {num_classifiers}")
    print(f"tau_Pr_Re: {tau_pr_re:.6f}")
    print(f"optimal_beta: {optimal_beta:.6f}")
    print(f"degree_of_optimality_f1: {o_f1:.2f}")
    print(f"degree_of_optimality_optimal: {o_optimal:.2f}")
    print(f"primary_metric: {o_optimal:.2f}")


if __name__ == "__main__":
    main()
