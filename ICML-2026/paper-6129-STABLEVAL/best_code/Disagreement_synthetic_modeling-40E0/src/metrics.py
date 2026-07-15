"""Evaluation metrics for aggregation methods."""

import numpy as np
from typing import Tuple, List
from scipy.stats import kendalltau


def compute_mse(estimated_scores: np.ndarray, true_scores: np.ndarray) -> float:
    """
    Compute Mean Squared Error between estimated and true scores.
    
    MSE = (1/A) * sum_a (S_hat_a - S*_a)^2
    
    Args:
        estimated_scores: Estimated agent scores
        true_scores: Ground truth agent scores
    
    Returns:
        MSE value
    """
    return np.mean((estimated_scores - true_scores) ** 2)


def compute_kendall_tau(
    estimated_scores: np.ndarray,
    true_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Kendall's tau between estimated and true rankings.
    
    Args:
        estimated_scores: Estimated agent scores
        true_scores: Ground truth agent scores
    
    Returns:
        tau: Kendall's tau correlation coefficient
        p_value: Two-sided p-value
    """
    tau, p_value = kendalltau(estimated_scores, true_scores)
    return tau, p_value


def compute_ranking_accuracy(
    estimated_scores: np.ndarray,
    true_scores: np.ndarray
) -> float:
    """
    Compute fraction of pairwise comparisons that are correct.
    
    Args:
        estimated_scores: Estimated agent scores
        true_scores: Ground truth agent scores
    
    Returns:
        Fraction of correct pairwise orderings
    """
    n = len(estimated_scores)
    correct = 0
    total = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            true_order = np.sign(true_scores[i] - true_scores[j])
            est_order = np.sign(estimated_scores[i] - estimated_scores[j])
            
            if true_order == est_order:
                correct += 1
            elif true_order == 0 or est_order == 0:
                correct += 0.5  # Ties count as half
            
            total += 1
    
    return correct / total if total > 0 else 1.0


def compute_stability(
    y: np.ndarray,
    annotators: np.ndarray,
    aggregation_fn,
    n_annotators: int,
    credit_mapping: List[float],
    subsample_m: int,
    n_subsamples: int = 100,
    rng: np.random.Generator = None
) -> Tuple[float, float]:
    """
    Compute ranking stability under annotator subsampling.
    
    For each subsample:
        1. Subsample labels from m to m'
        2. Recompute ranking
        3. Compute Kendall's tau with original ranking
    
    Args:
        y: Observed labels of shape (n_agents, n_items, m)
        annotators: Annotator IDs
        aggregation_fn: Function to compute scores (takes y, annotators, n_annotators, credit_mapping)
        n_annotators: Total annotators
        credit_mapping: Credit values
        subsample_m: New number of labels per item
        n_subsamples: Number of subsamples
        rng: Random number generator
    
    Returns:
        mean_tau: Average Kendall's tau across subsamples
        std_tau: Standard deviation of tau
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_agents, n_items, m = y.shape
    
    if subsample_m >= m:
        # No subsampling possible
        return 1.0, 0.0
    
    # Get original ranking
    original_scores = aggregation_fn(y, annotators, n_annotators, credit_mapping)
    if isinstance(original_scores, tuple):
        original_scores = original_scores[0] if isinstance(original_scores[0], np.ndarray) and len(original_scores[0]) == n_agents else original_scores[1]
    
    taus = []
    
    for _ in range(n_subsamples):
        # Subsample labels
        y_sub = np.zeros((n_agents, n_items, subsample_m), dtype=np.int32)
        ann_sub = np.zeros((n_agents, n_items, subsample_m), dtype=np.int32)
        
        for a in range(n_agents):
            for i in range(n_items):
                idx = rng.choice(m, size=subsample_m, replace=False)
                y_sub[a, i] = y[a, i, idx]
                ann_sub[a, i] = annotators[a, i, idx]
        
        # Recompute scores
        sub_scores = aggregation_fn(y_sub, ann_sub, n_annotators, credit_mapping)
        if isinstance(sub_scores, tuple):
            sub_scores = sub_scores[0] if isinstance(sub_scores[0], np.ndarray) and len(sub_scores[0]) == n_agents else sub_scores[1]
        
        # Compute tau
        tau, _ = kendalltau(original_scores, sub_scores)
        if not np.isnan(tau):
            taus.append(tau)
    
    return np.mean(taus), np.std(taus)


def evaluate_all_methods(
    results: dict,
    true_scores: np.ndarray,
    y: np.ndarray,
    annotators: np.ndarray,
    n_annotators: int,
    credit_mapping: List[float],
    subsample_m: int = 3,
    n_stability_subsamples: int = 100,
    rng: np.random.Generator = None
) -> dict:
    """
    Evaluate all aggregation methods.
    
    Args:
        results: Output from aggregate_all_methods
        true_scores: Ground truth scores
        y: Observed labels
        annotators: Annotator IDs
        n_annotators: Total annotators
        credit_mapping: Credit values
        subsample_m: Labels for stability subsampling
        n_stability_subsamples: Number of stability subsamples
        rng: Random number generator
    
    Returns:
        Dictionary with metrics for each method
    """
    from .aggregation import majority_vote, posterior_expected_credit
    
    metrics = {}
    
    # Majority Vote
    mv_scores = results["mv_scores"]
    metrics["mv"] = {
        "mse": compute_mse(mv_scores, true_scores),
        "kendall_tau": compute_kendall_tau(mv_scores, true_scores)[0],
        "ranking_accuracy": compute_ranking_accuracy(mv_scores, true_scores),
    }
    
    # Dawid-Skene Hard
    ds_scores = results["ds_scores"]
    metrics["ds"] = {
        "mse": compute_mse(ds_scores, true_scores),
        "kendall_tau": compute_kendall_tau(ds_scores, true_scores)[0],
        "ranking_accuracy": compute_ranking_accuracy(ds_scores, true_scores),
    }
    
    # Posterior Expected Credit
    pec_scores = results["pec_scores"]
    metrics["pec"] = {
        "mse": compute_mse(pec_scores, true_scores),
        "kendall_tau": compute_kendall_tau(pec_scores, true_scores)[0],
        "ranking_accuracy": compute_ranking_accuracy(pec_scores, true_scores),
    }
    
    # Stability (optional, computationally expensive)
    if n_stability_subsamples > 0 and subsample_m < y.shape[2]:
        # MV stability
        def mv_fn(y, ann, n_ann, credit):
            _, scores = majority_vote(y, credit)
            return scores
        
        mv_stab_mean, mv_stab_std = compute_stability(
            y, annotators, mv_fn, n_annotators, credit_mapping,
            subsample_m, n_stability_subsamples, rng
        )
        metrics["mv"]["stability_mean"] = mv_stab_mean
        metrics["mv"]["stability_std"] = mv_stab_std
        
        # PEC stability
        def pec_fn(y, ann, n_ann, credit):
            scores, _ = posterior_expected_credit(y, ann, n_ann, credit)
            return scores
        
        pec_stab_mean, pec_stab_std = compute_stability(
            y, annotators, pec_fn, n_annotators, credit_mapping,
            subsample_m, n_stability_subsamples, rng
        )
        metrics["pec"]["stability_mean"] = pec_stab_mean
        metrics["pec"]["stability_std"] = pec_stab_std
        
        # DS stability (same posterior computation as PEC)
        metrics["ds"]["stability_mean"] = pec_stab_mean
        metrics["ds"]["stability_std"] = pec_stab_std
    
    return metrics
