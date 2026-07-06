"""
ConfTS: Conformal Temperature Scaling (Xi et al. 2024, arXiv:2402.04344).

Optimizes a single temperature T for the softmax to minimize prediction set size
while maintaining conformal coverage, rather than minimizing ECE.

For conformal prediction, we want T that maximizes efficiency (smaller sets) while
maintaining coverage. The optimal T minimizes the gap between the conformal quantile
threshold and the nonconformity scores of true labels on the calibration set.
"""

import numpy as np
from typing import Optional, Tuple


def softmax_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to logits or probabilities."""
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got {T}")
    if abs(T - 1.0) < 1e-10:
        return logits
    
    # If inputs are already probabilities (all >= 0, sum to ~1), 
    # convert to log space, scale, convert back
    if np.all(logits >= 0) and abs(np.sum(logits, axis=-1).mean() - 1.0) < 0.01:
        # Inputs are probabilities - convert to log space safely
        eps = 1e-12
        clipped = np.clip(logits, eps, 1.0)
        logits_space = np.log(clipped)
    else:
        logits_space = logits
    
    scaled = logits_space / T
    # Stable softmax
    scaled_max = np.max(scaled, axis=-1, keepdims=True)
    exp_scaled = np.exp(scaled - scaled_max)
    probs = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
    return probs


def compute_aps_score(probs: np.ndarray, label: int) -> float:
    """APS nonconformity score for a single point."""
    p_y = float(probs[label])
    mask = probs > p_y
    mask[label] = False
    return float(probs[mask].sum())


def evaluate_set_efficiency(probs: np.ndarray, labels: np.ndarray, alpha: float, T: float) -> dict:
    """
    Evaluate efficiency of a conformal predictor with temperature T.
    
    Uses split conformal: calibrate q_hat on first half, test on second half.
    Returns average set size and coverage.
    """
    n = len(probs)
    n_cal = n // 2
    
    probs_T = softmax_temperature(probs, T)
    
    # Calibration
    cal_probs = probs_T[:n_cal]
    cal_labels = labels[:n_cal]
    scores = np.array([compute_aps_score(cal_probs[i], int(cal_labels[i])) for i in range(n_cal)])
    scores_sorted = np.sort(scores)
    k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n_cal)
    q_hat = float(scores_sorted[k - 1])
    
    # Test
    test_probs = probs_T[n_cal:]
    test_labels = labels[n_cal:]
    n_test = len(test_probs)
    
    set_sizes = []
    covered = 0
    for i in range(n_test):
        s = sum(1 for y in range(len(test_probs[i])) 
                if compute_aps_score(test_probs[i], y) <= q_hat)
        set_sizes.append(s)
        if compute_aps_score(test_probs[i], int(test_labels[i])) <= q_hat:
            covered += 1
    
    return {
        "avg_set_size": float(np.mean(set_sizes)),
        "coverage": covered / n_test,
        "q_hat": q_hat,
        "temperature": T,
    }


def optimize_temperature(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float,
    T_grid: Optional[np.ndarray] = None,
    min_coverage_margin: float = 0.005,
) -> Tuple[float, dict]:
    """
    Find optimal temperature T that minimizes set size while maintaining coverage.
    
    Args:
        cal_probs: Calibration probabilities [N, K]
        cal_labels: Calibration labels [N]
        alpha: Target miscoverage level
        T_grid: Grid of temperatures to search. Default: [0.1, 4.0] with fine grid.
        min_coverage_margin: Minimum coverage margin above 1-alpha to accept T.
    
    Returns:
        best_T: Optimal temperature
        info: Dict with grid search results
    """
    if T_grid is None:
        # Dense grid around T=1.0 with finer spacing near 1.0
        T_coarse = np.arange(0.2, 4.1, 0.2)
        T_fine = np.arange(0.8, 1.3, 0.05)
        T_grid = np.unique(np.concatenate([T_coarse, T_fine, [1.0]]))
        T_grid = np.sort(T_grid)
    
    target_coverage = 1.0 - alpha
    best_T = 1.0
    best_size = float("inf")
    results = []
    
    for T in T_grid:
        res = evaluate_set_efficiency(cal_probs, cal_labels, alpha, T)
        results.append(res)
        
        if res["coverage"] >= target_coverage - min_coverage_margin:
            if res["avg_set_size"] < best_size:
                best_size = res["avg_set_size"]
                best_T = T
    
    # If no T meets coverage constraint, pick the one with best coverage
    if best_size == float("inf"):
        best_coverage = 0.0
        for res in results:
            if res["coverage"] > best_coverage:
                best_coverage = res["coverage"]
                best_T = res["temperature"]
    
    return best_T, {"results": results, "best_size": best_size}


def apply_confts(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply ConfTS temperature scaling to probabilities."""
    if abs(T - 1.0) < 1e-10:
        return probs
    return softmax_temperature(probs, T)


__all__ = ["optimize_temperature", "apply_confts", "softmax_temperature"]
