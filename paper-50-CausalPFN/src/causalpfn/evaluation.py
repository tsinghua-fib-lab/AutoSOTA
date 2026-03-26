from typing import Tuple

import numpy as np

from .causal_estimator import CATEEstimator


def calculate_pehe(cate_true: np.ndarray, cate_pred: np.ndarray) -> float:
    """
    Calculate the PEHE (Precision in Estimation of Heterogeneous Effect) metric.
    """

    return np.sqrt(np.mean((cate_true - cate_pred) ** 2))


def calculate_calibration_curve_scores(
    *,
    fitted_estimator: CATEEstimator,
    X_test: np.ndarray,
    true_cate: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, float]:
    """
    Calculates the calibration curve and the Integrated Coverage Error (ICE).

    Returns:
        - coverage: The coverage of the calibration curve with shape (n_bins,).
        - ICE
    """

    alpha_values = np.linspace(0.01, 1.0, n_bins)  # significance levels

    X_test = X_test
    cates = true_cate  # shape: (n_samples,)

    coverage = np.zeros((len(alpha_values),))

    # Compute the calibration curve
    for i, alpha in enumerate(alpha_values):
        res = fitted_estimator.estimate_cate_CI(X=X_test, alpha=alpha, n_samples=5000)
        # 1 - alpha confidence interval
        lower_bound, upper_bound = res["lower_bound"], res["upper_bound"]  # shape: (n_samples,)
        # Compute the coverage
        coverage[i] = np.mean((cates >= lower_bound) & (cates <= upper_bound))

    # Compute ICEs
    ice = np.mean(coverage - (1 - alpha_values))

    return coverage, ice


def get_qini_curve(
    rct_treatments: np.ndarray,
    rct_outcomes: np.ndarray,
    estimated_cate: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calculate the Qini curve based on the estimated CATE values.
    See https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf
    """
    sorted_indices = np.argsort(estimated_cate)[::-1]

    treated_size = rct_treatments[sorted_indices].cumsum()
    control_size = (1 - rct_treatments[sorted_indices]).cumsum()
    treated_outcomes = (rct_outcomes[sorted_indices] * rct_treatments[sorted_indices]).cumsum()
    control_outcomes = (rct_outcomes[sorted_indices] * (1 - rct_treatments[sorted_indices])).cumsum()

    qini_curve = np.where(control_size != 0, treated_outcomes - control_outcomes * (treated_size / control_size), 0)
    if normalize:
        qini_curve /= np.abs(qini_curve[-1]) + 1e-8

    sum_qini = np.sum(qini_curve) / len(qini_curve)
    return qini_curve, sum_qini
