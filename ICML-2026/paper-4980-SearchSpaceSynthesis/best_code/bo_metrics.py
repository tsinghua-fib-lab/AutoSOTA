# =============================================================================
# SURROGATE MODEL METRICS
# =============================================================================
#
# This module implements statistical metrics used to evaluate
# surrogate model ranking performance.
#
#
# =============================================================================
# METRICS
# =============================================================================
#
# Kendall tau rank correlation
#
# Spearman rank correlation
#
#
# These metrics compare surrogate predictions against true objective values.
#
#
# =============================================================================
# IMPLEMENTATION DETAILS
# =============================================================================
#
# Use scipy.stats:
#
# from scipy.stats import kendalltau
# from scipy.stats import spearmanr
#
#
# =============================================================================
# FUNCTION
# =============================================================================
#
# compute_ranking_metrics(predictions, true_values)
#
#
# Inputs:
#
# predictions: array-like
# true_values: array-like
#
#
# Output:
#
# dictionary:
#
# {
#     "kendall_tau": float,
#     "spearman_rho": float
# }
#
#
# Handle cases where correlation cannot be computed by returning NaN.
#
#
# =============================================================================
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _safe_float(value: Any) -> float:
	try:
		if value is None:
			return float("nan")
		return float(value)
	except (TypeError, ValueError):
		return float("nan")


def _to_1d_float_array(values) -> np.ndarray:
	return np.asarray(values, dtype=float).reshape(-1)


def compute_ranking_metrics(predictions, true_values) -> dict[str, float]:
	"""Compute surrogate ranking correlations with NaN-safe fallbacks."""
	try:
		pred = _to_1d_float_array(predictions)
		true = _to_1d_float_array(true_values)
	except (TypeError, ValueError):
		return {"kendall_tau": float("nan"), "spearman_rho": float("nan")}

	if pred.shape[0] != true.shape[0] or pred.shape[0] < 2:
		return {"kendall_tau": float("nan"), "spearman_rho": float("nan")}

	finite_mask = np.isfinite(pred) & np.isfinite(true)
	pred = pred[finite_mask]
	true = true[finite_mask]

	if pred.shape[0] < 2:
		return {"kendall_tau": float("nan"), "spearman_rho": float("nan")}

	if np.allclose(pred, pred[0]) or np.allclose(true, true[0]):
		return {"kendall_tau": float("nan"), "spearman_rho": float("nan")}

	kendall_res = kendalltau(pred, true)
	spearman_res = spearmanr(pred, true)

	return {
		"kendall_tau": _safe_float(getattr(kendall_res, "correlation", np.nan)),
		"spearman_rho": _safe_float(getattr(spearman_res, "correlation", np.nan)),
	}
