# =============================================================================
# STRUCTURAL DISTANCE ANALYSIS
# =============================================================================
#
# This module performs structural distance analysis between Tree programs.
#
# Distance is computed using kernel-induced distances.
#
#
# =============================================================================
# IMPORT KERNELS
# =============================================================================
#
# from bayesian_optimization.graph_kernel import WeisfeilerLehmanKernel
#
# from bayesian_optimization.tree_kernel import OrderedRootedSubtreeKernel
#
#
# Instantiate kernels as:
#
# wl_kernel = WeisfeilerLehmanKernel(n_iter=1, normalize=True)
#
# tree_kernel = OrderedRootedSubtreeKernel(max_height=None, normalize=True)
#
#
# =============================================================================
# DISTANCE DEFINITION
# =============================================================================
#
# Kernel induced distance:
#
# d(x,y) = sqrt(k(x,x) + k(y,y) - 2*k(x,y))
#
#
# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================
#
# During aggregation:
#
# 1 load all evaluated programs
#
# 2 find best observed program
#
# x_best = argmin objective_value
#
#
# 3 find top-k programs
#
# sort by objective value
#
# select first TOP_K_PROGRAMS
#
#
# 4 compute distances
#
# distance_to_best
#
# distance_to_topk
#
#
# =============================================================================
# FUNCTIONS
# =============================================================================
#
# kernel_distance(kernel, x, y)
#
# compute_distance_to_best(df, kernel)
#
# compute_distance_to_topk(df, kernel)
#
#
# These functions should return new DataFrame columns.
#
#
# =============================================================================

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bo_experiment_config import TOP_K_PROGRAMS


def _to_kernel_input(x: Any) -> np.ndarray:
	return np.asarray([x], dtype=object)


def _safe_kernel_scalar(kernel, x: Any, y: Any) -> float:
	try:
		matrix = np.asarray(kernel(_to_kernel_input(x), _to_kernel_input(y)), dtype=float)
	except Exception:
		return float("nan")
	if matrix.size == 0:
		return float("nan")
	return float(matrix.reshape(-1)[0])


def kernel_distance(kernel, x: Any, y: Any) -> float:
	"""Compute kernel-induced distance d(x,y)=sqrt(kxx+kyy-2kxy)."""
	k_xx = _safe_kernel_scalar(kernel, x, x)
	k_yy = _safe_kernel_scalar(kernel, y, y)
	k_xy = _safe_kernel_scalar(kernel, x, y)

	if not (np.isfinite(k_xx) and np.isfinite(k_yy) and np.isfinite(k_xy)):
		return float("nan")

	dist_sq = max(k_xx + k_yy - 2.0 * k_xy, 0.0)
	return float(np.sqrt(dist_sq))


def _resolve_tree_column(df: pd.DataFrame, tree_column: str | None) -> str:
	if tree_column is not None:
		if tree_column not in df.columns:
			raise ValueError(f"tree column '{tree_column}' not found")
		return tree_column

	for candidate in ("program", "tree", "x", "candidate"):
		if candidate in df.columns:
			return candidate
	raise ValueError("No tree/program column found. Expected one of: program, tree, x, candidate")


def compute_distance_to_best(
	df: pd.DataFrame,
	kernel,
	*,
	objective_column: str = "objective_value",
	tree_column: str | None = None,
	output_column: str = "distance_to_best",
) -> pd.DataFrame:
	"""Append distance of each program to the best (minimum objective) program."""
	if df.empty:
		result = df.copy()
		result[output_column] = pd.Series(dtype=float)
		return result

	if objective_column not in df.columns:
		raise ValueError(f"objective column '{objective_column}' not found")

	resolved_tree_column = _resolve_tree_column(df, tree_column)
	result = df.copy()

	best_idx = result[objective_column].astype(float).idxmin()
	best_program = result.loc[best_idx, resolved_tree_column]

	result[output_column] = [
		kernel_distance(kernel, program, best_program)
		for program in result[resolved_tree_column].tolist()
	]
	return result


def compute_distance_to_topk(
	df: pd.DataFrame,
	kernel,
	*,
	top_k: int = TOP_K_PROGRAMS,
	objective_column: str = "objective_value",
	tree_column: str | None = None,
	output_column: str = "distance_to_topk",
) -> pd.DataFrame:
	"""Append mean distance of each program to the top-k (minimum objective) set."""
	if df.empty:
		result = df.copy()
		result[output_column] = pd.Series(dtype=float)
		return result

	if objective_column not in df.columns:
		raise ValueError(f"objective column '{objective_column}' not found")
	if top_k <= 0:
		raise ValueError("top_k must be > 0")

	resolved_tree_column = _resolve_tree_column(df, tree_column)
	result = df.copy()

	topk_df = result.nsmallest(min(top_k, len(result)), columns=objective_column)
	topk_programs = topk_df[resolved_tree_column].tolist()

	distances: list[float] = []
	for program in result[resolved_tree_column].tolist():
		vals = [kernel_distance(kernel, program, top_program) for top_program in topk_programs]
		finite_vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
		if finite_vals.size == 0:
			distances.append(float("nan"))
		else:
			distances.append(float(np.mean(finite_vals)))

	result[output_column] = distances
	return result
