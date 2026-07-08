"""
FILE: dro.py
INPUT: Normalized sample matrices and DRO calibration settings.
OUTPUT: DRO radius estimates and spectral self-regression coefficient matrices.
POS: Core optimization logic for DRO subspace clustering.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_ADMM_RHO = 1.0
DEFAULT_RHO_MU = 5.0
DEFAULT_RHO_TAU = 2.0
DEFAULT_TOL_ABS = 1e-5
DEFAULT_TOL_REL = 1e-4
DEFAULT_STEP_TOL_ABS = 1e-6
DEFAULT_STEP_TOL_REL = 1e-3
DEFAULT_STEP_TOL_FUNC = 1e-5
DEFAULT_STEP_LEARNING_RATE = 0.01
DEFAULT_STEP_LEARNING_RATE_DECAY = 0.01
DEFAULT_STEP_NESTEROV_GAMMA = 0.9
DEFAULT_STEP_ITERATIONS = 5000
DEFAULT_OUTER_ITERATIONS = 5000
DEFAULT_RESTARTS = 2

logger = logging.getLogger(__name__)


def dro_sqrt_delta(
    covariance: FloatArray,
    n_samples: int,
    samples: FloatArray | None = None,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    seed: int = 0,
    new_version: bool = True,
    diagonal_inverse: bool = True,
    use_simple_z: bool = True,
) -> float:
    """Calibrate sqrt(delta) for the DRO spectral penalty."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")

    dim = covariance.shape[0]
    rng = np.random.default_rng(seed)
    if use_simple_z:
        upsilon = np.expand_dims(np.diag(covariance), 1) @ np.expand_dims(np.diag(covariance), 0) + covariance**2
        z_values = rng.normal(size=(n_simulations, dim, dim)) * np.sqrt(upsilon)
    else:
        if samples is None:
            raise ValueError("samples must be provided when use_simple_z=False.")
        covariance_samples = np.cov((samples[:, :, None] * samples[:, None, :]).reshape([n_samples, dim * dim]).T)
        z_values = rng.multivariate_normal(np.zeros(covariance_samples.shape[0]), cov=covariance_samples, size=n_simulations)
        z_values = z_values.reshape([n_simulations, dim, dim])

    z_values = np.tril(z_values) + np.swapaxes(np.tril(z_values, -1), 1, 2)
    if new_version:
        if diagonal_inverse:
            diagonal = np.diag(covariance)
            if np.any(diagonal <= 0):
                raise ValueError("diagonal_inverse=True requires positive covariance diagonal entries.")
            covariance_inverse = np.diag(1.0 / diagonal)
        else:
            covariance_inverse = np.linalg.inv(covariance)
        radius_samples = (
            np.diagonal((z_values.swapaxes(1, 2) @ covariance_inverse @ z_values), axis1=1, axis2=2) / 4.0
        ).sum(axis=1)
    else:
        denominator = 4.0 * np.diagonal((z_values.swapaxes(1, 2) @ covariance @ z_values), axis1=1, axis2=2)
        if np.any(denominator == 0):
            raise ValueError("Legacy DRO calibration encountered zero denominator.")
        radius_samples = ((np.sum(z_values**2, axis=1) ** 2) / denominator).sum(axis=1)
    return float(np.sqrt(np.quantile(radius_samples, 1.0 - alpha) / n_samples))


def spectral_least_squares_svd(matrix: FloatArray, penalty: float) -> FloatArray:
    """Solve the spectral-norm proximal subproblem by singular value thresholding."""
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    rank = singular_values.shape[0]
    thresholds = np.clip(
        (singular_values.cumsum() - penalty / 2.0) / (np.arange(rank) + 1),
        a_min=np.pad(singular_values[1:], (0, 1), "constant"),
        a_max=singular_values,
    )
    loss = np.diag(np.square(singular_values - np.expand_dims(thresholds, 1)).cumsum(axis=1)) + penalty * thresholds
    selected = int(np.argmin(loss))
    adjusted = singular_values.copy()
    adjusted[: selected + 1] = thresholds[selected]
    return (left * adjusted) @ right


def _sqrt_ridge_matrix_grad(coefs: FloatArray, samples: FloatArray, shift: FloatArray, rho: float) -> tuple[FloatArray, float]:
    coefs_no_diag = coefs.copy()
    np.fill_diagonal(coefs_no_diag, 0.0)
    shifted = coefs_no_diag + shift
    residual = samples - samples.dot(coefs_no_diag)
    residual_norm = np.linalg.norm(residual, "fro")
    n_samples = samples.shape[0]
    if residual_norm == 0.0:
        gradient = rho * shifted
    else:
        gradient = rho * shifted - samples.T.dot(residual) / (np.sqrt(n_samples) * residual_norm)
    objective = residual_norm / np.sqrt(n_samples) + rho * np.linalg.norm(shifted, "fro") ** 2 / 2.0
    np.fill_diagonal(gradient, 0.0)
    return gradient, float(objective)


def gradient_descent(
    samples: FloatArray,
    shift: FloatArray,
    rho: float,
    learning_rate: float = DEFAULT_STEP_LEARNING_RATE,
    learning_rate_decay: float = DEFAULT_STEP_LEARNING_RATE_DECAY,
    nesterov_gamma: float = DEFAULT_STEP_NESTEROV_GAMMA,
    n_iterations: int = DEFAULT_STEP_ITERATIONS,
    tol_abs: float = DEFAULT_STEP_TOL_ABS,
    tol_rel: float = DEFAULT_STEP_TOL_REL,
    tol_func: float = DEFAULT_STEP_TOL_FUNC,
    max_restarts: int = DEFAULT_RESTARTS,
    seed: int = 0,
) -> FloatArray:
    """Optimize the smooth ADMM subproblem with bounded deterministic restarts."""
    if n_iterations <= 0 or max_restarts < 0:
        raise ValueError("n_iterations must be positive and max_restarts must be non-negative.")
    rng = np.random.default_rng(seed)
    dim = samples.shape[1]
    coefs = np.zeros((dim, dim))
    velocity = np.zeros_like(coefs)
    best_objective = np.inf
    best_coefs = coefs.copy()

    for restart_idx in range(max_restarts + 1):
        previous_objective = np.inf
        current_learning_rate = learning_rate * (0.9**restart_idx)
        for _ in range(n_iterations):
            lookahead = coefs - nesterov_gamma * velocity
            gradient, objective = _sqrt_ridge_matrix_grad(lookahead, samples, shift, rho)
            if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
                break
            if objective < best_objective:
                best_objective = objective
                best_coefs = coefs.copy()
            if objective > previous_objective:
                current_learning_rate *= np.exp(-learning_rate_decay)
            velocity = nesterov_gamma * velocity + current_learning_rate * gradient
            coefs = coefs - velocity
            coefs_nozero = coefs.copy()
            coefs_nozero[coefs_nozero == 0.0] = 1.0
            gradient_ok = bool(np.all(np.abs(gradient) <= tol_abs))
            update_ok = bool(np.all(np.abs(velocity / coefs_nozero) <= tol_rel))
            objective_ok = False
            if np.isfinite(previous_objective) and abs(previous_objective) > 0.0:
                relative_change = (objective - previous_objective) / abs(previous_objective)
                objective_ok = bool(-tol_func < relative_change <= 0.0)
            if gradient_ok or update_ok or objective_ok:
                np.fill_diagonal(coefs, 0.0)
                return coefs
            previous_objective = objective

        logger.info("Gradient descent restart %s reached without convergence.", restart_idx)
        coefs = best_coefs + 0.1 * rng.normal(size=(dim, dim))
        np.fill_diagonal(coefs, 0.0)
        velocity = np.zeros_like(coefs)

    raise RuntimeError(f"Gradient descent did not converge; best objective={best_objective:.8g}.")


def spectral_regression_admm(
    samples: FloatArray,
    penalty: float,
    rho: float = DEFAULT_ADMM_RHO,
    varying_rho_mu: float = DEFAULT_RHO_MU,
    varying_rho_tau: float = DEFAULT_RHO_TAU,
    tol_abs: float = DEFAULT_TOL_ABS,
    tol_rel: float = DEFAULT_TOL_REL,
    max_outer_iterations: int = DEFAULT_OUTER_ITERATIONS,
    step_seed: int = 0,
) -> FloatArray:
    """Run the spectral-regularized DRO self-regression ADMM solver."""
    if penalty <= 0.0 or rho <= 0.0:
        raise ValueError("penalty and rho must be positive.")
    if max_outer_iterations <= 0:
        raise ValueError("max_outer_iterations must be positive.")

    dim = samples.shape[1]
    coefs_step2 = np.zeros((dim, dim))
    dual = np.zeros((dim, dim))
    identity = np.eye(dim)

    for iteration in range(max_outer_iterations):
        shift = coefs_step2 - identity + dual
        coefs_step1 = gradient_descent(samples, shift, rho, seed=step_seed + iteration)
        previous_step2 = coefs_step2.copy()
        coefs_step2 = spectral_least_squares_svd(-coefs_step1 + identity - dual, penalty * 2.0 / rho)
        dual = dual + coefs_step1 + coefs_step2 - identity

        tol_primal = dim * tol_abs + tol_rel * max(
            np.linalg.norm(coefs_step1, "fro"), np.linalg.norm(coefs_step2, "fro"), np.sqrt(dim)
        )
        tol_dual = dim * tol_abs + tol_rel * rho * np.linalg.norm(dual, "fro")
        primal_error = np.linalg.norm(coefs_step1 + coefs_step2 - identity, "fro")
        dual_error = np.linalg.norm(-rho * (coefs_step2 - previous_step2), "fro")
        if primal_error <= tol_primal and dual_error <= tol_dual:
            np.fill_diagonal(coefs_step1, 0.0)
            return coefs_step1

        if varying_rho_mu > 0.0:
            if primal_error * tol_dual > varying_rho_mu * dual_error * tol_primal:
                rho *= varying_rho_tau
                dual /= varying_rho_tau
            elif dual_error * tol_primal > varying_rho_mu * primal_error * tol_dual:
                rho /= varying_rho_tau
                dual *= varying_rho_tau

    raise RuntimeError(
        "Spectral ADMM did not converge within "
        f"{max_outer_iterations} iterations; primal_error={primal_error:.8g}, dual_error={dual_error:.8g}."
    )


def compute_dro_coefficients(
    samples: FloatArray,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    seed: int = 0,
    admm_rho: float = DEFAULT_ADMM_RHO,
    max_outer_iterations: int = DEFAULT_OUTER_ITERATIONS,
) -> FloatArray:
    """Compute a DRO spectral self-regression matrix for normalized samples."""
    covariance = np.cov(samples.T)
    penalty = dro_sqrt_delta(
        covariance=covariance,
        n_samples=samples.shape[0],
        samples=samples,
        alpha=alpha,
        n_simulations=n_simulations,
        seed=seed,
        new_version=True,
        diagonal_inverse=True,
        use_simple_z=True,
    )
    return spectral_regression_admm(
        samples,
        penalty=penalty,
        rho=admm_rho,
        max_outer_iterations=max_outer_iterations,
        step_seed=seed,
    )
