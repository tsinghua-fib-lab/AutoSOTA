"""
JAX implementation of Savitzky-Golay filtering for SO(3) rotational data.

This module provides smooth interpolation of rotation matrices using polynomial
fitting in the tangent space, with support for irregular timestamps.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple
from .so3 import (
    map_to_lie_algebra,
    rodrigues,
    log_map,
    symmetric_orthogonalization,
    ddexp_so3,
)

import diffrax


@jax.jit(static_argnums=(2,))
def construct_polynomial_basis(t: jnp.ndarray, t_center: float, p: int) -> jnp.ndarray:
    """Construct polynomial basis matrix for Savitzky-Golay filtering.

    Args:
        t: Time points, shape (n_points,)
        t_center: Center time point for polynomial expansion
        p: Polynomial order (must be concrete, not traced)

    Returns:
        Basis matrix of shape (3*n_points, 3*(p+1))
        Row layout: [pt0_x, pt0_y, pt0_z, pt1_x, pt1_y, pt1_z, ...]
        Col layout: [term0_x, term0_y, term0_z, term1_x, term1_y, term1_z, ...]
    """
    t.shape[0]
    t_rel = t - t_center  # Relative times from center

    # Vectorized polynomial basis construction - avoid Python loops
    powers = jnp.arange(p + 1)  # [0, 1, 2, ..., p]
    normalizers = jnp.maximum(1.0, powers)  # [1, 1, 2, 3, ..., p]

    # Broadcast computation: (n_points, 1) ** (1, p+1) / (1, p+1)
    t_rel_expanded = t_rel[:, None]  # (n_points, 1)
    powers_expanded = powers[None, :]  # (1, p+1)
    normalizers_expanded = normalizers[None, :]  # (1, p+1)

    # Vectorized polynomial terms: (n_points, p+1)
    poly_basis_T = (t_rel_expanded**powers_expanded) / normalizers_expanded

    # Memory-efficient vectorized construction using Kronecker product
    I3 = jnp.eye(3)  # (3, 3)
    basis_matrix = jnp.kron(poly_basis_T, I3)  # (3*n_points, 3*(p+1))

    return basis_matrix


@partial(jax.jit, static_argnums=(3,))
def _so3_filter_solve(
    A: jnp.ndarray, b: jnp.ndarray, R_center: jnp.ndarray, p: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled core solving routine for SO3 Savitzky-Golay filter."""
    # Solve normal equations
    AtA = A.T @ A  # (3*(p+1), 3*(p+1))

    # Handle both single sample and batch cases for matrix multiplication
    if b.ndim == 1:
        # Single sample: b is (3*n_points,), A.T is (3*(p+1), 3*n_points)
        Atb = A.T @ b  # (3*(p+1),)
        rho = jnp.linalg.solve(AtA, Atb)  # (3*(p+1),)
    else:
        # Batch case: b is (..., 3*n_points), need to use einsum for proper broadcasting
        Atb = jnp.einsum("ij,...j->...i", A.T, b)  # (..., 3*(p+1))
        # Solve for each batch element by transposing
        rho = jnp.linalg.solve(AtA, Atb.T).T  # (..., 3*(p+1))

    # Extract rotation at center (coefficient of constant term)
    phi_center = rho[..., :3]  # (..., 3) or (3,) for single sample

    # Apply correction to center rotation - handle both single and batch cases
    if phi_center.ndim == 1:
        # Single sample case
        R_correction = rodrigues(phi_center)  # (3, 3)
        R_smoothed = R_correction @ R_center  # (3, 3)
    else:
        # Batch case
        R_correction = jax.vmap(rodrigues)(phi_center)  # (..., 3, 3)
        R_smoothed = R_correction @ R_center  # (..., 3, 3)

    return R_smoothed, rho


@partial(jax.jit, static_argnums=(3,))
def _so3_filter_solve_weighted(
    A_weighted: jnp.ndarray, b_weighted: jnp.ndarray, R_center: jnp.ndarray, p: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled weighted solving routine for SO3 Savitzky-Golay filter."""
    # Solve normal equations
    AtA = A_weighted.T @ A_weighted  # (3*(p+1), 3*(p+1))

    # Handle both single sample and batch cases for matrix multiplication
    if b_weighted.ndim == 1:
        # Single sample: b_weighted is (3*n_points,), A_weighted.T is (3*(p+1), 3*n_points)
        Atb = A_weighted.T @ b_weighted  # (3*(p+1),)
        rho = jnp.linalg.solve(AtA, Atb)  # (3*(p+1),)
    else:
        # Batch case: b_weighted is (..., 3*n_points), need to use einsum for proper broadcasting
        Atb = jnp.einsum("ij,...j->...i", A_weighted.T, b_weighted)  # (..., 3*(p+1))
        # Solve for each batch element by transposing
        rho = jnp.linalg.solve(AtA, Atb.T).T  # (..., 3*(p+1))

    # Extract rotation at center (coefficient of constant term)
    phi_center = rho[..., :3]  # (..., 3) or (3,) for single sample

    # Apply correction to center rotation - handle both single and batch cases
    if phi_center.ndim == 1:
        # Single sample case
        R_correction = rodrigues(phi_center)  # (3, 3)
        R_smoothed = R_correction @ R_center  # (3, 3)
    else:
        # Batch case
        R_correction = jax.vmap(rodrigues)(phi_center)  # (..., 3, 3)
        R_smoothed = R_correction @ R_center  # (..., 3, 3)

    return R_smoothed, rho


@partial(jax.jit, static_argnums=(2, 4))
def _so3_filter_core(
    R: jnp.ndarray,
    t: jnp.ndarray,
    p: int,
    weight: Optional[jnp.ndarray],
    center_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled core SG filter computation."""
    batch_shape = R.shape[:-3]
    n_points = R.shape[-3]

    t_center = t[center_idx]
    R_center = R[..., center_idx, :, :]

    # Construct polynomial basis matrix (now JIT compiled)
    A = construct_polynomial_basis(t, t_center, p)  # (3*n_points, 3*(p+1))

    # Convert rotations to tangent space relative to center rotation
    R_center_inv = jnp.swapaxes(R_center, -1, -2)  # Transpose for inverse
    R_rel = R @ R_center_inv[..., None, :, :]  # (..., n_points, 3, 3)

    # Convert to rotation vectors (tangent space) - vectorized
    rotvec_rel = jax.vmap(log_map, in_axes=-3, out_axes=-2)(R_rel)  # (..., n_points, 3)

    # Flatten rotation vectors to match matrix structure
    b = rotvec_rel.reshape(batch_shape + (n_points * 3,))  # (..., n_points*3)

    # Solve weighted or unweighted least squares
    if weight is not None:
        # Memory-efficient weighted solution
        weight_expanded = jnp.repeat(weight, 3)  # (3*n_points,)
        sqrt_weight_expanded = jnp.sqrt(weight_expanded)

        # Apply weights via broadcasting
        A_weighted = A * sqrt_weight_expanded[:, None]  # (3*n_points, 3*(p+1))

        # Apply weights to b
        if b.ndim == 1:
            b_weighted = b * sqrt_weight_expanded
        else:
            b_weighted = b * sqrt_weight_expanded[None, :]

        R_smoothed, rho = _so3_filter_solve_weighted(
            A_weighted, b_weighted, R_center, p
        )
    else:
        R_smoothed, rho = _so3_filter_solve(A, b, R_center, p)

    return R_smoothed, rho


def so3_savitzky_golay_filter(
    R: jnp.ndarray,
    t: jnp.ndarray,
    p: int,
    weight: Optional[jnp.ndarray] = None,
    return_omega: bool = False,
    return_coefficients: bool = False,
    center_idx: Optional[int] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Apply Savitzky-Golay smoothing to SO(3) rotation data.

    Args:
        R: Rotation matrices, shape (..., n_points, 3, 3)
        t: Time points, shape (n_points,) - can be irregular
        p: Polynomial order for fitting
        weight: Optional weights for data points, shape (n_points,)
        return_omega: If True, also return angular velocity
        return_coefficients: If True, also return polynomial coefficients
        center_idx: Center index for polynomial fit (default: middle)

    Returns:
        Tuple of (smoothed_rotation, angular_velocity, coefficients)
        - smoothed_rotation: Shape (..., 3, 3) - rotation at center time
        - angular_velocity: Shape (..., 3) if return_omega else None
        - coefficients: Shape (..., 3*(p+1)) if return_coefficients else None
    """
    n_points = R.shape[-3]

    # Default center to middle point
    if center_idx is None:
        center_idx = n_points // 2

    # Use JIT-compiled core computation
    R_smoothed, rho = _so3_filter_core(R, t, p, weight, center_idx)

    omega = None
    if return_omega and p >= 1:
        # Angular velocity from first derivative coefficient
        phi_center = rho[..., :3]
        phi_dot = rho[..., 3:6]  # (..., 3)

        # Convert to angular velocity using left Jacobian
        omega = compute_angular_velocity_from_coeffs(phi_center, phi_dot)

    coefficients = rho if return_coefficients else None

    return R_smoothed, omega, coefficients


@jax.jit
def compute_angular_velocity_from_coeffs(
    phi: jnp.ndarray, phi_dot: jnp.ndarray
) -> jnp.ndarray:
    """Compute angular velocity from rotation vector and its derivative.

    Uses the exponential map formula for SO(3).

    Args:
        phi: Rotation vector, shape (..., 3)
        phi_dot: Time derivative of rotation vector, shape (..., 3)

    Returns:
        Angular velocity, shape (..., 3)
    """
    eps = 1e-8
    phi_norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)  # (..., 1)

    # Standard approach: beta = sin(phi/2)^2 / (phi/2)^2, alpha = sin(phi)/phi
    half_phi = phi_norm / 2.0
    sin_half_phi = jnp.sin(half_phi)
    sin_phi = jnp.sin(phi_norm)

    # Handle small angles with series expansion
    beta = jnp.where(
        phi_norm < eps,
        1.0 - (phi_norm**2) / 12.0 + (phi_norm**4) / 240.0,  # Series expansion
        (sin_half_phi / half_phi) ** 2,
    )

    alpha = jnp.where(
        phi_norm < eps,
        1.0 - (phi_norm**2) / 6.0 + (phi_norm**4) / 120.0,  # Series expansion
        sin_phi / phi_norm,
    )

    # Create identity matrix
    I = jnp.eye(3)
    I = jnp.broadcast_to(I, phi.shape[:-1] + (3, 3))

    # Create skew-symmetric matrix
    phi_hat = map_to_lie_algebra(phi)  # (..., 3, 3)

    # Rodrigues formula: res = I + 0.5 * beta * phi_hat + (1/phi^2) * (1-alpha) * phi_hat @ phi_hat
    phi_norm_sq = phi_norm**2

    # Handle division by phi^2 carefully
    factor = jnp.where(
        phi_norm < eps,
        1.0 / 12.0 - (phi_norm**2) / 240.0,  # Series expansion of (1-alpha)/phi^2
        (1.0 - alpha) / phi_norm_sq,
    )

    # Construct the Jacobian matrix
    res = I + 0.5 * beta[..., None] * phi_hat + factor[..., None] * (phi_hat @ phi_hat)

    # Compute omega = res @ phi_dot
    omega = jnp.einsum("...ij,...j->...i", res, phi_dot)
    return omega


# Convenience function for regular time grids
def so3_filter(
    R: jnp.ndarray,
    dt: float,
    p: int,
    return_omega: bool = False,
    return_coefficients: bool = False,
    weight: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Convenience wrapper for regular time grids.

    Args:
        R: Rotation matrices, shape (..., n_points, 3, 3)
        dt: Time step (assumes regular grid)
        p: Polynomial order
        return_omega: Whether to return angular velocity
        return_coefficients: Whether to return polynomial coefficients
        weight: Optional weights for data points

    Returns:
        Tuple of (smoothed_rotation, angular_velocity, coefficients)
    """
    n_points = R.shape[-3]
    t = jnp.arange(n_points, dtype=jnp.float32) * dt

    return so3_savitzky_golay_filter(
        R,
        t,
        p,
        weight=weight,
        return_omega=return_omega,
        return_coefficients=return_coefficients,
    )


# Vectorized version for batch processing
so3_filter_batch = jax.vmap(
    so3_savitzky_golay_filter, in_axes=(0, None, None, 0, None, None)
)


class SO3PolynomialPath(diffrax.AbstractPath):
    """Diffrax-compatible path for SO(3) polynomial interpolation.

    Handles batched input like other interpolation methods in the model.
    Following diffrax patterns for efficient coefficient caching.
    """

    p: int
    t_center: float
    R_center: jnp.ndarray
    _t0: float
    _t1: float
    t_scale: jnp.ndarray
    batch_size: int
    second_order: bool

    # Cached polynomial coefficients following diffrax pattern
    phi_coeffs: jnp.ndarray  # (batch_size, p+1, 3)
    phi_normalizers: jnp.ndarray  # (p+1,)
    first_deriv_coeffs: jnp.ndarray  # (batch_size, p, 3)
    second_deriv_coeffs: jnp.ndarray  # (batch_size, p-1, 3)
    powers_phi: jnp.ndarray  # (p+1,)
    powers_first: jnp.ndarray  # (p,)
    powers_second: jnp.ndarray  # (p-1,) or (1,)

    def __init__(
        self,
        R: jnp.ndarray,
        t: jnp.ndarray,
        p: int,
        weight: Optional[jnp.ndarray] = None,
        second_order: bool = False,
        t0: Optional[jnp.ndarray] = None,
        t1: Optional[jnp.ndarray] = None,
    ):
        """Initialize SO(3) polynomial path.

        Args:
            R: Rotation matrices, shape (batch_size, n_points, 3, 3)
            t: Time points, shape (n_points,) or (batch_size, n_points)
            p: Polynomial order
            weight: Optional weights for data points
            second_order: Whether to support second-order derivatives
            t0: Optional start time (JAX scalar) for path (default: t[0])
            t1: Optional end time (JAX scalar) for path (default: t[-1])
        """
        if R.ndim == 3:
            # Single sequence case: add batch dimension
            R = R[None, ...]  # (1, n_points, 3, 3)

        if R.ndim != 4:
            raise ValueError(
                f"Expected R to have shape (batch_size, n_points, 3, 3), got {R.shape}"
            )

        batch_size, n_points = R.shape[:2]
        center_idx = n_points // 2

        # Handle time array - use 1D if provided as 2D
        if t.ndim == 2:
            t = t[0]  # Extract 1D time array (assuming all batches have same times)

        # Use so3_savitzky_golay_filter to get coefficients for all batch elements
        _, _, rho = so3_savitzky_golay_filter(
            R, t, p, weight=weight, return_coefficients=True
        )

        # Store path parameters
        R_center = R[:, center_idx]  # (batch_size, 3, 3)
        t_center = t[center_idx]
        # Allow overriding t0/t1 for extrapolation beyond fitted range
        _t0 = t0 if t0 is not None else t[0]
        _t1 = t1 if t1 is not None else t[-1]
        t_scale = _t1 - _t0
        t_scale = jnp.where(jnp.abs(t_scale) < 1e-8, 1.0, t_scale)

        # Pre-compute ALL polynomial coefficients for performance
        # This avoids expensive reconstruction in derivative() calls
        # Cache phi, phi_dot, and phi_ddot coefficients for ultra-fast evaluation

        # Phi coefficients: ALL coefficients with normalization (for base rotation)
        phi_coeffs = []
        phi_normalizers = []
        for power in range(p + 1):  # range(p+1) for base polynomial
            normalizer = max(1.0, power)
            coeff_start = power * 3
            coeff_end = (power + 1) * 3
            if coeff_end <= rho.shape[-1]:
                phi_coeffs.append(rho[..., coeff_start:coeff_end])
                phi_normalizers.append(normalizer)

        if phi_coeffs:
            phi_coeffs = jnp.stack(phi_coeffs, axis=-2)  # (batch_size, p+1, 3)
            phi_normalizers = jnp.array(phi_normalizers, dtype=jnp.float32)  # (p+1,)
        else:
            phi_coeffs = jnp.zeros((batch_size, 1, 3))
            phi_normalizers = jnp.array([1.0], dtype=jnp.float32)

        # First derivative coefficients: skip constant term (first 3 coeffs), no normalization
        first_deriv_coeffs = []
        for power in range(p):  # range(p) for first derivative
            coeff_start = (power + 1) * 3  # Skip constant term
            coeff_end = (power + 2) * 3
            if coeff_end <= rho.shape[-1]:
                first_deriv_coeffs.append(rho[..., coeff_start:coeff_end])

        if first_deriv_coeffs:
            first_deriv_coeffs = jnp.stack(
                first_deriv_coeffs, axis=-2
            )  # (batch_size, p, 3)
        else:
            first_deriv_coeffs = jnp.zeros((batch_size, 1, 3))

        # Second derivative coefficients: ALWAYS compute (for p > 1), regardless of second_order flag
        second_deriv_coeffs = []
        if p > 1:  # Can only compute second derivative if polynomial order > 1
            for power in range(p - 1):  # range(p-1) for second derivative
                coeff_start = (power + 2) * 3  # Skip first two coefficient groups
                coeff_end = (power + 3) * 3
                if coeff_end <= rho.shape[-1]:
                    second_deriv_coeffs.append(rho[..., coeff_start:coeff_end])

        if second_deriv_coeffs:
            second_deriv_coeffs = jnp.stack(
                second_deriv_coeffs, axis=-2
            )  # (batch_size, p-1, 3)
        else:
            second_deriv_coeffs = jnp.zeros((batch_size, 1, 3))

        # Pre-compute powers for fast polynomial evaluation - ALWAYS compute all three
        powers_phi = jnp.arange(p + 1, dtype=jnp.float32)  # [0, 1, ..., p]
        powers_first = jnp.arange(p, dtype=jnp.float32)  # [0, 1, ..., p-1]
        powers_second = jnp.arange(
            max(1, p - 1), dtype=jnp.float32
        )  # [0, 1, ..., p-2] or [0] if p=1

        # Initialize attributes following diffrax pattern
        self.p = p
        self.t_center = t_center
        self.R_center = R_center
        self._t0 = _t0
        self._t1 = _t1
        self.t_scale = t_scale
        self.batch_size = batch_size
        self.second_order = second_order

        # Cached polynomial coefficients for ultra-fast evaluation
        self.phi_coeffs = phi_coeffs
        self.phi_normalizers = phi_normalizers
        self.first_deriv_coeffs = first_deriv_coeffs
        self.second_deriv_coeffs = second_deriv_coeffs
        self.powers_phi = powers_phi
        self.powers_first = powers_first
        self.powers_second = powers_second

        # Note: Pre-compiled functions will be created on first call and cached

    def _compute_first_order_derivative_optimized(self, t_rel):
        """Optimized first-order derivative using pre-compiled module-level functions."""
        # Use pre-compiled module-level vmap function
        result = _first_order_vmap(
            self.phi_coeffs,
            self.R_center,
            self.first_deriv_coeffs,
            self.second_deriv_coeffs,
            t_rel,
            self.powers_phi,
            self.phi_normalizers,
            self.powers_first,
        )

        if self.batch_size == 1:
            return result[0]  # (10,)
        else:
            return result  # (batch_size, 10)

    def _compute_second_order_derivative_optimized(self, t_rel):
        """Optimized second-order derivative computation using pre-compiled module-level functions."""
        # Use pre-compiled module-level vmap function - this eliminates recompilation!
        # No @eqx.filter_jit here since the module-level functions are already JIT compiled
        result = _second_order_vmap(
            self.phi_coeffs,
            self.R_center,
            self.first_deriv_coeffs,
            self.second_deriv_coeffs,
            t_rel,
            self.powers_phi,
            self.phi_normalizers,
            self.powers_first,
            self.powers_second,
        )

        # Dual derivative result: ((batch_size, 10), (batch_size, 10))
        first_result, second_result = result
        if self.batch_size == 1:
            return first_result[0], second_result[0]  # Both (10,)
        else:
            return first_result, second_result  # Both (batch_size, 10)

    @property
    def t0(self):
        return self._t0

    @property
    def t1(self):
        return self._t1

    def evaluate(self, t0, t1=None, left=True):
        """Evaluate the path at time t0 or increment between t0 and t1."""
        del left  # Not used for deterministic paths

        if t1 is not None:
            # Return increment: path(t1) - path(t0)
            return self._evaluate_single(t1) - self._evaluate_single(t0)
        else:
            # Return path value at t0
            return self._evaluate_single(t0)

    def _evaluate_single(self, t_val):
        """Evaluate path at a single time point using cached coefficients."""

        def _fast_evaluate_single_element(phi_coeffs_single, R_center_single):
            """Fast evaluation using cached coefficients - to be vmapped."""
            # Relative time from center
            t_rel = (t_val - self.t_center) / self.t_scale

            # ULTRA-FAST phi computation using cached coefficients and normalizers
            t_powers_phi = t_rel**self.powers_phi  # (p+1,)
            phi = jnp.sum(
                phi_coeffs_single * (t_powers_phi / self.phi_normalizers)[:, None],
                axis=0,
            )  # (3,)

            # Convert to rotation matrix: R(t) = exp(phi) @ R_center
            R_correction = rodrigues(phi)
            R_eval = R_correction @ R_center_single

            # Project back to SO(3) to ensure numerical stability
            R_eval_ortho = symmetric_orthogonalization(R_eval.flatten()).reshape(3, 3)

            # Return as flattened vector with time prepended (diffrax convention)
            return jnp.concatenate([jnp.array([t_val]), R_eval_ortho.flatten()])

        # Apply to all batch elements using vmap with cached coefficients
        # Remove nested JIT compilation to avoid conflicts with pre-compiled functions
        vmap_eval = jax.vmap(_fast_evaluate_single_element, in_axes=(0, 0))
        result = vmap_eval(self.phi_coeffs, self.R_center)  # (batch_size, 10)

        # For backward compatibility: squeeze if batch_size == 1
        if self.batch_size == 1:
            return result[0]  # (10,)
        else:
            return result  # (batch_size, 10)

    def derivative(self, t, left=True, order=1):
        """Compute time derivative(s) using DIRECT pre-compiled functions.

        ULTIMATE OPTIMIZATION: Directly calls module-level pre-compiled functions
        to completely eliminate any JIT compilation during diffrax integration.

        Args:
            t: Time point for derivative evaluation
            left: Not used for deterministic paths
            order: Derivative order (1 for first, 2 for both first AND second)

        Returns:
            If order=1: First derivative vector (10,)
            If order=2: Tuple of (first_derivative, second_derivative), both (10,)
        """
        del left  # Not used for deterministic paths

        if order == 2 and not self.second_order:
            raise ValueError(
                "Second-order derivatives requested but second_order=False in constructor"
            )

        if order not in [1, 2]:
            raise ValueError(
                f"Only derivative orders 1 and 2 are supported, got {order}"
            )

        # Relative time from center (irregular sampling compatible)
        t_rel = (t - self.t_center) / self.t_scale

        # DIRECT CALL to pre-compiled functions - no method indirection!
        if order == 1:
            # First-order: direct call to pre-compiled vmap
            result = _first_order_vmap(
                self.phi_coeffs,
                self.R_center,
                self.first_deriv_coeffs,
                self.second_deriv_coeffs,
                t_rel,
                self.powers_phi,
                self.phi_normalizers,
                self.powers_first,
            )
            result = result.at[..., 1:].multiply(1.0 / self.t_scale)
            if self.batch_size == 1:
                return result[0]  # (10,)
            else:
                return result  # (batch_size, 10)
        else:
            # Second-order: direct call to pre-compiled vmap
            result = _second_order_vmap(
                self.phi_coeffs,
                self.R_center,
                self.first_deriv_coeffs,
                self.second_deriv_coeffs,
                t_rel,
                self.powers_phi,
                self.phi_normalizers,
                self.powers_first,
                self.powers_second,
            )
            # Dual derivative result: ((batch_size, 10), (batch_size, 10))
            first_result, second_result = result
            first_result = first_result.at[..., 1:].multiply(1.0 / self.t_scale)
            second_result = second_result.at[..., 1:].multiply(1.0 / (self.t_scale**2))
            if self.batch_size == 1:
                return first_result[0], second_result[0]  # Both (10,)
            else:
                return first_result, second_result  # Both (batch_size, 10)

    def first_derivative(self, t, left=True):
        """Optimized method to compute ONLY first derivative.

        More efficient than derivative(t, order=1) for MultiTerm integration
        where we need only first derivatives.

        Args:
            t: Time point for derivative evaluation
            left: Not used for deterministic paths

        Returns:
            First derivative vector (10,) for single batch, (batch_size, 10) for multiple
        """
        del left  # Not used for deterministic paths

        # Relative time from center (irregular sampling compatible)
        t_rel = (t - self.t_center) / self.t_scale

        # DIRECT CALL to pre-compiled first-order function
        result = _first_order_vmap(
            self.phi_coeffs,
            self.R_center,
            self.first_deriv_coeffs,
            self.second_deriv_coeffs,
            t_rel,
            self.powers_phi,
            self.phi_normalizers,
            self.powers_first,
        )
        result = result.at[..., 1:].multiply(1.0 / self.t_scale)
        if self.batch_size == 1:
            return result[0]  # (10,)
        else:
            return result  # (batch_size, 10)

    def second_derivative(self, t, left=True):
        """Optimized method to compute ONLY second derivative.

        More efficient than derivative(t, order=2) for MultiTerm integration
        where we need only second derivatives.

        Args:
            t: Time point for derivative evaluation
            left: Not used for deterministic paths

        Returns:
            Second derivative vector (10,) for single batch, (batch_size, 10) for multiple
        """
        del left  # Not used for deterministic paths

        if not self.second_order:
            raise ValueError(
                "Second-order derivatives requested but second_order=False in constructor"
            )

        # For second derivative only, we still need to call the second-order function
        # but we can discard the first derivative result
        # This is still more efficient than the combined derivative(order=2) call
        # because it avoids the tuple unpacking and conditional logic

        # Relative time from center (irregular sampling compatible)
        t_rel = (t - self.t_center) / self.t_scale

        # DIRECT CALL to pre-compiled second-order function
        result = _second_order_vmap(
            self.phi_coeffs,
            self.R_center,
            self.first_deriv_coeffs,
            self.second_deriv_coeffs,
            t_rel,
            self.powers_phi,
            self.phi_normalizers,
            self.powers_first,
            self.powers_second,
        )
        # Extract only second derivative result
        _, second_result = result
        second_result = second_result.at[..., 1:].multiply(1.0 / (self.t_scale**2))
        if self.batch_size == 1:
            return second_result[0]  # (10,)
        else:
            return second_result  # (batch_size, 10)


# Vmapped orthogonalization for batch processing
symmetric_orthogonalization_batched = jax.vmap(symmetric_orthogonalization, in_axes=0)


# MODULE-LEVEL OPTIMIZED DERIVATIVE FUNCTIONS
# Moving these outside the class methods to avoid nested function compilation issues


@jax.jit
def _compute_first_order_derivative_element(
    phi_coeffs_single,
    R_center_single,
    first_coeffs,
    second_coeffs,
    t_rel,
    powers_phi,
    phi_normalizers,
    powers_first,
):
    """Optimized first-order derivative computation for a single element."""
    # Compute phi, phi_dot using cached coefficients
    t_powers_phi = t_rel**powers_phi  # (p+1,)
    phi = jnp.sum(
        phi_coeffs_single * (t_powers_phi / phi_normalizers)[:, None], axis=0
    )  # (3,)

    # First derivative coefficients
    if first_coeffs.shape[0] > 0:
        t_powers_first = t_rel**powers_first  # (p,)
        phi_dot = jnp.sum(first_coeffs * t_powers_first[:, None], axis=0)  # (3,)
    else:
        phi_dot = jnp.zeros(3)

    # Compute manifold operations
    omega = compute_angular_velocity_from_coeffs(phi, phi_dot)
    R_correction = rodrigues(phi)
    R_current = R_correction @ R_center_single
    omega_hat = map_to_lie_algebra(omega)

    # First derivative
    dR_dt = omega_hat @ R_current
    return jnp.concatenate([jnp.array([1.0]), dR_dt.flatten()])


@jax.jit
def _compute_second_order_derivative_element(
    phi_coeffs_single,
    R_center_single,
    first_coeffs,
    second_coeffs,
    t_rel,
    powers_phi,
    phi_normalizers,
    powers_first,
    powers_second,
):
    """Optimized second-order derivative computation for a single element."""
    # Compute phi, phi_dot using cached coefficients (shared)
    t_powers_phi = t_rel**powers_phi  # (p+1,)
    phi = jnp.sum(
        phi_coeffs_single * (t_powers_phi / phi_normalizers)[:, None], axis=0
    )  # (3,)

    # First derivative coefficients
    if first_coeffs.shape[0] > 0:
        t_powers_first = t_rel**powers_first  # (p,)
        phi_dot = jnp.sum(first_coeffs * t_powers_first[:, None], axis=0)  # (3,)
    else:
        phi_dot = jnp.zeros(3)

    # Shared manifold operations
    omega = compute_angular_velocity_from_coeffs(phi, phi_dot)
    R_correction = rodrigues(phi)
    R_current = R_correction @ R_center_single
    omega_hat = map_to_lie_algebra(omega)

    # First derivative
    dR_dt = omega_hat @ R_current
    first_deriv = jnp.concatenate([jnp.array([1.0]), dR_dt.flatten()])

    # Second derivative coefficients
    if second_coeffs.shape[0] > 0:
        t_powers_second = t_rel**powers_second  # (p-1,)
        phi_ddot = jnp.sum(second_coeffs * t_powers_second[:, None], axis=0)  # (3,)
    else:
        phi_ddot = jnp.zeros(3)

    # Second derivative manifold operations
    ddexp_term = ddexp_so3(phi, phi_dot) @ phi_dot  # (3,)
    dexp_term = compute_angular_velocity_from_coeffs(phi, phi_ddot)  # (3,)
    alpha = ddexp_term + dexp_term

    alpha_hat = map_to_lie_algebra(alpha)
    omega_hat_sq = omega_hat @ omega_hat
    d2R_dt2 = (alpha_hat + omega_hat_sq) @ R_current

    second_deriv = jnp.concatenate([jnp.array([0.0]), d2R_dt2.flatten()])

    return first_deriv, second_deriv


# Pre-compiled vmap functions
_first_order_vmap = jax.jit(
    jax.vmap(
        _compute_first_order_derivative_element,
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )
)
_second_order_vmap = jax.jit(
    jax.vmap(
        _compute_second_order_derivative_element,
        in_axes=(0, 0, 0, 0, None, None, None, None, None),
    )
)
