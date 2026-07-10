"""
Curvature-parameterized Hyperboloid model (extrinsic coordinates), adapted from:
https://geomstats.github.io/_modules/geomstats/geometry/hyperboloid.html#Hyperboloid

Main change: introduce curvature kappa < 0 with radius R = 1 / sqrt(-kappa),
and scale all computations accordingly.
"""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.base import LevelSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.minkowski import MinkowskiMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out





def _as_hyp_f64(x):
    """Coerce points/tangents to float64 before hyperbolic (Minkowski) numerics."""
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64) if x.dtype != torch.float64 else x
    return gs.array(np.asarray(x, dtype=np.float64))


class MinkowskiMetricDeviceSafe(MinkowskiMetric):
    """Minkowski metric with a CUDA-safe inner product.

    The constant ``metric_matrix_`` is created on CPU at init; with the PyTorch
    backend, ``squared_norm`` / ``inner_product`` must einsum against tensors
    on the same device as the tangent vectors.
    """

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        inner_prod_mat = self.metric_matrix(base_point)
        ref = tangent_vec_a
        if hasattr(inner_prod_mat, "device") and hasattr(ref, "device"):
            if inner_prod_mat.device != ref.device or inner_prod_mat.dtype != ref.dtype:
                inner_prod_mat = inner_prod_mat.to(
                    device=ref.device, dtype=ref.dtype
                )
        aux = gs.einsum("...j,...jk->...k", tangent_vec_a, inner_prod_mat)
        return gs.dot(aux, tangent_vec_b)


def _taylor_exp_even_func_device_safe(point, taylor_function, order=5, tol=None):
    """geomstats ``taylor_exp_even_func`` with Taylor coeffs on ``point``'s device.

    The stock helper builds coefficient tensors on CPU while ``point`` can be CUDA,
    which breaks the einsum in ``exp`` / ``log``.
    """
    if tol is None:
        tol = utils.EPSILON
    coeffs = gs.array(taylor_function["coefficients"][:order])
    if hasattr(point, "device"):
        coeffs = coeffs.to(device=point.device, dtype=point.dtype)
    powers = gs.stack([point**k for k in range(order)], axis=0)
    approx = gs.einsum("k,k...->...", coeffs, powers)
    point_ = gs.where(gs.abs(point) <= tol, tol, point)
    exact = taylor_function["function"](gs.sqrt(point_))
    return gs.where(gs.abs(point) < tol, approx, exact)


class HyperboloidKappa(_Hyperbolic, LevelSet):
    r"""n-dimensional hyperbolic space in the hyperboloid model with curvature kappa<0.

    Points are embedded in Minkowski space R^{n+1} with signature (-,+,...,+) and satisfy:
        <x, x>_L = -R^2,   where R = 1 / sqrt(-kappa).

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    curvature : float
        Negative sectional curvature (kappa < 0). Default: -1.0.
    """

    def __init__(self, dim, curvature=-1.0, equip=True):
        if curvature >= 0:
            raise ValueError("HyperboloidKappa requires curvature < 0.")
        self.coords_type = "extrinsic"
        self.dim = dim
        self.curvature = float(curvature)
        self.radius = 1.0 / math.sqrt(-self.curvature)
        super().__init__(dim=dim, intrinsic=False, equip=equip)

    @staticmethod
    def default_metric():
        return HyperboloidMetricKappa

    def _define_embedding_space(self):
        space = Euclidean(self.dim + 1, equip=False)
        space.equip_with_metric(MinkowskiMetricDeviceSafe)
        return space

    def submersion(self, point):
        """Level-set function defining the hyperboloid: <x,x>_L + R^2 = 0."""
        point = _as_hyp_f64(point)
        return self.embedding_space.metric.squared_norm(point) + (self.radius ** 2)

    def tangent_submersion(self, vector, point):
        """Derivative of the submersion: <vector, point>_L."""
        vector = _as_hyp_f64(vector)
        point = _as_hyp_f64(point)
        return self.embedding_space.metric.inner_product(vector, point)

    def regularize(self, point):
        """Rescale point to satisfy <x,x>_L = -R^2 (canonical sheet)."""
        point = _as_hyp_f64(point)
        sq_norm = self.embedding_space.metric.squared_norm(point)
        # Handle numerical precision issues: if squared norm is too close to zero,
        # clip it to a small negative value to prevent division by zero
        # (Minkowski squared norm should be negative for hyperboloid points)
        eps = 1e-10
        sq_norm = gs.clip(sq_norm, -math.inf, -eps)
        real_norm = gs.sqrt(gs.abs(sq_norm))
        # Want |<x,x>_L| = R^2, so scale by (R / real_norm)
        return gs.einsum("...i,...->...i", point, self.radius / real_norm)

    def to_tangent(self, vector, base_point):
        """Project a vector in Minkowski space to T_{base_point}H."""
        vector = _as_hyp_f64(vector)
        base_point = _as_hyp_f64(base_point)
        sq_norm = self.embedding_space.metric.squared_norm(base_point)  # should be -R^2
        inner_prod = self.embedding_space.metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        return vector - gs.einsum("...,...j->...j", coef, base_point)

    # NOTE: projection() and intrinsic/extrinsic conversions in Geomstats route through
    # _Hyperbolic.change_coordinates_system and assume the kappa=-1 conventions.
    # If you need those, we can also adapt them, but you asked for "just the hyperboloid".


class HyperboloidMetricKappa(RiemannianMetric):
    """Riemannian metric operations on HyperboloidKappa."""

    def metric_matrix(self, base_point=None):
        if base_point is not None:
            base_point = _as_hyp_f64(base_point)
        return self._space.embedding_space.metric.metric_matrix(base_point)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        tangent_vec_a = _as_hyp_f64(tangent_vec_a)
        tangent_vec_b = _as_hyp_f64(tangent_vec_b)
        if base_point is not None:
            base_point = _as_hyp_f64(base_point)
        return self._space.embedding_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

    def squared_norm(self, vector, base_point=None):
        vector = _as_hyp_f64(vector)
        if base_point is not None:
            base_point = _as_hyp_f64(base_point)
        return self._space.embedding_space.metric.squared_norm(vector)

    def dist(self, point_a, point_b):
        point_a = _as_hyp_f64(point_a)
        point_b = _as_hyp_f64(point_b)
        embedding_metric = self._space.embedding_space.metric

        # Debug guard: fail early if NaNs appear
        if gs.any(gs.isnan(point_a)) or gs.any(gs.isnan(point_b)):
            print(point_a, point_b)
            raise ValueError("NaN in dist inputs (point_a or point_b).")

        sq_norm_a = embedding_metric.squared_norm(point_a)
        sq_norm_b = embedding_metric.squared_norm(point_b)

        if gs.any(gs.isnan(sq_norm_a)) or gs.any(gs.isnan(sq_norm_b)):
            raise ValueError("NaN in squared_norm inside dist.")

        inner_prod = embedding_metric.inner_product(point_a, point_b)

        den2 = gs.abs(sq_norm_a * sq_norm_b)
        den2 = gs.clip(den2, 0.0, math.inf)      # keeps it nonnegative; NaN already handled above
        den = gs.sqrt(den2)
        den = gs.clip(den, 1e-15, math.inf)

        cosh_angle = -inner_prod / den
        cosh_angle = gs.clip(cosh_angle, 1.0, 1e24)
        return self._space.radius * gs.arccosh(cosh_angle)


    def squared_dist(self, point_a, point_b):
        return self.dist(point_a, point_b) ** 2

    def exp(self, tangent_vec, base_point):
        """Riemannian exponential with curvature kappa<0 (radius R)."""
        tangent_vec = _as_hyp_f64(tangent_vec)
        base_point = _as_hyp_f64(base_point)
        R = self._space.radius
        sq_norm_tv = self._space.embedding_space.metric.squared_norm(tangent_vec)
        sq_norm_tv = gs.clip(sq_norm_tv, 0, math.inf)
        norm_tv = gs.sqrt(sq_norm_tv)

        # Work with z = ||v|| / R
        z = norm_tv / R
        z2 = z**2

        coef_1 = _taylor_exp_even_func_device_safe(z2, utils.cosh_close_0, order=5)
        coef_2 = _taylor_exp_even_func_device_safe(z2, utils.sinch_close_0, order=5)

        # exp = cosh(z)*base + R*sinch(z)*tangent_vec
        exp = gs.einsum("...,...j->...j", coef_1, base_point) + gs.einsum(
            "...,...j->...j", R * coef_2, tangent_vec
        )
        return self._space.regularize(exp)

    def log(self, point, base_point):
        """Riemannian logarithm with curvature kappa<0 (radius R)."""
        point = _as_hyp_f64(point)
        base_point = _as_hyp_f64(base_point)
        R = self._space.radius

        # In unit-curvature code, angle = dist(base, point).
        # Here dist already includes R, so use angle = dist / R.
        angle = self.dist(base_point, point) / R
        angle2 = angle**2

        coef_1 = _taylor_exp_even_func_device_safe(
            angle2, utils.inv_sinch_close_0, order=4
        )
        coef_2 = _taylor_exp_even_func_device_safe(
            angle2, utils.inv_tanh_close_0, order=4
        )

        # Same algebraic form as Geomstats; only "angle" is scaled.
        log_term_1 = gs.einsum("...,...j->...j", coef_1, point)
        log_term_2 = -gs.einsum("...,...j->...j", coef_2, base_point)
        return log_term_1 + log_term_2

    def injectivity_radius(self, base_point=None):
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
