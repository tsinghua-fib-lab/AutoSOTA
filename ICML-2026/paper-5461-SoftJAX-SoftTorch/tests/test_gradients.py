"""Gradient correctness tests: autodiff vs finite differences.

Exhaustively tests all (op, method, mode) combinations to ensure
analytic gradients match central finite differences in float64.
"""

import jax.numpy as jnp
import pytest

import softjax as sj

from . import common


# JAX config (jax_enable_x64, matmul_precision) set in common.py

# ---------------------------------------------------------------------------
# Methods per function (same as test_arraywise.py)
# ---------------------------------------------------------------------------

VALUE_METHODS = [
    "softsort",
    "neuralsort",
    "fast_soft_sort",
    "smooth_sort",
    "ot",
    "sorting_network",
]
ARG_METHODS = ["softsort", "neuralsort", "ot", "sorting_network"]

FUNCTION_METHODS = {
    "sort": VALUE_METHODS,
    "rank": VALUE_METHODS,
    "max": VALUE_METHODS,
    "min": VALUE_METHODS,
    "median": VALUE_METHODS,
    "quantile": VALUE_METHODS,
    "top_k": VALUE_METHODS,
    "argsort": ARG_METHODS,
    "argmax": ARG_METHODS,
    "argmin": ARG_METHODS,
    "argmedian": ARG_METHODS,
    "argquantile": ARG_METHODS,
}

MODES = ["smooth", "c0", "c1", "c2"]


def _skip_unsupported(method, mode):
    if method == "smooth_sort" and mode != "smooth":
        pytest.skip("smooth_sort only supports smooth mode")


# ---------------------------------------------------------------------------
# Loss functions (scalar output for jax.grad)
# ---------------------------------------------------------------------------


def _make_loss(op, method, mode, softness=1.0):
    """Build a scalar loss function for the given (op, method, mode).

    Uses a weighted sum rather than plain sum to avoid trivially constant
    outputs. For example, sum(rank(x)) = n*(n+1)/2 (constant) so its gradient
    is zero. A weighted sum produces non-trivial gradients for all ops.
    """
    # Weights for non-trivial loss; length must match output size
    w5 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    w2 = jnp.array([1.0, 2.0], dtype=jnp.float64)

    def loss(x):
        kwargs = dict(softness=softness, mode=mode, method=method)
        if op == "quantile":
            return jnp.sum(w5 * sj.quantile(x, 0.5, axis=-1, **kwargs))
        elif op == "argquantile":
            out = sj.argquantile(x, 0.5, axis=-1, **kwargs)
            return jnp.sum(w5 * out)
        elif op == "top_k":
            vals, _ = sj.top_k(x, k=2, axis=-1, **kwargs)
            return jnp.sum(w2 * vals)
        elif op in ("argmax", "argmin"):
            out = getattr(sj, op)(x, axis=-1, **kwargs)
            return jnp.sum(w5 * out)
        elif op == "argmedian":
            out = sj.argmedian(x, axis=-1, **kwargs)
            return jnp.sum(w5 * out)
        elif op == "argsort":
            out = sj.argsort(x, axis=-1, **kwargs)
            return jnp.sum(w5[None, :] * out)
        elif op == "rank":
            out = sj.rank(x, axis=-1, **kwargs)
            return jnp.sum(w5 * out)
        else:
            out = getattr(sj, op)(x, axis=-1, **kwargs)
            return jnp.sum(w5 * out)

    return loss


# ---------------------------------------------------------------------------
# Build exhaustive parametrization
# ---------------------------------------------------------------------------

_CASES = []
for op, methods in FUNCTION_METHODS.items():
    for method in methods:
        for mode in MODES:
            if method == "smooth_sort" and mode != "smooth":
                continue
            _CASES.append((op, method, mode))

_IDS = [f"{op}_{method}_{mode}" for op, method, mode in _CASES]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,method,mode", _CASES, ids=_IDS)
def test_grad_vs_finite_diff(op, method, mode):
    """Analytic gradient matches finite differences for all (op, method, mode)."""
    # Non-uniform spacing avoids degenerate finite-diff behavior with standardized
    # ops (uniform spacing means a shift perturbation barely changes the standardized
    # values, making finite diff noisy while autodiff correctly gives ~0).
    x = jnp.array([-0.8, -0.1, 0.3, 0.5, 1.2], dtype=jnp.float64)
    loss = _make_loss(op, method, mode)
    # OT c0 (L2) transport plan is Gamma_ij = (1/tau)*max(f_i+g_j-C_ij, 0), which
    # is C0 but not C1 at support boundaries. The implicit function theorem requires
    # the dual Hessian, which involves d/dy max(S,0) = Heaviside (discontinuous).
    # This makes the implicit diff linear system ill-conditioned, causing inherently
    # lower gradient accuracy. Higher smoothness modes (c1/c2/smooth) don't have
    # this issue because their P^(q-1) terms are differentiable.
    # smooth_sort uses LBFGS with a custom VJP, which also has solver-dependent accuracy.
    if method == "ot" and mode == "c0":
        common.assert_grad_matches_finite_diff(
            loss,
            x,
            rtol=2e-1,
            atol=2e-1,
            msg=f"{op} {method} {mode}",
        )
    elif method in ("ot", "smooth_sort"):
        common.assert_grad_matches_finite_diff(
            loss,
            x,
            rtol=5e-2,
            atol=5e-2,
            msg=f"{op} {method} {mode}",
        )
    else:
        common.assert_grad_matches_finite_diff(loss, x, msg=f"{op} {method} {mode}")
