"""
Pluggable extrapolation schemes for Neural CDE models.

These schemes take reconstruction data and future time points,
and return a control path that covers the full time range
(reconstruction + future), with the future portion extrapolated.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
from diffrax._custom_types import RealScalarLike
from taming_the_ito_lyon.config.config_options import ExtrapolationSchemeType


class ExtrapolationScheme(Protocol):
    """Protocol for extrapolation schemes."""

    @abstractmethod
    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.AbstractPath, jax.Array]:
        """Create a control path that covers both reconstruction and future times.

        Fits the control on reconstruction data, then extrapolates to future times.

        Args:
            t_all: All timestamps of shape (T_total,) covering reconstruction + future
            x_all: Input data of shape (T_total, D), but only first n_recon are used
            n_recon: Number of reconstruction points to fit on (split index)

        Returns:
            Tuple of:
            - control: A diffrax.AbstractPath covering [t_all[0], t_all[-1]]
            - t_all: The full time array passed in
        """
        ...


class LinearScheme(eqx.Module):
    """Linear interpolation on reconstruction, linear extrapolation for future.

    - Within t_recon: linear interpolation between data points
    - Beyond t_recon: continues linearly based on last segment's slope
    """

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.LinearInterpolation, jax.Array]:
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all and x_all must have length >= n_recon, got {t_all.shape[0]} and {x_all.shape[0]}"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]
        # Prepend time channel: (n_recon, D) -> (n_recon, 1+D)
        ys_with_time = jnp.concatenate([t_recon[:, None], x_recon], axis=1)
        # Fit only on reconstruction data - diffrax will extrapolate linearly
        control = diffrax.LinearInterpolation(ts=t_recon, ys=ys_with_time)
        return control, t_all


class HermiteScheme(eqx.Module):
    """Cubic spline on reconstruction, polynomial extrapolation for future.

    - Within t_recon: cubic spline interpolation
    - Beyond t_recon: continues using polynomial from last segment (can be unstable)
    """

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.CubicInterpolation, jax.Array]:
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all and x_all must have length >= n_recon, got {t_all.shape[0]} and {x_all.shape[0]}"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]
        # Prepend time channel: (n_recon, D) -> (n_recon, 1+D)
        ys_with_time = jnp.concatenate([t_recon[:, None], x_recon], axis=1)
        # Fit only on reconstruction data - diffrax will extrapolate
        coeffs = diffrax.backward_hermite_coefficients(ts=t_recon, ys=ys_with_time)
        control = diffrax.CubicInterpolation(ts=t_recon, coeffs=coeffs)
        return control, t_all


class WeightedSGScheme(eqx.Module):
    """Savitzky-Golay polynomial fit on reconstruction, polynomial extrapolation.

    Fits a single polynomial to the reconstruction data, providing smooth
    extrapolation to future times.

    Uses the same numerical stability features as the SO3 implementation:
    - Center-based polynomial expansion
    - Normalized polynomial basis (t^k / max(1, k))
    - Raw weights with softplus for positivity (no softmax)
    """

    weights: jax.Array
    poly_order: int = eqx.field(static=True)

    def __init__(self, num_points: int, poly_order: int = 3, *, key: jax.Array):
        self.weights = (
            jax.random.normal(key, (num_points,)) * 0.1
        )  # Raw weights, learnable
        self.poly_order = poly_order

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[_PolynomialPath, jax.Array]:
        """Create polynomial path that extrapolates smoothly to future."""
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all (shape {t_all.shape}) and x_all (shape {x_all.shape}) must have length >= n_recon ({n_recon})"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]
        seq_len = n_recon

        # Resize weights if needed
        if self.weights.shape[0] != seq_len:
            weights = jnp.interp(
                jnp.linspace(0, 1, seq_len),
                jnp.linspace(0, 1, self.weights.shape[0]),
                self.weights,
            )
        else:
            weights = self.weights

        # Ensure positive weights
        weights = jax.nn.softplus(weights)

        # Center-based polynomial expansion (like SO3 version)
        center_idx = seq_len // 2
        t_center = t_recon[center_idx]

        # Normalized polynomial basis: t^k / max(1, k)
        normalizers = jnp.maximum(1.0, jnp.arange(self.poly_order + 1))

        def fit_channel(x_channel: jax.Array) -> jax.Array:
            """Fit polynomial and return coefficients."""
            t_rel = t_recon - t_center
            # Vandermonde with normalized basis
            V = jnp.vander(t_rel, N=self.poly_order + 1, increasing=True)
            V = V / normalizers[None, :]

            # Weighted least squares
            sqrt_W = jnp.sqrt(weights)
            V_w = V * sqrt_W[:, None]
            b_w = x_channel * sqrt_W

            coeffs = jnp.linalg.lstsq(V_w, b_w)[0]
            return coeffs

        # Fit polynomial for each channel
        poly_coeffs = jax.vmap(fit_channel, in_axes=1, out_axes=0)(x_recon)

        control = _PolynomialPath(
            poly_coeffs=poly_coeffs,
            t_center=t_center,
            normalizers=normalizers,
            t0=t_all[0],
            t1=t_all[-1],  # Extends to future!
        )
        return control, t_all


class _PolynomialPath(diffrax.AbstractPath):
    """Diffrax-compatible path backed by a polynomial."""

    poly_coeffs: jax.Array  # (D, poly_order+1)
    t_center: jax.Array
    normalizers: jax.Array
    powers_phi: jax.Array  # (poly_order+1,)
    powers_first: jax.Array  # (poly_order,)
    t0: RealScalarLike  # type: ignore
    t1: RealScalarLike  # type: ignore

    def __init__(
        self,
        poly_coeffs: jax.Array,
        t_center: jax.Array,
        normalizers: jax.Array,
        t0: RealScalarLike,
        t1: RealScalarLike,
    ) -> None:
        object.__setattr__(self, "poly_coeffs", poly_coeffs)
        object.__setattr__(self, "t_center", t_center)
        object.__setattr__(self, "normalizers", normalizers)
        poly_order = poly_coeffs.shape[1] - 1
        # Pre-compute power arrays for efficient derivative computation
        object.__setattr__(
            self, "powers_phi", jnp.arange(poly_order + 1, dtype=jnp.float32)
        )
        object.__setattr__(
            self, "powers_first", jnp.arange(poly_order, dtype=jnp.float32)
        )
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            t_rel = t0 - self.t_center
            t_powers = (t_rel**self.powers_phi) / self.normalizers
            x_t = self.poly_coeffs @ t_powers
            # Prepend time channel: (D,) -> (1+D,)
            return jnp.concatenate([jnp.array([t0]), x_t])
        else:
            return self.evaluate(t1) - self.evaluate(t0)

    def derivative(self, t, left=True, order=1):
        """Compute time derivative using analytic polynomial derivative.

        Args:
            t: Time point for derivative evaluation
            left: Not used for deterministic paths
            order: Derivative order (only 1 is currently supported)

        Returns:
            First derivative vector of shape (1+D,) = [1, dx/dt]
        """
        del left
        if order != 1:
            raise ValueError(f"Only derivative order 1 is supported, got {order}")

        t_rel = t - self.t_center
        # For derivative: d/dt [t^k / max(1,k)] =
        #   - k=0: 0 (constant term)
        #   - k>=1: t^(k-1) (no extra normalizer needed for derivative)
        # So we use coeffs[:, 1:] with powers_first = arange(poly_order)
        if self.poly_coeffs.shape[1] > 1:
            # Derivative coefficients: skip constant term
            deriv_coeffs = self.poly_coeffs[:, 1:]  # (D, poly_order)
            t_powers_deriv = t_rel**self.powers_first  # (poly_order,)
            x_dot = deriv_coeffs @ t_powers_deriv  # (D,)
        else:
            # Constant polynomial (order 0) has zero derivative
            x_dot = jnp.zeros(self.poly_coeffs.shape[0])

        # Time derivative is constant 1.0
        # Return shape (1+D,) = [1, dx/dt]
        return jnp.concatenate([jnp.array([1.0]), x_dot])


class PiecewiseMLPScheme(eqx.Module):
    """Piecewise MLP extrapolation: ground-truth recon, MLP for future.

    - For reconstruction times (t <= t_recon[-1]): uses linear interpolation of the
      observed reconstruction data.
    - For future times (t > t_recon[-1]): uses a conditioned decoder MLP.

    The decoder output is shifted by a constant vector to ensure continuity at the
    split time t_recon[-1].
    """

    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    input_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    n_recon: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        n_recon: int,
        context_dim: int = 16,
        hidden_dim: int = 32,
        depth: int = 2,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.n_recon = n_recon

        self.encoder = eqx.nn.MLP(
            in_size=input_dim * n_recon,
            out_size=context_dim,
            width_size=hidden_dim,
            depth=depth,
            key=k1,
        )
        self.decoder = eqx.nn.MLP(
            in_size=1 + context_dim,
            out_size=input_dim,
            width_size=hidden_dim,
            depth=depth,
            key=k2,
        )

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.AbstractPath, jax.Array]:
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all and x_all must have length >= n_recon, got {t_all.shape[0]} and {x_all.shape[0]}"
            )
        if n_recon != self.n_recon:
            raise ValueError(
                f"n_recon mismatch: scheme was initialized with {self.n_recon}, got {n_recon}"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]

        context = self.encoder(jnp.reshape(x_recon, (-1,)))  # (context_dim,)

        ys_with_time = jnp.concatenate([t_recon[:, None], x_recon], axis=1)
        recon_control = diffrax.LinearInterpolation(ts=t_recon, ys=ys_with_time)

        mlp_control = _MLPPath(
            decoder=self.decoder,
            context=context,
            t0=t_all[0],
            t1=t_all[-1],
        )
        t_recon_end = t_recon[-1]
        x_recon_end = x_recon[-1]
        x_mlp_end = mlp_control.evaluate(t_recon_end)[1:]
        mlp_shift = x_recon_end - x_mlp_end

        control = _PiecewiseReconMLPPath(
            recon_control=recon_control,
            decoder=self.decoder,
            context=context,
            t0=t_all[0],
            t1=t_all[-1],
            t_recon_end=t_recon_end,
            mlp_shift=mlp_shift,
        )
        return control, t_all


class _MLPPath(diffrax.AbstractPath):
    """Diffrax-compatible path backed by a conditioned MLP."""

    decoder: eqx.nn.MLP
    context: jax.Array
    t0: RealScalarLike  # type: ignore[assignment]
    t1: RealScalarLike  # type: ignore[assignment]

    def __init__(
        self,
        decoder: eqx.nn.MLP,
        context: jax.Array,
        t0: RealScalarLike,
        t1: RealScalarLike,
    ) -> None:
        object.__setattr__(self, "decoder", decoder)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            # Normalize time to [0, 1] based on full range
            t_norm = (t0 - self.t0) / (self.t1 - self.t0 + 1e-8)
            # Concatenate time and context
            mlp_input = jnp.concatenate([jnp.atleast_1d(t_norm), self.context])
            x_t = self.decoder(mlp_input)
            # Prepend time channel: (D,) -> (1+D,)
            return jnp.concatenate([jnp.array([t0]), x_t])
        else:
            return self.evaluate(t1) - self.evaluate(t0)


class _PiecewiseReconMLPPath(diffrax.AbstractPath):
    """Ground-truth recon control, MLP for future times.

    Uses linear interpolation of the observed reconstruction data for
    t <= t_recon_end. For later times, uses the conditioned decoder MLP and an
    additive shift chosen to ensure continuity at t_recon_end.
    """

    recon_control: diffrax.LinearInterpolation
    decoder: eqx.nn.MLP
    context: jax.Array
    t_recon_end: RealScalarLike
    mlp_shift: jax.Array
    t0: RealScalarLike  # type: ignore[assignment]
    t1: RealScalarLike  # type: ignore[assignment]

    def __init__(
        self,
        recon_control: diffrax.LinearInterpolation,
        decoder: eqx.nn.MLP,
        context: jax.Array,
        t0: RealScalarLike,
        t1: RealScalarLike,
        t_recon_end: RealScalarLike,
        mlp_shift: jax.Array,
    ) -> None:
        object.__setattr__(self, "recon_control", recon_control)
        object.__setattr__(self, "decoder", decoder)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)
        object.__setattr__(self, "t_recon_end", t_recon_end)
        object.__setattr__(self, "mlp_shift", mlp_shift)

    def _eval_mlp(self, t: RealScalarLike) -> jax.Array:
        # Normalize time to [0, 1] based on full range.
        t_norm = (t - self.t0) / (self.t1 - self.t0 + 1e-8)
        mlp_input = jnp.concatenate([jnp.atleast_1d(t_norm), self.context])
        x_t = self.decoder(mlp_input) + self.mlp_shift
        return jnp.concatenate([jnp.array([t]), x_t])

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            use_recon = t0 <= self.t_recon_end
            return jax.lax.cond(
                use_recon,
                lambda _: self.recon_control.evaluate(t0),
                lambda _: self._eval_mlp(t0),
                operand=None,
            )

        # Diffrax convention: increments are differences of values.
        return self.evaluate(t1) - self.evaluate(t0)


class SO3SGScheme(eqx.Module):
    """SO(3) Savitzky-Golay scheme for rotation matrix data.

    Uses manifold-aware polynomial fitting in the tangent space (Lie algebra).
    Ensures outputs remain on SO(3) through exponential map and orthogonalization.

    Handles flattened (T, 9) driver data from SO(3) datasets, which is the standard
    format used by the SO3DynamicsSim dataloader.
    """

    poly_order: int = eqx.field(static=True)
    weights: jax.Array | None = None

    def __init__(
        self,
        poly_order: int = 3,
        num_points: int | None = None,
        *,
        key: jax.Array | None = None,
    ):
        self.poly_order = poly_order
        # Optional: learn weights (like WeightedSGScheme)
        if num_points is not None and key is not None:
            self.weights = jax.random.normal(key, (num_points,)) * 0.1
        else:
            self.weights = None

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.AbstractPath, jax.Array]:
        from taming_the_ito_lyon.utils.savitzky_golay_so3 import SO3PolynomialPath

        if t_all.shape[0] < n_recon:
            raise ValueError("t_all must have length >= n_recon")

        # Use FULL time array for polynomial, not just reconstruction
        # The polynomial will fit on first n_recon points but cover full range
        x_recon = x_all[:n_recon]
        t_recon = t_all[:n_recon]

        # Reshape to (n_recon, 3, 3)
        if x_recon.ndim == 2 and x_recon.shape[-1] == 9:
            x_recon = x_recon.reshape(x_recon.shape[0], 3, 3)

        # Add batch dimension
        R = x_recon[None, ...]  # (1, n_recon, 3, 3)

        # Process weights - must be batched!
        weights = None
        if self.weights is not None:
            if self.weights.shape[0] != n_recon:
                weights_1d = jnp.interp(
                    jnp.linspace(0, 1, n_recon),
                    jnp.linspace(0, 1, self.weights.shape[0]),
                    jax.nn.softplus(self.weights),
                )
            else:
                weights_1d = jax.nn.softplus(self.weights)
            # Broadcast to batch dimension
            weights = weights_1d[None, :]  # (1, n_recon)

        # Create path with FULL time range for extrapolation
        # The key: pass t_all (not t_recon) so path covers full range
        control = SO3PolynomialPath(
            R=R,
            t=t_recon,  # CHANGED: Full time array
            p=self.poly_order,
            weight=weights,  # Now (1, n_recon) or None
            t0=t_all[0],
            t1=t_all[-1],
        )

        return control, t_all


def create_scheme(
    name: ExtrapolationSchemeType,
    *,
    input_dim: int | None = None,
    num_points: int | None = None,
    poly_order: int = 3,
    hidden_dim: int = 32,
    mlp_depth: int = 2,
    key: jax.Array | None = None,
) -> ExtrapolationScheme:
    """Factory function for extrapolation schemes.

    Args:
        name: One of ExtrapolationSchemeType
        input_dim: Required for 'mlp' scheme
        num_points: Required for 'sg' and 'mlp' schemes (reconstruction length)
        poly_order: Polynomial order for 'sg' scheme (default: 3)
        hidden_dim: Hidden dimension for 'mlp' scheme
        mlp_depth: Depth for 'mlp' scheme
        key: Random key (required for 'sg' and 'mlp')

    Returns:
        An ExtrapolationScheme instance

    Extrapolation behavior:
        - 'linear': Linear continuation from last segment
        - 'hermite': Polynomial from last cubic segment (can be unstable)
        - 'sg': Smooth polynomial extrapolation (fitted to all recon data)
        - 'so3_sg': SO(3) manifold-aware SG for rotation data (flattened as 9D)
        - 'piecewiseMLP': Ground-truth reconstruction + MLP future extrapolation
    """
    match name:
        case "linear":
            return LinearScheme()
        case "hermite":
            return HermiteScheme()
        case "sg":
            if num_points is None or key is None:
                raise ValueError("'sg' scheme requires num_points and key")
            return WeightedSGScheme(
                num_points=num_points, poly_order=poly_order, key=key
            )
        case "so3_sg":
            return SO3SGScheme(poly_order=poly_order, num_points=num_points, key=key)
        case "piecewiseMLP":
            if input_dim is None or key is None or num_points is None:
                raise ValueError(
                    "'piecewiseMLP' scheme requires input_dim, num_points, and key"
                )
            return PiecewiseMLPScheme(
                input_dim=input_dim,
                n_recon=num_points,
                context_dim=hidden_dim // 2,
                hidden_dim=hidden_dim,
                depth=mlp_depth,
                key=key,
            )
        case _:
            raise ValueError(
                f"Unknown scheme: {name}. Use 'linear', 'hermite', 'sg', 'so3_sg', or 'piecewiseMLP'"
            )
