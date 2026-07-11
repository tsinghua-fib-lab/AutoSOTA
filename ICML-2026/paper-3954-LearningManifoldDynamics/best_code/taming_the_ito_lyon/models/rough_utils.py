from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import georax
import jax
import jax.numpy as jnp
import roughrax
import pysiglib.jax_api as pysiglib
from roughrax._bases import make_planar_tree_basis, make_tree_basis


def compute_disjoint_signature_times(
    ts: jax.Array, signature_window_size: int
) -> jax.Array:
    """Return knot times for disjoint rough-path signature windows."""
    step = int(signature_window_size)
    if step <= 0:
        raise ValueError("signature_window_size must be a positive integer.")

    num_points = int(ts.shape[0])
    if num_points < 2:
        raise ValueError("ts must contain at least two time points.")

    start_indices = jnp.arange(0, num_points - 1, step, dtype=jnp.int32)
    return jnp.concatenate([ts[start_indices], ts[-1:]], axis=0)


class ItoCorrectedSignatureInterpolation(roughrax.SignatureInterpolation):
    """roughrax-compatible branched log-signature interpolation with Itô QV."""

    brownian_channels: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        control: diffrax.AbstractPath,
        signature_knots: jax.Array,
        depth: int,
        *,
        brownian_channels: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(control, signature_knots, depth, "ito")
        self.brownian_channels = brownian_channels

    def _correction(self, windows: jax.Array) -> jax.Array:
        dim = int(windows.shape[-1])
        num_windows = int(windows.shape[0])
        num_segments = int(windows.shape[1]) - 1
        correction = jnp.zeros(
            (num_windows, num_segments, dim, dim),
            dtype=windows.dtype,
        )
        if dim < 2 or num_segments < 1:
            return correction.reshape((num_windows, num_segments, dim * dim))

        # The first channel is time in this codebase's unconditional controls.
        # Using the path's time channel, rather than solver timestamps, leaves the
        # synthetic zero basepoint and padded tail with zero quadratic variation.
        dt = jnp.diff(windows[..., 0], axis=1)
        brownian_channels = (
            tuple(range(1, dim))
            if self.brownian_channels is None
            else self.brownian_channels
        )
        for channel in brownian_channels:
            correction = correction.at[:, :, int(channel), int(channel)].set(dt)
        return correction.reshape((num_windows, num_segments, dim * dim))

    def materialise(
        self,
        geometry: georax.Manifold[Any],
    ) -> roughrax.SignatureInterpolation:
        if self.coeffs is not None:
            return self

        control_ts = getattr(self.control, "ts")
        ys = jnp.asarray(getattr(self.control, "ys"))
        dim = int(ys.shape[-1])
        num_intervals = self.ts.shape[0] - 1
        num_control_intervals = control_ts.shape[0] - 1
        if num_intervals < 1:
            raise ValueError("signature_knots must contain at least two points.")
        if num_control_intervals % num_intervals != 0:
            raise ValueError(
                "signature_knots must evenly subdivide the control sample grid."
            )

        stride = num_control_intervals // num_intervals
        windows = jnp.stack(
            [ys[j * stride : (j + 1) * stride + 1] for j in range(num_intervals)]
        )

        planar = not isinstance(geometry, georax.Euclidean)
        basis = (
            make_planar_tree_basis(self.depth, dim)
            if planar
            else make_tree_basis(self.depth, dim)
        )
        pysiglib.prepare_branched_sig(dim, self.depth, planar=planar)
        coeffs = pysiglib.branched_log_sig(
            windows,
            self.depth,
            tree_order="canonical",
            planar=planar,
            correction=self._correction(windows),
        )

        out = ItoCorrectedSignatureInterpolation(
            self.control,
            self.ts,
            self.depth,
            brownian_channels=self.brownian_channels,
        )
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "basis", basis)
        return out
