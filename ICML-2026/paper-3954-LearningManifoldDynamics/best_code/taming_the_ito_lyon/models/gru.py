"""
Simple GRU model with the same high-level interface as the CDE/RDE models:

- Standard mode: model(ts, x) where x are raw control values sampled at `ts`.
- Extrapolation mode: model(ts, x) where x are raw control values and an
  extrapolation_scheme is provided; the scheme builds a control path that covers
  reconstruction + future, and we evaluate it at `ts` to obtain a discrete sequence.

Additionally supports a Stochastax manifold, retracting the hidden state and the
readout outputs (mirroring the behavior in `bnrde.py`).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import Manifold
from stochastax.manifolds.spd import SPDManifold

from .extrapolation import ExtrapolationScheme


class GRU(eqx.Module):
    """A simple GRU sequence model with optional extrapolation and manifold support."""

    initial_cond_mlp: eqx.nn.MLP
    cell: eqx.nn.GRUCell
    readout_layer: eqx.nn.Linear

    # Static configuration
    manifold: Manifold = eqx.field(static=True)
    hidden_manifold: Manifold = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        gru_state_dim: int,
        output_path_dim: int,
        mlp_hidden_dim: int,
        initial_cond_mlp_depth: int,
        *,
        key: jax.Array,
        manifold: Manifold,
        hidden_manifold: Manifold | None = None,
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)

        self.manifold = manifold
        self.hidden_manifold = manifold if hidden_manifold is None else hidden_manifold
        self.readout_activation = readout_activation
        self.evolving_out = bool(evolving_out)
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon

        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=gru_state_dim,
            width_size=mlp_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cell = eqx.nn.GRUCell(
            input_size=input_path_dim, hidden_size=gru_state_dim, key=k2
        )
        self.readout_layer = eqx.nn.Linear(
            in_features=gru_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )

    def _forward_from_x(self, x: jax.Array) -> jax.Array:
        """Run the GRU over a discrete input sequence x of shape (T, C)."""
        x0 = x[0]
        h0 = self.hidden_manifold.retract(self.initial_cond_mlp(x0))

        def step(h: jax.Array, xt: jax.Array) -> tuple[jax.Array, jax.Array]:
            h_new = self.cell(xt, h)
            h_new = self.hidden_manifold.retract(h_new)
            return h_new, h_new

        # We keep h0 as the hidden state at time ts[0] (like diffeqsave does).
        hs_rest = jax.lax.scan(step, h0, x[1:])[1]
        hs = jnp.concatenate([h0[None, :], hs_rest], axis=0)
        return hs

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        def apply_single(h: jax.Array) -> jax.Array:
            y = self.readout_activation(self.readout_layer(h))
            if isinstance(self.manifold, SPDManifold):
                matrix = SPDManifold.unvech(y)
                return SPDManifold.retract(matrix)
            return self.manifold.retract(y)

        return jax.vmap(apply_single)(hidden_states)

    def __call__(
        self,
        control_values: jax.Array,
    ) -> jax.Array:
        length = control_values.shape[0]
        ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)  # (T,)
        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None, (
                "n_recon must be set when using extrapolation_scheme"
            )
            control, _ = self.extrapolation_scheme.create_control(
                ts, control_values, self.n_recon
            )
            x_eval = jax.vmap(control.evaluate)(ts)
        else:
            x_eval = control_values
        hidden = self._forward_from_x(x_eval)

        if self.evolving_out:
            return self._apply_readout(hidden)

        final_output = self.readout_activation(self.readout_layer(hidden[-1]))
        if isinstance(self.manifold, SPDManifold):
            matrix = SPDManifold.unvech(final_output)
            return SPDManifold.retract(matrix)
        return self.manifold.retract(final_output)
