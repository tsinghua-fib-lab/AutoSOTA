"""Stacked LSTM sequence model with the same high-level interface as `GRU`."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import Manifold
from stochastax.manifolds.spd import SPDManifold

from .extrapolation import ExtrapolationScheme


def _build_input_sequence(
    control_values: jax.Array,
    extrapolation_scheme: ExtrapolationScheme | None,
    n_recon: int | None,
) -> jax.Array:
    if extrapolation_scheme is None:
        return control_values

    assert n_recon is not None, "n_recon must be set when using extrapolation_scheme"
    length = int(control_values.shape[0])
    ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)
    control, _ = extrapolation_scheme.create_control(ts, control_values, n_recon)
    return jax.vmap(control.evaluate)(ts)


def _retract_output(manifold: Manifold, y: jax.Array) -> jax.Array:
    if isinstance(manifold, SPDManifold):
        return SPDManifold.retract(SPDManifold.unvech(y))
    return manifold.retract(y)


class LSTM(eqx.Module):
    """A simple stacked LSTM sequence model."""

    initial_cond_mlp: eqx.nn.MLP
    cells: tuple[eqx.nn.LSTMCell, ...]
    readout_layer: eqx.nn.Linear

    manifold: Manifold = eqx.field(static=True)
    hidden_manifold: Manifold = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        lstm_state_dim: int,
        output_path_dim: int,
        mlp_hidden_dim: int,
        initial_cond_mlp_depth: int,
        *,
        key: jax.Array,
        manifold: Manifold,
        hidden_manifold: Manifold | None = None,
        num_layers: int = 2,
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        keys = jr.split(key, num_layers + 2)
        init_key, readout_key = keys[0], keys[1]
        cell_keys = keys[2:]

        self.manifold = manifold
        self.hidden_manifold = manifold if hidden_manifold is None else hidden_manifold
        self.readout_activation = readout_activation
        self.evolving_out = bool(evolving_out)
        self.num_layers = int(num_layers)
        self.hidden_size = int(lstm_state_dim)
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon

        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=num_layers * lstm_state_dim,
            width_size=mlp_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jax.nn.softplus,
            key=init_key,
        )

        cells: list[eqx.nn.LSTMCell] = []
        for idx, cell_key in enumerate(cell_keys):
            layer_input_dim = input_path_dim if idx == 0 else lstm_state_dim
            cells.append(
                eqx.nn.LSTMCell(
                    input_size=layer_input_dim,
                    hidden_size=lstm_state_dim,
                    key=cell_key,
                )
            )
        self.cells = tuple(cells)
        self.readout_layer = eqx.nn.Linear(
            in_features=lstm_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=readout_key,
        )

    def _initial_states(self, x0: jax.Array) -> tuple[tuple[jax.Array, jax.Array], ...]:
        h0 = self.initial_cond_mlp(x0).reshape(self.num_layers, self.hidden_size)
        h0 = jax.vmap(self.hidden_manifold.retract)(h0)
        c0 = jnp.zeros_like(h0)
        return tuple((h0[i], c0[i]) for i in range(self.num_layers))

    def _step_layers(
        self,
        x_t: jax.Array,
        states: tuple[tuple[jax.Array, jax.Array], ...],
    ) -> tuple[jax.Array, tuple[tuple[jax.Array, jax.Array], ...]]:
        next_states: list[tuple[jax.Array, jax.Array]] = []
        layer_input = x_t
        for cell, (h, c) in zip(self.cells, states, strict=True):
            h_next, c_next = cell(layer_input, (h, c))
            h_next = self.hidden_manifold.retract(h_next)
            next_states.append((h_next, c_next))
            layer_input = h_next
        return layer_input, tuple(next_states)

    def _forward_from_x(self, x: jax.Array) -> jax.Array:
        states0 = self._initial_states(x[0])
        y0 = states0[-1][0]

        if int(x.shape[0]) == 1:
            return y0[None, :]

        def scan_step(
            states: tuple[tuple[jax.Array, jax.Array], ...],
            x_t: jax.Array,
        ) -> tuple[tuple[tuple[jax.Array, jax.Array], ...], jax.Array]:
            y_t, next_states = self._step_layers(x_t, states)
            return next_states, y_t

        ys_rest = jax.lax.scan(scan_step, states0, x[1:])[1]
        return jnp.concatenate([y0[None, :], ys_rest], axis=0)

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        def apply_single(h: jax.Array) -> jax.Array:
            y = self.readout_activation(self.readout_layer(h))
            return _retract_output(self.manifold, y)

        return jax.vmap(apply_single)(hidden_states)

    def __call__(self, control_values: jax.Array) -> jax.Array:
        x_eval = _build_input_sequence(
            control_values, self.extrapolation_scheme, self.n_recon
        )
        hidden = self._forward_from_x(x_eval)

        if self.evolving_out:
            return self._apply_readout(hidden)

        y = self.readout_activation(self.readout_layer(hidden[-1]))
        return _retract_output(self.manifold, y)
