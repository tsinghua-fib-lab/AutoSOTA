"""Multi-layer xLSTM sequence model built from stacked mLSTM blocks."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from stochastax.manifolds import Manifold

from .extrapolation import ExtrapolationScheme
from .xlstm import (
    XLSTMArgs,
    XLSTMLayer,
    _build_input_sequence,
    _retract_output,
)


class StackedXLSTM(eqx.Module):
    """Stack several xLSTM layers over a projected input sequence."""

    input_proj: eqx.nn.Linear
    layers: tuple[XLSTMLayer, ...]
    norm: eqx.nn.LayerNorm
    readout_layer: eqx.nn.Linear

    manifold: Manifold = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    args: XLSTMArgs = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        output_path_dim: int,
        *,
        d_model: int,
        n_heads: int,
        num_layers: int,
        key: jax.Array,
        manifold: Manifold,
        d_conv: int = 4,
        xlstm_expand: int = 2,
        ffn_expand: int = 2,
        use_ffn: bool = True,
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        keys = jr.split(key, num_layers + 2)
        proj_key, readout_key = keys[0], keys[1]
        layer_keys = keys[2:]

        self.args = XLSTMArgs(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=num_layers,
            d_conv=d_conv,
            xlstm_expand=xlstm_expand,
            ffn_expand=ffn_expand,
            use_ffn=use_ffn,
        )
        self.manifold = manifold
        self.readout_activation = readout_activation
        self.evolving_out = bool(evolving_out)
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon
        self.num_layers = int(num_layers)

        self.input_proj = eqx.nn.Linear(
            input_path_dim,
            d_model,
            use_bias=True,
            key=proj_key,
        )
        self.layers = tuple(
            XLSTMLayer(layer_key, args=self.args) for layer_key in layer_keys
        )
        self.norm = eqx.nn.LayerNorm(d_model, use_weight=True, use_bias=True)
        self.readout_layer = eqx.nn.Linear(
            d_model,
            output_path_dim,
            use_bias=True,
            key=readout_key,
        )

    def _forward_from_x(self, x: jax.Array) -> jax.Array:
        h = jax.vmap(self.input_proj)(x)
        mask = jnp.tril(jnp.ones((h.shape[0], h.shape[0]), dtype=bool))
        for layer in self.layers:
            h = layer(h, mask)
        return jax.vmap(self.norm)(h)

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

    @eqx.filter_jit
    def step(
        self,
        x_t: jax.Array,
        state: tuple[
            tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]] | None, ...
        ]
        | None,
    ) -> tuple[
        jax.Array,
        tuple[tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]] | None, ...],
    ]:
        if state is None:
            state = tuple([None] * self.num_layers)

        h = self.input_proj(x_t)
        next_state: list[
            tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]] | None
        ] = []
        for layer, layer_state in zip(self.layers, state, strict=True):
            h, next_layer_state = layer.step(h, layer_state)
            next_state.append(next_layer_state)

        h = self.norm(h)
        y = self.readout_activation(self.readout_layer(h))
        return _retract_output(self.manifold, y), tuple(next_state)
