"""Neural Rough Differential Equation backed by roughrax."""

from collections.abc import Callable

import diffrax
import equinox as eqx
import georax
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import pysiglib
import roughrax
from stochastax.manifolds import Manifold
from stochastax.manifolds.spd import SPDManifold

from .extrapolation import ExtrapolationScheme
from .rough_utils import compute_disjoint_signature_times


def _lyndon_logsig_size(input_path_dim: int, signature_depth: int) -> int:
    return len(tuple(pysiglib.lyndon_words(int(input_path_dim), int(signature_depth))))


class NRDEFunc(eqx.Module):
    """NRDE vector field returning direct log-signature columns."""

    vf_mlp: eqx.nn.MLP
    cde_state_dim: int = eqx.field(static=True)
    logsig_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        cde_state_dim: int,
        logsig_size: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        key: jax.Array,
    ) -> None:
        self.cde_state_dim = int(cde_state_dim)
        self.logsig_size = int(logsig_size)
        self.vf_mlp = eqx.nn.MLP(
            in_size=self.cde_state_dim,
            out_size=self.cde_state_dim
            * self.logsig_size,  # NRDE outputs one element per log-signature coefficient
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jnn.softplus,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        # Return flattened direct columns so roughrax never confuses depth-1 NRDE
        # columns with first-level vector fields when dimensions coincide.
        return self.vf_mlp(y)


class NeuralRDE(eqx.Module):
    """Neural Rough Differential Equation with Euclidean hidden dynamics."""

    # Modules
    initial: eqx.nn.MLP
    cde_func: NRDEFunc
    readout_layer: eqx.nn.Linear

    # Static configuration
    manifold: type[Manifold] = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    prepend_zero_basepoint: bool = eqx.field(static=True)

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

    solver: diffrax.AbstractSolver = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        vf_hidden_dim: int,
        init_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_mlp_depth: int,
        signature_depth: int,
        signature_window_size: int,
        *,
        key: jax.Array,
        manifold: type[Manifold],
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
        dt0: float | None = None,
        evolving_out: bool = True,
        prepend_zero_basepoint: bool = False,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        del stepsize_controller, dt0

        k1, k2, k3 = jr.split(key, 3)
        logsig_size = _lyndon_logsig_size(input_path_dim, signature_depth)

        # Initial state from initial control value (matches NCDE style)
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=init_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = NRDEFunc(
            cde_state_dim=cde_state_dim,
            logsig_size=logsig_size,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            key=k2,
        )
        self.readout_layer = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )
        self.readout_activation = readout_activation
        self.manifold = manifold
        self.signature_depth = signature_depth
        self.signature_window_size = signature_window_size
        self.evolving_out = evolving_out
        self.prepend_zero_basepoint = prepend_zero_basepoint
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon

        self.solver = solver
        self.adjoint = adjoint

    def _maybe_prepend_zero_basepoint(
        self, ts: jax.Array, control_values: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if not self.prepend_zero_basepoint:
            return ts, control_values
        if int(ts.shape[0]) < 2:
            raise ValueError(
                "Expected at least two timestamps when prepending a basepoint."
            )

        dt = ts[1] - ts[0]
        ts_aug = jnp.concatenate([ts[:1] - dt, ts], axis=0)
        values_aug = jnp.concatenate(
            [
                jnp.zeros(
                    (1, int(control_values.shape[-1])),
                    dtype=control_values.dtype,
                ),
                control_values,
            ],
            axis=0,
        )

        # Keep the disjoint-window partition valid after the synthetic prefix point.
        step = int(self.signature_window_size)
        remainder = (int(values_aug.shape[0]) - 1) % step
        if remainder == 0:
            return ts_aug, values_aug

        pad_points = step - remainder
        ts_pad = ts_aug[-1] + dt * jnp.arange(1, pad_points + 1, dtype=ts.dtype)
        values_pad = jnp.repeat(values_aug[-1:], pad_points, axis=0)
        return (
            jnp.concatenate([ts_aug, ts_pad], axis=0),
            jnp.concatenate([values_aug, values_pad], axis=0),
        )

    def _project_readout(self, activation: jax.Array) -> jax.Array:
        if issubclass(self.manifold, SPDManifold):
            matrix = SPDManifold.unvech(activation)
            return SPDManifold.retract(matrix)
        if activation.shape[-1] == 9:
            return self.manifold.retract(jnp.reshape(activation, (3, 3)))
        return self.manifold.retract(activation)

    def _forward_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        h0 = self.initial(control_values[0])
        ys, _ = self._solve_from_values(ts, control_values, h0)
        return ys

    def _solve_from_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
        y0: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        driver = diffrax.LinearInterpolation(ts=ts, ys=control_values)
        signature_ts = compute_disjoint_signature_times(
            ts, int(self.signature_window_size)
        )
        control = roughrax.SignatureInterpolation(
            driver,
            signature_ts,
            depth=int(self.signature_depth),
            solution="stratonovich",
        )

        def vector_field(y: jax.Array) -> jax.Array:
            return self.cde_func(y)

        term = roughrax.RoughTerm(vector_field, control, georax.Euclidean())
        solution = diffrax.diffeqsolve(
            term,
            roughrax.LogODE(self.solver),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(signature_ts),
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=self.adjoint,
            max_steps=int(signature_ts.shape[0]) + 4,
        )
        assert solution.ys is not None
        return solution.ys, solution.stats

    def _forward_with_control(
        self,
        ts: jax.Array,
        control: diffrax.AbstractPath,
    ) -> jax.Array:
        control_values = jax.vmap(control.evaluate)(ts)
        return self._forward_with_values(ts, control_values)

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        """Apply readout to hidden states with manifold-aware output projection."""

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            return self._project_readout(activation)

        return jax.vmap(apply_single)(hidden_states)

    def __call__(
        self,
        control_values: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Parameters
        - control_values: shape (T, C). Control values.

        Returns
        - If self.evolving_out is False: shape (out_size,)
        - If self.evolving_out is True: shape (T, out_size)
        """
        length, ts, control_values = self._prepare_control(control_values)
        h0 = self.initial(control_values[0])
        hidden_over_time, _ = self._solve_from_values(ts, control_values, h0)
        outputs = self._apply_readout(hidden_over_time)

        if self.prepend_zero_basepoint:
            outputs = outputs[1 : 1 + length]

        if self.evolving_out:
            return outputs
        return outputs[-1]

    def _prepare_control(
        self,
        control_values: jax.Array,
    ) -> tuple[int, jax.Array, jax.Array]:
        length = control_values.shape[0]
        ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)

        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None
            control, _ = self.extrapolation_scheme.create_control(
                ts, control_values, self.n_recon
            )
            control_values = jax.vmap(control.evaluate)(ts)

        ts, control_values = self._maybe_prepend_zero_basepoint(ts, control_values)
        return int(length), ts, control_values
