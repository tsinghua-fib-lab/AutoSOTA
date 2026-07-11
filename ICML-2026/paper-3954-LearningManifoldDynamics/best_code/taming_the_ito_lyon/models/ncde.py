"""
Neural Controlled Differential Equation model. Taken from the Diffrax documentation.

https://docs.kidger.site/diffrax/examples/neural_cde/
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import diffrax
from typing import Callable

from stochastax.manifolds import Manifold
from stochastax.manifolds.spd import SPDManifold
from .extrapolation import ExtrapolationScheme


class CDEFunc(eqx.Module):
    """
    Vector field for a Neural CDE.

    Given hidden state y in R^{cde_state_dim}, returns matrix in
    R^{cde_state_dim x data_size} which is multiplied against dx/dt.
    """

    input_path_dim: int
    vf_mlp: eqx.nn.MLP
    cde_state_dim: int

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        *,
        key: jax.Array,
    ) -> None:
        # Vector field
        self.input_path_dim = input_path_dim
        self.cde_state_dim = cde_state_dim
        self.vf_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=cde_state_dim
            * input_path_dim,  # Shaped as such to reshape into (cde_state_dim, input_path_dim) matrix for dx/dt multiplication
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jnn.softplus,
            # Bound the vector field to avoid exploding hidden dynamics on long,
            # finely sampled controls such as PPG-DaLiA.
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args
        out = self.vf_mlp(y)
        return out.reshape(self.cde_state_dim, self.input_path_dim)


class NeuralCDE(eqx.Module):
    """
    Neural Controlled Differential Equation model.

    Usage
    - Provide `ts` and either a `diffrax` control path or cubic interpolation coeffs.
    - The model solves the induced ODE and applies a readout on the hidden state.
    """

    # Modules
    initial_cond_mlp: eqx.nn.MLP
    cde_func: CDEFunc
    readout_layer: eqx.nn.Linear

    # Static configuration
    manifold: type[Manifold] = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    control_interpolation: str = eqx.field(static=True)

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

    # Solver configuration
    solver: diffrax.AbstractAdaptiveSolver = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        init_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        *,
        key: jax.Array,
        manifold: type[Manifold],
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Tsit5(),
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller: diffrax.AbstractStepSizeController,
        dt0: float | None = None,
        evolving_out: bool,
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
        control_interpolation: str = "hermite_cubic",
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)

        # Modules
        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            # The CDE/ODE state dimension must match the vector-field input dimension.
            # `init_hidden_dim` controls the *width* of this MLP, not the output size.
            out_size=cde_state_dim,
            width_size=init_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = CDEFunc(
            input_path_dim=input_path_dim,
            cde_state_dim=cde_state_dim,
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

        # Static configuration
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon
        self.evolving_out = evolving_out
        self.manifold = manifold
        self.readout_activation = readout_activation
        self.control_interpolation = control_interpolation

        # Solver configuration
        self.solver = solver
        self.adjoint = adjoint
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        """Apply readout to hidden states, converting from 6D to 3x3 rotation matrices."""

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            if issubclass(self.manifold, SPDManifold):
                matrix = SPDManifold.unvech(activation)
                return SPDManifold.retract(matrix)
            return self.manifold.retract(activation)

        return jax.vmap(apply_single)(hidden_states)

    def _scaled_dt0_for_observation_grid(self, length: int) -> float | None:
        """Rescale `dt0` when solving on the native sample-index grid."""
        if self.dt0 is None or length <= 1:
            return self.dt0
        return float(self.dt0) * float(length - 1)

    def _forward_with_control(
        self,
        ts: jax.Array,
        control: diffrax.AbstractPath,
        *,
        dt0: float | None,
    ) -> jax.Array:
        """Core forward pass given control path (standard Neural CDE).

        We use the provided control path directly in a Diffrax `ControlTerm`, i.e.
        we solve the vanilla Neural CDE driven by the original interpolation.
        """
        x0 = control.evaluate(ts[0])
        y0 = self.initial_cond_mlp(x0)

        term = diffrax.ControlTerm(self.cde_func, control).to_ode()

        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
            adjoint=self.adjoint,
        )
        assert solution.ys is not None
        return solution.ys

    def __call__(
        self,
        control_values: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Standard mode (self.extrapolation_scheme is None):
            model(control_values) -> outputs

        Extrapolation mode (self.extrapolation_scheme is set):
            model(control_values) -> outputs
            where control_values has shape (T_total, C) and only the first n_recon points are used
            to fit the control; the remainder is extrapolated.
        """
        length = control_values.shape[0]
        if self.extrapolation_scheme is not None:
            ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)  # (T,)
            assert self.n_recon is not None, (
                "n_recon must be set when using extrapolation_scheme"
            )
            control, _ = self.extrapolation_scheme.create_control(
                ts, control_values, self.n_recon
            )
            hidden = self._forward_with_control(ts, control, dt0=self.dt0)
            outputs = self._apply_readout(hidden)

            return outputs
        else:
            # Use the native sample-index grid for standard controls. This keeps the
            # interpolation derivative at the scale of per-sample increments instead of
            # amplifying it by `length - 1` on a normalized [0, 1] grid.
            ts = jnp.arange(length, dtype=control_values.dtype)
            dt0 = self._scaled_dt0_for_observation_grid(int(length))
            if self.control_interpolation == "hermite_cubic":
                coeffs = diffrax.backward_hermite_coefficients(ts=ts, ys=control_values)
                control = diffrax.CubicInterpolation(ts, coeffs)
            elif self.control_interpolation == "linear":
                control = diffrax.LinearInterpolation(ts, control_values)
            else:
                raise ValueError(
                    f"Unknown control_interpolation={self.control_interpolation!r}. "
                    "Expected 'hermite_cubic' or 'linear'."
                )
            hidden = self._forward_with_control(ts, control, dt0=dt0)

            if self.evolving_out:
                return self._apply_readout(hidden)
            return self._apply_readout(hidden[-1:])[0]
