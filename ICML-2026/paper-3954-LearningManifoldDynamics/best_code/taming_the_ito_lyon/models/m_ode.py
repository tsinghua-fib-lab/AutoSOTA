from __future__ import annotations

import math

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from stochastax.manifolds import Manifold


def _identity(x: jax.Array) -> jax.Array:
    return x


class LocalCoordinateVectorField(eqx.Module):
    """Local coordinate vector field g(t, u, x_k) -> R^m."""

    mlp: eqx.nn.MLP
    local_dim: int = eqx.field(static=True)
    output_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        local_dim: int,
        anchor_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        output_scale: float = 1.0,
        activation: Callable[[jax.Array], jax.Array] = jnn.softplus,
        key: jax.Array,
    ) -> None:
        self.local_dim = int(local_dim)
        self.output_scale = float(output_scale)
        self.mlp = eqx.nn.MLP(
            in_size=self.local_dim + int(anchor_dim) + 1,
            out_size=self.local_dim,
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=activation,
            final_activation=_identity,
            key=key,
        )

    def __call__(
        self,
        t: jax.typing.ArrayLike,
        u: jax.Array,
        anchor: jax.Array,
    ) -> jax.Array:
        anchor_flat = jnp.ravel(anchor)
        t_arr = jnp.asarray([t], dtype=u.dtype)
        inputs = jnp.concatenate([u, anchor_flat.astype(u.dtype), t_arr], axis=0)
        return self.output_scale * jnp.tanh(self.mlp(inputs))


class ManifoldNeuralODE(eqx.Module):
    """Initial-condition-only manifold Neural ODE baseline.

    This matches the repo's standard `model(control_values)` interface, but only
    uses the first observed state as the initial condition. The remainder of the
    input path is ignored.

    To keep the implementation simple and compatible with the existing
    `stochastax` API, the moving chart uses the approximation

        Retr_x(v) := manifold.retract(x + v).
    """

    vector_field: LocalCoordinateVectorField

    manifold: type[Manifold] = eqx.field(static=True)
    local_dim: int = eqx.field(static=True)
    future_only_loss: bool = eqx.field(static=True)
    steps_per_segment: int = eqx.field(static=True)
    solver: diffrax.AbstractAdaptiveSolver = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        local_dim: int,
        anchor_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        manifold: type[Manifold],
        key: jax.Array,
        output_scale: float = 1.0,
        activation: Callable[[jax.Array], jax.Array] = jnn.softplus,
        future_only_loss: bool = True,
        steps_per_segment: int = 4,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Tsit5(),
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller: diffrax.AbstractStepSizeController,
        dt0: float | None = None,
    ) -> None:
        self.local_dim = int(local_dim)
        self.manifold = manifold
        self.future_only_loss = bool(future_only_loss)
        self.steps_per_segment = max(1, int(steps_per_segment))
        self.vector_field = LocalCoordinateVectorField(
            local_dim=local_dim,
            anchor_dim=anchor_dim,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            output_scale=output_scale,
            activation=activation,
            key=key,
        )
        self.solver = solver
        self.adjoint = adjoint
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0

    def _frame(self, anchor: jax.Array) -> jax.Array:
        ambient_dim = math.prod(anchor.shape)
        if self.local_dim > ambient_dim:
            raise ValueError(
                f"local_dim={self.local_dim} exceeds ambient_dim={ambient_dim}."
            )

        canonical_basis = jnp.eye(ambient_dim, dtype=anchor.dtype).reshape(
            (ambient_dim,) + anchor.shape
        )
        tangent_basis = jax.vmap(lambda v: self.manifold.project_to_tangent(anchor, v))(
            canonical_basis
        )
        tangent_basis = tangent_basis.reshape(ambient_dim, ambient_dim).T
        u, _, _ = jnp.linalg.svd(tangent_basis, full_matrices=False)
        return u[:, : self.local_dim]

    def _coordinates_to_tangent(self, anchor: jax.Array, u: jax.Array) -> jax.Array:
        frame = self._frame(anchor)
        tangent_flat = frame @ u
        return tangent_flat.reshape(anchor.shape)

    def _chart_map(self, anchor: jax.Array, u: jax.Array) -> jax.Array:
        tangent = self._coordinates_to_tangent(anchor, u)
        return self.manifold.retract(anchor + tangent)

    def _solve_segment(
        self,
        anchor: jax.Array,
        t0: jax.Array,
        t1: jax.Array,
    ) -> jax.Array:
        local_state0 = jnp.zeros((self.local_dim,), dtype=anchor.dtype)
        segment_dt = (t1 - t0) / jnp.asarray(self.steps_per_segment, dtype=anchor.dtype)

        def ode_func(t: jax.Array, u: jax.Array, args: None) -> jax.Array:
            del args
            return self.vector_field(t, u, anchor)

        # Keep each chart solve cheap and predictable. Using a fixed small number
        # of substeps per adjacent save interval avoids PID step-collapse inside
        # the moving retraction chart.
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(ode_func),
            solver=self.solver,
            t0=t0,
            t1=t1,
            dt0=segment_dt,
            y0=local_state0,
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            adjoint=self.adjoint,
            max_steps=self.steps_per_segment + 1,
        )
        assert solution.ys is not None
        return self._chart_map(anchor, solution.ys[0])

    def _solve_from_initial(self, x0: jax.Array, ts: jax.Array) -> jax.Array:
        if ts.ndim != 1:
            raise ValueError(f"Expected `ts` to have shape (T,), got {ts.shape}.")
        if ts.shape[0] == 0:
            raise ValueError("Expected at least one save time.")
        if ts.shape[0] == 1:
            return x0[None, ...]

        interval_bounds = jnp.stack([ts[:-1], ts[1:]], axis=1)

        def step(anchor: jax.Array, bounds: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_anchor = self._solve_segment(anchor, bounds[0], bounds[1])
            return next_anchor, next_anchor

        _, xs = jax.lax.scan(step, x0, interval_bounds)
        return jnp.concatenate([x0[None, ...], xs], axis=0)

    def _extract_initial_condition(self, control_values: jax.Array) -> jax.Array:
        if control_values.ndim == 3:
            return self.manifold.retract(control_values[0])

        if control_values.ndim != 2:
            raise ValueError(
                "Expected control_values with shape (T, C) or (T, ..., ...), "
                f"got {control_values.shape}."
            )

        x0 = control_values[0]
        if x0.shape[-1] == 9:
            return self.manifold.retract(x0.reshape(3, 3))
        return self.manifold.retract(x0)

    def __call__(self, control_values: jax.Array) -> jax.Array:
        if control_values.shape[0] == 0:
            raise ValueError("Expected at least one observed time step.")

        ts = jnp.linspace(
            0.0,
            1.0,
            control_values.shape[0],
            dtype=control_values.dtype,
        )
        x0 = self._extract_initial_condition(control_values)
        return self._solve_from_initial(x0, ts)
