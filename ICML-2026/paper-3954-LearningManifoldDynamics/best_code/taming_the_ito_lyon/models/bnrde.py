import math
import os
from collections.abc import Callable
from typing import Literal

import diffrax
import equinox as eqx
import georax
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import roughrax
from stochastax.manifolds import SO3, EuclideanSpace, Manifold
from stochastax.manifolds.spd import SPDManifold

from taming_the_ito_lyon.config.config_options import HiddenStateMode, RoughSolution

from .extrapolation import ExtrapolationScheme
from .rough_utils import ItoCorrectedSignatureInterpolation, compute_disjoint_signature_times


def lipswish(x: jax.Array) -> jax.Array:
    return 0.909 * jnn.silu(x)


def _spd_n_from_dim(dim: int) -> int:
    disc = 8 * int(dim) + 1
    root = math.isqrt(disc)
    if root * root != disc:
        raise ValueError(f"Expected triangular SPD dimension, got {dim}.")
    return (root - 1) // 2


def _geometry_for_manifold(
    manifold: type[Manifold],
    state_param_dim: int,
) -> tuple[georax.Manifold, tuple[int, ...], int]:
    if manifold is EuclideanSpace:
        return georax.Euclidean(), (int(state_param_dim),), int(state_param_dim)
    if manifold is SO3:
        if int(state_param_dim) not in (6, 9):
            raise ValueError(
                "SO3 initializes from a 6D representation or a direct 3x3 state, "
                f"so initial_state_param_dim must be 6 or 9; got {state_param_dim}."
            )
        return georax.SO(3), (3, 3), 3
    if manifold is SPDManifold:
        n = _spd_n_from_dim(state_param_dim)
        return georax.SPD(n), (n, n), n * (n + 1) // 2
    raise ValueError(f"Unsupported BNRDE manifold: {manifold}.")


class BNRDEFunc(eqx.Module):
    """BNRDE vector field returning georax frame coefficients."""

    input_path_dim: int = eqx.field(static=True)
    frame_dim: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    vf_mlp: eqx.nn.MLP

    def __init__(
        self,
        *,
        input_path_dim: int,
        frame_dim: int,
        state_shape: tuple[int, ...],
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        key: jax.Array,
    ) -> None:
        self.input_path_dim = int(input_path_dim)
        self.frame_dim = int(frame_dim)
        self.state_shape = tuple(int(i) for i in state_shape)
        self.vf_mlp = eqx.nn.MLP(
            in_size=math.prod(self.state_shape),
            out_size=self.input_path_dim * self.frame_dim,
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        return self.vf_mlp(jnp.ravel(y)).reshape(self.input_path_dim, self.frame_dim)


class BNRDE(eqx.Module):
    """Branched/rough neural differential equation backed by roughrax."""

    initial_cond_mlp: eqx.nn.MLP
    vector_field: BNRDEFunc
    readout_layer: eqx.nn.Linear | None
    extrapolation_scheme: ExtrapolationScheme | None

    data_manifold: type[Manifold] = eqx.field(static=True)
    hidden_state_mode: HiddenStateMode = eqx.field(static=True)
    geometry: georax.Manifold
    state_shape: tuple[int, ...] = eqx.field(static=True)
    rough_solution: Literal["ito", "stratonovich"] = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)
    prepend_zero_basepoint: bool = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

    solver: diffrax.AbstractSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)
    adjoint: diffrax.AbstractAdjoint = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        initial_state_param_dim: int,
        output_path_dim: int,
        initial_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        signature_depth: int,
        signature_window_size: int,
        *,
        key: jax.Array,
        data_manifold: type[Manifold],
        hidden_state_mode: HiddenStateMode,
        rough_solution: RoughSolution | Literal["ito", "stratonovich"],
        solver: diffrax.AbstractSolver,
        stepsize_controller: diffrax.AbstractStepSizeController | None = None,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        prepend_zero_basepoint: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        if hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD and not isinstance(
            solver, (georax.CG2, georax.CFEES25)
        ):
            raise ValueError(
                "hidden_state_mode='problem_manifold' requires solver to be "
                "georax.CG2() or georax.CFEES25()."
            )

        hidden_manifold = (
            EuclideanSpace
            if hidden_state_mode == HiddenStateMode.EUCLIDEAN
            else data_manifold
        )
        geometry, state_shape, frame_dim = _geometry_for_manifold(
            hidden_manifold,
            initial_state_param_dim,
        )

        if hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD:
            expected_output_dim = (
                frame_dim if data_manifold is SPDManifold else math.prod(state_shape)
            )
            if int(output_path_dim) != expected_output_dim:
                raise ValueError(
                    "BNRDE with hidden_state_mode='problem_manifold' uses the "
                    "integrated manifold state as the output, so output_path_dim must "
                    f"match {expected_output_dim}; got {output_path_dim}."
                )

        k_init, k_readout, k_vf = jr.split(key, 3)
        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=initial_state_param_dim,
            width_size=initial_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=lipswish,
            key=k_init,
        )
        self.vector_field = BNRDEFunc(
            input_path_dim=input_path_dim,
            frame_dim=frame_dim,
            state_shape=state_shape,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            key=k_vf,
        )
        self.readout_layer = (
            eqx.nn.Linear(
                in_features=initial_state_param_dim,
                out_features=output_path_dim,
                use_bias=True,
                key=k_readout,
            )
            if hidden_state_mode == HiddenStateMode.EUCLIDEAN
            else None
        )

        self.data_manifold = data_manifold
        self.hidden_state_mode = hidden_state_mode
        self.geometry = geometry
        self.state_shape = state_shape
        self.rough_solution = str(rough_solution)
        self.readout_activation = readout_activation
        self.signature_depth = int(signature_depth)
        self.signature_window_size = int(signature_window_size)
        self.evolving_out = evolving_out
        self.prepend_zero_basepoint = (
            prepend_zero_basepoint
            and os.environ.get("TIL_BNRDE_PREPEND_ZERO_BASEPOINT", "1") != "0"
        )
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon
        self.solver = solver
        self.stepsize_controller = (
            stepsize_controller
            if stepsize_controller is not None
            else diffrax.ConstantStepSize()
        )
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

    def _initial_hidden(self, x0: jax.Array) -> jax.Array:
        raw = self.initial_cond_mlp(x0)
        if self.hidden_state_mode == HiddenStateMode.EUCLIDEAN:
            return raw
        if self.data_manifold is SO3:
            if raw.shape[-1] == math.prod(self.state_shape):
                return SO3.retract(raw.reshape(self.state_shape))
            return SO3.retract(raw)
        if self.data_manifold is SPDManifold:
            sym = SPDManifold.unvech(raw)
            sym = 0.5 * (sym + jnp.swapaxes(sym, -1, -2))
            evals, evecs = jnp.linalg.eigh(sym)
            evals = jnp.clip(evals, -8.0, 8.0)
            return (evecs * jnp.exp(evals)[..., None, :]) @ jnp.swapaxes(evecs, -1, -2)
        raise ValueError(f"Could not initialize BNRDE state for {self.data_manifold}.")

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        if self.hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD:
            return hidden_states

        assert self.readout_layer is not None

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            if self.data_manifold is SPDManifold:
                matrix = SPDManifold.unvech(activation)
                return SPDManifold.retract(matrix)
            return self.data_manifold.retract(activation)

        return jax.vmap(apply_single)(hidden_states)

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
        if (
            self.rough_solution == "ito"
            and os.environ.get("TIL_BNRDE_ITO_CORRECTION", "0") == "1"
        ):
            control = ItoCorrectedSignatureInterpolation(
                driver,
                signature_ts,
                depth=int(self.signature_depth),
            )
        else:
            control = roughrax.SignatureInterpolation(
                driver,
                signature_ts,
                depth=int(self.signature_depth),
                solution=self.rough_solution,
            )

        def vector_field(y: jax.Array) -> jax.Array:
            return self.vector_field(y)

        term = roughrax.RoughTerm(vector_field, control, self.geometry)
        if (
            os.environ.get("TIL_BNRDE_LEGACY_PIECEWISE", "0") == "1"
            and isinstance(self.geometry, georax.Euclidean)
        ):
            return self._solve_piecewise_logode(ts, signature_ts, term, y0)

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

    def _solve_piecewise_logode(
        self,
        ts: jax.Array,
        signature_ts: jax.Array,
        term: roughrax.RoughTerm,
        y0: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Legacy MNRDE-style integration of the frozen log-ODE per window."""
        assert term.control.coeffs is not None
        step = int(self.signature_window_size)
        num_windows = int(term.control.coeffs.shape[0])
        state_shape = tuple(int(i) for i in jnp.shape(y0))
        outputs = jnp.zeros(
            (num_windows, step + 1, *state_shape),
            dtype=y0.dtype,
        )
        step_counts = jnp.zeros((num_windows,), dtype=jnp.int32)

        def body(
            i: jax.Array,
            carry: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            y, out, counts = carry
            start_index = i * step
            ts_window = jax.lax.dynamic_slice(ts, (start_index,), (step + 1,))
            t0 = ts_window[0]
            t1 = ts_window[-1]
            dzdt = term.control.coeffs[i] / (t1 - t0)

            def ode_func(
                t: jax.typing.ArrayLike,
                y: jax.Array,
                args: None,
            ) -> jax.Array:
                del args
                vf = term.vf(t, y, None)
                return jnp.tensordot(dzdt, vf, axes=1)

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(ode_func),
                self.solver,
                t0=t0,
                t1=t1,
                dt0=0.01
                if isinstance(self.stepsize_controller, diffrax.ConstantStepSize)
                else None,
                y0=y,
                stepsize_controller=self.stepsize_controller,
                saveat=diffrax.SaveAt(ts=ts_window),
                adjoint=self.adjoint,
                max_steps=9999,
            )
            assert solution.ys is not None
            out = out.at[i].set(solution.ys)
            counts = counts.at[i].set(solution.stats["num_steps"])
            return solution.ys[-1], out, counts

        _, outputs, step_counts = jax.lax.fori_loop(
            0,
            num_windows,
            body,
            (y0, outputs, step_counts),
        )
        first = outputs[0]
        if num_windows == 1:
            ys = first
        else:
            rest = outputs[1:, 1:, ...].reshape(
                ((num_windows - 1) * step, *state_shape)
            )
            ys = jnp.concatenate([first, rest], axis=0)
        return ys, {"num_steps": jnp.sum(step_counts)}

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

        ts_aug, control_values_aug = self._maybe_prepend_zero_basepoint(
            ts, control_values
        )
        return int(length), ts_aug, control_values_aug

    def __call__(self, control_values: jax.Array) -> jax.Array:
        length, ts, control_values = self._prepare_control(control_values)
        h0 = self._initial_hidden(control_values[0])
        hidden, _ = self._solve_from_values(ts, control_values, h0)
        outputs = self._apply_readout(hidden)

        if self.prepend_zero_basepoint:
            outputs = outputs[1 : 1 + length]

        if self.evolving_out:
            return outputs
        return outputs[-1]
