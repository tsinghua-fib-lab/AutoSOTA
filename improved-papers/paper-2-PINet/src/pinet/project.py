"""Implementation of the projection layer."""

from functools import partial
from typing import Callable, Optional

import jax
from jax import numpy as jnp

from .constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
)
from .dataclasses import EquilibrationParams, ProjectionInstance
from .equilibration import ruiz_equilibration
from .solver import build_iteration_step, initialize


class Project:
    """Projection layer implemented via Douglas-Rachford."""

    eq_constraint: EqualityConstraint = None
    ineq_constraint: AffineInequalityConstraint = None
    box_constraint: BoxConstraint = None
    unroll: bool = False

    def __init__(
        self,
        eq_constraint: EqualityConstraint = None,
        ineq_constraint: AffineInequalityConstraint = None,
        box_constraint: BoxConstraint = None,
        unroll: bool = False,
        equilibration_params: EquilibrationParams = EquilibrationParams(),
    ) -> None:
        """Initialize projection layer.

        Args:
            eq_constraint (EqualityConstraint): Equality constraint.
            ineq_constraint (AffineInequalityConstraint): Inequality constraint.
            box_constraint (BoxConstraint): Box constraint.
            unroll (bool): Use loop unrolling for backpropagation.
            equilibration_params (EquilibrationParams): Parameters for equilibration.
        """
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.unroll = unroll
        self.equilibration_params = equilibration_params
        self.setup()

    def setup(self) -> None:
        """Setup the projection layer."""
        constraints = [
            c
            for c in (self.eq_constraint, self.box_constraint, self.ineq_constraint)
            if c
        ]
        assert len(constraints) > 0, "At least one constraint must be provided."
        self.dim = constraints[0].dim

        is_single_simple_constraint = (
            self.ineq_constraint is None and len(constraints) == 1
        )

        self.dim_lifted = self.dim
        self.step_iteration = lambda s_prev, yraw, sigma, omega: s_prev
        self.step_final = self._project_single
        self.single_constraint = constraints[0]
        self.d_r = jnp.ones((1, self.single_constraint.n_constraints, 1))
        self.d_c = jnp.ones((1, self.single_constraint.dim, 1))
        if not is_single_simple_constraint:
            # Constraints need to be parsed
            if self.ineq_constraint is not None:
                self.dim_lifted += self.ineq_constraint.n_constraints
            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
            )
            (self.lifted_eq_constraint, self.lifted_box_constraint, self.lift) = (
                parser.parse(method=None)
            )
            # Only equilibrate when we have a single A
            if (
                not self.lifted_eq_constraint.var_A
                and self.lifted_eq_constraint.A.shape[0] == 1
            ):
                scaled_A, self.d_r, self.d_c = ruiz_equilibration(
                    self.lifted_eq_constraint.A[0], self.equilibration_params
                )
                # Update A in lifted equality and setup projection
                self.lifted_eq_constraint.A = scaled_A.reshape(
                    1,
                    self.lifted_eq_constraint.A.shape[1],
                    self.lifted_eq_constraint.A.shape[2],
                )
                self.d_r = self.d_r.reshape(1, -1, 1)
                self.d_c = self.d_c.reshape(1, -1, 1)
            else:
                # No equilibration for variable A
                n_ineq = (
                    self.ineq_constraint.n_constraints
                    if self.ineq_constraint is not None
                    else 0
                )
                self.d_r = jnp.ones((1, self.eq_constraint.n_constraints + n_ineq, 1))
                self.d_c = jnp.ones((1, self.dim_lifted, 1))

            self.lifted_eq_constraint.method = "pinv"
            self.lifted_eq_constraint.setup()

            # Scale the equality RHS
            self.lifted_eq_constraint.b *= self.d_r
            # Scale the lifted box constraints
            mask = self.lifted_box_constraint.mask
            scale = self.d_c[:, mask, :]
            self.lifted_box_constraint.scale = 1 / scale
            self.lifted_box_constraint.ub *= self.lifted_box_constraint.scale
            self.lifted_box_constraint.lb *= self.lifted_box_constraint.scale

            self.step_iteration, self.step_final = build_iteration_step(
                self.lifted_eq_constraint,
                self.lifted_box_constraint,
                self.dim,
                self.d_c[:, : self.dim, :],
            )

        project_fn = (
            _project_general
            if (self.unroll or is_single_simple_constraint)
            else _project_general_custom
        )

        static_args = (
            ["n_iter"]
            if (self.unroll or is_single_simple_constraint)
            else ["n_iter", "n_iter_bwd", "fpi"]
        )

        self._project = jax.jit(
            partial(
                project_fn,
                initialize_fn=self.initialize,
                step_iteration=self.step_iteration,
                step_final=self.step_final,
                dim_lifted=self.dim_lifted,
                d_r=self.d_r,
                d_c=self.d_c,
            ),
            static_argnames=static_args,
        )

        # jit correctly the call method
        self.call = self._project

    def initialize(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Returns a zero initial value for the governing sequence.

        Args:
            yraw (ProjectionInstance): Point to be projected data.

        Returns:
            ProjectionInstance: Initial value for the governing sequence.
        """
        return initialize(
            yraw=yraw,
            ineq_constraint=self.ineq_constraint,
            box_constraint=self.box_constraint,
            dim=self.dim,
            dim_lifted=self.dim_lifted,
            d_r=self.d_r,
        )

    def cv(self, y: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            y (ProjectionInstance): Point to be evaluated.
                Shape of y.x is (batch_size, dimension, 1).

        Returns:
            jnp.ndarray: Constraint violation for each point in the batch.
        """
        if y.x.shape[1] != self.dim_lifted:
            y = self.lift(y)
        return jnp.maximum(
            self.lifted_eq_constraint.cv(y),
            self.lifted_box_constraint.cv(y),
        )

    def call_and_check(
        self,
        sigma=1.0,
        omega=1.7,
        check_every=10,
        tol=1e-3,
        max_iter=100,
        reduction="max",
    ) -> Callable[[ProjectionInstance], tuple[jnp.ndarray, bool, int]]:
        """Returns a function that projects input and checks constraint violation.

        Args:
            check_every (int): Frequency of checking constraint violation.
            tol (float): Tolerance for constraint violation.
            max_iter (int): Maximum number of iterations for checking.
            reduction (str): Method to reduce constraint violations among a batch.
            Valid options are: "max" meaning that maximum cv is less that tol;
            "mean" meaning that mean cv is less than tol;
            or a number in [0,1] meaning the percentage of instances
            with cv less than tol.

        Returns:
            Callable: Takes as input the points to be projected and any specifications for
            the constraints (e.g., the value of b for variable b equality constraints.).
            Returns an approximately project and a flag showing whether the termination
            condition was satisfied.
        """

        @jax.jit
        def check(inp: ProjectionInstance) -> bool:
            if reduction == "max":
                return jnp.max(self.cv(inp)) < tol
            elif reduction == "mean":
                return jnp.mean(self.cv(inp)) < tol
            elif isinstance(reduction, float) and 0 < reduction < 1:
                return jnp.mean(self.cv(inp) < tol) >= reduction
            else:
                raise ValueError(
                    f"Invalid reduction method {reduction}. "
                    "Valid options are: 'max', 'mean', or a float in (0, 1)."
                )

        def project_and_check(
            yraw: ProjectionInstance,
        ) -> tuple[jnp.ndarray, bool, int]:
            # Executed iterations
            iter_exec = 0
            terminate = False
            # Call the projection function with all given arguments.
            y0 = self.initialize(yraw)
            while not (terminate or iter_exec >= max_iter):
                xproj, y = self.call(
                    s0=y0,
                    yraw=yraw,
                    sigma=sigma,
                    omega=omega,
                    n_iter=check_every,
                )
                y0 = y
                iter_exec += check_every
                terminate = check(xproj)

            return xproj, terminate, iter_exec

        return project_and_check

    def _project_single(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Project a batch of points with single constraint.

        Args:
            x (jnp.ndarray): Point to be projected.
                Shape (batch_size, dimension, 1).

        Returns:
            jnp.ndarray: The projected point for each point in the batch.
        """
        if yraw.eq and yraw.eq.A is not None:
            Apinv = jnp.linalg.pinv(yraw.eq.A)
            yraw = yraw.update(eq=yraw.eq.update(Apinv=Apinv))

        return self.single_constraint.project(yraw)


# Project general
def _project_general(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    d_r: jnp.ndarray,
    d_c: jnp.ndarray,
    yraw: ProjectionInstance,
    s0: Optional[ProjectionInstance] = None,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
) -> tuple[ProjectionInstance, ProjectionInstance]:
    """Project a batch of points using Douglas-Rachford.

    Args:
        step_iteration (callable): Function for the iteration step.
        step_final (callable): Function for the final step.
        dim_lifted (int): Dimension of the lifted space.
        d_r (jnp.ndarray): Scaling factor for the rows.
        d_c (jnp.ndarray): Scaling factor for the columns.
        yraw (jnp.ndarray): Point to be projected.
            Shape (batch_size, dimension, 1).
        s0 (ProjectionInstance, optional): Initial value for the governing sequence.
        sigma (float): ADMM parameter.
        omega (float): ADMM parameter.
        n_iter (int): Number of iterations to run.

    Returns:
        tuple[ProjectionInstance, ProjectionInstance]: First output is the projected
            point, and second output is the value of the governing sequence.
    """
    if n_iter > 0:
        s0 = initialize_fn(yraw) if s0 is None else s0
        sk, _ = jax.lax.scan(
            lambda s_prev, _: (
                step_iteration(s_prev, yraw, sigma, omega),
                None,
            ),
            s0,
            None,
            length=n_iter,
        )
    else:
        sk = yraw
    y = step_final(sk).x[:, : yraw.x.shape[1], :]
    y_scaled = y * d_c[:, : yraw.x.shape[1], :]

    # Unscale the output
    return yraw.update(x=y_scaled), sk


@partial(
    jax.custom_vjp,
    nondiff_argnames=[
        "initialize_fn",
        "step_iteration",
        "step_final",
        "dim_lifted",
        "n_iter",
        "n_iter_bwd",
        "fpi",
    ],
)
def _project_general_custom(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    d_r: jnp.ndarray,
    d_c: jnp.ndarray,
    yraw: ProjectionInstance,
    s0: Optional[ProjectionInstance] = None,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[ProjectionInstance, ProjectionInstance]:
    return _project_general(
        initialize_fn=initialize_fn,
        step_iteration=step_iteration,
        step_final=step_final,
        dim_lifted=dim_lifted,
        d_r=d_r,
        d_c=d_c,
        s0=s0,
        yraw=yraw,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )


def _project_general_fwd(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    d_r: jnp.ndarray,
    d_c: jnp.ndarray,
    yraw: ProjectionInstance,
    s0: Optional[ProjectionInstance] = None,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[
    tuple[ProjectionInstance, ProjectionInstance],
    tuple[ProjectionInstance, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float],
]:
    # unpack trailing options that belong only to custom vjp
    y, sK = _project_general_custom(
        initialize_fn=initialize_fn,
        step_iteration=step_iteration,
        step_final=step_final,
        dim_lifted=dim_lifted,
        d_r=d_r,
        d_c=d_c,
        s0=s0,
        yraw=yraw,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )

    return (y, sK), (sK, yraw, d_r, d_c, sigma, omega)


def _project_general_bwd(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    n_iter: int,
    n_iter_bwd: int,
    fpi: bool,
    residuals: tuple,
    cotangent: jnp.ndarray,
) -> tuple[None, None, jnp.ndarray, None, None, None]:
    """Backward pass for custom vjp.

    This function computes the vjp for the projection using the
    implicit function theorem.
    Note that, the arguments are:
    (i) any arguments for the
    forward that are not jnp.ndarray;
    (ii) residuals: tuple with auxiliary data from the forward pass;
    (iii) cotangent: incoming cotangents.
    The function returns a tuple where each element corresponds
    to a jnp.ndarray from the input.

    Args:
        step_iteration (callable): Function for the iteration step.
        step_final (callable): Function for the final step.
        dim_lifted (int): Dimension of the lifted space.
        n_iter (int): Number of iterations to run.
        n_iter_bwd (int): Number of iterations for backward pass.
        fpi (bool): Whether to use fixed-point iteration.
        residuals (tuple): Auxiliary data from the forward pass.
        cotangent (tuple): Incoming cotangents.

    Returns:
        tuple: The computed cotangent for the projection.
    """
    sK, yraw, _, d_c, sigma, omega = residuals
    cotangent_zk1, _ = cotangent

    _, iteration_vjp = jax.vjp(
        lambda xx: step_iteration(xx, yraw, sigma, omega),
        sK,
    )
    _, iteration_vjp2 = jax.vjp(lambda xx: step_iteration(sK, xx, sigma, omega), yraw)
    _, equality_vjp = jax.vjp(lambda xx: step_final(xx), sK)

    # Rescale the gradient
    cotangent_zk1 = cotangent_zk1.x * d_c[:, : yraw.x.shape[1], :]

    # Compute VJP of cotangent with projection before auxiliary
    cotangent_eq_6 = equality_vjp(
        sK.update(
            x=jnp.concatenate(
                [
                    cotangent_zk1,
                    jnp.zeros(
                        (cotangent_zk1.shape[0], dim_lifted - cotangent_zk1.shape[1], 1)
                    ),
                ],
                axis=1,
            )
        )
    )[0].x
    # Run iteration
    if fpi:

        def body_fn(x, _):
            vjp = iteration_vjp(x)[0].x
            return sK.update(x=(vjp + cotangent_eq_6)), None

        cotangent_eq_7, _ = jax.lax.scan(
            body_fn,
            sK.update(x=jnp.zeros((cotangent_zk1.shape[0], dim_lifted, 1))),
            None,
            length=n_iter_bwd,
        )
    else:
        cotangent_eq_7 = jax.scipy.sparse.linalg.bicgstab(
            A=lambda x: x - iteration_vjp(sK.update(x=x))[0].x,
            b=cotangent_eq_6,
            maxiter=n_iter_bwd,
        )[0]
        cotangent_eq_7 = sK.update(x=cotangent_eq_7)

    thevjp = iteration_vjp2(cotangent_eq_7)[0]

    return (None, None, thevjp, None, None, None)


_project_general_custom.defvjp(_project_general_fwd, _project_general_bwd)
