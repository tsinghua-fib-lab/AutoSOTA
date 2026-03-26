"""Module for setting HCNN models for the benchmarks."""

import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
)

from .other_projections import get_cvxpy_projection, get_jaxopt_projection


def setup_pinet(
    hyperparameters: dict,
    eq_constraint: Optional[EqualityConstraint] = None,
    ineq_constraint: Optional[AffineInequalityConstraint] = None,
    box_constraint: Optional[BoxConstraint] = None,
    setup_reps: int = -1,
):
    """Setup of pinet projection layer."""
    projection_layer = Project(
        ineq_constraint=ineq_constraint,
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        unroll=hyperparameters["unroll"],
        equilibration_params=EquilibrationParams(**hyperparameters["equilibrate"]),
    )

    setup_time = 0.0
    if setup_reps > 0:
        start_setup_time = time.time()
        for _ in range(setup_reps):
            _ = Project(
                ineq_constraint=ineq_constraint,
                eq_constraint=eq_constraint,
                box_constraint=box_constraint,
                unroll=hyperparameters["unroll"],
                equilibration_params=EquilibrationParams(
                    **hyperparameters["equilibrate"]
                ),
            )
        setup_time = (time.time() - start_setup_time) / (max(setup_reps, 1))
        print(f"Time to create constraints: {setup_time:.5f} seconds")

    kw = (
        {}
        if hyperparameters["unroll"]
        else {
            "n_iter_bwd": hyperparameters["n_iter_bwd"],
            "fpi": hyperparameters["fpi"],
        }
    )

    def project(x, b):
        inp = ProjectionInstance(
            x=x[..., None], eq=EqualityConstraintsSpecification(b=b)
        )
        return projection_layer.call(
            yraw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_train"],
            **kw,
        )[0].x[..., 0]

    def project_test(x, b):
        inp = ProjectionInstance(
            x=x[..., None], eq=EqualityConstraintsSpecification(b=b)
        )
        return projection_layer.call(
            yraw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_test"],
            **kw,
        )[0].x[..., 0]

    return project, project_test, setup_time


def setup_jaxopt(
    A: jnp.ndarray,
    b: jnp.ndarray,
    C: jnp.ndarray,
    ub: jnp.ndarray,
    setup_reps: int,
    hyperparameters: dict,
):
    """Setup of jaxopt projection layer.

    Args:
        A (jnp.ndarray): Coefficient matrix for the equality constraint.
        b (jnp.ndarray): Right-hand side vector for the equality constraint.
        C (jnp.ndarray): Coefficient matrix for the inequality constraint.
        ub (jnp.ndarray): Upper bounds for the inequality constraint.
        setup_reps (int): Number of repetitions for setup timing.
            For jaxopt, we do not time the setup.
        hyperparameters (dict): Hyperparameters for the projection.

    Returns:
        project (Callable): Function to project the input.
        project_test (Callable): Function to project the input in test mode.
        setup_time (float): We always return 0.0 for jaxopt setup time.
    """
    project = get_jaxopt_projection(
        A=A[0, :, :],
        C=C[0, :, :],
        d=ub[0, :, 0],
        dim=A.shape[2],
        tol=hyperparameters["jaxopt_tol"],
    )
    project_test = project
    setup_time = 0.0
    return project, project_test, setup_time


def setup_cvxpy(
    A: jnp.ndarray,
    b: jnp.ndarray,
    C: jnp.ndarray,
    ub: jnp.ndarray,
    setup_reps: int,
    hyperparameters: dict,
) -> tuple[
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    float,
]:
    """Setup of cvxpy projection layer.

    Args:
        A (jnp.ndarray): Coefficient matrix for the equality constraint.
        b (jnp.ndarray): Right-hand side vector for the equality constraint.
        C (jnp.ndarray): Coefficient matrix for the inequality constraint.
        ub (jnp.ndarray): Upper bounds for the inequality constraint.
        setup_reps (int): Number of repetitions for setup timing.
            For cvxpy, we do not time the setup.
        hyperparameters (dict): Hyperparameters for the projection.

    Returns:
        project (Callable): Function to project the input.
        project_test (Callable): Function to project the input in test mode.
        setup_time (float): We always return 0.0 for cvxpy setup time
    """
    cvxpy_proj = get_cvxpy_projection(
        A=A[0, :, :],
        C=C[0, :, :],
        d=ub[0, :, 0],
        dim=A.shape[2],
    )

    def project(xx, bb):
        return cvxpy_proj(
            xx,
            bb[:, :, 0],
            solver_args={
                "verbose": False,
                "eps_abs": hyperparameters["cvxpy_tol"],
                "eps_rel": hyperparameters["cvxpy_tol"],
            },
        )[0]

    project_test = project
    setup_time = 0.0
    return project, project_test, setup_time


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    project: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    project_test: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    dim: int
    features_list: list
    activation: nn.Module = nn.relu
    raw_train: bool = False
    raw_test: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        b: jnp.ndarray,
        test: bool,
    ):
        """Forward pass of the MLP with projection.

        Args:
            x (jnp.ndarray): Input data.
            b (jnp.ndarray): Right-hand side vector for the equality constraint.
            test (bool): Whether to use the test projection.

        Returns:
            jnp.ndarray: Output of the MLP after projection.
        """
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.dim)(x)
        if test and (not self.raw_test):
            x = self.project_test(x, b)
        elif (not test) and (not self.raw_train):
            x = self.project(x, b)
        return x


def build_model_and_train_step(
    *,
    rng_key: jax.random.PRNGKey,
    dim: int,
    features_list: list,
    activation: nn.Module,
    project,
    project_test,
    raw_train: bool,
    raw_test: bool,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    example_x: jnp.ndarray,
    example_b: jnp.ndarray,
    jit: bool = True,
) -> tuple[
    nn.Module,
    dict,
    Callable[[TrainState, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, dict]],
]:
    """Build the model and the training step function.

    Args:
        rng_key (jax.random.PRNGKey): Random key for initialization.
        dim (int): Dimension of the input.
        features_list (list): List of features for the MLP.
        activation (nn.Module): Activation function to use in the MLP.
        project (Callable): Function to project the input during training.
        project_test (Callable): Function to project the input during testing.
        raw_train (bool): Whether to use raw training data without projection.
        raw_test (bool): Whether to use raw test data without projection.
        loss_fn (Callable): Loss function to be used during training.
        example_x (jnp.ndarray): Example input for model initialization.
        example_b (jnp.ndarray):
            Example right-hand side vector for the equality constraint.
        jit (bool): Whether to JIT compile the training step.

    Returns:
        model (nn.Module): The constructed model.
        params (dict): Initial parameters of the model.
        train_step (Callable): Function to perform a training step.
    """
    model = HardConstrainedMLP(
        project=project,
        project_test=project_test,
        dim=dim,
        features_list=features_list,
        activation=activation,
        raw_train=raw_train,
        raw_test=raw_test,
    )

    params = model.init(
        rng_key,
        x=example_x,
        b=example_b,
        test=False,
    )

    def train_step(state, x_batch: jnp.ndarray, b_batch: jnp.ndarray):
        def _loss(p):
            preds = state.apply_fn({"params": p}, x=x_batch, b=b_batch, test=False)
            return loss_fn(preds, b_batch).mean()

        loss, grads = jax.value_and_grad(_loss)(state.params)
        return loss, state.apply_gradients(grads=grads)

    if jit:
        train_step = jax.jit(train_step)

    return model, params, train_step


def setup_model(
    rng_key: jax.random.PRNGKey,
    hyperparameters: dict,
    proj_method: str,
    A: jnp.ndarray,
    X: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    batched_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    setup_reps: int = 10,
) -> tuple[
    nn.Module,
    dict,
    float,
    Callable[[TrainState, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, dict]],
]:
    """Receives problem (hyper)parameters and returns the model and its parameters.

    Args:
        rng_key (jax.random.PRNGKey): Random key for initialization.
        hyperparameters (dict): Hyperparameters for the model and projection.
        proj_method (str): Method for projection ('pinet', 'jaxopt', 'cvxpy').
        A (jnp.ndarray): Coefficient matrix for the equality constraint.
        X (jnp.ndarray): Right-hand side vector for the equality constraint.
        G (jnp.ndarray): Coefficient matrix for the inequality constraint.
        h (jnp.ndarray): Upper bounds for the inequality constraint.
        batched_loss (Callable): Loss function to be used during training.
        setup_reps (int): Number of repetitions for setup timing.

    Returns:
        model (nn.Module): The constructed model.
        params (dict): Initial parameters of the model.
        setup_time (float): Time taken to set up the projection layer.
        train_step (Callable): Function to perform a training step.
    """
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    setups = {"pinet": setup_pinet, "jaxopt": setup_jaxopt, "cvxpy": setup_cvxpy}
    if proj_method not in setups:
        raise ValueError(f"Projection method not valid: {proj_method}")

    if proj_method == "pinet":
        eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
        ineq_constraint = AffineInequalityConstraint(
            C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
        )
        project, project_test, setup_time = setups[proj_method](
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            hyperparameters=hyperparameters,
        )
    else:
        project, project_test, setup_time = setups[proj_method](
            A=A, b=X, C=G, ub=h, setup_reps=setup_reps, hyperparameters=hyperparameters
        )

    model, params, train_step = build_model_and_train_step(
        rng_key=rng_key,
        dim=A.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        project=project,
        project_test=project_test,
        raw_train=hyperparameters.get("raw_train", False),
        raw_test=hyperparameters.get("raw_test", False),
        loss_fn=batched_loss,
        example_x=X[:2, :, 0],
        example_b=X[:2],
        jit=(proj_method != "cvxpy"),
    )

    return model, params, setup_time, train_step
