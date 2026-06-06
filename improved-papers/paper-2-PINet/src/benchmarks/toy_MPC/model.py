"""Module for setting up Pinet models for toy MPC."""

from typing import Any, Callable

import jax.numpy as jnp
from flax import linen as nn
from jax import random as jrnd

from pinet import BoxConstraint, BoxConstraintSpecification, EqualityConstraint
from src.benchmarks.model import build_model_and_train_step, setup_pinet


def setup_model(
    rng_key: jrnd.PRNGKey,
    hyperparameters: dict[str, Any],
    A: jnp.ndarray,
    X: jnp.ndarray,
    b: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Receives problem (hyper)parameters and returns the model and its parameters.

    Args:
        rng_key (jrnd.PRNGKey): Random key for initialization.
        hyperparameters (dict[str, Any]): Hyperparameters for the model.
        A (jnp.ndarray): Coefficient matrix for the equality constraint.
        X (jnp.ndarray): Input data for the model.
        b (jnp.ndarray): Right-hand side vector for the equality constraint.
        lb (jnp.ndarray): Lower bounds for the box constraint.
        ub (jnp.ndarray): Upper bounds for the box constraint.
        batched_objective (Callable[[jnp.ndarray], jnp.ndarray]): Function to compute
            the objective value for the model predictions.

    Returns:
        model (nn.Module): The Pinet model.
        params (dict[str, Any]): Parameters of the model.
        train_step (Callable): Function to perform a training step.
    """
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    # Constraints + projection layer
    eq_constraint = EqualityConstraint(A=A, b=b, method=None, var_b=True)
    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub))
    project, project_test, _ = setup_pinet(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        hyperparameters=hyperparameters,
    )

    # Reuse the shared builder; adapt the loss to ignore b
    model, params, train_step = build_model_and_train_step(
        rng_key=rng_key,
        dim=A.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        project=project,
        project_test=project_test,
        raw_train=hyperparameters.get("raw_train", False),
        raw_test=hyperparameters.get("raw_test", False),
        loss_fn=lambda preds, _b: batched_objective(preds),
        example_x=X[:1, :, 0],
        example_b=b[:1],
        jit=True,
    )

    return model, params, train_step
