"""Test the HardConstrainedMLP on 2d with 1 equality constraint."""

from itertools import product

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from pinet import EqualityConstraint, Project, ProjectionInstance

jax.config.update("jax_enable_x64", True)
# Random seeds
SEEDS = [24, 42]
VALID_METHODS = ["pinv"]


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    eq_constraint: EqualityConstraint

    def setup(self):
        self.project = Project(eq_constraint=self.eq_constraint)

    @nn.compact
    def __call__(self, x, step):
        x = nn.Dense(64)(x)
        x = nn.softplus(x)
        x = nn.Dense(64)(x)
        x = nn.softplus(x)
        x = nn.Dense(2)(x)
        x = self.project.call(yraw=ProjectionInstance(x=x[..., None]))[0].x.squeeze(-1)
        return x


@pytest.mark.parametrize("method, seed", product(VALID_METHODS, SEEDS))
def test_equality_constraint_2d(method, seed):
    """Test HardConstrainedMLP on a 2D fitting with equality constraints.

    The problem consists in fitting a (1D) parametrization of a 2D curve.
    The output is constrained to lie on the y=x line, i.e., one
    equality constraint.
    """
    # Test parameters
    N_SAMPLES = 2000
    LEARNING_RATE = 5e-3
    N_EPOCHS = 2000

    # Generate dataset
    x = jnp.linspace(0.0, 1.0, N_SAMPLES).reshape(-1, 1)
    y = jnp.hstack((x, x + jnp.sin(1.0 * jnp.pi * x)))

    # Define and initialize the hard-constrained MLP
    # Define equality constraint LHS and RHS
    A = jnp.expand_dims(jnp.array([[1.0, -1.0]]), axis=0)
    b = jnp.zeros(shape=(1, 1, 1))
    # Instantiate equality constraint
    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    model = HardConstrainedMLP(eq_constraint)
    params = model.init(jax.random.PRNGKey(seed=seed), jnp.ones((1, 1)), 0)
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    # Train the MLP
    @jax.jit
    def train_step(state, x_batch, y_batch, step):
        def loss_fn(params):
            predictions = state.apply_fn({"params": params}, x_batch, step)
            return jnp.mean((predictions - y_batch) ** 2)

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    # Run training
    for step in range(N_EPOCHS):
        state = train_step(state, x, y, step)

    # Get predictions
    predictions = model.apply({"params": state.params}, x, 100000)

    # Project ground truth onto the y=x axis
    projected_y = jnp.repeat(jnp.average(y, axis=1, keepdims=True), repeats=2, axis=1)

    atol = 3e-2
    rtol = 1e-3
    assert jnp.allclose(predictions, projected_y, atol=atol, rtol=rtol)

    # Plot dataset and print extra results
    # Create a scatter plot
    extra_results = False
    if extra_results:
        # Plot ground truth
        scatter_gt = plt.scatter(y[:, 0], y[:, 1], c=x, cmap="viridis", label="GT")
        plt.plot(y[:, 0], y[:, 1], c="blue", label="Predictions")

        # Plot predictions
        plt.scatter(
            predictions[:, 0],
            predictions[:, 1],
            c=x,
            cmap="viridis",
            label="Predictions",
        )
        plt.plot(predictions[:, 0], predictions[:, 1], c="black", label="Predictions")

        # Plot ground truth projected onto the y=x line
        # Add a little offset for visual clarity
        plotting_offset = 0.05
        plt.scatter(
            projected_y[:, 0] + plotting_offset,
            projected_y[:, 1] - plotting_offset,
            c=x,
            cmap="viridis",
            label="Project GT",
        )
        plt.plot(
            projected_y[:, 0] + plotting_offset,
            projected_y[:, 1] - plotting_offset,
            c="red",
            label="Project GT",
        )

        # Add a colorbar
        plt.colorbar(scatter_gt, label="Parametric value x")

        # Label axes
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")

        # Show plot
        plt.axis("equal")
        plt.show()
