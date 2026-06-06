"""Test the HardConstrainedMLP on the clipped sine function."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from pinet import BoxConstraint, BoxConstraintSpecification, Project, ProjectionInstance


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    box_constraint: BoxConstraint

    def setup(self):
        self.project = Project(box_constraint=self.box_constraint)

    @nn.compact
    def __call__(self, x, step):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = self.project.call(yraw=ProjectionInstance(x=x[..., None]))[0].x.squeeze(-1)
        return x


@pytest.mark.parametrize(
    "seed",
    [0],
)  # Add more seeds as needed
def test_clipped_sine(seed: int):
    """Test if the HardConstrainedMLP fits max(min(sin(x), 1-EPS), EPS).

    The training objective is to fit the sine function with a MLP, but the
    hard constraint is that the predictions must be clipped to the range
    [EPS, 1 - EPS]. This test checks if the projection layer effectively
    clips the predictions to the desired range.
    """
    # Test params
    EPS = 0.1
    N_SAMPLES = 1000
    LEARNING_RATE = 1e-5
    N_EPOCHS = 10000

    # Generate dataset
    x = jnp.linspace(-jnp.pi, jnp.pi, N_SAMPLES).reshape(-1, 1)
    y = jnp.sin(x)

    # Define and initialize the hard constrained MLP
    model = HardConstrainedMLP(
        box_constraint=BoxConstraint(
            BoxConstraintSpecification(
                lb=jnp.array([EPS]).reshape((1, 1, 1)),
                ub=jnp.array([1 - EPS]).reshape((1, 1, 1)),
            )
        )
    )
    params = model.init(jax.random.PRNGKey(seed), jnp.ones([1, 1]), 0)
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

    for step in range(N_EPOCHS):
        state = train_step(state, x, y, step)

    # Get predictions
    predictions = model.apply({"params": state.params}, x, 100000)

    # Clip y to meet the constraints
    clipped_y = jnp.clip(y, EPS, 1 - EPS)

    # Check if predictions meet the condition
    error = jnp.abs(predictions - clipped_y).mean()
    assert (
        error < 1e-1
    ), f"Predictions do not meet the clipping condition. Mean error: {error}"
