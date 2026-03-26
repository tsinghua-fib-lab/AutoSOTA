"""Test the HardConstrainedMLP on the clipped (v2) sine function."""

import cvxpy as cp
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from pinet import AffineInequalityConstraint, Project, ProjectionInstance

jax.config.update("jax_enable_x64", True)


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    project: Project

    @nn.compact
    def __call__(self, x, _, sigma=1.0, omega=1.7):
        x = nn.Dense(64)(x)
        x = nn.softplus(x)
        x = nn.Dense(64)(x)
        x = nn.softplus(x)
        x = nn.Dense(1)(x)
        x = self.project.call(
            yraw=ProjectionInstance(x=x.reshape((x.shape[0], x.shape[1], 1))),
            sigma=sigma,
            omega=omega,
            n_iter=100,
            n_iter_bwd=50,
            fpi=False,
        )[0].x.squeeze(-1)
        return x


SEEDS = [42]
COEFFICIENTS = [0.25, 0.5, 0.75]
LOWER_BOUNDS = [-1.0]
UPPER_BOUNDS = [1.0]


@pytest.mark.parametrize(
    "seed, C, lb, ub",
    [
        (s, c, lb, ub)
        for s in SEEDS
        for c in COEFFICIENTS
        for lb in LOWER_BOUNDS
        for ub in UPPER_BOUNDS
    ],
)  # Add more seeds as needed
def test_clipped_sine(seed: int, C: float, lb: float, ub: float):
    """Test if the HardConstrainedMLP fits max(sin(x), Ax + lb).

    The training objective is to fit the sine function with a MLP, but the
    hard constraint is that the predictions must satisfy the inequality
    lb <= A sin(x) <= ub.

    Args:
        seed (int): Random seed for reproducibility.
        C (float): Coefficient of the affine inequality constraint.
        lb (float): Lower bound of the affine inequality constraint.
        ub (float): Upper bound of the affine inequality constraint.
    """
    # Test params
    N_SAMPLES = 1000
    LEARNING_RATE = 1e-3
    N_EPOCHS = 5000
    PLOT_RESULTS = False

    # Generate dataset
    x = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, N_SAMPLES).reshape(-1, 1)
    y = jnp.sin(x)
    lower_bound = C * x + lb
    upper_bound = C * x + ub

    # Define the affine inequality constraint
    ineq_constraint = AffineInequalityConstraint(
        C=jnp.array([1]).reshape((1, 1, 1)),
        lb=lower_bound.reshape((N_SAMPLES, x.shape[1], 1)),
        ub=upper_bound.reshape((N_SAMPLES, x.shape[1], 1)),
    )
    projection_layer = Project(ineq_constraint=ineq_constraint, unroll=False)
    # Define and initialize the hard constrained MLP
    model = HardConstrainedMLP(project=projection_layer)
    params = model.init(jax.random.PRNGKey(seed), x, 0)
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
    predictions = state.apply_fn({"params": state.params}, x, 10000)

    # Clip y to meet the constraints via cvxpy
    projected_y = []
    for _x, _y in zip(x, y):
        # Create a scalar optimization variable.
        z = cp.Variable()

        # Formulate the objective: minimize (z - y)^2.
        objective = cp.Minimize((z - _y) ** 2)

        # Define the constraints: z must lie in [A*x + lb, A*x + ub].
        constraints = [z >= C * _x + lb, z <= C * _x + ub]

        # Solve the problem.
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Append the optimal value.
        projected_y.append(z.value)

    projected_y = jnp.array(projected_y).reshape(-1, 1)

    # Plot dataset and print extra results
    if PLOT_RESULTS:
        from matplotlib import pyplot as plt

        # Plot the original function, its projection, and the constraint bounds.
        plt.figure(figsize=(8, 5))
        # Compute and plot the lower and upper bounds.
        upper_bound = C * x + ub  # 0.5*x + 0.5
        plt.plot(x, lower_bound, linestyle="--", label="Lower Bound")
        plt.plot(x, upper_bound, linestyle="--", label="Upper Bound")
        # Plot the original and projected functions.
        plt.plot(x, y, label=r"Original $\sin(x)$", linestyle="dashed", color="blue")
        plt.plot(x, projected_y, label=r"Projected $\sin(x)$", color="red")
        plt.plot(x, predictions, label=r"Predicted $\sin(x)$", color="green")

        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.legend()
        plt.title(r"Projection of $\sin(x)$ onto Constraint Bounds")
        plt.show()

    # Check if predictions meet the condition
    assert jnp.allclose(
        predictions, projected_y, atol=1e-1
    ), "The MLP predictions do not meet the clipping condition."
