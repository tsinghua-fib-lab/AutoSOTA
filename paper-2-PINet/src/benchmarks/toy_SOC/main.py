"""This module implements the solver for random second-order cone parametric problems."""

# %% Imports
import cvxpy as cp
import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax import config as jconf
from jax import custom_vjp as _custom_vjp
from jax import jit, lax
from jax import numpy as jnp
from jax import random as jrnd
from jax import value_and_grad, vjp
from jax.scipy.sparse.linalg import bicgstab

# Use 64 bit precision for numerical stability
jconf.update("jax_enable_x64", True)

# %%
# Problem dimensions
n = 250
m = 250
# Key
key = jrnd.PRNGKey(1)

use_custom_vjp = True
if use_custom_vjp:
    custom_vjp = _custom_vjp
else:

    def custom_vjp(f):
        """No op."""
        return f


# %% Projections
def _project_soc(z: jnp.ndarray) -> jnp.ndarray:
    """Project onto the second-order cone (SOC) constraint.

    Args:
        z (jnp.ndarray):
            Input array of shape (B, m + 1, 1) where the last column is the SOC radius.

    Returns:
        jnp.ndarray:
            Projected array of the same shape as `z`, satisfying the SOC constraint.
    """
    eps = 1e-12
    u, t = z[:, :-1], z[:, -1:]
    norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)

    proj1 = z
    proj2 = jnp.zeros_like(z)
    proj3 = (
        (t + norm_u)
        / 2
        * jnp.concatenate((u / (norm_u + eps), jnp.ones_like(t)), axis=1)
    )

    when1 = norm_u <= t
    when2 = norm_u <= -t

    return jnp.where(when1, proj1, jnp.where(when2, proj2, proj3))


project_soc = jit(_project_soc)


# %% Generate random data
def rand_sparse_mask(
    key: jrnd.PRNGKey,
    shape: tuple,
    sparsity: float = 0.01,
    dtype: jnp.dtype = jnp.float64,
):
    """Return a dense tensor whose entries are 0 with prob = `sparsity`.

    Args:
        key (jax.random.PRNGKey): Random key for generating the tensor.
        shape (tuple): Shape of the tensor to be generated.
        sparsity (float): Probability of an entry being zero. Default is 0.01.
        dtype (jnp.dtype): Data type of the tensor. Default is jnp.float64

    Returns:
        jnp.ndarray:
            A tensor of the specified shape with random values and a mask applied.
    """
    key_val, key_mask = jrnd.split(key)

    # Non-zero density is 1 − sparsity
    density = 1.0 - sparsity

    values = jrnd.uniform(key_val, shape, dtype, minval=-1, maxval=1)
    mask = jrnd.bernoulli(key_mask, p=density, shape=shape)
    return values * mask.astype(dtype)


keyA, key = jrnd.split(key)
A = rand_sparse_mask(keyA, (m, n))


def generate_problem(key: jrnd.PRNGKey, B: int):
    """Generate a random linear problem with SOC constraints.

    Args:
        key (jax.random.PRNGKey): Random key for generating the problem.
        B (int): Number of problem instances to generate.

    Returns:
        tuple:
            - b (jnp.ndarray):
                Right-hand side of the equality constraints, shape (B, m, 1).
            - c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
            - x (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
            - s (jnp.ndarray):
                Optimal dual solution satisfying the SOC constraint, shape (B, m, 1).
    """
    keyz, keyx = jrnd.split(key)
    z = jrnd.uniform(keyz, (B, m, 1), minval=-1, maxval=1)
    s = project_soc(z)
    y = s - z

    # Generate the primal solution x
    x = jrnd.uniform(keyx, (B, n, 1), minval=-1, maxval=1)
    b = A @ x + s
    c = -A.T @ y

    return b, c, x, s


def objective(x: jnp.ndarray, c: jnp.ndarray):
    """Compute the objective value for the linear problem.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).

    Returns:
        jnp.ndarray: Objective value, shape (B, 1).
    """
    return jnp.sum(c * x, axis=(1, 2), keepdims=True)


# %% CV and RS
def constraint_violation_eq(x: jnp.ndarray, s: jnp.ndarray, b: jnp.ndarray):
    """Compute the constraint violation for Ax = b.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        s (jnp.ndarray): Dual solution, shape (B, m, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jnp.ndarray: Constraint violation, shape (B, 1).
    """
    return jnp.linalg.norm(A @ x + s - b, ord=jnp.inf, axis=-1)


def constraint_violation_soc(s: jnp.ndarray):
    """Compute the constraint violation for the SOC constraint.

    Args:
        s (jnp.ndarray): Dual solution, shape (B, m + 1, 1).

    Returns:
        jnp.ndarray: Constraint violation, shape (B, 1).
    """
    u = s[:, :-1]
    t = s[:, -1:]
    u_norm = jnp.linalg.norm(u, axis=1, keepdims=True)

    return jnp.maximum(u_norm - t, 0.0)


def relative_suboptimality(x: jnp.ndarray, xstar: jnp.ndarray, c: jnp.ndarray):
    """Compute the relative suboptimality of the solution.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).

    Returns:
        jnp.ndarray: Relative suboptimality, shape (B, 1).
    """
    optimal_val = objective(xstar, c)
    candidate_val = objective(x, c)
    return jnp.abs(candidate_val - optimal_val) / (jnp.abs(optimal_val) + 1e-12)


def print_stats(
    x: jnp.ndarray, s: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, xstar: jnp.ndarray
):
    """Print the statistics of the solution.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        s (jnp.ndarray): Dual solution, shape (B, m, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
    """
    cv_eq = constraint_violation_eq(x, s, b)
    cv_soc = constraint_violation_soc(s)
    rs = relative_suboptimality(x, xstar, c)

    print("=========== Solution statistics ===========")
    # mean, std, max, min
    print(
        f"""CV (Ax = b): {jnp.mean(cv_eq):.15f} ± {jnp.std(cv_eq):.15f}
          in [{jnp.min(cv_eq):.15f}, {jnp.max(cv_eq):.15f}]"""
    )
    print(
        f"""CV (SOC): {jnp.mean(cv_soc):.15f} ± {jnp.std(cv_soc):.15f}
          in [{jnp.min(cv_soc):.15f}, {jnp.max(cv_soc):.15f}]"""
    )
    print(
        f"""RS: {jnp.mean(rs):.15f} ± {jnp.std(rs):.15f}
          in [{jnp.min(rs):.15f}, {jnp.max(rs):.15f}]"""
    )


# %% Validate the problem
B = 1024
# Symbolic problem
b, c, xstar, sstar = generate_problem(key, B)

# %% CVXPY
A_np = np.asarray(A)
x_var = cp.Variable(n)
s_var = cp.Variable(m)
b_par = cp.Parameter(m)
c_par = cp.Parameter(n)

constraints = [A_np @ x_var + s_var == b_par, cp.SOC(s_var[-1], s_var[:-1])]

problem = cp.Problem(cp.Minimize(c_par @ x_var), constraints)
x_sol = []
s_sol = []
for i in range(B):
    b_par.value = np.asarray(b[i]).ravel()  # shape (m,)
    c_par.value = np.asarray(c[i]).ravel()  # shape (n,)

    problem.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)

    if x_var.value is None:
        raise RuntimeError(f"sample {i}: {problem.status}")
    x_sol.append(x_var.value.reshape(n, 1))
    s_sol.append(s_var.value.reshape(m, 1))

x_cvxpy = jnp.asarray(x_sol).reshape(B, n, 1)
s_cvxpy = jnp.asarray(s_sol).reshape(B, m, 1)

# Print the statistics of the solution
print_stats(x_cvxpy, s_cvxpy, b, c, xstar)

# %% Use our solver
n_iter_forward = 1e3
n_iter_backward = 200
sigma = 0.1
omega = 1.8

Aaug = jnp.concatenate((A, jnp.eye(m)), axis=1)
assert Aaug.shape == (
    m,
    m + n,
), f"Augmented matrix A should have shape ({m}, {m + n}), instead: {Aaug.shape}"

Aaug_inv = jnp.linalg.pinv(Aaug)


def project_pinv_vb(xs: jnp.ndarray, b: jnp.ndarray):
    """Project onto the pseudo-inverse of the augmented matrix A.

    Args:
        xs (jnp.ndarray):
            Input array of shape (B, m + n, 1) where the first n columns
            are the primal variables and the last m columns are residuals.

        b (jnp.ndarray):
            Right-hand side of the equality constraints, shape (B, m, 1).
    """
    return xs - Aaug_inv @ (Aaug @ xs - b)


def step_iteration(yraw: jnp.ndarray, sk: jnp.ndarray, b: jnp.ndarray):
    """Perform one iteration of the forward step.

    Args:
        yraw (jnp.ndarray):
            Raw input array of shape (B, m + n, 1) where the first n columns
            are the primal variables and the last m columns are residuals.

        sk (jnp.ndarray):
            Governing sequence, shape (B, m + n, 1).

        b (jnp.ndarray):
            Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jnp.ndarray: Updated governing sequence, shape (B, m + n, 1).
    """
    zk = project_pinv_vb(sk, b)
    reflect = 2 * zk - sk
    toproj = (reflect - 2 * sigma * yraw) / (1 + 2 * sigma)
    tk1 = toproj[:, :n]
    tk2 = project_soc(toproj[:, n:])
    tk = jnp.concatenate((tk1, tk2), axis=1)
    return sk + omega * (tk - zk)


def step_final(s, b):
    """Retrieve the final result from the forward step.

    Args:
        s (jnp.ndarray): Governing sequence, shape (B, m + n, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jnp.ndarray: Final projected value, shape (B, m + n, 1).
    """
    return project_pinv_vb(s, b)


@custom_vjp
def project(
    s0: jnp.ndarray,
    yraw: jnp.ndarray,
    b: jnp.ndarray,
):
    """Project the raw input onto the feasible set defined by the constraints.

    Args:
        s0 (jnp.ndarray): Initial governing sequence, shape (B, m + n, 1).
        yraw (jnp.ndarray): Raw input array, shape (B, m + n, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        tuple:
            - zk1 (jnp.ndarray): Projected value, shape (B, m + n, 1).
            - sk (jnp.ndarray): Final governing sequence, shape (B, m + n, 1).
    """
    sk = s0
    sk, _ = lax.scan(
        lambda sk, _: (
            step_iteration(yraw.reshape((yraw.shape[0], yraw.shape[1], 1)), sk, b),
            None,
        ),
        sk,
        xs=None,
        length=n_iter_forward,
    )

    # NOTE: There is no auxiliary variable in this case
    zk1 = step_final(sk, b).reshape(yraw.shape)

    # return values and residuals
    return zk1, sk


def _project_fwd(s0: jnp.ndarray, yraw: jnp.ndarray, b: jnp.ndarray):
    """Forward pass of the projection function.

    Args:
        s0 (jnp.ndarray): Initial governing sequence, shape (B, m + n, 1).
        yraw (jnp.ndarray): Raw input array, shape (B, m + n, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        - tuple:
            - zk1 (jnp.ndarray): Projected value, shape (B, m + n, 1).
            - sk (jnp.ndarray): Final governing sequence, shape (B, m + n, 1).
        - tuple:
            - (sk, yraw, b): Residuals for the backward pass.
    """
    zk1, sk = project(s0, yraw, b)
    return (zk1, sk), (sk, yraw.reshape((yraw.shape[0], yraw.shape[1], 1)), b)


def _project_bwd(residuals: tuple, cotangent: tuple):
    """Backward pass of the projection function.

    Args:
        residuals (tuple): Residuals from the forward pass, containing:
            - sk (jnp.ndarray): Governing sequence, shape (B, m + n, 1).
            - yraw (jnp.ndarray): Raw input array, shape (B, m + n, 1).
            - b (jnp.ndarray):
                Right-hand side of the equality constraints, shape (B, m, 1).

        cotangent (tuple): Cotangent vector from the backward pass, containing:
            - cotangent_zk1 (jnp.ndarray):
                Cotangent vector for the projected value, shape (B, m + n, 1).
            - cotangent_sk (jnp.ndarray):
                Cotangent vector for the governing sequence, shape (B, m + n, 1).

    Returns:
        tuple:
            - None: Placeholder for the vjp wrt to sk (the DRA governing sequence).
            - thevjp (jnp.ndarray):
                The vjp wrt to yraw, shape (B, m + n, 1).
            - None: Placeholder for the vjp wrt to b
                (the right-hand side of the equality constraints).
    """
    sk, yraw, b = residuals
    cotangent_zk1, _ = cotangent

    # Compute the vjp of the iteration step wrt to the DRA governing sequence
    _, iteration_vjp = vjp(lambda xx: step_iteration(yraw, xx, b), sk)
    # Compute the vjp of the iteration step wrt to the value to be projected
    _, iteration_vjp2 = vjp(lambda xx: step_iteration(xx, sk, b), yraw)
    # Compute the vjp of the final step wrt to DRA governing sequence
    _, equality_vjp = vjp(lambda xx: step_final(xx, b), sk)

    cotangent_eq_6 = equality_vjp(cotangent_zk1)[0]

    def Aop(xx):
        return xx - iteration_vjp(xx)[0]

    cotangent_eq_7 = bicgstab(Aop, cotangent_eq_6, maxiter=n_iter_backward)[0]

    thevjp = iteration_vjp2(cotangent_eq_7)[0]

    # We only care about the vjp wrt to yraw
    # So, we return None for the vjp wrt to sk (the DRA governing sequence)
    # and None for the vjp wrt to b (the right-hand side of the equality constraints)
    return (None, thevjp, None)


if use_custom_vjp:
    project.defvjp(_project_fwd, _project_bwd)

# %% Test the projection
# To test the correctness of the projection, we can sample random points,
# project them, and check if the result has no constraint violation.


def test_projection(
    b: jnp.ndarray, c: jnp.ndarray, xstar: jnp.ndarray, sstar: jnp.ndarray
):
    """Test the projection function on random samples.

    Args:
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
        sstar (jnp.ndarray): Optimal dual solution, shape (B, m, 1).
    """
    n_samples = b.shape[0]
    yraw = jrnd.uniform(key, (n_samples, n + m, 1))

    x = yraw[:, :n]
    s = yraw[:, n:]
    cv_eq_raw = constraint_violation_eq(x, s, b)
    cv_soc_raw = constraint_violation_soc(s)
    if jnp.all(cv_eq_raw < 1e-6) and jnp.all(cv_soc_raw < 1e-6):
        print(f"Sample {i}: No constraint violation in the raw samples.")

    cv_eq_opt = constraint_violation_eq(xstar, sstar, b)
    cv_soc_opt = constraint_violation_soc(sstar)
    if jnp.any(cv_eq_opt > 1e-6) or jnp.any(cv_soc_opt > 1e-6):
        print(f"Optimal sample: {cv_eq_opt.max()=}, {cv_soc_opt.max()=}")

    # Project the point
    y, _ = project(jnp.zeros_like(yraw), yraw, b)
    x = y[:, :n]
    s = y[:, n:]

    # Check the constraint violation
    cv_eq = constraint_violation_eq(x, s, b)
    cv_eq_soc = constraint_violation_soc(s)
    if (
        jnp.any(cv_eq > 1e-6)
        or jnp.any(cv_eq_soc > 1e-6)
        or jnp.isnan(cv_eq).any()
        or jnp.isnan(cv_eq_soc).any()
    ):
        print(f"Projection failed: {cv_eq.max()=}, {cv_eq_soc.max()=}")
        print_stats(x, s, b, c, xstar.reshape(-1, n, 1))
    print("All projections passed.")


test_projection(b, c, xstar, sstar)


# %% Simple MLP
class HardConstrainedMLP(nn.Module):
    """A simple MLP model for solving the hard constrained problem."""

    layers: list[int]

    @nn.compact
    def __call__(
        self,
        input: dict[str, jnp.ndarray],
    ):
        """Call the NN.

        Args:
            input (dict):
                Dictionary containing the input data with keys "b" and "c".

        Returns:
            jnp.ndarray:
                Output of the MLP, projected onto the feasible set.
        """
        b, c = input["b"].squeeze(-1), input["c"].squeeze(-1)
        x = jnp.concatenate((b, c), axis=-1)
        for layer_size in self.layers:
            x = nn.relu(nn.Dense(layer_size)(x))
        # Final layer to project
        x = nn.Dense(n + m)(x).reshape((x.shape[0], n + m, 1))
        x = project(jnp.zeros_like(x), x, b.reshape((b.shape[0], -1, 1)))[0]
        return x


# %% Train the MLP
BATCH_SIZE = 512
N_EPOCHS = 1000
LEARNING_RATE = 1e-3
key_train, key_init = jrnd.split(key)


# Batcher
def make_batch(key: jax.random.PRNGKey, batch_size: int = BATCH_SIZE):
    """Generate a batch of random problems.

    Args:
        key (jax.random.PRNGKey): Random key for generating the batch.
        batch_size (int): Number of problem instances in the batch.

    Returns:
        tuple:
            - dict: Input data containing "b" and "c".
            - jnp.ndarray: Optimal primal solution, shape (B, n, 1).
            - jnp.ndarray: Optimal dual solution, shape (B, m, 1).
    """
    key_prob, key = jrnd.split(key)
    b, c, xstar, sstar = generate_problem(key_prob, batch_size)
    return {
        "input": {"b": b, "c": c},
        "xstar": xstar,
        "sstar": sstar,
    }, key


# %% Initialize the model
model = HardConstrainedMLP(layers=[200, 200])

# Sample one batch only to create shapes for initialisation
batch, key = make_batch(key_init, batch_size=1)
key, key_init = jrnd.split(key)
params = model.init(key_init, batch["input"])

tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# %% Training
@jit
def loss_fn(params: dict, input: dict):
    """Compute the loss function and auxiliary values.

    Args:
        params (dict): Model parameters.
        input (dict): Input data containing "b" and "c".

    Returns:
        tuple:
            - loss (jnp.ndarray): Mean objective value.
            - aux (tuple): Auxiliary values containing constraint violations.
    """
    c = input["c"]
    pred = model.apply(params, input)
    x = pred[:, :n]
    s = pred[:, n:]
    objective_value = objective(x, c)
    return jnp.mean(objective_value), (x, s)


@jit
def train_step(state: train_state.TrainState, batch: dict):
    """Perform a single training step.

    Args:
        state (TrainState): Current state of the model.
        batch (dict): Input data containing "b" and "c".

    Returns:
        tuple:
            - state (TrainState): Updated state of the model.
            - loss (jnp.ndarray): Loss value after the step.
            - aux (tuple): Auxiliary values containing constraint violations.
    """
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, batch["input"])
    state = state.apply_gradients(grads=grads)
    return state, loss, aux


# Training loop
for epoch in range(1, N_EPOCHS + 1):
    epoch_losses = []
    key_train, key = jrnd.split(key_train)
    batch, key_train = make_batch(key)
    state, l, (x, s) = train_step(state, batch)
    cv_eq = constraint_violation_eq(x, s, batch["input"]["b"])
    cv_soc = constraint_violation_soc(s)
    rs = relative_suboptimality(x, batch["xstar"], batch["input"]["c"])
    if epoch % 10 == 0 or epoch == 1:
        print(
            f"""[{epoch:03d}/{N_EPOCHS}]
            \tloss = {l:.4e}
            \t{cv_eq.max()=}
            \t{cv_soc.max()=}
            \t{rs.max()=}
            """
        )

# %% Validation
key_test, key = jrnd.split(key_train)
val_batch, _ = make_batch(key_test, batch_size=BATCH_SIZE)
pred_val = model.apply(state.params, val_batch["input"])
b = val_batch["input"]["b"]

x_pred = pred_val[:, :n]
s_pred = pred_val[:, n:]

print_stats(
    x_pred, s_pred, val_batch["input"]["b"], val_batch["input"]["c"], val_batch["xstar"]
)

# %%
