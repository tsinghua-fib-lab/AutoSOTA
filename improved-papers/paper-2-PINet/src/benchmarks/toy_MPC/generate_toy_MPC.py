"""Generate toy MPC problem data."""

import pathlib

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# Enable saving the dataset
SAVE_RESULTS = True
# Define parameters
# We are planning in 2D
base_dim = 2
# Horizon
T = 20
# State and input dimension
dimx = base_dim * (T + 1)
dimu = base_dim * T
# Objective tradeoff
alpha = 1.0
# One instance of initial point
x0 = jnp.array([0, 4])
# Target point
xhat = jnp.array([3, -12])
# The decision variable looks like
# [x_0, x_1, ..., x_T, u_0, u_1, ..., u_{T-1}] = [x; u]
# Setup equality constraints
# They look like [A_x, A_u] [x; u] == [x_0; 0]
A_x = jnp.diag(jnp.ones(T + 1)) - jnp.diag(jnp.ones(T), k=-1)
A_x = jnp.kron(A_x, jnp.eye(base_dim))
A_u = jnp.diag(jnp.ones(T), k=-1)[:, :-1]
A_u = jnp.kron(A_u, jnp.eye(base_dim))
A = jnp.concatenate((A_x, A_u), axis=1)
# Setup Inequality constraints
# -boundx <= x <= boundx
# -boundu <= u <= boundu
boundx = 10
boundu = 1
lbx = -boundx * jnp.ones((dimx, 1))
ubx = boundx * jnp.ones((dimx, 1))
lbu = -boundu * jnp.ones((dimu, 1))
ubu = boundu * jnp.ones((dimu, 1))
# Solve one instance of the MPC problem
z = cp.Variable(dimx + dimu)
xinit = cp.Parameter(base_dim)
constraints = [
    A @ z == cp.hstack([xinit, jnp.zeros(dimx - base_dim)]),
    z[:dimx] >= lbx,
    z[:dimx] <= ubx,
    z[dimx:] >= lbu,
    z[dimx:] <= ubu,
]
objective = cp.Minimize(
    cp.sum_squares(z[:dimx] - jnp.tile(xhat, T + 1)) + alpha * cp.sum_squares(z[dimx:])
)
problem = cp.Problem(objective, constraints)
# Setup problem parameter
xinit.value = np.array(x0)
problem.solve()

# Solve many problems
SEED = 42
NUM_EXAMPLES = 10000
# Randomly generate dataset initial conditions
key = jax.random.PRNGKey(SEED)
x0set = jax.random.uniform(
    key, shape=(NUM_EXAMPLES, base_dim), minval=-boundx, maxval=boundx
)
objectives = jnp.zeros(NUM_EXAMPLES)
Ystar = jnp.zeros((NUM_EXAMPLES, dimx + dimu))
print(f"Solving {NUM_EXAMPLES} problem instances")
for idx in tqdm(range(NUM_EXAMPLES)):
    xinit.value = np.array(x0set[idx])
    problem.solve()
    objectives = objectives.at[idx].set(problem.value)
    Ystar = Ystar.at[idx, :].set(jnp.array(z.value))
# Generate matrices with appropriate shape
As = A.reshape((1, A.shape[0], A.shape[1]))
lbxs = lbx.reshape(1, -1, 1)
ubxs = ubx.reshape(1, -1, 1)
lbus = lbu.reshape(1, -1, 1)
ubus = ubu.reshape(1, -1, 1)
x0sets = x0set.reshape((NUM_EXAMPLES, base_dim, 1))
xhat = xhat.reshape((base_dim, 1))
if SAVE_RESULTS:
    datasets_path = pathlib.Path(__file__).parent.resolve() / "datasets"
    filename = f"toy_MPC_seed{SEED}_examples{NUM_EXAMPLES}.npz"
    path = datasets_path / filename
    jnp.savez(
        path,
        As=As,
        lbxs=lbxs,
        ubxs=ubxs,
        lbus=lbus,
        ubus=ubus,
        x0sets=x0sets,
        xhat=xhat,
        objectives=objectives,
        Ystar=Ystar,
        T=T,
        base_dim=base_dim,
        alpha=alpha,
    )
