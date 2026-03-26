"""Generate simple QP problem.

Using the approach from the DC3 paper:
https://arxiv.org/pdf/2104.12225
"""

import os

import cvxpy as cp
import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


def solve_problems(
    Q: jnp.ndarray,
    p: jnp.ndarray,
    A: jnp.ndarray,
    X: jnp.ndarray,
    G: jnp.ndarray,
    h: jnp.ndarray,
    convex: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the optimal solutions for problem instances.

    Args:
        Q (jnp.ndarray): Quadratic term of the objective function.
        p (jnp.ndarray): Linear term of the objective function.
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        X (jnp.ndarray): Input data for equality constraints.
        G (jnp.ndarray): Coefficient matrix for inequality constraints.
        h (jnp.ndarray): Right-hand side of the inequality constraints.
        convex (bool): Whether the problem is convex or not.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            A tuple containing the objectives and optimal solutions.
    """
    # Dimension of decision variable
    Y_DIM = Q.shape[2]
    N_SAMPLES = X.shape[0]
    if convex:
        print("Solving problem instances")
        objectives = jnp.zeros(N_SAMPLES)
        Ystar = jnp.zeros((N_SAMPLES, Y_DIM, 1))
        for problem_idx in tqdm(range(N_SAMPLES)):
            ycp = cp.Variable(Y_DIM)
            constraints = [
                A[0, :, :] @ ycp == X[problem_idx, :, 0],
                G[0, :, :] @ ycp <= h[0, :, 0],
            ]
            objective = cp.Minimize(0.5 * cp.quad_form(ycp, Q[0, :, :]) + p.T @ ycp)
            problem = cp.Problem(objective=objective, constraints=constraints)
            problem.solve(solver=cp.OSQP)
            objectives = objectives.at[problem_idx].set(problem.value)
            Ystar = Ystar.at[problem_idx, :, :].set(jnp.expand_dims(ycp.value, axis=1))
    else:
        raise NotImplementedError

    return objectives, Ystar


if __name__ == "__main__":
    # Save dataset flag
    SAVE_DATASET = True

    # Parameters setup
    SEED = 42
    NUM_VAR = 100
    NUM_INEQ = 50
    NUM_EQ = 50
    NUM_EXAMPLES = 200
    CONVEX = True

    # Setup keys
    key = jax.random.PRNGKey(SEED)
    key = jax.random.split(key, 5)

    # Generate matrices
    Q = jnp.expand_dims(
        jnp.diag(jax.random.uniform(key[0], shape=(NUM_VAR,), minval=0.0, maxval=1.0)),
        axis=0,
    )
    p = jax.random.uniform(key[1], shape=(1, NUM_VAR, 1), minval=0.0, maxval=1.0)
    A = jax.random.normal(key[2], shape=(1, NUM_EQ, NUM_VAR))
    X = jax.random.uniform(
        key[3], shape=(NUM_EXAMPLES, NUM_EQ, 1), minval=-1.0, maxval=1.0
    )
    G = jax.random.normal(key[4], shape=(1, NUM_INEQ, NUM_VAR))
    h = jnp.expand_dims(jnp.sum(jnp.abs(G @ jnp.linalg.pinv(A[0])), axis=1), axis=2)

    if SAVE_DATASET:
        # Create the datasets directory if it doesn't exist
        datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")

        # Solve all the problem instances
        objectives, Ystar = solve_problems(Q, p[0, :, :], A, X, G, h, CONVEX)

        # Define the filename and save the dataset
        if CONVEX:
            filename = os.path.join(
                datasets_dir,
                (
                    f"SimpleQP_seed{SEED}_var{NUM_VAR}_ineq{NUM_INEQ}"
                    f"_eq{NUM_EQ}_examples{NUM_EXAMPLES}.npz"
                ),
            )
        else:
            filename = os.path.join(
                datasets_dir,
                (
                    f"Simple_nonconvex_seed{SEED}_var{NUM_VAR}_ineq{NUM_INEQ}"
                    f"_eq{NUM_EQ}_examples{NUM_EXAMPLES}.npz"
                ),
            )
        jnp.savez(
            filename, Q=Q, p=p, A=A, X=X, G=G, h=h, objectives=objectives, Ystar=Ystar
        )
