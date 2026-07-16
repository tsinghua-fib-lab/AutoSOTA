import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Configuration (can be updated via set_grid_size())
N = 32  # Default grid size (changed from 64 to 32 for faster training)
dx = 1.0 / N
dy = 1.0 / N
x_grid = jnp.linspace(0, 1, N)
y_grid = jnp.linspace(0, 1, N)
xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')

nu = 0.2    # Diffusion coefficient
sigma = 0.15  # Width of the actuator influence (slightly larger for 2D)
fixed_dt = 0.001

def set_grid_size(n_grid):
    """
    Reconfigure the grid resolution.

    Call this before using the solver if you want a different grid size.

    Args:
        n_grid: Number of grid points per dimension (e.g., 32 or 64)
    """
    global N, dx, dy, x_grid, y_grid, xx, yy
    N = n_grid
    dx = 1.0 / N
    dy = 1.0 / N
    x_grid = jnp.linspace(0, 1, N)
    y_grid = jnp.linspace(0, 1, N)
    xx, yy = jnp.meshgrid(x_grid, y_grid, indexing='ij')

def build_implicit_matrix_1d(N, r):
    """
    Builds the LHS matrix for Crank-Nicolson: (I + r/2 * L)
    where L is the Laplacian operator in 1D.
    This is reused from the 1D heat solver.
    """
    main_diag = jnp.ones(N) * (1 + r)
    off_diag = jnp.ones(N - 1) * (-r / 2.0)

    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

    # Dirichlet Boundary Conditions (zero at ends)
    A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
    A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)
    return A

def forcing_fn(xx, yy, xi, u):
    """
    2D Gaussian actuator forcing.

    Args:
        xx, yy: Meshgrid arrays (N, N)
        xi: Actuator positions (M, 2) where xi[:, 0] = x, xi[:, 1] = y
        u: Forcing intensities (M,) - scalar control per actuator

    Returns:
        B: Forcing field (N, N)
    """
    # Compute squared distance: (M, N, N)
    dx_sq = (xx[None, :, :] - xi[:, 0, None, None])**2
    dy_sq = (yy[None, :, :] - xi[:, 1, None, None])**2
    sq_dist = dx_sq + dy_sq

    # Gaussian basis
    basis = jnp.exp(-sq_dist / (2 * sigma**2))

    # Sum contributions: (M,) * (M, N, N) -> (N, N)
    B = jnp.sum(u[:, None, None] * basis, axis=0)
    return B

@jit
def adi_step(z, xi, u, v):
    """
    ADI Crank-Nicolson step for 2D heat equation.

    Uses Alternating Direction Implicit (ADI) method - Peaceman-Rachford scheme:
    - Step 1: Implicit in x, explicit in y
    - Step 2: Implicit in y, explicit in x

    Args:
        z: State (N, N)
        xi: Actuator positions (M, 2)
        u: Forcing intensities (M,)
        v: Actuator velocities (M, 2)

    Returns:
        z_next: Updated state (N, N)
        xi_next: Updated positions (M, 2)
    """
    dt = fixed_dt
    r = nu * dt / (dx**2)

    # Precompute forcing
    B = forcing_fn(xx, yy, xi, u)

    # Build 1D tridiagonal matrix (reused for both directions)
    A = build_implicit_matrix_1d(N, r)

    # --- STEP 1: Implicit in X, Explicit in Y ---
    # For each row j (y-direction), solve tridiagonal system in x-direction

    def solve_x_row(j):
        """Solve implicit x-direction for row j."""
        # Boundary rows - Dirichlet BC means z=0
        def boundary_case():
            return jnp.zeros(N)

        def interior_case():
            z_row = z[:, j]  # Current row

            # Explicit y-diffusion contribution
            # d²z/dy² ≈ (z[:, j-1] - 2*z[:, j] + z[:, j+1]) / dy²
            z_left = z[:, j-1]
            z_right = z[:, j+1]

            # Explicit part: (I + r/2 * L_y) z + dt/2 * B
            diff_y = (r / 2.0) * (z_left - 2*z_row + z_right)

            rhs = z_row + diff_y + (dt / 2.0) * B[:, j]

            # Enforce boundary conditions in RHS
            rhs = rhs.at[0].set(0.0)
            rhs = rhs.at[-1].set(0.0)

            # Solve tridiagonal system
            return jnp.linalg.solve(A, rhs)

        return jax.lax.cond(
            (j == 0) | (j == N-1),
            boundary_case,
            interior_case
        )

    z_star_rows = jax.vmap(solve_x_row)(jnp.arange(N))  # (N, N)
    z_star = z_star_rows.T  # Transpose to correct orientation

    # --- STEP 2: Implicit in Y, Explicit in X ---
    # For each column i (x-direction), solve tridiagonal system in y-direction

    def solve_y_col(i):
        """Solve implicit y-direction for column i."""
        def boundary_case():
            return jnp.zeros(N)

        def interior_case():
            z_col = z_star[i, :]  # Current column

            z_down = z_star[i-1, :]
            z_up = z_star[i+1, :]

            # Explicit part in x
            diff_x = (r / 2.0) * (z_down - 2*z_col + z_up)

            rhs = z_col + diff_x + (dt / 2.0) * B[i, :]

            rhs = rhs.at[0].set(0.0)
            rhs = rhs.at[-1].set(0.0)

            return jnp.linalg.solve(A, rhs)

        return jax.lax.cond(
            (i == 0) | (i == N-1),
            boundary_case,
            interior_case
        )

    z_next_cols = jax.vmap(solve_y_col)(jnp.arange(N))  # (N, N)
    z_next = z_next_cols  # Already correct orientation

    # Update actuator positions
    xi_next = xi + dt * v
    xi_next = jnp.clip(xi_next, 0.0, 1.0)

    return z_next, xi_next

@partial(jax.jit, static_argnums=(4, 5))
def solve_with_policy(z_init, xi_init, z_target, params, policy_apply_fn, t_steps):
    """
    Main solver loop with policy integration.

    Args:
        z_init: Initial state (N, N)
        xi_init: Initial positions (M, 2)
        z_target: Target state (N, N)
        params: Neural network parameters (PyTree)
        policy_apply_fn: Policy function
        t_steps: Number of timesteps

    Returns:
        Tuple of (z_traj, xi_traj, u_traj, v_traj)
    """
    def step_fn(carry, _):
        z_curr, xi_curr = carry

        # Policy inference
        u, v = policy_apply_fn(params, z_curr, z_target, xi_curr)

        # Physics step
        z_next, xi_next = adi_step(z_curr, xi_curr, u, v)

        return (z_next, xi_next), (z_next, xi_next, u, v)

    _, trajectory = jax.lax.scan(
        step_fn,
        (z_init, xi_init),
        None,
        length=t_steps
    )

    return trajectory