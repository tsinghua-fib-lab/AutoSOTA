import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Configuration (from first code)
N = 100
dx = 1.0 / N
x_grid = jnp.linspace(0, 1, N)
nu = 0.2    # Diffusion coefficient
sigma = 0.1  # Width of the actuator influence
fixed_dt = 0.001 

def build_implicit_matrix(N, r):
    """
    Builds the LHS matrix for Crank-Nicolson: (I - r/2 * L)
    where L is the Laplacian operator.
    """
    main_diag = jnp.ones(N) * (1 + r)
    off_diag = jnp.ones(N - 1) * (-r / 2.0)
    
    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    
    # Dirichlet Boundary Conditions (zero at ends)
    A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
    A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)
    return A

def forcing_fn(x_grid, xi, u):
    """Gaussian actuator forcing."""
    sq_dist = (x_grid[None, :] - xi[:, None])**2
    basis = jnp.exp(-sq_dist / (2 * sigma**2))
    return jnp.sum(u[:, None] * basis, axis=0)

@jit
def implicit_step(z, xi, u, v):
    """
    A semi-implicit step:
    1. Crank-Nicolson for the Diffusion (z)
    2. Explicit Euler for Actuator Position (xi)
    """
    dt = fixed_dt
    r = nu * dt / (dx**2)
    
    # 1. Precompute the Implicit Matrix
    A = build_implicit_matrix(N, r)
    
    # 2. Build the RHS (Explicit part of Crank-Nicolson + Forcing)
    z_interior = z[1:-1]
    z_left = z[:-2]
    z_right = z[2:]
    
    # Explicit diffusion part: (I + r/2 * L)z
    diff_explicit = (r / 2.0) * (z_left - 2 * z_interior + z_right)
    
    # Add Forcing (evaluated at current xi)
    f_t = forcing_fn(x_grid, xi, u)
    
    rhs_interior = z_interior + diff_explicit + dt * f_t[1:-1]
    
    rhs = jnp.zeros_like(z)
    rhs = rhs.at[1:-1].set(rhs_interior)
    
    # 3. Solve for z_next
    z_next = jnp.linalg.solve(A, rhs)
    
    # 4. Update xi (Explicitly, as it only depends on v). Then clipping for ensuring safety
    xi_next = xi + dt * v
    xi_next = jnp.clip(xi_next, 0.0, 1.0)
    
    return z_next, xi_next

@partial(jax.jit, static_argnums=(4, 5)) # 4 is policy_apply_fn, 5 is t_steps
def solve_with_policy(z_init, xi_init, z_target, params, policy_apply_fn, t_steps):
    """
    The core loop is now inside the solver.
    At each step: Policy -> Action -> PDE Step.
    """
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        
        # 1. Neural Network Inference (Inside the Solver Loop)
        u, v = policy_apply_fn(params, z_curr, z_target, xi_curr)
        
        # 2. Physics Step
        z_next, xi_next = implicit_step(z_curr, xi_curr, u, v)
        
        # Carry the state, and output the state + control for loss calculation
        return (z_next, xi_next), (z_next, xi_next, u, v)

    _, trajectory = jax.lax.scan(
        step_fn, 
        (z_init, xi_init), 
        None, 
        length=t_steps
    )
    
    return trajectory # (z_traj, xi_traj, u_traj, v_traj)