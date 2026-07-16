import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# --- Configuration ---
N_grid = 100 
L = 1.0
dx = L / N_grid
dt = 0.001   # Time step
sigma = 0.05 # Actuator width


def forcing_fn_1d(xi, u, N):
    """
    Calculates the 1D Gaussian influence of actuators.
    xi: (M,) positions
    u:  (M,) intensities
    """
    x_coords = jnp.linspace(0, 1, N)
    def single_actuator(pos, intensity):
        dist_sq = (x_coords - pos)**2
        return intensity * jnp.exp(-dist_sq / (2 * sigma**2))
    
    forcings = jax.vmap(single_actuator)(xi, u)
    return jnp.sum(forcings, axis=0)

def solve_tridiagonal_diffusion(z_explicit, r, N):
    # Dirichlet (z=0 at boundaries)
    d = jnp.ones(N) * (1 + 2 * r)
    d = d.at[0].set(1.0)
    d = d.at[-1].set(1.0)

    ld = jnp.ones(N) * (-r)
    ld = ld.at[0].set(0.0)

    ud = jnp.ones(N) * (-r)
    ud = ud.at[-1].set(0.0)
    
    rhs_values = z_explicit.at[0].set(0.0).at[-1].set(0.0)
    rhs = rhs_values[:, jnp.newaxis]
    
    out = jax.lax.linalg.tridiagonal_solve(ld, d, ud, rhs)
    return out.ravel()

@jit
def fkpp_step_1d(z, xi, u, v, nu=0.005, rho=3.0):
    """
    Refactored to accept nu and rho as parameters.
    """
    N = z.shape[0]
    
    # 1. Reaction + Forcing (Explicit)
    f_t = forcing_fn_1d(xi, u, N) 
    # Use the passed 'rho'
    reaction = rho * z * (1.0 - z)
    z_explicit = z + dt * (reaction + f_t)

    # 2. Diffusion (Implicit)
    # Use the passed 'nu'
    r = nu * dt / (dx**2)
    z_next = solve_tridiagonal_diffusion(z_explicit, r, N)
    
    # 3. Updates & Clipping
    z_next = jnp.clip(z_next, 0.0, 1.0)
    xi_next = jnp.clip(xi + dt * v, 0.0, 1.0)

    return z_next, xi_next

@partial(jax.jit, static_argnums=(4, 5))
def solve_with_policy(z_init, xi_init, z_target, params, policy_apply_fn, t_steps, key, 
                      nu=0.005, rho=3.0,       
                      noise_u=0.0, noise_z=0.0):
    """
    FKPP Loop with Noise Injection and dynamic Physics Parameters.
    """
    def step_fn(carry, _):
        z_curr, xi_curr, current_key = carry
        
        # Split keys for this step
        k_sensor, k_actuator, next_key = jax.random.split(current_key, 3)

        # 1. Add Sensor Noise (What the policy sees)
        z_observed = z_curr + noise_z * jax.random.normal(k_sensor, z_curr.shape)

        # 2. Policy Inference (using observed state)
        u, v = policy_apply_fn(params, z_observed, z_target, xi_curr)

        # 3. Add Actuator Noise (What the physics gets)
        u_noisy = u + noise_u * jax.random.normal(k_actuator, u.shape)
        
        # 4. FKPP Physics Step (Passing nu and rho down)
        z_next, xi_next = fkpp_step_1d(z_curr, xi_curr, u_noisy, v, nu, rho)
        
        return (z_next, xi_next, next_key), (z_next, xi_next, u_noisy, v)

    _, trajectory = jax.lax.scan(
        step_fn, 
        (z_init, xi_init, key),
        None, 
        length=t_steps
    )
    
    return trajectory