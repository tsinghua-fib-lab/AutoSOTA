"""
Data Utilities for 1D Kuramoto-Sivashinsky (KS)
Generates chaotic initial conditions by starting from random noise and 
evolving the system autonomously (no control) for a warm-up period.

Reference:
"Using a random initial condition, we let the system evolve autonomously 
for 100 time units before activating the agent." [Peitz et al., 2024]
"""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
from functools import partial
import tesseracts.ks1d.solver as solver

def generate_random_noise(key, N_grid, L, scale=0.01):
    """
    Generates initial random noise with zero mean (conservation of mass).
    """
    # Simple Gaussian noise
    u = jax.random.normal(key, shape=(N_grid,)) * scale
    
    # Enforce Zero Mean (Standard for KS to prevent drift)
    u = u - jnp.mean(u)
    
    # Enforce Periodic Boundaries (implicit in array, but good for smoothing)
    # The spectral solver handles this, but starting smooth helps stability.
    # Optional: Apply a slight smoothing kernel if N is very large, 
    # but pure spectral evolution usually handles white noise fine.
    return u

@partial(jax.jit, static_argnames=("N_grid", "L", "warmup_time", "dt"))
def evolve_to_attractor(key, N_grid, L, warmup_time=1000.0, dt=0.05):
    """
    Evolves random noise for 'warmup_time' to reach the chaotic attractor.
    Updated to handle dynamic operator generation.
    """
    # 1. Generate seed noise
    u_init = generate_random_noise(key, N_grid, L)
    u_hat = jnp.fft.rfft(u_init)
    
    # 2. Pre-calculate Operators (since they are now removed from global scope)
    dx = L / N_grid
    k = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
    L_linear = k**2 - k**4
    
    # Determine steps
    steps = int(warmup_time / dt)
    
    # Dummy inputs for the solver
    xi_dummy = jnp.zeros(1) 
    u_control_dummy = jnp.zeros(1)

    def warmup_step(carry, _):
        u_hat_curr, u_curr = carry
        
        # Apply solver step with explicit operators and parameters
        u_hat_next, u_next = solver.ks_spectral_step(
            u_hat_curr, 
            u_curr, 
            xi_dummy, 
            u_control_dummy,
            k=k,               # Required arg
            L_linear=L_linear, # Required arg
            N=N_grid,          # Ensure consistency
            L=L,               # Ensure consistency
            dt=dt
        )
        return (u_hat_next, u_next), None

    # print(steps)
    
    # Run the loop
    (u_hat_final, u_final), _ = jax.lax.scan(
        warmup_step,
        (u_hat, u_init),
        None,
        length=steps
    )
    
    return u_final

def get_batch_initial_conditions(key, batch_size, N_grid, L):
    """
    Generates a batch of fully developed chaotic states.
    """
    keys = jax.random.split(key, batch_size)
    
    # Vectorize the warm-up process
    # We map over keys to get different chaotic realizations
    batch_u = jax.vmap(evolve_to_attractor, in_axes=(0, None, None))(
        keys, N_grid, L
    )
    
    return batch_u

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Ensure project root is in sys.path
    script_dir = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.append(str(script_dir))

    
    import tesseracts.ks1d.solver as solver
    print("Generating KS Chaotic Attractor samples...")
    # Parameters from the paper for the "large" domain case
    L_paper = 200.0 
    N_paper = 256 # Higher resolution for L=200
    
    key = jax.random.PRNGKey(42)
    
    # Generate samples
    # This might take a moment to compile and run 100 time units
    u_samples = get_batch_initial_conditions(key, batch_size=3, N_grid=N_paper, L=L_paper)
    
    x_grid = jnp.linspace(0, L_paper, N_paper, endpoint=False)
    
    plt.figure(figsize=(12, 6))
    for i, u in enumerate(u_samples):
        plt.plot(x_grid, u, label=f"Sample {i+1}")
        
    plt.title(f"Fully Developed KS Chaos (L={L_paper}, T_warmup=100)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save generic name for valid verification
    plt.savefig("ks_initial_conditions.png")
    print("Saved ks_initial_conditions.png")