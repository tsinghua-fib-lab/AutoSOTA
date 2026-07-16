"""
Data Utilities for 2D Kuramoto-Sivashinsky (KS)
Generates chaotic initial conditions by evolving the system autonomously.
Reuses the core solver to ensure physical consistency.
"""
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Enable x64 (Crucial for spectral accuracy)
jax.config.update("jax_enable_x64", True)

from tesseracts.ks2d.solver import ks_spectral_step_etdrk4, precompute_etdrk4_coeffs

# --- 1. Initialization Logic ---

def generate_random_noise_2d(key, N_grid, L, scale=1.0):
    """Generates a smooth random initial condition on a periodic grid."""
    x = jnp.linspace(0, L, N_grid, endpoint=False)
    X, Y = jnp.meshgrid(x, x)
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Mode 1: L-periodic base waves
    phase1_x = jax.random.uniform(k1, minval=0, maxval=2*jnp.pi)
    phase1_y = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    u = jnp.sin(2*jnp.pi*X/L + phase1_x) * jnp.cos(2*jnp.pi*Y/L + phase1_y)
    
    # Mode 2: Higher frequency perturbation
    phase2 = jax.random.uniform(k3, minval=0, maxval=2*jnp.pi)
    u += 0.5 * jnp.sin(4*jnp.pi*X/L + phase2)
    
    # Mode 3: Random white noise
    u += 0.05 * jax.random.normal(k4, shape=(N_grid, N_grid))
    
    # Normalize to standard deviation
    u = (u - jnp.mean(u)) 
    u = u / (jnp.std(u) + 1e-6) * scale
    
    return u

@partial(jax.jit, static_argnames=("N_grid", "L", "warmup_time", "dt"))
def evolve_to_attractor(key, N_grid, L, warmup_time=200.0, dt=0.005):
    """
    Evolves random noise into fully developed chaos.
    """
    # 1. Generate random seed
    u_init = generate_random_noise_2d(key, N_grid, L)
    u_hat = jnp.fft.rfftn(u_init)
    
    # 2. Setup Spectral Grid (Must match solver.py logic)
    dx = L / N_grid
    kx_vec = 2 * jnp.pi * jnp.fft.fftfreq(N_grid, d=dx)
    ky_vec = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
    KX, KY = jnp.meshgrid(kx_vec, ky_vec, indexing='ij')
    
    # Linear Operator
    q_sq = KX**2 + KY**2
    L_linear = q_sq - q_sq**2
    
    # 3. Precompute ETDRK4 Coeffs
    etdrk4_coeffs = precompute_etdrk4_coeffs(L_linear, dt)
    
    # De-aliasing Mask (Standard 2/3 Rule)
    k_max_x = jnp.max(jnp.abs(kx_vec))
    k_max_y = jnp.max(jnp.abs(ky_vec))
    mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
    mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
    dealias_mask = (mask_x & mask_y).astype(jnp.float32)

    # 4. Simulation Loop
    steps = int(warmup_time / dt)
    
    # DUMMY CONTROL: Zero intensity, valid shape
    # We pass this to the solver so we can reuse the same function
    xi_dummy = jnp.zeros((1, 2)) 
    u_control_dummy = jnp.zeros(1)

    def warmup_step(carry, _):
        u_hat_curr, u_curr = carry
        
        # We use the EXACT same step function as training
        u_hat_next, u_next = ks_spectral_step_etdrk4(
            u_hat_curr, u_curr, xi_dummy, u_control_dummy,
            KX, KY, etdrk4_coeffs, dealias_mask,
            N=N_grid, L=L, dt=dt, sigma=1.2
        )
        return (u_hat_next, u_next), None

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
    Generates a batch of chaotic states.
    Uses vmap to generate 'batch_size' simulations in parallel.
    """
    keys = jax.random.split(key, batch_size)
    
    evolve_fn = partial(
        evolve_to_attractor, 
        N_grid=N_grid, 
        L=L, 
        warmup_time=200.0, # Long enough to forget initial noise
        dt=0.005           # Safe time step for 2D KS
    )
    
    batch_u = jax.vmap(evolve_fn)(keys)
    return batch_u

# --- 2. Main Execution Block ---

if __name__ == "__main__":
    print("--- 2D KS Data Generation ---")

    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys
    
    # Ensure project root is in sys.path
    script_dir = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.append(str(script_dir))    
    from tesseracts.ks2d.solver import ks_spectral_step_etdrk4, precompute_etdrk4_coeffs

    
    # Config matching your Training Script
    L_domain = 32.0  
    N_grid = 64    
    n_samples = 3
    
    key = jax.random.PRNGKey(101)
    
    print(f"Generating {n_samples} samples (L={L_domain}, N={N_grid})...")
    start = jax.numpy.zeros(1) # Trigger JIT
    
    try:
        u_samples = get_batch_initial_conditions(key, n_samples, N_grid, L_domain)
        # Block until ready
        u_samples.block_until_ready()
        
        print(f"Shape: {u_samples.shape}")
        print(f"Stats: Min {u_samples.min():.2f} | Max {u_samples.max():.2f} | Mean {u_samples.mean():.2f}")
        
        # Save Preview
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            if i < n_samples:
                ax = axes[i]
                im = ax.imshow(u_samples[i], origin='lower', cmap='RdBu_r', extent=[0, L_domain, 0, L_domain])
                ax.set_title(f"Sample {i}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig("ks2d_preview.png")
        print("Preview saved to ks2d_preview.png")
        
        # Save Data
        save_path = "ks2d_chaotic_ics_64.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(u_samples, f)
        print(f"Data saved to {save_path}")
        
    except Exception as e:
        print(f"Error during generation: {e}")