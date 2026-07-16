"""
Data Utilities for 2D Decaying Isotropic Turbulence
Generates chaotic initial conditions by evolving random spectral noise.
Reuses the core solver to ensure physical consistency.
"""
import jax
import jax.numpy as jnp
from functools import partial
import pickle
from pathlib import Path

# Enable x64 (Crucial for spectral accuracy)
jax.config.update("jax_enable_x64", True)

import tesseracts.turbulence2d.solver as solver

# --- 1. Initialization Logic ---

def generate_spectral_noise(key, N_grid, L, energy=1.0):
    """
    Generates a random initial vorticity field in SPECTRAL space.
    Matches the Julia logic: Random phase with specific energy spectrum.
    """
    # 1. Get Grid Specs
    kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_grid, L)
    
    # 2. Generate Random Complex Noise
    # Shape: (N, N) complex128
    w_hat = jax.random.normal(key, (N_grid, N_grid), dtype=jnp.complex128)
    
    # 3. Band-pass Filter (Smoothness)
    # We dampen high frequencies to start with large smooth structures
    # Peak wavenumber around k=10
    k_peak = 10.0 * (2 * jnp.pi / L)
    filter_mask = jnp.exp(-k_sq / (2 * k_peak**2))
    w_hat = w_hat * filter_mask
    
    # 4. Normalize Energy
    # Calculate kinetic energy: E = 0.5 * sum(|u_hat|^2 + |v_hat|^2)
    # u_hat = i*ky*psi, v_hat = -i*kx*psi, psi = -w_hat/k^2
    psi_hat = -w_hat * k_inv
    u_hat = (1j * ky) * psi_hat
    v_hat = (-1j * kx) * psi_hat
    
    current_energy = 0.5 * jnp.sum(jnp.abs(u_hat)**2 + jnp.abs(v_hat)**2) / (N_grid**4)
    scaling_factor = jnp.sqrt(energy / (current_energy + 1e-10))
    
    return w_hat * scaling_factor

@partial(jax.jit, static_argnames=("N_grid", "L", "warmup_time", "dt", "viscosity"))
def evolve_to_chaos(key, N_grid, L, warmup_time=2.0, dt=0.02, viscosity=5e-5):
    """
    Evolves random spectral noise into fully developed chaotic turbulence.
    Returns the SPECTRAL state (w_hat).
    """
    # 1. Generate Initial Random State
    w_hat_init = generate_spectral_noise(key, N_grid, L)
    
    # 2. Setup Solver Constants
    kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_grid, L)
    
    # 3. Setup Dummy Control Inputs (Autonomous Evolution)
    # The solver expects forcing_hat and u_cmd. We pass zeros.
    # Dummy shape (1, N, N) for forcing, (1,) for control
    zeros_forcing = jnp.zeros((1, N_grid, N_grid), dtype=jnp.complex128)
    zeros_cmd = jnp.zeros(1) 
    
    # 4. Define Loop
    # Calculate substeps to match solver stability logic
    substeps_per_step = int(16 * N_grid * dt)
    dt_phys = dt / substeps_per_step
    total_steps = int(warmup_time / dt)
    
    def warmup_step(w_curr, _):
        # We use a nested loop for sub-stepping, just like the main solver
        def physics_substep(i, w):
            return solver.rk4_step(
                w, dt_phys, kx, ky, k_sq, k_inv, 
                viscosity, zeros_forcing, zeros_cmd
            )
        
        w_next = jax.lax.fori_loop(0, substeps_per_step, physics_substep, w_curr)
        return w_next, None

    # 5. Run Scan (Evolution)
    w_hat_final, _ = jax.lax.scan(
        warmup_step,
        w_hat_init,
        None,
        length=total_steps
    )
    
    return w_hat_final

def get_batch_initial_conditions(key, batch_size, N_grid, L, warmup_time=2.0, viscosity=5e-5):
    """
    Generates a batch of chaotic spectral states.
    Uses vmap to generate 'batch_size' simulations in parallel.
    """
    keys = jax.random.split(key, batch_size)
    
    evolve_fn = partial(
        evolve_to_chaos, 
        N_grid=N_grid, 
        L=L, 
        warmup_time=warmup_time,
        dt=0.02,
        viscosity=viscosity
    )
    
    # VMAP over the keys to run simulations in parallel
    batch_w_hat = jax.vmap(evolve_fn)(keys)
    return batch_w_hat

# --- 2. Main Execution Block ---

if __name__ == "__main__":
    print("--- 2D Turbulence Data Generation ---")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Configuration
    L_domain = 1.0
    N_grid = 128
    n_samples = 3
    warmup = 2.0  # seconds
    
    key = jax.random.PRNGKey(42)
    
    print(f"Generating {n_samples} samples (L={L_domain}, N={N_grid})...")
    print(f"Warmup time: {warmup}s (to reach developed chaos)")
    
    try:
        # Generate Batch (Spectral States)
        w_hat_samples = get_batch_initial_conditions(key, n_samples, N_grid, L_domain, warmup)
        
        # Block until ready (force computation)
        w_hat_samples.block_until_ready()
        
        print(f"Generation Complete.")
        print(f"Spectral Shape: {w_hat_samples.shape}")
        
        # Convert to Physical Space for Visualization/Check
        # Note: ifft2 returns complex, we take real part
        w_phys_samples = jax.vmap(lambda x: jnp.fft.ifft2(x).real)(w_hat_samples)
        
        print(f"Physical Stats: Min {w_phys_samples.min():.2f} | Max {w_phys_samples.max():.2f}")
        
        # Save Preview
        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
        if n_samples == 1: axes = [axes]
        
        for i in range(n_samples):
            ax = axes[i]
            im = ax.imshow(w_phys_samples[i], origin='lower', cmap='RdBu_r', 
                           extent=[0, L_domain, 0, L_domain])
            ax.set_title(f"Sample {i} (t={warmup}s)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig("turbulence_ic_preview.png")
        print("Preview saved to 'turbulence_ic_preview.png'")
        
        # Save Data (Saving the SPECTRAL states)
        save_path = "turbulence_chaotic_ics_128.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(w_hat_samples, f)
        print(f"Spectral data saved to {save_path}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()