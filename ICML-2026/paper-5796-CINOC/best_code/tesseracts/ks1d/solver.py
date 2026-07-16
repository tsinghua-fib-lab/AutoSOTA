import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import matplotlib.pyplot as plt

# --- 1. Physics Helper Functions (No Globals) ---

def forcing_fn_1d(xi_fixed, u_intensities, N, L, sigma):
    """
    Calculates the 1D Gaussian influence of STATIC actuators.
    """
    x_coords = jnp.linspace(0, L, N, endpoint=False)
    
    def single_actuator(pos, intensity):
        # Periodic distance for influence
        dist = jnp.abs(x_coords - pos)
        dist = jnp.minimum(dist, L - dist) # Handle periodicity
        return intensity * jnp.exp(-(dist**2) / (2 * sigma**2))
    
    forcings = jax.vmap(single_actuator)(xi_fixed, u_intensities)
    return jnp.sum(forcings, axis=0)

def ks_spectral_step(
    u_hat, 
    u_field, 
    xi_fixed, 
    u_control, 
    k, 
    L_linear, 
    N=256, 
    L=64.0, 
    dt=0.05, 
    sigma=1.0
):
    """
    Semi-Implicit Crank-Nicolson Spectral Step.
    """
    # 1. Non-linear term (computed in real space)
    # u * u_x = 0.5 * d/dx (u^2)
    u_sq = u_field ** 2
    u_sq_hat = jnp.fft.rfft(u_sq)
    
    # Nonlinearity in Fourier space: -0.5 * i * k * FFT(u^2)
    nonlinear_term_hat = -0.5 * (1j * k) * u_sq_hat
    
    # 2. Forcing term (computed in real space)
    f_field = forcing_fn_1d(xi_fixed, u_control, N, L, sigma)
    f_hat = jnp.fft.rfft(f_field)

    # 3. Time Stepping (Crank-Nicolson for Linear, Explicit for Non-linear/Forcing)
    denom = 1.0 - (dt / 2.0) * L_linear
    numer = (1.0 + (dt / 2.0) * L_linear) * u_hat + dt * (nonlinear_term_hat + f_hat)
    
    u_hat_next = numer / denom
    
    # 4. Recover real space for next iteration/logging
    u_next = jnp.fft.irfft(u_hat_next)
    
    return u_hat_next, u_next

# --- 2. Main Solver Loop with Noise ---

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'N_grid'])
def solve_with_policy(
    u_init, 
    xi_fixed, 
    u_target, 
    params, 
    policy_apply_fn, 
    t_steps,
    key,                # <--- Added Key
    N_grid=256, 
    L=64.0, 
    dt=0.05, 
    sigma=1.0,
    noise_u=0.0,        # <--- Added Actuator Noise Magnitude
    noise_z=0.0         # <--- Added Sensor/State Noise Magnitude
):
    """
    KS Loop using dynamic N_grid and L, with Sensor and Actuator noise.
    """
    # Recalculate dx and k based on current N_grid and L
    dx = L / N_grid
    k = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
    L_linear = k**2 - k**4
    
    u_hat_init = jnp.fft.rfft(u_init)

    def step_fn(carry, _):
        u_hat_curr, u_curr, current_key = carry
        
        # Split keys for this step
        k_sensor, k_actuator, next_key = jax.random.split(current_key, 3)
        
        # 1. Add Sensor Noise (What the policy sees)
        # We add noise to the real-space representation
        u_observed = u_curr + noise_z * jax.random.normal(k_sensor, u_curr.shape)
        
        # 2. Policy determines control intensities (using observed state)
        u_control = policy_apply_fn(params, u_observed, u_target, xi_fixed)
        
        # 3. Add Actuator Noise (What the physics gets)
        u_control_noisy = u_control + noise_u * jax.random.normal(k_actuator, u_control.shape)
        
        # 4. Physics step (Uses actual previous state, but noisy control)
        u_hat_next, u_next = ks_spectral_step(
            u_hat_curr, u_curr, xi_fixed, u_control_noisy, 
            k, L_linear, 
            N=N_grid, L=L, dt=dt, sigma=sigma
        )
        
        v_dummy = jnp.zeros_like(u_control) 
        
        # Pass next_key to the next iteration
        return (u_hat_next, u_next, next_key), (u_next, xi_fixed, u_control_noisy, v_dummy)

    # Initialize scan with key
    _, trajectory = jax.lax.scan(
        step_fn, 
        (u_hat_init, u_init, key), 
        None, 
        length=t_steps
    )
    
    return trajectory

# --- 3. Example Usage ---
if __name__ == "__main__":
    # --- Configuration for this run ---
    N_GRID = 256
    L_DOMAIN = 64.0
    DT = 0.05
    N_STEPS = 2000

    # --- Dummy Policy ---
    def dummy_policy_fn(params, u_curr, u_target, xi_fixed):
        n_actuators = xi_fixed.shape[0]
        # Just return zeros, but now physics will add noise to this 0.0
        return jnp.zeros((n_actuators,))

    # --- Initialization ---
    key = jax.random.PRNGKey(0) # Main random key
    n_actuators = 4
    actuator_positions = jnp.linspace(0, L_DOMAIN, n_actuators, endpoint=False)

    # Initial Condition
    x = jnp.linspace(0, L_DOMAIN, N_GRID, endpoint=False)
    u0 = (jnp.sin(2 * jnp.pi * x / L_DOMAIN) + 
          0.5 * jnp.sin(4 * jnp.pi * x / L_DOMAIN) + 
          0.1 * jax.random.normal(jax.random.PRNGKey(42), (N_GRID,)))
          
    u_target = jnp.zeros_like(u0)

    print(f"Running simulation with N={N_GRID}, L={L_DOMAIN}...")
    print("Injecting Actuator Noise (0.5) and Sensor Noise (0.1)...")
    
    # --- Run ---
    trajectory = solve_with_policy(
        u0, 
        actuator_positions, 
        u_target, 
        None, 
        dummy_policy_fn, 
        N_STEPS,
        key,              # Pass the key
        N_grid=N_GRID,
        L=L_DOMAIN,
        dt=DT,
        noise_u=0.,      
        noise_z=0.       
    )
    
    u_history, xi_history, control_history, _ = trajectory

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    im = plt.imshow(u_history, aspect='auto', extent=[0, L_DOMAIN, N_STEPS * DT, 0], cmap='RdBu_r')
    plt.colorbar(im, label='u(x, t)')
    plt.xlabel('Spatial Domain (x)')
    plt.ylabel('Time (t)')
    plt.title(f'KS Trajectory with Noise (L={L_DOMAIN})')
    
    for pos in actuator_positions:
        plt.axvline(x=pos, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('ks_trajectory_noisy.png')
    print("Plot saved as 'ks_trajectory_noisy.png'")