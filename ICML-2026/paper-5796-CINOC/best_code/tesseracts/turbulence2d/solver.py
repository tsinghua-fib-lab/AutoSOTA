"""
2D Decaying Isotropic Turbulence Solver
Spectral Solver (FFT + RK4 + De-aliasing)
Pure JAX Implementation
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange

# Enable 64-bit precision to match
jax.config.update("jax_enable_x64", True)

# --- 1. Physics Core (Spectral & RK4) ---

def get_spectral_grid(N, L):
    """Generates wavenumbers for N x N grid on domain size L."""
    # kx: [0, 1, ..., N/2, -N/2+1, ..., -1] * (2pi/L)
    k = jnp.fft.fftfreq(N, d=L/N) * 2 * jnp.pi
    kx, ky = jnp.meshgrid(k, k, indexing='xy')
    k_sq = kx**2 + ky**2
    # Avoid division by zero for inverse laplacian (streamfunction)
    k_sq_inv = jnp.where(k_sq == 0, 0, 1.0 / k_sq)
    return kx, ky, k_sq, k_sq_inv

def compute_forcing_profile(centers_x, centers_y, N, L, sigma):
    """
    Pre-computes the Gaussian forcing pattern in Spectral Space.
    """
    x = jnp.linspace(0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x, indexing='xy')
    
    # Create batch of Gaussians in Physical Space
    # Shape: (Num_Actuators, N, N)
    dx = X[None, :, :] - centers_x[:, None, None]
    dy = Y[None, :, :] - centers_y[:, None, None]
    
    # Periodic distance handling
    dx = dx - L * jnp.round(dx / L)
    dy = dy - L * jnp.round(dy / L)
    
    # Gaussian blob
    r2 = dx**2 + dy**2
    blob = jnp.exp(-r2 / (2 * sigma**2))
    
    # Normalize
    # Here we normalize so peak is 1.0 before weighting
    blob = blob / jnp.max(blob, axis=(1,2), keepdims=True)
    
    # Transform to Spectral Space
    return jnp.fft.fft2(blob)

def dealias_pad(hat_field, N_pad):
    """Pads spectral field to N_pad (3/2 rule) for de-aliasing."""
    N = hat_field.shape[0]
    N_mid = N // 2
    
    # Construct padded array
    padded = jnp.zeros((N_pad, N_pad), dtype=hat_field.dtype)
    
    # Copy quadrants: (TopLeft, TopRight, BotLeft, BotRight)
    # This preserves wavenumbers while inserting zeros in high freq
    padded = padded.at[0:N_mid, 0:N_mid].set(hat_field[0:N_mid, 0:N_mid])
    padded = padded.at[0:N_mid, N_pad-N_mid:].set(hat_field[0:N_mid, N_mid:])
    padded = padded.at[N_pad-N_mid:, 0:N_mid].set(hat_field[N_mid:, 0:N_mid])
    padded = padded.at[N_pad-N_mid:, N_pad-N_mid:].set(hat_field[N_mid:, N_mid:])
    
    return padded

def dealias_truncate(hat_field_padded, N):
    """Truncates padded spectral field back to N."""
    N_pad = hat_field_padded.shape[0]
    N_mid = N // 2
    
    new_field = jnp.zeros((N, N), dtype=hat_field_padded.dtype)
    
    # Extract quadrants
    new_field = new_field.at[0:N_mid, 0:N_mid].set(hat_field_padded[0:N_mid, 0:N_mid])
    new_field = new_field.at[0:N_mid, N_mid:].set(hat_field_padded[0:N_mid, N_pad-N_mid:])
    new_field = new_field.at[N_mid:, 0:N_mid].set(hat_field_padded[N_pad-N_mid:, 0:N_mid])
    new_field = new_field.at[N_mid:, N_mid:].set(hat_field_padded[N_pad-N_mid:, N_pad-N_mid:])
    
    # Normalization scale factor for FFT padding
    scale = (N / N_pad)**2
    return new_field * scale

def physics_rhs(w_hat, kx, ky, k_sq, k_sq_inv, viscosity, forcing_hat, u_cmd, dealias=True):
    """
    Computes dw_hat/dt = - (u dot grad) w + nu * laplace(w) + forcing
    """
    N = w_hat.shape[0]
    
    # 1. Streamfunction: psi_hat = - w_hat / k^2
    psi_hat = -w_hat * k_sq_inv
    
    # 2. Velocity in Spectral Space: u = dpsi/dy, v = -dpsi/dx
    # derivative in spectral = i * k * val
    u_hat = (1j * ky) * psi_hat
    v_hat = (-1j * kx) * psi_hat
    
    # 3. Non-linear Term (Advection): - (u*dw/dx + v*dw/dy)
    # Compute derivatives of w in spectral
    w_x_hat = (1j * kx) * w_hat
    w_y_hat = (1j * ky) * w_hat
    
    if dealias:
        # 3/2 Padding Rule
        M = int(3 * N / 2)
        u_phys = jnp.fft.ifft2(dealias_pad(u_hat, M)).real
        v_phys = jnp.fft.ifft2(dealias_pad(v_hat, M)).real
        w_x_phys = jnp.fft.ifft2(dealias_pad(w_x_hat, M)).real
        w_y_phys = jnp.fft.ifft2(dealias_pad(w_y_hat, M)).real
        
        # Convection in Physical Space
        conv_phys = u_phys * w_x_phys + v_phys * w_y_phys
        
        # Back to Spectral + Truncate
        conv_hat = dealias_truncate(jnp.fft.fft2(conv_phys), N)
    else:
        # No de-aliasing (faster, less accurate)
        u_phys = jnp.fft.ifft2(u_hat).real
        v_phys = jnp.fft.ifft2(v_hat).real
        w_x_phys = jnp.fft.ifft2(w_x_hat).real
        w_y_phys = jnp.fft.ifft2(w_y_hat).real
        conv_hat = jnp.fft.fft2(u_phys * w_x_phys + v_phys * w_y_phys)

    # 4. Diffusion: nu * laplace(w) -> -nu * k^2 * w_hat
    diff_term = -viscosity * k_sq * w_hat
    
    # 5. Forcing
    # forcing_hat is (Num_Actuators, N, N), u_cmd is (Num_Actuators,)
    # We sum over actuators
    active_forcing = jnp.sum(forcing_hat * u_cmd[:, None, None], axis=0)
    
    return -conv_hat + diff_term + active_forcing

def rk4_step(w_hat, dt, kx, ky, k_sq, k_sq_inv, viscosity, forcing_hat, u_cmd):
    """Standard RK4 Integration."""
    def f(w):
        return physics_rhs(w, kx, ky, k_sq, k_sq_inv, viscosity, forcing_hat, u_cmd)
    
    k1 = f(w_hat)
    k2 = f(w_hat + 0.5 * dt * k1)
    k3 = f(w_hat + 0.5 * dt * k2)
    k4 = f(w_hat + dt * k3)
    
    return w_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 2. Solver Interface ---

def solve_with_policy(
    policy_fn,
    domain_size=1.0,
    grid_res=128,
    viscosity=5e-5,
    dt=0.02,
    t_max=6.0,
    actuator_grid_shape=(8, 8),
    sigma=0.05#0.2
):
    # 1. Setup Grid & Physics Constants
    N = grid_res
    L = domain_size
    kx, ky, k_sq, k_inv = get_spectral_grid(N, L)
    
    # 2. Setup Actuators
    nx_act, ny_act = actuator_grid_shape
    x_coords = np.linspace(0, L, nx_act, endpoint=False) + L/(2*nx_act)
    y_coords = np.linspace(0, L, ny_act, endpoint=False) + L/(2*ny_act)
    
    centers_list = [[x, y] for x in x_coords for y in y_coords]
    centers_x = jnp.array([c[0] for c in centers_list])
    centers_y = jnp.array([c[1] for c in centers_list])
    
    # Pre-compute forcing profiles (Batch, N, N)
    print("Pre-computing spectral forcing profiles...")
    forcing_hat = compute_forcing_profile(centers_x, centers_y, N, L, sigma)
    
    # 3. Initial Condition (Random Spectral Noise)
    # Random phase, specific energy spectrum
    key = jax.random.PRNGKey(42)
    # Simple Kolmogorov-like spectrum initialization
    w_hat = jax.random.normal(key, (N, N), dtype=jnp.complex128)
    # Filter to low frequencies for smooth IC
    filter_mask = jnp.exp(-k_sq / (2 * (10 * 2*jnp.pi/L)**2))
    w_hat = w_hat * filter_mask
    
    # Normalize Energy (approx)
    psi_hat = -w_hat * k_inv
    u_hat = (1j * ky) * psi_hat
    v_hat = (-1j * kx) * psi_hat
    energy = 0.5 * jnp.sum(jnp.abs(u_hat)**2 + jnp.abs(v_hat)**2) / (N**4)
    w_hat = w_hat / jnp.sqrt(energy) # Scale to E=1.0
    
    # 4. Simulation Constants
    substeps = int(16 * grid_res * dt) 
    dt_phys = dt / substeps
    total_steps = int(t_max / dt)
    
    # 5. JIT Compiled Inner Loop
    @jax.jit
    def physics_loop(w_in, u_cmd_val):
        def body_fun(i, w):
            return rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, viscosity, forcing_hat, u_cmd_val)
        return jax.lax.fori_loop(0, substeps, body_fun, w_in)
    
    # Warmup
    print("Compiling JAX kernel...")
    zeros_cmd = jnp.zeros(len(centers_list))
    _ = physics_loop(w_hat, zeros_cmd)
    
    history = []
    times = []
    
    print(f"Running {total_steps} macro-steps (Spectral + RK4)...")
    start_time = time.time()
    
    current_w_hat = w_hat
    
    for i in trange(total_steps):
        t = i * dt
        
        # Get Physical State for Policy (Output real field)
        w_phys = jnp.fft.ifft2(current_w_hat).real
        
        # Policy Step (CPU or GPU)
        # Note: Policy sees physical grid (N,N), returns action vector
        u_action = policy_fn(w_phys, centers_list, t)
        
        # Ensure action is JAX array
        u_action = jnp.array(u_action)
        
        # Physics Step (Spectral)
        current_w_hat = physics_loop(current_w_hat, u_action)
        
        if i % 5 == 0:
            # Store Physical representation
            history.append(jnp.fft.ifft2(current_w_hat).real)
            times.append(t)
            
    print(f"Simulation done in {time.time()-start_time:.2f}s")
    return np.array(history), times

# --- 3. Visualization & Test ---

# --- 3. Visualization & Test ---

if __name__ == "__main__":
    
    # A simple policy (e.g., zero control)
    def zero_policy(w_state, centers, t):
        return jnp.zeros(len(centers))

    # Run Solver with params matching the training script
    # T_max = 150 control steps * 5 physics substeps * 0.01 dt = 7.5s
    w_history, times = solve_with_policy(
        policy_fn=zero_policy,
        grid_res=64,          # Matches 'N_grid'
        viscosity=5e-4,       # Matches 'viscosity'
        dt=0.01,              # Matches 'dt'
        t_max=7.5,            # Matches training horizon (150 * 5 * 0.01)
        domain_size=1.0,      # Matches 'L_domain'
        actuator_grid_shape=(8, 8) # Matches 'n_agents': 64
    )
    
    # Plotting
    print("\nVisualizing...")
    num_plots = 6
    indices = np.linspace(0, len(w_history)-1, num_plots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    w_max = np.max(np.abs(w_history[0]))
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        w_field = w_history[idx]
        t_val = times[idx]
        
        im = ax.imshow(w_field, origin='lower', cmap='RdBu_r', 
                       vmin=-w_max, vmax=w_max,
                       extent=[0, 1.0, 0, 1.0])
        ax.set_title(rf"$\omega$ at t={t_val:.2f}")
        ax.axis('off')

    plt.suptitle(r"2D Turbulence (Spectral RK4) - Training Config", fontsize=16)
    plt.tight_layout()
    plt.savefig("turbulence_spectral_jax_new.png", dpi=150)
    print("Plot saved to 'turbulence_spectral_jax_new.png'")