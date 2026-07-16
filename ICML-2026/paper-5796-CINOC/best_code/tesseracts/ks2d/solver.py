import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import matplotlib.pyplot as plt

# --- 1. Physics Helper Functions (2D) ---

def forcing_fn_2d(xi_fixed, u_intensities, N, L, sigma):
    """Calculates the 2D Gaussian influence of STATIC actuators."""
    x = jnp.linspace(0, L, N, endpoint=False)
    y = jnp.linspace(0, L, N, endpoint=False)

    # --- indexing='ij' ---
    # Matches the spectral solver's grid layout (Row=X, Col=Y)
    X, Y = jnp.meshgrid(x, y, indexing='ij') 
    
    def single_actuator(pos, intensity):
        dx = jnp.abs(X - pos[0])
        dx = jnp.minimum(dx, L - dx)
        dy = jnp.abs(Y - pos[1])
        dy = jnp.minimum(dy, L - dy)
        dist_sq = dx**2 + dy**2
        return intensity * jnp.exp(-dist_sq / (2 * sigma**2))
    
    forcings = jax.vmap(single_actuator)(xi_fixed, u_intensities)
    return jnp.sum(forcings, axis=0)

def precompute_etdrk4_coeffs(L_linear, dt):
    """Precomputes the stability coefficients for ETDRK4."""
    ch = L_linear * dt
    tol = 1e-4
    is_small = jnp.abs(ch) < tol
    safe_ch = jnp.where(is_small, 1.0, ch) 
    
    f1 = jnp.where(is_small, 1.0 + ch/2.0 + ch**2/6.0, (jnp.exp(ch) - 1.0) / safe_ch)
    f2 = jnp.where(is_small, 0.5 + ch/6.0 + ch**2/24.0, (jnp.exp(ch) - ch - 1.0) / (safe_ch**2))
    f3 = jnp.where(is_small, 1.0/6.0 + ch/24.0 + ch**2/120.0, (jnp.exp(ch) - 0.5*ch**2 - ch - 1.0) / (safe_ch**3))
    
    E = jnp.exp(ch)
    E2 = jnp.exp(ch / 2.0)
    Q = dt * f1
    P1 = dt * (f1 - 3*f2 + 4*f3)
    P2 = dt * (2*f2 - 4*f3)
    P3 = dt * (2*f2 - 4*f3)
    P4 = dt * (-f2 + 4*f3)
    
    return E, E2, Q, P1, P2, P3, P4

def get_nonlinear(u_hat, kx, ky, dealias_mask, N):
    u_hat_clean = u_hat * dealias_mask
    u_x_hat = 1j * kx * u_hat_clean
    u_y_hat = 1j * ky * u_hat_clean
    u_x = jnp.fft.irfftn(u_x_hat, s=(N, N))
    u_y = jnp.fft.irfftn(u_y_hat, s=(N, N))
    nonlinear_field = 0.5 * (u_x**2 + u_y**2)
    nl_hat = -jnp.fft.rfftn(nonlinear_field)
    nl_hat = nl_hat.at[0, 0].set(0.0) 
    return nl_hat

def ks_spectral_step_etdrk4(u_hat, u_curr_dummy, xi_fixed, u_control, kx, ky, etdrk4_coeffs, dealias_mask, N=128, L=64.0, dt=0.05, sigma=1.2):
    E, E2, Q, P1, P2, P3, P4 = etdrk4_coeffs
    f_field = forcing_fn_2d(xi_fixed, u_control, N, L, sigma)
    f_hat = jnp.fft.rfftn(f_field)
    
    def NL_fn(uh):
        return get_nonlinear(uh, kx, ky, dealias_mask, N) + f_hat

    Nu_n = NL_fn(u_hat)
    a = E2 * u_hat + Q * Nu_n * 0.5
    Na = NL_fn(a)
    b = E2 * u_hat + Q * Na * 0.5
    Nb = NL_fn(b)
    c = E2 * a + Q * (2.0 * Nb - Nu_n) * 0.5 
    Nc = NL_fn(c)
    
    u_hat_next = (E * u_hat + P1 * Nu_n + P2 * Na + P3 * Nb + P4 * Nc)
    u_next = jnp.fft.irfftn(u_hat_next, s=(N, N))
    return u_hat_next, u_next

# --- 2. Smooth Initial Condition Generator ---
def generate_random_noise_2d(key, N_grid, L, scale=1.0):
    """
    Generates a smooth random initial condition.
    """
    x = jnp.linspace(0, L, N_grid, endpoint=False)
    X, Y = jnp.meshgrid(x, x)
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Mode 1: L-periodic
    phase1_x = jax.random.uniform(k1, minval=0, maxval=2*jnp.pi)
    phase1_y = jax.random.uniform(k2, minval=0, maxval=2*jnp.pi)
    u = jnp.sin(2*jnp.pi*X/L + phase1_x) * jnp.cos(2*jnp.pi*Y/L + phase1_y)
    
    # Mode 2: L/2-periodic
    phase2 = jax.random.uniform(k3, minval=0, maxval=2*jnp.pi)
    u += 0.5 * jnp.sin(4*jnp.pi*X/L + phase2)
    
    # Small noise
    u += 0.05 * jax.random.normal(k4, shape=(N_grid, N_grid))
    
    # Normalize
    u = (u - jnp.mean(u)) 
    std_val = jnp.std(u)
    # Safety check: avoid division by zero if std is tiny
    safe_std = jnp.where(std_val < 1e-6, 1.0, std_val)
    u = u / safe_std * scale
    
    return u

@partial(jax.jit, static_argnames=['policy_apply_fn', 't_steps', 'substeps', 'N_grid'])
def solve_with_policy(
    u_init,
    xi_fixed,
    u_target,
    params,
    policy_apply_fn,
    t_steps,
    substeps=1,
    N_grid=128,
    L=64.0,
    dt=0.01,
    sigma=1.2,
    key=None,           # <--- Added Key for noise generation
    noise_u=0.0,        # <--- Actuator Noise Magnitude
    noise_z=0.0         # <--- Sensor/State Noise Magnitude
):
    """
    Full simulation loop controllable by a policy with Action Repetition.

    Structure:
    - Outer Loop (t_steps): Policy Inference (Control Update)
      - Inner Loop (substeps): Physics Evolution (ETDRK4) holding control constant

    Noise Injection:
    - noise_z: Added to state observation before policy sees it (sensor noise)
    - noise_u: Added to control output before physics applies it (actuator noise)
    """
    # Handle default key
    if key is None:
        key = jax.random.PRNGKey(0)

    # Setup spectral frequencies
    dx = L / N_grid
    kx_vec = 2 * jnp.pi * jnp.fft.fftfreq(N_grid, d=dx)
    ky_vec = 2 * jnp.pi * jnp.fft.rfftfreq(N_grid, d=dx)
    
    KX, KY = jnp.meshgrid(kx_vec, ky_vec, indexing='ij')
    q_sq = KX**2 + KY**2
    L_linear = q_sq - q_sq**2
    
    # De-aliasing Mask
    k_max_x = jnp.max(jnp.abs(kx_vec))
    k_max_y = jnp.max(jnp.abs(ky_vec))
    mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
    mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
    dealias_mask = (mask_x & mask_y).astype(jnp.float32)

    # Precompute Coeffs
    etdrk4_coeffs = precompute_etdrk4_coeffs(L_linear, dt)
    
    u_hat_init = jnp.fft.rfftn(u_init)

    # --- The Outer Loop Function (Policy Step) ---
    def step_fn_outer(carry, _):
        u_hat_curr, u_curr, current_key = carry

        # Split keys for this step
        k_sensor, k_actuator, next_key = jax.random.split(current_key, 3)

        # 1. Add Sensor Noise (What the policy sees)
        u_observed = u_curr + noise_z * jax.random.normal(k_sensor, u_curr.shape)

        # 2. Policy acts based on observed state
        #    This happens once every 'substeps' physics steps
        u_control = policy_apply_fn(params, u_observed, u_target, xi_fixed)

        # 3. Add Actuator Noise (What the physics gets)
        u_control_noisy = u_control + noise_u * jax.random.normal(k_actuator, u_control.shape)
        
        # --- The Inner Loop Function (Physics Substep) ---
        def step_fn_inner(carry_inner, _):
            u_h, u_c = carry_inner

            # Physics evolves, but u_control_noisy is constant (Zero-Order Hold)
            u_h_next, u_c_next = ks_spectral_step_etdrk4(
                u_h, u_c, xi_fixed, u_control_noisy,
                KX, KY, etdrk4_coeffs, dealias_mask,
                N=N_grid, L=L, dt=dt, sigma=sigma
            )
            return (u_h_next, u_c_next), None

        # 4. Run physics for 'substeps' iterations
        (u_hat_next, u_next), _ = jax.lax.scan(
            step_fn_inner,
            (u_hat_curr, u_curr),
            None,
            length=substeps
        )

        # Return state and trajectory info (pass next_key to next iteration)
        v_dummy = jnp.zeros_like(u_control_noisy)
        return (u_hat_next, u_next, next_key), (u_next, xi_fixed, u_control_noisy, v_dummy)

    # Run the outer control loop (initialize with key in carry)
    _, trajectory = jax.lax.scan(
        step_fn_outer,
        (u_hat_init, u_init, key),
        None,
        length=t_steps
    )

    return trajectory

# --- 3. Example Usage ---
if __name__ == "__main__":
    N_GRID = 128  
    L_DOMAIN = 64.0 
    DT = 0.01 
    N_STEPS = 2000 

    def dummy_policy_fn(params, u_curr, u_target, xi_fixed):
        return jnp.zeros((xi_fixed.shape[0],))

    actuator_positions = jnp.array([[16., 16.], [48., 16.], [16., 48.], [48., 48.]])
    u_target = jnp.zeros((N_GRID, N_GRID))

    # --- TESTING THE NEW I.C. HERE ---
    print("Generating Smooth Random I.C...")
    key = jax.random.PRNGKey(42)
    u0 = generate_random_noise_2d(key, N_GRID, L_DOMAIN, scale=1.0)
    
    print(f"I.C. Stats -> Min: {u0.min():.3f}, Max: {u0.max():.3f}, Mean: {u0.mean():.3f}")

    print(f"Simulating {N_STEPS} steps with ETDRK4...")
    trajectory = solve_with_policy(
        u0, actuator_positions, u_target, None, dummy_policy_fn, 
        N_STEPS, N_grid=N_GRID, L=L_DOMAIN, dt=DT
    )
    
    u_history, _, _, _ = trajectory
    
    # Check for NaNs at the end
    if jnp.isnan(u_history[-1]).any():
        print("!!! FAILURE: Simulation resulted in NaNs !!!")
    else:
        print("SUCCESS: Simulation finished with valid numbers.")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    indices = jnp.linspace(0, N_STEPS - 1, 6, dtype=int)

    for i, idx in enumerate(indices):
        ax = axes[i]
        time = idx * DT
        im = ax.imshow(u_history[idx], extent=[0, L_DOMAIN, 0, L_DOMAIN], 
                       origin='lower', cmap='RdBu_r', vmin=-1.2, vmax=1.2)
        ax.set_title(f"t = {time:.2f}")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='u(x, y)')
    plt.suptitle(f"2D KS Check (New IC + dt={DT})", fontsize=16)
    plt.savefig('ks_2d_ic_test.png', bbox_inches='tight')
    print("Test plot saved as 'ks_2d_ic_test.png'")