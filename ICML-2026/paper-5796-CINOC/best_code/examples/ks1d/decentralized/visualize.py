"""
Conference-Quality Visualization for KS-1D Centralized DPC
Replicates the "Chaos -> Control" transition plot style.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics
from models.policy_ks1d import DecentralizedControlNet
import tesseracts.ks1d.solver as solver 

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

N_GRID = 128
L_DOMAIN = 22.0
N_AGENTS = 8
T_CONTROL = 200    # Steps of control
T_PRE_SHOW = 100    # Steps of chaos to show BEFORE control starts
DT = 0.05         # Standardized DT

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    """Configure matplotlib for academic/conference style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 24,
        "font.size": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.titlesize": 26,
        "figure.titlesize": 28,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
    }
    plt.rcParams.update(tex_fonts)

def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    dummy_u = jnp.zeros((N_GRID,))
    dummy_xi = jnp.linspace(0, L_DOMAIN, N_AGENTS)
    init_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def generate_stitched_trajectory(key, dynamics, params, xi_fixed):
    """
    Generates a continuous trajectory:
    Phase 1: Spin-up (Chaos) -> We keep the last T_PRE_SHOW steps.
    Phase 2: Control (Stabilization) -> We run T_CONTROL steps.
    """
    warmup_steps = 2000 
    DT = 0.05  # Standardized DT
    SIGMA = 1.0 # Standardized Actuator Width
    
    u_noise = jax.random.normal(key, shape=(N_GRID,)) * 0.01
    u_noise = u_noise - jnp.mean(u_noise)
    u_hat_init = jnp.fft.rfft(u_noise)
    
    # Pre-calculate operators for the warm-up phase
    dx = L_DOMAIN / N_GRID
    k = 2 * jnp.pi * jnp.fft.rfftfreq(N_GRID, d=dx)
    L_linear = k**2 - k**4

    # 1. Physics (Uncontrolled Warm-up)
    def step_fn(carry, _):
        u_hat_curr, u_curr = carry
        u_zero = jnp.zeros_like(xi_fixed)
        
        # FIX: Pass all required arguments to the spectral step
        u_hat_next, u_next = solver.ks_spectral_step(
            u_hat_curr, u_curr, xi_fixed, u_zero, 
            k=k, L_linear=L_linear, 
            N=N_GRID, L=L_DOMAIN, dt=DT, sigma=SIGMA
        )
        return (u_hat_next, u_next), (u_next, u_zero)

    (u_hat_ready, u_ready), (u_traj_warmup, _) = jax.lax.scan(
        step_fn, (u_hat_init, u_noise), None, length=warmup_steps
    )
    
    # 2. Control Phase
    u_target = jnp.zeros_like(u_ready)
    u_traj_ctrl, _, u_ctrl_ctrl, _ = dynamics.unroll_controlled(
        u_ready, xi_fixed, u_target, params, T_CONTROL, 
        N_grid=N_GRID, L=L_DOMAIN, dt=DT, sigma=SIGMA
    )
    
    # 3. Stitch Data
    u_pre = u_traj_warmup[-T_PRE_SHOW:]
    u_post = u_traj_ctrl
    
    # FIX: Update the forcing reconstruction to include L and sigma
    def get_forcing(u_intensities):
        return solver.forcing_fn_1d(xi_fixed, u_intensities, N_GRID, L_DOMAIN, SIGMA)
    
    f_pre = jnp.zeros((T_PRE_SHOW, N_GRID)) 
    f_post = jax.vmap(get_forcing)(u_ctrl_ctrl)
    
    u_full = jnp.concatenate([u_pre, u_post], axis=0)
    f_full = jnp.concatenate([f_pre, f_post], axis=0)
    
    return u_full, f_full

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paper_style(u_full, f_full, example_idx=1):
    setup_academic_style()
    
    # Convert to Numpy and Transpose for (Space x Time)
    data_u = np.array(u_full).T
    data_f = np.array(f_full).T
    
    # Construct Time Axis relative to Control Start (t=0)
    total_steps = T_PRE_SHOW + T_CONTROL
    t_start = -T_PRE_SHOW * DT
    t_end = T_CONTROL * DT
    
    extent = [t_start, t_end, 0, L_DOMAIN]
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, width_ratios=[30, 1], height_ratios=[1, 1], wspace=0.02, hspace=0.15)
    
    # --- Top: State Evolution ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(data_u, aspect='auto', cmap='RdBu_r', origin='lower', 
                     extent=extent, vmin=-3, vmax=3)
    
    # Add vertical line at t=0 (Control On)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.8)
    ax1.text(t_start + 1, L_DOMAIN * 0.9, "Uncontrolled", fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(1, L_DOMAIN * 0.9, "Controlled", fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.set_ylabel(r"Position $x$")
    ax1.set_title(r"(a) Kuramoto-Sivashinsky Stabilization", loc='left', fontweight='bold')
    ax1.tick_params(labelbottom=False)
    
    cax1 = fig.add_subplot(gs[0, 1])
    plt.colorbar(im1, cax=cax1, label=r"$u(x,t)$")
    
    # --- Bottom: Control Forcing ---
    ax2 = fig.add_subplot(gs[1, 0])
    # Use symmetric limit for colorbar
    f_lim = np.max(np.abs(data_f)) * 0.8
    if f_lim < 1e-4: f_lim = 1.0
    
    im2 = ax2.imshow(data_f, aspect='auto', cmap='bwr', origin='lower', 
                     extent=extent, vmin=-f_lim, vmax=f_lim)
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.8)
    
    ax2.set_ylabel(r"Position $x$")
    ax2.set_xlabel(r"Time $t$ (seconds)")
    ax2.set_title(r"(b) Actuator Forcing", loc='left', fontweight='bold')
    
    cax2 = fig.add_subplot(gs[1, 1])
    plt.colorbar(im2, cax=cax2, label=r"Forcing $f(x,t)$")
    
    save_path = Path("figures/images/vanilla") / f"ks_paper_plot_ex{example_idx}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

def main():
    print("=" * 60)
    print("  KS VISUALIZATION (CHAOS -> CONTROL)")
    print("=" * 60)
    
    model = DecentralizedControlNet(features=(64, 64), L_domain=L_DOMAIN)
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    try:
        params = load_params(model, 'ks_centralized_params.msgpack')
        print(f"✓ Loaded trained parameters")
    except FileNotFoundError:
        print("✗ Error: Params not found.")
        return

    # Fixed Actuators
    xi_fixed = jnp.linspace(0.0, L_DOMAIN, N_AGENTS, endpoint=False) + (L_DOMAIN/N_AGENTS)/2
    
    key = jax.random.PRNGKey(42)
    
    for i in range(3):
        print(f"Generating Example {i+1}...")
        key, subkey = jax.random.split(key)
        
        # Run the full pipeline: Warmup -> Stitch -> Control
        u_full, f_full = generate_stitched_trajectory(subkey, dynamics, params, xi_fixed)
        
        plot_paper_style(u_full, f_full, example_idx=i+1)

if __name__ == "__main__":
    main()