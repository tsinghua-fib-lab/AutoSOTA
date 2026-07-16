"""
Generate visualizations for multiple Kuramoto-Sivashinsky (KS) 1D scenarios
using pre-trained decentralized controllers. Each scenario varies in domain size,
grid resolution, and number of actuators. The resulting state and control
evolution are stitched together and visualized in a paper-style format.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import os
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
# CONFIGURATION & SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

n_agents_list = [200, 30, 80]
L_domain_list = [500.0, 64.0, 200.0]
N_grid_list = [1024, 256, 512] 

T_CONTROL = 200     # Steps of control
T_PRE_SHOW = 100    # Steps of chaos to show BEFORE control starts
DT = 0.05           # Standardized DT

MODEL_PATH = "multiple_experiments/models"
PICS_PATH = "multiple_experiments/pics"
os.makedirs(PICS_PATH, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 20, # Slightly reduced for multi-panel
        "font.size": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.titlesize": 22,
        "figure.titlesize": 24,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
    }
    plt.rcParams.update(tex_fonts)

def load_params(model, filepath, n_grid, l_domain, n_agents):
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    dummy_u = jnp.zeros((n_grid,))
    dummy_xi = jnp.linspace(0, l_domain, n_agents)
    init_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def generate_stitched_trajectory(key, dynamics, params, xi_fixed, n_grid, l_domain):
    warmup_steps = 2000 
    SIGMA = 1.0 
    
    u_noise = jax.random.normal(key, shape=(n_grid,)) * 0.01
    u_noise = u_noise - jnp.mean(u_noise)
    u_hat_init = jnp.fft.rfft(u_noise)
    
    dx = l_domain / n_grid
    k = 2 * jnp.pi * jnp.fft.rfftfreq(n_grid, d=dx)
    L_linear = k**2 - k**4

    def step_fn(carry, _):
        u_hat_curr, u_curr = carry
        u_zero = jnp.zeros_like(xi_fixed)
        u_hat_next, u_next = solver.ks_spectral_step(
            u_hat_curr, u_curr, xi_fixed, u_zero, 
            k=k, L_linear=L_linear, 
            N=n_grid, L=l_domain, dt=DT, sigma=SIGMA
        )
        return (u_hat_next, u_next), (u_next, u_zero)

    (u_hat_ready, u_ready), (u_traj_warmup, _) = jax.lax.scan(
        step_fn, (u_hat_init, u_noise), None, length=warmup_steps
    )
    
    u_target = jnp.zeros_like(u_ready)
    u_traj_ctrl, _, u_ctrl_ctrl, _ = dynamics.unroll_controlled(
        u_ready, xi_fixed, u_target, params, T_CONTROL, 
        N_grid=n_grid, L=l_domain, dt=DT, sigma=SIGMA
    )
    
    u_pre = u_traj_warmup[-T_PRE_SHOW:]
    u_post = u_traj_ctrl
    
    def get_forcing(u_intensities):
        return solver.forcing_fn_1d(xi_fixed, u_intensities, n_grid, l_domain, SIGMA)
    
    f_pre = jnp.zeros((T_PRE_SHOW, n_grid)) 
    f_post = jax.vmap(get_forcing)(u_ctrl_ctrl)
    
    return jnp.concatenate([u_pre, u_post], axis=0), jnp.concatenate([f_pre, f_post], axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paper_style(u_full, f_full, n_grid, l_domain, n_agents, filename):
    setup_academic_style()
    data_u = np.array(u_full).T
    data_f = np.array(f_full).T
    
    t_start = -T_PRE_SHOW * DT
    t_end = T_CONTROL * DT
    extent = [t_start, t_end, 0, l_domain]
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, width_ratios=[30, 1], height_ratios=[1, 1], wspace=0.02, hspace=0.15)
    
    # State Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(data_u, aspect='auto', cmap='RdBu_r', origin='lower', extent=extent, vmin=-3, vmax=3)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.8)
    ax1.text(t_start + (t_end-t_start)*0.05, l_domain * 0.85, "Uncontrolled", fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(t_end*0.05, l_domain * 0.85, "Controlled", fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax1.set_ylabel(r"Position $x$")
    ax1.set_title(f"State $u(x,t)$: L={l_domain}, Agents={n_agents}", loc='left', fontweight='bold')
    ax1.tick_params(labelbottom=False)
    
    cax1 = fig.add_subplot(gs[0, 1])
    plt.colorbar(im1, cax=cax1, label=r"$u(x,t)$")
    
    # Control Forcing
    ax2 = fig.add_subplot(gs[1, 0])
    f_lim = np.max(np.abs(data_f)) * 0.8
    if f_lim < 1e-4: f_lim = 1.0
    im2 = ax2.imshow(data_f, aspect='auto', cmap='bwr', origin='lower', extent=extent, vmin=-f_lim, vmax=f_lim)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.8)
    ax2.set_ylabel(r"Position $x$")
    ax2.set_xlabel(r"Time $t$ (seconds)")
    ax2.set_title(f"Actuator Forcing $f(x,t)$", loc='left', fontweight='bold')
    
    cax2 = fig.add_subplot(gs[1, 1])
    plt.colorbar(im2, cax=cax2, label=r"$f(x,t)$")
    
    full_save_path = os.path.join(PICS_PATH, filename)
    plt.savefig(full_save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization: {full_save_path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" MULTI-SCENARIO KS VISUALIZATION")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)

    for i in range(len(n_agents_list)):
        n_agents = n_agents_list[i]
        L_domain = L_domain_list[i]
        N_grid = N_grid_list[i]
        
        print(f"\nScenario {i+1}: N={N_grid}, L={L_domain}, A={n_agents}")
        
        model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)
        dynamics = PDEDynamics(policy_apply_fn=model.apply)
        
        # Match the filename used in the training script
        model_filename = f"ks_params_N{N_grid}_L{int(L_domain)}_A{n_agents}.msgpack"
        model_full_path = os.path.join(MODEL_PATH, model_filename)
        
        try:
            params = load_params(model, model_full_path, N_grid, L_domain, n_agents)
        except FileNotFoundError:
            print(f"✗ Skipping Scenario {i+1}: Weights not found at {model_full_path}")
            continue

        xi_fixed = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
        
        # Generate and plot
        key, subkey = jax.random.split(key)
        u_full, f_full = generate_stitched_trajectory(subkey, dynamics, params, xi_fixed, N_grid, L_domain)
        
        plot_name = f"viz_N{N_grid}_L{int(L_domain)}.png"
        plot_paper_style(u_full, f_full, N_grid, L_domain, n_agents, plot_name)

if __name__ == "__main__":
    main()