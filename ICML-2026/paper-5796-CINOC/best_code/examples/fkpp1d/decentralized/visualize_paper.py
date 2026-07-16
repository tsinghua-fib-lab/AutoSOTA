"""
Paper-Quality Visualization for 1D Fisher-KPP Equation.
Three rows:
- Top Row: Natural Evolution (Space-Time Heatmap)
- Middle Row: Controlled Evolution (Space-Time Heatmap with Agent Paths)
- Bottom Row: Metrics (MSE, Agent Speed, Control Intensity)

Style: contourf (Space-Time), cmcrameri colormaps, Times New Roman fonts.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path
import cmcrameri.cm as cmc

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Imports using the high-level wrappers
from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'n_pde': 100,
    'n_agents': 20,
    'T_steps': 300,
    'params_file': 'decentralized_params.msgpack'
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath):
    """Load trained parameters from msgpack file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found. Run training first.")
    
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((CONFIG['n_pde'],))
    dummy_xi = jnp.zeros((CONFIG['n_agents'],)) 
    init_params = model.init(key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def zero_policy_apply(params, obs_z, target_z, obs_xi):
    """Dummy policy that returns zero control for 'Natural Evolution'."""
    # Matches the signature expected by DecentralizedControlNet.apply
    # Arguments: params, observed_state, target_state, agent_positions
    
    n_agents = obs_xi.shape[0]
    u = jnp.zeros((n_agents,))
    v = jnp.zeros((n_agents,))
    return u, v

def get_initial_conditions(key):
    """Generate initial and target fields (1D GRF)."""
    k1, k2 = jax.random.split(key, 2)
    _, z_init = data_utils.generate_grf(k1, n_points=CONFIG['n_pde'], length_scale=0.2)
    _, z_target = data_utils.generate_grf(k2, n_points=CONFIG['n_pde'], length_scale=0.4)
    xi_init = jnp.linspace(0.1, 0.9, CONFIG['n_agents'])
    return z_init, z_target, xi_init

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spacetime_heatmap(ax, z_traj, xi_traj, title, v_min=0, v_max=1, show_agents=False):
    """Plot a Space-Time heatmap (Time on Y, Space on X)."""
    T, N = z_traj.shape
    x = np.linspace(0, 1, N)
    t = np.arange(T)
    X, Time = np.meshgrid(x, t)
    
    # Contourf plot
    cf = ax.contourf(
        X, Time, z_traj,
        levels=100,
        cmap=cmc.batlow, # Sequential colormap
        vmin=v_min, vmax=v_max,
        extend='both'
    )
    
    # Overlay Agent Paths
    if show_agents and xi_traj is not None:
        ax.plot(xi_traj, t, color='white', alpha=0.3, linewidth=0.8, linestyle='-')
        ax.scatter(xi_traj[0], [0]*xi_traj.shape[1], c='white', s=2, alpha=0.8)
        
    ax.set_title(title, pad=10)
    ax.set_ylabel("Time Step")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, T)
    
    if title == "Natural Evolution":
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Space ($x$)")
    return cf

def plot_metrics_row(ax_list, mse_ctrl, mse_unctrl, avg_speed, control_intensity):
    """Plot metrics in bottom row."""
    time_steps = np.arange(len(mse_ctrl))
    
    # 1. MSE Tracking
    ax = ax_list[0]
    ax.plot(time_steps, mse_unctrl, color='gray', linestyle='--', lw=1.2, label='Uncontrolled')
    ax.plot(time_steps, mse_ctrl, color='navy', lw=1.5, label='Controlled')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    # CHANGED: Legend moved to center right to avoid covering data
    ax.legend(fontsize=8, loc='center right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Tracking Error", fontsize=10)

    # 2. Agent Speed
    ax = ax_list[1]
    ax.plot(time_steps, avg_speed, color='forestgreen', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(r'Avg Velocity $|v|$')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Agent Agility", fontsize=10)

    # 3. Control Intensity
    ax = ax_list[2]
    ax.plot(time_steps, control_intensity, color='purple', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg Input $|u|$')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Control Effort", fontsize=10)

def create_paper_figure(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
                        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
                        save_name="fkpp1d_paper_figure.pdf"):
    setup_paper_style()
    
    # Ensure paper directory exists
    output_dir = Path("figures/images/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / save_name
    
    fig = plt.figure(figsize=(8, 7.5)) # Slightly taller for better spacing
    
    # CHANGED: Increased hspace from 0.3 to 0.5 to prevent overlap between
    # the "Space (x)" label of the middle row and the titles of the bottom row.
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6], width_ratios=[1, 0.05],
                           hspace=0.5, wspace=0.05)
    
    v_max = max(jnp.max(z_traj_ctrl), jnp.max(z_traj_unctrl))
    v_min = min(jnp.min(z_traj_ctrl), jnp.min(z_traj_unctrl))
    
    # Row 1: Natural
    ax_nat = fig.add_subplot(gs[0, 0])
    cf = plot_spacetime_heatmap(ax_nat, z_traj_unctrl, None, "Natural Evolution", 
                                v_min, v_max, show_agents=False)
    
    # Colorbar
    cax = fig.add_subplot(gs[0:2, 1])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    
    # Row 2: Controlled
    ax_ctrl = fig.add_subplot(gs[1, 0])
    plot_spacetime_heatmap(ax_ctrl, z_traj_ctrl, xi_traj_ctrl, "Controlled Evolution (DPC)", 
                           v_min, v_max, show_agents=True)
    
    # Row 3: Metrics
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 0], 
                                                  wspace=0.4)
    ax_m1 = fig.add_subplot(gs_metrics[0, 0])
    ax_m2 = fig.add_subplot(gs_metrics[0, 1])
    ax_m3 = fig.add_subplot(gs_metrics[0, 2])
    
    plot_metrics_row([ax_m1, ax_m2, ax_m3], mse_ctrl, mse_unctrl, avg_speed, control_intensity)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved paper figure to {save_path}")
    
    png_path = save_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to {png_path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FKPP 1D - Paper Figure Generation")
    print("=" * 60)
    
    # 1. Setup Model
    model = DecentralizedControlNet(features=(64, 64))
    
    # 2. Setup Dynamics Wrappers (Native JAX)
    # Controlled Dynamics (uses loaded model)
    dynamics_ctrl = PDEDynamics(policy_apply_fn=model.apply)
    # Natural Dynamics (uses Zero Policy)
    dynamics_nat = PDEDynamics(policy_apply_fn=zero_policy_apply)

    # 3. Load Parameters
    try:
        print(f"\nLoading parameters from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
        print("✓ Parameters loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # 4. Generate Scenario
    print("\nGenerating test scenario...")
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    z_init, z_target, xi_init = get_initial_conditions(init_key)
    
    # 5. Run Simulations
    print(f"\nRunning controlled trajectory ({CONFIG['T_steps']} steps)...")
    # PDEDynamics handles the key splitting and rollout internally
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = dynamics_ctrl.unroll_controlled(
        z_init, xi_init, z_target, params, CONFIG['T_steps']
    )
    
    print(f"Running natural (uncontrolled) trajectory ({CONFIG['T_steps']} steps)...")
    # Pass dummy params for zero policy (it ignores them anyway)
    z_traj_unctrl, xi_traj_unctrl, _, _ = dynamics_nat.unroll_controlled(
        z_init, xi_init, z_target, params, CONFIG['T_steps']
    )
    
    # 6. Compute Metrics
    print("\nComputing metrics...")
    mse_ctrl = jnp.mean((z_traj_ctrl - z_target[None, :])**2, axis=1)
    mse_unctrl = jnp.mean((z_traj_unctrl - z_target[None, :])**2, axis=1)
    avg_speed = jnp.mean(jnp.abs(v_traj_ctrl), axis=1)
    control_intensity = jnp.mean(jnp.abs(u_traj_ctrl), axis=1)
    
    print(f"  Final MSE (Controlled):   {float(mse_ctrl[-1]):.6f}")
    print(f"  Final MSE (Uncontrolled): {float(mse_unctrl[-1]):.6f}")
    
    # 7. Generate Figure
    print("\nGenerating paper figure...")
    create_paper_figure(
        z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
        mse_ctrl, mse_unctrl, avg_speed, control_intensity, 
        save_name="fkpp1d_paper_figure.pdf"
    )
        
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)