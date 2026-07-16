"""
Paper-Quality Visualization for 1D Heat Equation.
Three rows:
- Top Row: Natural Evolution (Space-Time Heatmap)
- Middle Row: Controlled Evolution (Space-Time Heatmap with Agent Paths)
- Bottom Row: Metrics (MSE, Agent Speed, Control Intensity)

Style: contourf (Space-Time), cmcrameri colormaps, Times New Roman fonts.
Requires: pip install cmcrameri
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

# --- Path Setup for Imports ---
# We need the project root to import 'dynamics_dual', 'models', etc.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Imports using the high-level wrappers
from dynamics_dual import PDEDynamics
from models.policy import DecentralizedControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'n_pde': 100,
    'n_agents': 8,
    'T_steps': 300,
    'params_file': 'decentralized_params.msgpack' 
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    font_family = "serif"
    font_serif = ["Times New Roman", "Times", "DejaVu Serif"]
    
    try:
        import matplotlib.font_manager
        found = matplotlib.font_manager.findfont("Times New Roman")
        if "Times New Roman" not in found:
             print("Note: 'Times New Roman' font not found. Falling back to default serif.")
             font_serif = ["DejaVu Serif"]
    except:
        pass

    plt.rcParams.update({
        "font.family": font_family,
        "font.serif": font_serif,
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
        print(f"Warning: {filepath} not found. Using random weights for demo.")
        return None
    
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((CONFIG['n_pde'],))
    dummy_xi = jnp.zeros((CONFIG['n_agents'],)) 
    init_params = model.init(key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def zero_policy_apply(params, obs_z, target_z, obs_xi):
    """Dummy policy that returns zero control for 'Natural Evolution'."""
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

def plot_spacetime_heatmap(ax, z_traj, xi_traj, title, v_min=None, v_max=None, show_agents=False):
    """Plot a Space-Time heatmap (Time on Y, Space on X)."""
    T, N = z_traj.shape
    x = np.linspace(0, 1, N)
    t = np.arange(T)
    X, Time = np.meshgrid(x, t)
    
    # Use 'lajolla' (diverging/sequential) or 'batlow' from cmcrameri
    cmap = cmc.lajolla
    
    # Contourf plot
    cf = ax.contourf(
        X, Time, z_traj,
        levels=100,
        cmap=cmap,
        vmin=v_min, vmax=v_max,
        extend='both'
    )
    
    # Overlay Agent Paths
    if show_agents and xi_traj is not None:
        # Plot trails
        ax.plot(xi_traj, t, color='white', alpha=0.4, linewidth=1.0, linestyle='-')
        # Plot starting positions (at time 0)
        ax.scatter(xi_traj[0], [0]*xi_traj.shape[1], c='white', s=5, alpha=0.9, zorder=10)
        
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
    ax.plot(time_steps, control_intensity, color='firebrick', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg Input $|u|$')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Control Effort", fontsize=10)

def create_paper_figure(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
                        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
                        save_name="heat1d_paper_figure__.pdf"):
    setup_paper_style()
    
    fig = plt.figure(figsize=(8, 7.5)) 
    
    # Grid layout: 3 rows, 2 columns (2nd col is strictly for colorbar)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6], width_ratios=[1, 0.05],
                           hspace=0.5, wspace=0.05)
    
    # Determine Color Limits
    v_max = max(jnp.max(z_traj_ctrl), jnp.max(z_traj_unctrl))
    v_min = min(jnp.min(z_traj_ctrl), jnp.min(z_traj_unctrl))
    
    # Row 1: Natural (Uncontrolled)
    ax_nat = fig.add_subplot(gs[0, 0])
    cf = plot_spacetime_heatmap(ax_nat, z_traj_unctrl, None, "Natural Evolution", 
                                v_min, v_max, show_agents=False)
    
    # Shared Colorbar (spans top two rows)
    cax = fig.add_subplot(gs[0:2, 1])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("Temperature ($z$)", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    # Row 2: Controlled (DPC)
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
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✓ Saved paper figure to {save_name}")
    
    # Handle PNG replacement robustly using pathlib
    png_path = Path(save_name).with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to {png_path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Heat 1D - Paper Figure Generation")
    print("=" * 60)
    
    # 1. Setup Model (No Tesseract Solver needed)
    model = DecentralizedControlNet(features=(64, 64))
    
    # 2. Setup Dynamics Wrappers
    dynamics_ctrl = PDEDynamics(policy_apply_fn=model.apply)
    dynamics_nat = PDEDynamics(policy_apply_fn=zero_policy_apply)

    # 3. Load Parameters
    # Define current directory for finding parameters and saving output
    current_dir = Path(__file__).resolve().parent

    params_path = current_dir / CONFIG['params_file']
    if not params_path.exists():
         # Fallback just in case
         params_path = CONFIG['params_file']

    params = load_params(model, params_path)
    
    if params is None:
        # Init random params if file missing so script still runs
        key = jax.random.PRNGKey(0)
        dummy_z = jnp.zeros((CONFIG['n_pde'],))
        dummy_xi = jnp.zeros((CONFIG['n_agents'],))
        params = model.init(key, dummy_z, dummy_z, dummy_xi)
        print("Using initialized (random) parameters for visualization test.")
    else:
        print("✓ Parameters loaded successfully")
        
    # 4. Generate Scenario
    print("\nGenerating test scenario...")
    key = jax.random.PRNGKey(1234)
    key, init_key = jax.random.split(key)
    z_init, z_target, xi_init = get_initial_conditions(init_key)
    
    # 5. Run Simulations
    print(f"\nRunning controlled trajectory ({CONFIG['T_steps']} steps)...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = dynamics_ctrl.unroll_controlled(
        z_init, xi_init, z_target, params, CONFIG['T_steps']
    )
    
    print(f"Running natural (uncontrolled) trajectory ({CONFIG['T_steps']} steps)...")
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
    
    # 7. Generate Figure with Output Folder Logic
    print("\nGenerating paper figure...")

    # --- SAVE LOGIC START ---
    # Define output directory RELATIVE TO THIS SCRIPT
    # This puts the folder in the same directory where this .py file lives
    output_dir = current_dir / "figures" / "images" / "paper_viz"

    # Create directory if it doesn't exist (parents=True creates intermediate dirs)
    print(f"Ensuring output directory exists: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define full path for the PDF filename
    full_save_path = output_dir / "heat1d_paper_figure_final.pdf"
    # --- SAVE LOGIC END ---

    create_paper_figure(
        z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
        save_name=str(full_save_path) # Pass the full path string
    )
    
print("\n" + "=" * 60)
print("Done!")
print("=" * 60)