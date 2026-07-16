"""
Paper-Quality Visualization for 2D Heat Equation (No Obstacles).
Three rows:
- Top Row: Natural Evolution (Uncontrolled)
- Middle Row: Controlled Evolution (DPC)
- Bottom Row: Metrics (MSE, Agent Speed, Control Intensity)

Style: contourf, cmcrameri colormaps, Times New Roman fonts.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import sys
import flax.serialization
from pathlib import Path
import cmcrameri.cm as cmc  # Crameri scientific colormaps

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Direct imports
from tesseracts.solverHeat2D_decentralized import solver
from models.policy import DecentralizedHeat2DControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'n_grid': 32,
    'n_agents': 16,
    'T_steps': 300,
    
    # Time snapshots to display (indices)
    'snapshot_times': [0, 30, 80, 150, 299],  # 5 snapshots
    
    # UPDATED: Matches the training script output file
    'params_file': 'decentralized_params_heat2d.msgpack' 
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings - Times New Roman for papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        
        # Font sizes for two-column paper
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        
        # Line widths
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        
        # Spines
        "axes.spines.top": True,
        "axes.spines.right": True,
        
        # High-quality output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath):
    """Load trained parameters from msgpack file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found. Run the training script first.")
    
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((CONFIG['n_grid'], CONFIG['n_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def rollout_uncontrolled(z_init, xi_init, T_steps):
    """Rollout with zero control inputs."""

    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_zero = jnp.zeros(xi_curr.shape[0])
        v_zero = jnp.zeros_like(xi_curr)
        # Using solver for pure physics step
        z_next, xi_next = solver.adi_step(z_curr, xi_curr, u_zero, v_zero)
        return (z_next, xi_next), (z_next, xi_next, u_zero, v_zero)

    _, (z_traj, xi_traj, u_traj, v_traj) = jax.lax.scan(
        step_fn, (z_init, xi_init), None, length=T_steps
    )
    return z_traj, xi_traj, u_traj, v_traj

def get_initial_conditions(key):
    """Generate initial and target temperature fields."""
    _, k1, k2 = jax.random.split(key, 3)
    
    xx, yy, z_init = data_utils.generate_grf_2d(k1, n_points=CONFIG['n_grid'])
    _, _, z_target = data_utils.generate_grf_2d(k2, n_points=CONFIG['n_grid'])
    
    # Initialize agents in grid pattern at exact positions [0.2, 0.4, 0.6, 0.8]
    n_side = int(jnp.sqrt(CONFIG['n_agents']))
    positions_1d = jnp.array([0.2, 0.4, 0.6, 0.8])[:n_side]
    xi_init = []
    for i in range(n_side):
        for j in range(n_side):
            if len(xi_init) < CONFIG['n_agents']:
                xi_init.append([float(positions_1d[i]), float(positions_1d[j])])
    xi_init = jnp.array(xi_init)
    
    return z_init, z_target, xi_init

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_snapshot_row(
    ax_list,
    z_traj,
    xi_traj,
    timesteps,
    v_lim,
    show_agents=False,
    z_ref=None,
    ref_alphas=None,
    ref_linewidths=None,
):
    """Plot a row of temperature field snapshots using contourf."""
    N = CONFIG['n_grid']
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(-v_lim, v_lim, 100)
    ref_levels = np.linspace(-v_lim, v_lim, 11)
    zero_gap = 0.02 * v_lim
    ref_levels = ref_levels[np.abs(ref_levels) >= zero_gap]
    if ref_levels.size < 4:
        ref_levels = np.linspace(-v_lim, v_lim, 11)
    z_ref_np = np.array(z_ref) if z_ref is not None else None
    norm = mcolors.Normalize(vmin=-v_lim, vmax=v_lim)
    
    cf = None
    for i, t in enumerate(timesteps):
        ax = ax_list[i]
        z_snap = np.array(z_traj[t])
        
        cf = ax.contourf(
            X, Y, z_snap,
            levels=levels,
            cmap=cmc.vik,  # Crameri diverging colormap
            extend='both'
        )

        # Reference target contour (dashed, lightly shaded)
        if z_ref_np is not None:
            alpha = ref_alphas[i] if ref_alphas is not None else 1.0
            lw = ref_linewidths[i] if ref_linewidths is not None else 0.8
            cs = ax.contour(
                X, Y, z_ref_np,
                levels=ref_levels,
                cmap=cmc.vik,
                norm=norm,
                linestyles='--',
                linewidths=lw,
                alpha=alpha,
                zorder=6
            )
            shadow_effect = (
                pe.SimpleLineShadow(offset=(0.4, -0.4), shadow_color='black', alpha=0.15)
                if hasattr(pe, "SimpleLineShadow")
                else pe.Stroke(linewidth=lw + 0.3, foreground='black', alpha=0.15)
            )
            cs.set_path_effects([shadow_effect, pe.Normal()])
        
        # Optionally show agent positions
        if show_agents and xi_traj is not None:
            xi = xi_traj[t]
            ax.scatter(xi[:, 0], xi[:, 1], c='#E74C3C', s=16,
                       edgecolors='white', linewidths=0.8, zorder=10, alpha=1.0)
        
        # Set axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        
        # Time label below
        ax.set_xlabel(f"t={t}", fontsize=11)
    
    return cf

def plot_metrics_row(ax_list, mse_ctrl, mse_unctrl, avg_speed, control_intensity):
    """Plot the three metrics subplots with clean styling."""
    time_steps = np.arange(len(mse_ctrl))
    
    # MSE Tracking Error
    ax = ax_list[0]
    ax.plot(time_steps, mse_unctrl, color='gray', linestyle='--', lw=1.2, 
            label='Uncontrolled')
    ax.plot(time_steps, mse_ctrl, color='navy', lw=1.5, label='Controlled')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='center right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Agent Speed
    ax = ax_list[1]
    ax.plot(time_steps, avg_speed, color='forestgreen', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(r'Avg $\dot{\boldsymbol{\xi}}$')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Control Intensity
    ax = ax_list[2]
    ax.plot(time_steps, control_intensity, color='purple', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg |u|')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_paper_figure(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
                        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
                        save_name="heat2d_no_obs_paper_figure.pdf"):
    """
    Create the paper-quality figure with:
    - Row 1: Natural Evolution (Uncontrolled)
    - Row 2: Controlled Evolution (DPC)
    - Row 3: Metrics (MSE, Agent Speed, Control Intensity)
    """
    setup_paper_style()
    
    n_snaps = len(CONFIG['snapshot_times'])
    fig = plt.figure(figsize=(7.5, 5.2))
    
    # Main Layout: 3 rows
    # Rows 1-2: snapshots with colorbar
    # Row 3: 3 metric plots
    gs_main = gridspec.GridSpec(3, n_snaps + 1, 
                                 width_ratios=[1]*n_snaps + [0.08],
                                 height_ratios=[1, 1, 0.8],
                                 hspace=0.35, wspace=0.03)
    
    # Row 1: Natural Evolution (top)
    ax_row1 = [fig.add_subplot(gs_main[0, j]) for j in range(n_snaps)]
    
    # Row 2: Controlled Evolution
    ax_row2 = [fig.add_subplot(gs_main[1, j]) for j in range(n_snaps)]
    
    # Determine color limits
    v_lim = max(float(jnp.max(jnp.abs(z_traj_ctrl))),
                float(jnp.max(jnp.abs(z_traj_unctrl)))) * 0.9
    
    # Plot Natural Evolution (top row) - no reference contour
    cf = plot_snapshot_row(
        ax_row1,
        z_traj_unctrl,
        None,
        CONFIG['snapshot_times'],
        v_lim,
        show_agents=False,
        z_ref=None,
    )
    ax_row1[2].set_title("Natural Evolution", pad=4)
    
    # Plot Controlled Evolution (middle row)
    plot_snapshot_row(
        ax_row2,
        z_traj_ctrl,
        xi_traj_ctrl,
        CONFIG['snapshot_times'],
        v_lim,
        show_agents=True,
        z_ref=z_target,
    )
    ax_row2[2].set_title("Controlled Evolution", pad=4)
    
    # Add colorbar on the right (spans rows 0-1)
    cax = fig.add_subplot(gs_main[0:2, -1])
    cbar = fig.colorbar(cf, cax=cax, format='%.2f')
    cbar.ax.tick_params(labelsize=10)
    
    # Row 3: Metrics (shifted right - smaller spacer, reduced gaps)
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2, :], 
                                                  wspace=0.6, width_ratios=[0.15, 1, 1, 1])
    ax_metrics = [fig.add_subplot(gs_metrics[0, j]) for j in range(1, 4)] # Skip first column
    
    # Plot metrics
    plot_metrics_row(ax_metrics, mse_ctrl, mse_unctrl, avg_speed, control_intensity)
    
    # Save figure
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved paper figure to {save_name}")
    
    # Also save PNG version
    png_name = save_name.replace('.pdf', '.png')
    plt.savefig(png_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved PNG version to {png_name}")
    
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Heat2D (No Obstacles) - Paper Figure Generation")
    print("=" * 60)
    
    # 1. Initialize Model
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    
    # 2. Load Parameters
    try:
        print(f"\nLoading parameters from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
        print("✓ Parameters loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        print("Run training script first to generate parameters.")
        sys.exit(1)
    
    # 3. Generate Initial Conditions
    print("\nGenerating test scenario...")
    key = jax.random.PRNGKey(1234)
    z_init, z_target, xi_init = get_initial_conditions(key)
    
    # 4. Run Simulations using direct solver
    print(f"\nRunning controlled trajectory ({CONFIG['T_steps']} steps)...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = solver.solve_with_policy(
        z_init, xi_init, z_target, params, model.apply, CONFIG['T_steps']
    )
    
    print(f"Running uncontrolled trajectory ({CONFIG['T_steps']} steps)...")
    z_traj_unctrl, xi_traj_unctrl, _, _ = rollout_uncontrolled(
        z_init, xi_init, CONFIG['T_steps']
    )
    
    # 5. Compute Metrics
    print("\nComputing metrics...")
    mse_ctrl = jnp.mean((z_traj_ctrl - z_target[None, :, :])**2, axis=(1, 2))
    mse_unctrl = jnp.mean((z_traj_unctrl - z_target[None, :, :])**2, axis=(1, 2))
    
    # Agent speeds
    speeds_ctrl = jnp.sqrt(jnp.sum(v_traj_ctrl**2, axis=-1))
    avg_speed = jnp.mean(speeds_ctrl, axis=1)
    
    # Control intensity
    control_intensity = jnp.mean(jnp.abs(u_traj_ctrl), axis=1)
    
    print(f"  Final MSE (Controlled):   {float(mse_ctrl[-1]):.6f}")
    print(f"  Final MSE (Uncontrolled): {float(mse_unctrl[-1]):.6f}")
    
    # 6. Generate Figure
    print("\nGenerating paper figure...")
    save_dir = Path("figures/images/paper_viz")
    save_dir.mkdir(parents=True, exist_ok=True)

    create_paper_figure(
        z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
        save_name=str(save_dir / "heat2d_no_obs_paper_figure.pdf")
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
