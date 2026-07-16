"""
Paper-Quality Visualization for 2D Heat Equation with Obstacles.
Three rows:
- Top Row: Natural Evolution (Uncontrolled)
- Middle Row: Controlled Evolution (DPC)
- Bottom Row: Metrics (MSE, Agent Speed, Control Intensity)

Same style as ks2d_correct: contourf, cmcrameri colormaps, Times New Roman fonts.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
import matplotlib.animation as animation
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

# Direct imports - no tesseract needed
from tesseracts.solverHeat2D_decentralized import solver
from models.policy import DecentralizedHeat2DControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Obstacle configuration: [x_center, y_center, radius]
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   # Diagonal line obstacle 1
    [0.50, 0.50, 0.06],   # Diagonal line obstacle 2 (center)
    [0.70, 0.70, 0.06],   # Diagonal line obstacle 3
])

CONFIG = {
    'n_grid': 32,
    'n_agents': 16,
    'T_steps': 300,
    
    # Time snapshots to display (indices)
    'snapshot_times': [0, 30, 80, 150, 299],  # 5 snapshots
    
    'params_file': 'decentralized_params_heat2d_obstacles.msgpack'
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

def draw_obstacles(ax, obstacles):
    """Draw circular obstacles on axis."""
    for obs in obstacles:
        x, y, r = float(obs[0]), float(obs[1]), float(obs[2])
        circle = Circle((x, y), r, color='#FF8C00', alpha=0.80,
                        edgecolor='black', linewidth=0.8, zorder=15)
        ax.add_patch(circle)

def load_params(model, filepath):
    """Load trained parameters from msgpack file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
    
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((CONFIG['n_grid'], CONFIG['n_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def rollout_uncontrolled(z_init, xi_init, T_steps):
    """Rollout with zero control inputs."""
    from tesseracts.solverHeat2D_decentralized import solver

    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_zero = jnp.zeros(xi_curr.shape[0])
        v_zero = jnp.zeros_like(xi_curr)
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
        
        # Draw obstacles
        draw_obstacles(ax, OBSTACLES)
        
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
# 5. MAIN FIGURE / GIF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_paper_figure(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
                        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
                        save_name="figures/images/paper/heat2d_paper_figure.pdf"):
    """
    Create the static paper-quality figure.
    """
    setup_paper_style()
    
    n_snaps = len(CONFIG['snapshot_times'])
    fig = plt.figure(figsize=(7.5, 5.2))  
    
    gs_main = gridspec.GridSpec(3, n_snaps + 1, 
                                 width_ratios=[1]*n_snaps + [0.08],
                                 height_ratios=[1, 1, 0.8],
                                 hspace=0.35, wspace=0.03)  
    
    ax_row1 = [fig.add_subplot(gs_main[0, j]) for j in range(n_snaps)]
    ax_row2 = [fig.add_subplot(gs_main[1, j]) for j in range(n_snaps)]
    
    v_lim = max(float(jnp.max(jnp.abs(z_traj_ctrl))),
                float(jnp.max(jnp.abs(z_traj_unctrl)))) * 0.9
    
    cf = plot_snapshot_row(
        ax_row1, z_traj_unctrl, None, CONFIG['snapshot_times'],
        v_lim, show_agents=False, z_ref=None,
    )
    ax_row1[2].set_title("Natural Evolution", pad=4)
    
    plot_snapshot_row(
        ax_row2, z_traj_ctrl, xi_traj_ctrl, CONFIG['snapshot_times'],
        v_lim, show_agents=True, z_ref=z_target,
    )
    ax_row2[2].set_title("Controlled Evolution", pad=4)
    
    cax = fig.add_subplot(gs_main[0:2, -1])
    cbar = fig.colorbar(cf, cax=cax, format='%.2f')  
    cbar.ax.tick_params(labelsize=10)
    
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2, :], 
                                                   wspace=0.6, width_ratios=[0.15, 1, 1, 1])
    ax_metrics = [fig.add_subplot(gs_metrics[0, j]) for j in range(1, 4)] 
    
    plot_metrics_row(ax_metrics, mse_ctrl, mse_unctrl, avg_speed, control_intensity)
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved static paper figure to {save_name}")
    
    png_name = save_name.replace('.pdf', '.png')
    plt.savefig(png_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved static PNG version to {png_name}")
    plt.close()

def create_gif(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
               mse_ctrl, mse_unctrl, avg_speed, control_intensity,
               save_name="figures/images/paper/heat2d_paper_animation.gif",
               skip_frames=3):
    """
    Create a paper-quality animated GIF replicating the structure of the static figure.
    """
    setup_paper_style()
    # Taller narrower figure for single-column animation
    fig = plt.figure(figsize=(6.5, 7.5)) 
    
    # 3 rows x 2 cols layout
    gs_main = gridspec.GridSpec(3, 2, 
                                 width_ratios=[1, 0.05],
                                 height_ratios=[1, 1, 0.8],
                                 hspace=0.35, wspace=0.1)
    
    ax_unctrl = fig.add_subplot(gs_main[0, 0])
    ax_ctrl = fig.add_subplot(gs_main[1, 0])
    cax = fig.add_subplot(gs_main[0:2, 1])
    
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[2, :], wspace=0.4)
    ax_m1 = fig.add_subplot(gs_metrics[0, 0])
    ax_m2 = fig.add_subplot(gs_metrics[0, 1])
    ax_m3 = fig.add_subplot(gs_metrics[0, 2])
    
    N = CONFIG['n_grid']
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    v_lim = max(float(jnp.max(jnp.abs(z_traj_ctrl))),
                float(jnp.max(jnp.abs(z_traj_unctrl)))) * 0.9
    levels = np.linspace(-v_lim, v_lim, 100)
    
    ref_levels = np.linspace(-v_lim, v_lim, 11)
    zero_gap = 0.02 * v_lim
    ref_levels = ref_levels[np.abs(ref_levels) >= zero_gap]
    if ref_levels.size < 4:
        ref_levels = np.linspace(-v_lim, v_lim, 11)
        
    z_ref_np = np.array(z_target)
    norm = mcolors.Normalize(vmin=-v_lim, vmax=v_lim)
    
    # Initial render for colorbar hook
    cf = ax_unctrl.contourf(X, Y, np.array(z_traj_unctrl[0]), levels=levels, cmap=cmc.vik, extend='both')
    cbar = fig.colorbar(cf, cax=cax, format='%.2f')
    cbar.ax.tick_params(labelsize=10)
    
    time_steps = np.arange(len(mse_ctrl))
    T = len(time_steps)
    
    # Calculate limits so the axes stay fixed and don't jitter during the animation
    mse_min = float(np.min([np.min(mse_ctrl), np.min(mse_unctrl)]))
    mse_max = float(np.max([np.max(mse_ctrl), np.max(mse_unctrl)]))
    speed_max = float(np.max(avg_speed))
    ctrl_max = float(np.max(control_intensity))
    
    frames = np.arange(0, T, skip_frames)
    if frames[-1] != T - 1:
        frames = np.append(frames, T - 1)
        
    def update(frame):
        # Clear specific subplots explicitly
        ax_unctrl.clear()
        ax_ctrl.clear()
        ax_m1.clear()
        ax_m2.clear()
        ax_m3.clear()
        
        # --- Row 1: Natural Evolution (Uncontrolled) ---
        ax_unctrl.contourf(X, Y, np.array(z_traj_unctrl[frame]), levels=levels, cmap=cmc.vik, extend='both')
        draw_obstacles(ax_unctrl, OBSTACLES)
        ax_unctrl.set_title(f"Natural Evolution (t={frame})", pad=4)
        ax_unctrl.set_xticks([])
        ax_unctrl.set_yticks([])
        ax_unctrl.set_aspect('equal')
        
        # --- Row 2: Controlled Evolution (DPC) ---
        ax_ctrl.contourf(X, Y, np.array(z_traj_ctrl[frame]), levels=levels, cmap=cmc.vik, extend='both')
        cs = ax_ctrl.contour(X, Y, z_ref_np, levels=ref_levels, cmap=cmc.vik, norm=norm, linestyles='--', linewidths=0.8, alpha=1.0, zorder=6)
        shadow_effect = pe.SimpleLineShadow(offset=(0.4, -0.4), shadow_color='black', alpha=0.15) if hasattr(pe, "SimpleLineShadow") else pe.Stroke(linewidth=0.8 + 0.3, foreground='black', alpha=0.15)
        cs.set_path_effects([shadow_effect, pe.Normal()])
        
        draw_obstacles(ax_ctrl, OBSTACLES)
        xi = xi_traj_ctrl[frame]
        ax_ctrl.scatter(xi[:, 0], xi[:, 1], c='#E74C3C', s=16, edgecolors='white', linewidths=0.8, zorder=10, alpha=1.0)
        ax_ctrl.set_title(f"Controlled Evolution (t={frame})", pad=4)
        ax_ctrl.set_xticks([])
        ax_ctrl.set_yticks([])
        ax_ctrl.set_aspect('equal')
        
        # Reset plot borders after clearing
        for ax in [ax_unctrl, ax_ctrl]:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)
                
        # --- Row 3: Metrics Evolution ---
        curr_t = time_steps[:frame+1]
        
        # Plot 1: MSE
        # Render a faint outline of the entire trajectory as background
        ax_m1.plot(time_steps, mse_unctrl, color='gray', linestyle='--', lw=1.2, alpha=0.3)
        ax_m1.plot(time_steps, mse_ctrl, color='navy', lw=1.5, alpha=0.3)
        # Render bold trajectory up to current frame
        ax_m1.plot(curr_t, mse_unctrl[:frame+1], color='gray', linestyle='--', lw=1.5, label='Uncontrolled')
        ax_m1.plot(curr_t, mse_ctrl[:frame+1], color='navy', lw=2.0, label='Controlled')
        ax_m1.set_yscale('log')
        ax_m1.set_xlabel('Time Step')
        ax_m1.set_ylabel('MSE')
        ax_m1.set_xlim(0, T)
        ax_m1.set_ylim(max(1e-5, mse_min * 0.5), mse_max * 1.5)
        ax_m1.legend(fontsize=8, loc='upper right', framealpha=0.9)
            
        # Plot 2: Average Speed
        ax_m2.plot(time_steps, avg_speed, color='forestgreen', lw=1.5, alpha=0.3)
        ax_m2.plot(curr_t, avg_speed[:frame+1], color='forestgreen', lw=2.0)
        ax_m2.set_xlabel('Time Step')
        ax_m2.set_ylabel(r'Avg $\dot{\boldsymbol{\xi}}$')
        ax_m2.set_xlim(0, T)
        ax_m2.set_ylim(0, speed_max * 1.1)
        
        # Plot 3: Control Intensity
        ax_m3.plot(time_steps, control_intensity, color='purple', lw=1.5, alpha=0.3)
        ax_m3.plot(curr_t, control_intensity[:frame+1], color='purple', lw=2.0)
        ax_m3.set_xlabel('Time Step')
        ax_m3.set_ylabel('Avg |u|')
        ax_m3.set_xlim(0, T)
        ax_m3.set_ylim(0, ctrl_max * 1.1)
        
        for ax in [ax_m1, ax_m2, ax_m3]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
    # Compile animation 
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=80)
    anim.save(save_name, writer='pillow', dpi=150)
    print(f"✓ Saved animation GIF to {save_name}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Heat2D Obstacles - Paper Figure Generation")
    print("=" * 60)
    output_dir = Path("figures/images/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # 1. Initialize Model
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    
    # 2. Load Parameters
    try:
        print(f"\nLoading parameters from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
        print("✓ Parameters loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        print("Run training first to generate parameters.")
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
    
    # 6. Generate Static Figure
    print("\nGenerating static paper figure...")
    create_paper_figure(
        z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
        save_name="figures/images/paper/heat2d_paper_figure.pdf"
    )
    
    # 7. Generate GIF Animation
    print("\nGenerating animation (GIF)...")
    create_gif(
        z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
        save_name="figures/images/paper/heat2d_paper_animation.gif",
        skip_frames=3  # Skip a few frames to make compilation faster & output lighter
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)