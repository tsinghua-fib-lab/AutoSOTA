"""
Paper-Quality Visualization for 2D Heat Equation with Obstacles.
Layout:
- Top Row (Side-by-Side): Natural Evolution (Left) | Controlled Evolution (Right)
- Bottom Row: Metrics (MSE, Agent Speed, Control Intensity)
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

# Direct imports
from tesseracts.solverHeat2D_decentralized import solver
from models.policy import DecentralizedHeat2DControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Obstacle configuration: [x_center, y_center, radius]
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   
    [0.50, 0.50, 0.06],   
    [0.70, 0.70, 0.06],   
])

CONFIG = {
    'n_grid': 32,
    'n_agents': 16,
    'T_steps': 300,
    'snapshot_times': [0, 30, 80, 150, 299],  
    'params_file': 'decentralized_params_heat2d_obstacles.msgpack'
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
        "savefig.pad_inches": 0.02,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_obstacles(ax, obstacles):
    for obs in obstacles:
        x, y, r = float(obs[0]), float(obs[1]), float(obs[2])
        circle = Circle((x, y), r, color='#FF8C00', alpha=0.80,
                        edgecolor='black', linewidth=0.8, zorder=15)
        ax.add_patch(circle)

def load_params(model, filepath):
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
    _, k1, k2 = jax.random.split(key, 3)
    xx, yy, z_init = data_utils.generate_grf_2d(k1, n_points=CONFIG['n_grid'])
    _, _, z_target = data_utils.generate_grf_2d(k2, n_points=CONFIG['n_grid'])
    
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
    ax_list, z_traj, xi_traj, timesteps, v_lim, show_agents=False, z_ref=None,
):
    N = CONFIG['n_grid']
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    levels = np.linspace(-v_lim, v_lim, 100)
    
    ref_levels = np.linspace(-v_lim, v_lim, 11)
    ref_levels = ref_levels[np.abs(ref_levels) >= 0.02 * v_lim]
    if ref_levels.size < 4:
        ref_levels = np.linspace(-v_lim, v_lim, 11)
        
    norm = mcolors.Normalize(vmin=-v_lim, vmax=v_lim)
    z_ref_np = np.array(z_ref) if z_ref is not None else None
    
    cf = None
    for i, t in enumerate(timesteps):
        ax = ax_list[i]
        cf = ax.contourf(X, Y, np.array(z_traj[t]), levels=levels, cmap=cmc.vik, extend='both')

        if z_ref_np is not None:
            cs = ax.contour(X, Y, z_ref_np, levels=ref_levels, cmap=cmc.vik, norm=norm,
                            linestyles='--', linewidths=0.8, alpha=1.0, zorder=6)
            shadow = pe.SimpleLineShadow(offset=(0.4, -0.4), shadow_color='black', alpha=0.15) if hasattr(pe, "SimpleLineShadow") else pe.Stroke(linewidth=1.1, foreground='black', alpha=0.15)
            cs.set_path_effects([shadow, pe.Normal()])
        
        draw_obstacles(ax, OBSTACLES)
        
        if show_agents and xi_traj is not None:
            xi = xi_traj[t]
            ax.scatter(xi[:, 0], xi[:, 1], c='#E74C3C', s=16, edgecolors='white', linewidths=0.8, zorder=10)
        
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        ax.set_xlabel(f"t={t}", fontsize=11)
    
    return cf

def plot_metrics_row(ax_list, mse_ctrl, mse_unctrl, avg_speed, control_intensity):
    time_steps = np.arange(len(mse_ctrl))
    
    ax_list[0].plot(time_steps, mse_unctrl, color='gray', linestyle='--', lw=1.2, label='Uncontrolled')
    ax_list[0].plot(time_steps, mse_ctrl, color='navy', lw=1.5, label='Controlled')
    ax_list[0].set_xlabel('Time Step')
    ax_list[0].set_ylabel('MSE')
    ax_list[0].set_yscale('log')
    ax_list[0].legend(fontsize=8, loc='upper right', framealpha=0.9)
    
    ax_list[1].plot(time_steps, avg_speed, color='forestgreen', lw=1.5)
    ax_list[1].set_xlabel('Time Step')
    ax_list[1].set_ylabel(r'Avg $\dot{\boldsymbol{\xi}}$')
    ax_list[1].set_ylim(bottom=0)
    
    ax_list[2].plot(time_steps, control_intensity, color='purple', lw=1.5)
    ax_list[2].set_xlabel('Time Step')
    ax_list[2].set_ylabel('Avg |u|')
    ax_list[2].set_ylim(bottom=0)
    
    for ax in ax_list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN FIGURE / GIF GENERATION (HORIZONTAL ALIGNMENT)
# ═══════════════════════════════════════════════════════════════════════════════

def create_paper_figure(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
                        mse_ctrl, mse_unctrl, avg_speed, control_intensity,
                        save_name="figures/images/paper_gif/heat2d_paper_figure.pdf"):
    setup_paper_style()
    n_snaps = len(CONFIG['snapshot_times'])
    
    # Wider figure for side-by-side presentation
    fig = plt.figure(figsize=(14, 4.5)) 
    
    # Main grid: 2 rows (1 for images, 1 for metrics)
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 0.7], hspace=0.35)
    
    # Upper Row: Natural(5) | Spacer | Controlled(5) | Colorbar
    width_ratios = [1]*n_snaps + [0.2] + [1]*n_snaps + [0.1]
    gs_upper = gridspec.GridSpecFromSubplotSpec(1, n_snaps*2 + 2, subplot_spec=gs_main[0], 
                                                width_ratios=width_ratios, wspace=0.05)
    
    ax_natural = [fig.add_subplot(gs_upper[0, j]) for j in range(n_snaps)]
    ax_ctrl = [fig.add_subplot(gs_upper[0, j + n_snaps + 1]) for j in range(n_snaps)]
    
    v_lim = max(float(jnp.max(jnp.abs(z_traj_ctrl))), float(jnp.max(jnp.abs(z_traj_unctrl)))) * 0.9
    
    cf = plot_snapshot_row(ax_natural, z_traj_unctrl, None, CONFIG['snapshot_times'], v_lim, False, None)
    plot_snapshot_row(ax_ctrl, z_traj_ctrl, xi_traj_ctrl, CONFIG['snapshot_times'], v_lim, True, z_target)
    
    # Centered Titles above blocks
    ax_natural[n_snaps // 2].set_title("Natural Evolution (Uncontrolled)", pad=12)
    ax_ctrl[n_snaps // 2].set_title("Controlled Evolution (DPC)", pad=12)
    
    cax = fig.add_subplot(gs_upper[0, -1])
    cbar = fig.colorbar(cf, cax=cax, format='%.2f')
    cbar.ax.tick_params(labelsize=10)
    
    # Lower Row: 3 metrics spread evenly
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.25)
    ax_metrics = [fig.add_subplot(gs_metrics[0, j]) for j in range(3)]
    plot_metrics_row(ax_metrics, mse_ctrl, mse_unctrl, avg_speed, control_intensity)
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved static paper figure to {save_name}")
    plt.savefig(save_name.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def create_gif(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target,
               mse_ctrl, mse_unctrl, avg_speed, control_intensity,
               save_name="figures/images/paper/heat2d_paper_animation.gif", skip_frames=3):
    setup_paper_style()
    
    # Adjusted figure size for side-by-side animation
    fig = plt.figure(figsize=(9, 4.5)) 
    
    # Main grid: 2 rows (images, metrics)
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 0.7], hspace=0.4)
    
    # Upper Row: Natural | Controlled | Colorbar
    gs_upper = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], 
                                                width_ratios=[1, 1, 0.05], wspace=0.15)
    ax_unctrl = fig.add_subplot(gs_upper[0, 0])
    ax_ctrl = fig.add_subplot(gs_upper[0, 1])
    cax = fig.add_subplot(gs_upper[0, 2])
    
    # Lower Row: 3 metrics
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.35)
    ax_m1 = fig.add_subplot(gs_metrics[0, 0])
    ax_m2 = fig.add_subplot(gs_metrics[0, 1])
    ax_m3 = fig.add_subplot(gs_metrics[0, 2])
    
    N, T = CONFIG['n_grid'], len(mse_ctrl)
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    
    v_lim = max(float(jnp.max(jnp.abs(z_traj_ctrl))), float(jnp.max(jnp.abs(z_traj_unctrl)))) * 0.9
    levels = np.linspace(-v_lim, v_lim, 100)
    
    ref_levels = np.linspace(-v_lim, v_lim, 11)
    ref_levels = ref_levels[np.abs(ref_levels) >= 0.02 * v_lim]
    if ref_levels.size < 4: ref_levels = np.linspace(-v_lim, v_lim, 11)
        
    z_ref_np = np.array(z_target)
    norm = mcolors.Normalize(vmin=-v_lim, vmax=v_lim)
    
    # Initial colorbar hook
    cf = ax_unctrl.contourf(X, Y, np.array(z_traj_unctrl[0]), levels=levels, cmap=cmc.vik, extend='both')
    cbar = fig.colorbar(cf, cax=cax, format='%.2f')
    
    time_steps = np.arange(T)
    mse_min, mse_max = float(np.min([np.min(mse_ctrl), np.min(mse_unctrl)])), float(np.max([np.max(mse_ctrl), np.max(mse_unctrl)]))
    speed_max, ctrl_max = float(np.max(avg_speed)), float(np.max(control_intensity))
    
    frames = np.arange(0, T, skip_frames)
    if frames[-1] != T - 1: frames = np.append(frames, T - 1)
        
    def update(frame):
        ax_unctrl.clear(); ax_ctrl.clear(); ax_m1.clear(); ax_m2.clear(); ax_m3.clear()
        
        # --- Left: Natural Evolution ---
        ax_unctrl.contourf(X, Y, np.array(z_traj_unctrl[frame]), levels=levels, cmap=cmc.vik, extend='both')
        draw_obstacles(ax_unctrl, OBSTACLES)
        ax_unctrl.set_title(f"Natural Evolution (t={frame})", pad=8)
        ax_unctrl.set_xticks([]); ax_unctrl.set_yticks([]); ax_unctrl.set_aspect('equal')
        
        # --- Right: Controlled Evolution ---
        ax_ctrl.contourf(X, Y, np.array(z_traj_ctrl[frame]), levels=levels, cmap=cmc.vik, extend='both')
        cs = ax_ctrl.contour(X, Y, z_ref_np, levels=ref_levels, cmap=cmc.vik, norm=norm, linestyles='--', linewidths=0.8)
        shadow = pe.SimpleLineShadow(offset=(0.4, -0.4), shadow_color='black', alpha=0.15) if hasattr(pe, "SimpleLineShadow") else pe.Stroke(linewidth=1.1, foreground='black', alpha=0.15)
        cs.set_path_effects([shadow, pe.Normal()])
        draw_obstacles(ax_ctrl, OBSTACLES)
        
        xi = xi_traj_ctrl[frame]
        ax_ctrl.scatter(xi[:, 0], xi[:, 1], c='#E74C3C', s=16, edgecolors='white', linewidths=0.8, zorder=10)
        ax_ctrl.set_title(f"Controlled Evolution (t={frame})", pad=8)
        ax_ctrl.set_xticks([]); ax_ctrl.set_yticks([]); ax_ctrl.set_aspect('equal')
        
        for ax in [ax_unctrl, ax_ctrl]:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)
                
        # --- Metrics Row ---
        curr_t = time_steps[:frame+1]
        
        ax_m1.plot(time_steps, mse_unctrl, color='gray', linestyle='--', lw=1.2, alpha=0.3)
        ax_m1.plot(time_steps, mse_ctrl, color='navy', lw=1.5, alpha=0.3)
        ax_m1.plot(curr_t, mse_unctrl[:frame+1], color='gray', linestyle='--', lw=1.5)
        ax_m1.plot(curr_t, mse_ctrl[:frame+1], color='navy', lw=2.0)
        ax_m1.set_yscale('log'); ax_m1.set_xlabel('Time'); ax_m1.set_ylabel('MSE')
        ax_m1.set_xlim(0, T); ax_m1.set_ylim(max(1e-5, mse_min * 0.5), mse_max * 1.5)
            
        ax_m2.plot(time_steps, avg_speed, color='forestgreen', lw=1.5, alpha=0.3)
        ax_m2.plot(curr_t, avg_speed[:frame+1], color='forestgreen', lw=2.0)
        ax_m2.set_xlabel('Time'); ax_m2.set_ylabel(r'Avg $\dot{\boldsymbol{\xi}}$')
        ax_m2.set_xlim(0, T); ax_m2.set_ylim(0, speed_max * 1.1)
        
        ax_m3.plot(time_steps, control_intensity, color='purple', lw=1.5, alpha=0.3)
        ax_m3.plot(curr_t, control_intensity[:frame+1], color='purple', lw=2.0)
        ax_m3.set_xlabel('Time'); ax_m3.set_ylabel('Avg |u|')
        ax_m3.set_xlim(0, T); ax_m3.set_ylim(0, ctrl_max * 1.1)
        
        for ax in [ax_m1, ax_m2, ax_m3]:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            
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
    
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    
    try:
        print(f"\nLoading parameters from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
    except Exception as e:
        print(f"Error: {e}\nRun training first to generate parameters.")
        sys.exit(1)
    
    print("\nGenerating test scenario...")
    z_init, z_target, xi_init = get_initial_conditions(jax.random.PRNGKey(1234))
    
    print(f"\nRunning controlled trajectory ({CONFIG['T_steps']} steps)...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = solver.solve_with_policy(
        z_init, xi_init, z_target, params, model.apply, CONFIG['T_steps']
    )
    
    print(f"Running uncontrolled trajectory ({CONFIG['T_steps']} steps)...")
    z_traj_unctrl, xi_traj_unctrl, _, _ = rollout_uncontrolled(z_init, xi_init, CONFIG['T_steps'])
    
    print("\nComputing metrics...")
    mse_ctrl = jnp.mean((z_traj_ctrl - z_target[None, :, :])**2, axis=(1, 2))
    mse_unctrl = jnp.mean((z_traj_unctrl - z_target[None, :, :])**2, axis=(1, 2))
    avg_speed = jnp.mean(jnp.sqrt(jnp.sum(v_traj_ctrl**2, axis=-1)), axis=1)
    control_intensity = jnp.mean(jnp.abs(u_traj_ctrl), axis=1)
    
    print(f"  Final MSE (Controlled):   {float(mse_ctrl[-1]):.6f}")
    print(f"  Final MSE (Uncontrolled): {float(mse_unctrl[-1]):.6f}")
    
    print("\nGenerating static paper figure...")
    create_paper_figure(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target, mse_ctrl, mse_unctrl, avg_speed, control_intensity)
    
    print("\nGenerating animation (GIF)...")
    create_gif(z_traj_ctrl, z_traj_unctrl, xi_traj_ctrl, z_target, mse_ctrl, mse_unctrl, avg_speed, control_intensity)
    
    print("\n" + "=" * 60 + "\nDone!\n" + "=" * 60)