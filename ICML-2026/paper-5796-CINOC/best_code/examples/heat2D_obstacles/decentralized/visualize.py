"""
2D Heat Equation Control with Obstacles - Decentralized Visualization
Creates publication-quality figures comparing controlled vs uncontrolled evolution
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
import sys
from pathlib import Path
import flax.serialization
import numpy as np

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
import data_utils

# Obstacle configuration: [x_center, y_center, radius]
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   # Diagonal line obstacle 1
    [0.50, 0.50, 0.06],   # Diagonal line obstacle 2 (center)
    [0.70, 0.70, 0.06],   # Diagonal line obstacle 3
])

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
    plt.rcParams.update(tex_fonts)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_obstacles(ax, obstacles):
    """Draw circular obstacles on axis."""
    for obs in obstacles:
        x, y, r = obs
        circle = Circle((x, y), r, color='gray', alpha=0.5,
                       edgecolor='black', linewidth=1.5, zorder=5)
        ax.add_patch(circle)

def load_params(model, filepath, n_grid=32, n_agents=16):
    """Load trained parameters from msgpack file."""
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n_grid, n_grid))
    dummy_xi = jnp.zeros((n_agents, 2))
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

def get_log_timesteps(T_steps, n_points=6):
    """
    Generate logarithmically-spaced timesteps emphasizing early dynamics.
    First 80 steps are more densely sampled.
    """
    # Create log-like spacing in first 80 steps
    early_steps = np.logspace(0, np.log10(80), n_points-2, dtype=int)
    # Add some later timesteps
    late_steps = np.linspace(100, T_steps-1, 2, dtype=int)
    timesteps = np.concatenate([early_steps, late_steps])
    timesteps = np.unique(timesteps)  # Remove duplicates
    timesteps[0] = 0  # Ensure we start at 0
    return timesteps

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  2D HEAT EQUATION WITH OBSTACLES - DECENTRALIZED VISUALIZATION")
    print("=" * 70)

    # Create output directory
    output_dir = Path("figures/images/vanilla")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    # Configuration
    n_grid = 32
    n_agents = 16
    T_steps = 300

    # Initialize model and dynamics (no Tesseract)
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    dynamics = PDEDynamics(policy_apply_fn=model.apply)

    try:
        params = load_params(model, 'decentralized_params_heat2d_obstacles.msgpack', n_grid, n_agents)
        print(f"✓ Loaded trained parameters ({n_agents} agents)")
    except FileNotFoundError:
        print("✗ Error: decentralized_params_heat2d_obstacles.msgpack not found")
        return

    # Generate single test scenario (using scenario 1 from original)
    print("\n▶ Generating test scenario...")
    key = jax.random.PRNGKey(1234)
    key, k1, k2 = jax.random.split(key, 3)

    xx, yy, z_init = data_utils.generate_grf_2d(k1, n_points=n_grid)
    _, _, z_target = data_utils.generate_grf_2d(k2, n_points=n_grid)

    # Initialize agents in grid pattern at exact positions [0.2, 0.4, 0.6, 0.8]
    n_side = int(jnp.sqrt(n_agents))
    positions_1d = jnp.array([0.2, 0.4, 0.6, 0.8])[:n_side]
    xi_init = []
    for i in range(n_side):
        for j in range(n_side):
            if len(xi_init) < n_agents:
                xi_init.append([float(positions_1d[i]), float(positions_1d[j])])
    xi_init = jnp.array(xi_init)

    print(f"Obstacles at: {OBSTACLES.tolist()}")
    print("▶ Running controlled trajectory...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )

    print("▶ Running uncontrolled trajectory...")
    z_traj_unctrl, xi_traj_unctrl, u_traj_unctrl, v_traj_unctrl = rollout_uncontrolled(
        z_init, xi_init, T_steps
    )

    print("✓ Trajectories generated")

    # Compute metrics
    mse_ctrl = jnp.mean((z_traj_ctrl - z_target[None, :, :])**2, axis=(1, 2))
    mse_unctrl = jnp.mean((z_traj_unctrl - z_target[None, :, :])**2, axis=(1, 2))

    # Agent speeds (magnitude of velocity)
    speeds_ctrl = jnp.sqrt(jnp.sum(v_traj_ctrl**2, axis=-1))  # (T, n_agents)
    avg_speed_ctrl = jnp.mean(speeds_ctrl, axis=1)  # (T,)

    # Control intensity (mean absolute control)
    control_intensity = jnp.mean(jnp.abs(u_traj_ctrl), axis=1)  # (T,)

    # Get log-spaced timesteps for field plots
    timesteps = get_log_timesteps(T_steps, n_points=6)
    n_cols = len(timesteps)

    print(f"\n▶ Creating visualization at timesteps: {timesteps}")

    # Create figure: 3 rows (uncontrolled, controlled, error) × n_cols + 1 row for metrics
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(
        4,
        n_cols + 1,
        height_ratios=[1, 1, 1, 1.2],
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.12,
        wspace=0.25,
        left=0.08,
        right=0.96,
        top=0.95,
        bottom=0.06,
    )

    # Determine global color scale
    vmin = min(jnp.min(z_init), jnp.min(z_target),
               jnp.min(z_traj_ctrl), jnp.min(z_traj_unctrl))
    vmax = max(jnp.max(z_init), jnp.max(z_target),
               jnp.max(z_traj_ctrl), jnp.max(z_traj_unctrl))

    # Error color scale (tracking error)
    error_ctrl = jnp.abs(z_traj_ctrl - z_target[None, :, :])
    error_unctrl = jnp.abs(z_traj_unctrl - z_target[None, :, :])
    error_max = max(jnp.max(error_ctrl), jnp.max(error_unctrl))

    # Control intensity color scale
    u_min = jnp.min(u_traj_ctrl)
    u_max = jnp.max(u_traj_ctrl)

    # Plot field snapshots
    for col_idx, t in enumerate(timesteps):
        # Row 1: Uncontrolled Evolution
        ax = fig.add_subplot(gs[0, col_idx])
        im = ax.imshow(z_traj_unctrl[t], origin='lower', extent=[0, 1, 0, 1],
                      cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        draw_obstacles(ax, OBSTACLES)  # Draw obstacles
        if col_idx == 0:
            ax.set_ylabel('Uncontrolled\nEvolution', fontsize=11, fontweight='bold')
        ax.set_title(f't={t}', fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if col_idx > 0:
            ax.set_yticklabels([])

        # Row 2: DPC Controlled Evolution
        ax = fig.add_subplot(gs[1, col_idx])
        im = ax.imshow(z_traj_ctrl[t], origin='lower', extent=[0, 1, 0, 1],
                      cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
        draw_obstacles(ax, OBSTACLES)  # Draw obstacles

        # Overlay actuators with control intensity as color
        u_colors = u_traj_ctrl[t]
        norm = Normalize(vmin=u_min, vmax=u_max)
        scatter = ax.scatter(xi_traj_ctrl[t, :, 0], xi_traj_ctrl[t, :, 1],
                           c=u_colors, cmap='YlOrRd', norm=norm,
                           s=25, edgecolors='black', linewidths=0.5, zorder=10)

        if col_idx == 0:
            ax.set_ylabel('DPC Controlled\nEvolution', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if col_idx > 0:
            ax.set_yticklabels([])

        # Row 3: Tracking Error
        ax = fig.add_subplot(gs[2, col_idx])
        im_err = ax.imshow(error_ctrl[t], origin='lower', extent=[0, 1, 0, 1],
                          cmap='hot', vmin=0, vmax=error_max, interpolation='nearest')
        draw_obstacles(ax, OBSTACLES)  # Draw obstacles

        # Overlay actuators (same positions, cyan dots)
        ax.scatter(xi_traj_ctrl[t, :, 0], xi_traj_ctrl[t, :, 1],
                  c='cyan', s=20, edgecolors='black', linewidths=0.5,
                  zorder=10, alpha=0.8)

        if col_idx == 0:
            ax.set_ylabel('Tracking\nError', fontsize=11, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if col_idx > 0:
            ax.set_yticklabels([])

    # Add colorbars (aligned to each field row)
    # Field colorbar (temperature)
    cax1 = fig.add_subplot(gs[0, -1])
    cb1 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                     cmap='RdBu_r'),
                      cax=cax1, label='Temperature')
    cb1.ax.tick_params(labelsize=8)

    # Control intensity colorbar
    cax2 = fig.add_subplot(gs[1, -1])
    cb2 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=u_min, vmax=u_max),
                                     cmap='YlOrRd'),
                      cax=cax2, label='Control u')
    cb2.ax.tick_params(labelsize=8)

    # Error colorbar
    cax3 = fig.add_subplot(gs[2, -1])
    cb3 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=error_max),
                                     cmap='hot'),
                      cax=cax3, label='|Error|')
    cb3.ax.tick_params(labelsize=8)

    # Row 4: Time-series metrics (3 subplots)
    metrics_gs = gs[3, :].subgridspec(1, 3, wspace=0.3)
    # MSE Tracking Error
    ax_mse = fig.add_subplot(metrics_gs[0, 0])
    ax_mse.plot(mse_unctrl, 'b-', lw=1.5, label='Uncontrolled', alpha=0.8)
    ax_mse.plot(mse_ctrl, 'r-', lw=1.5, label='DPC Controlled', alpha=0.8)
    time_err = np.arange(len(mse_ctrl))
    ax_mse.fill_between(time_err, mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    ax_mse.set_xlabel('Time Step', fontsize=10)
    ax_mse.set_ylabel('MSE', fontsize=10)
    ax_mse.set_title('MSE Tracking Error', fontsize=11, fontweight='bold')
    ax_mse.set_yscale('log')
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(fontsize=9, loc='center right')

    # Agent Speed
    ax_speed = fig.add_subplot(metrics_gs[0, 1])
    ax_speed.plot(avg_speed_ctrl, 'g-', lw=1.5, alpha=0.8)
    ax_speed.set_xlabel('Time Step', fontsize=10)
    ax_speed.set_ylabel('Avg Speed', fontsize=10)
    ax_speed.set_title('Agent Speed', fontsize=11, fontweight='bold')
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_ylim(bottom=0)

    # Control Intensity
    ax_control = fig.add_subplot(metrics_gs[0, 2])
    ax_control.plot(control_intensity, 'm-', lw=1.5, alpha=0.8)
    ax_control.set_xlabel('Time Step', fontsize=10)
    ax_control.set_ylabel('Avg |u|', fontsize=10)
    ax_control.set_title('Control Intensity', fontsize=11, fontweight='bold')
    ax_control.grid(True, alpha=0.3)
    ax_control.set_ylim(bottom=0)

    # Add overall title
    fig.suptitle('2D Heat Equation with Obstacles: Decentralized DPC Control',
                fontsize=14, fontweight='bold', y=0.98)

    # Save as PDF (vector graphics)
    pdf_path = 'figures/images/vanilla/heat2d_obstacles_decentralized_visualization.pdf'
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {pdf_path}")
    
    # Also save as high-res PNG
    png_path = 'figures/images/vanilla/heat2d_obstacles_decentralized_visualization.png'
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Print final metrics
    print(f"\n{'─'*70}")
    print(f"  FINAL METRICS")
    print(f"{'─'*70}")
    print(f"  Final MSE (Controlled):   {mse_ctrl[-1]:.6f}")
    print(f"  Final MSE (Uncontrolled): {mse_unctrl[-1]:.6f}")
    print(f"  Improvement:              {(1 - mse_ctrl[-1]/mse_unctrl[-1])*100:.1f}%")
    print(f"{'─'*70}")

    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
