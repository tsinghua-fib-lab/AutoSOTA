"""
2D Heat Equation Control - Decentralized Animation
Creates 2×2 animated visualization with uncontrolled/controlled/error fields + MSE
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for animation style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 14,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
    }
    plt.rcParams.update(tex_fonts)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_2x2_animation(z_traj_unctrl, z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl,
                        z_target, fps=30, duration=10):
    """
    Create 2×2 animation:
    [Uncontrolled Evolution]  [DPC Controlled Evolution]
    [Tracking Error]          [MSE Plot]
    """
    setup_style()

    # Convert to numpy
    z_unctrl = np.array(z_traj_unctrl)
    z_ctrl = np.array(z_traj_ctrl)
    z_target_np = np.array(z_target)
    xi_ctrl = np.array(xi_traj_ctrl)
    u_ctrl = np.array(u_traj_ctrl)

    T = z_ctrl.shape[0]

    # Compute metrics
    mse_ctrl = np.mean((z_ctrl - z_target_np[None, :, :])**2, axis=(1, 2))
    mse_unctrl = np.mean((z_unctrl - z_target_np[None, :, :])**2, axis=(1, 2))
    error_ctrl = np.abs(z_ctrl - z_target_np[None, :, :])

    # Color scales
    vmin = min(z_unctrl.min(), z_ctrl.min(), z_target_np.min())
    vmax = max(z_unctrl.max(), z_ctrl.max(), z_target_np.max())
    error_max = error_ctrl.max()
    u_min, u_max = u_ctrl.min(), u_ctrl.max()

    # Calculate frames
    total_frames = fps * duration
    frame_indices = np.linspace(0, T-1, total_frames).astype(int)

    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.3,
        wspace=0.55,
        left=0.08,
        right=0.9,
        top=0.93,
        bottom=0.08,
    )

    ax1 = fig.add_subplot(gs[0, 0])  # Uncontrolled
    ax2 = fig.add_subplot(gs[0, 1])  # Controlled
    ax3 = fig.add_subplot(gs[1, 0])  # Tracking Error
    ax4 = fig.add_subplot(gs[1, 1])  # MSE

    # Panel 1: Uncontrolled Evolution
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel(r'Position $x$', fontsize=12)
    ax1.set_ylabel(r'Position $y$', fontsize=12)
    ax1.set_title('Uncontrolled Evolution', fontsize=13, fontweight='bold')
    im1 = ax1.imshow(z_unctrl[0], origin='lower', extent=[0, 1, 0, 1],
                     cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Panel 2: DPC Controlled Evolution
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel(r'Position $x$', fontsize=12)
    ax2.set_ylabel(r'Position $y$', fontsize=12)
    ax2.set_title('DPC Controlled Evolution', fontsize=13, fontweight='bold')
    im2 = ax2.imshow(z_ctrl[0], origin='lower', extent=[0, 1, 0, 1],
                     cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Actuators with control intensity
    norm_u = Normalize(vmin=u_min, vmax=u_max)
    scatter2 = ax2.scatter([], [], c=[], cmap='YlOrRd', norm=norm_u,
                          s=30, edgecolors='black', linewidths=0.6, zorder=10)

    # Panel 3: Tracking Error
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_xlabel(r'Position $x$', fontsize=12)
    ax3.set_ylabel(r'Position $y$', fontsize=12)
    ax3.set_title('Tracking Error', fontsize=13, fontweight='bold')
    im3 = ax3.imshow(error_ctrl[0], origin='lower', extent=[0, 1, 0, 1],
                     cmap='hot', vmin=0, vmax=error_max, interpolation='nearest')

    # Actuators (cyan markers)
    scatter3 = ax3.scatter([], [], c='cyan', s=25, edgecolors='black',
                          linewidths=0.6, zorder=10, alpha=0.8)

    # Panel 4: MSE Evolution
    ax4.set_xlim([0, T])
    ax4.set_ylim([min(mse_ctrl.min(), mse_unctrl.min())*0.5,
                  max(mse_ctrl.max(), mse_unctrl.max())*1.2])
    ax4.set_xlabel(r'Time Step', fontsize=12)
    ax4.set_ylabel(r'MSE', fontsize=12)
    ax4.set_title('MSE Tracking Error', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    line_unctrl, = ax4.plot([], [], 'b-', lw=2.5, label='Uncontrolled', alpha=0.8)
    line_ctrl, = ax4.plot([], [], 'r-', lw=2.5, label='DPC Controlled', alpha=0.8)
    time_marker = ax4.axvline(x=0, color='green', lw=2, alpha=0.7, linestyle='--')
    ax4.legend(fontsize=11, loc='center right')

    # Add colorbars (aligned to each field subplot)
    cbar_pad = 0.015
    cbar_width = 0.02
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    cax1 = fig.add_axes([pos1.x1 + cbar_pad, pos1.y0, cbar_width, pos1.height])
    cax2 = fig.add_axes([pos2.x1 + cbar_pad, pos2.y0, cbar_width, pos2.height])
    cax3 = fig.add_axes([pos3.x1 + cbar_pad, pos3.y0, cbar_width, pos3.height])

    cb1 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                      cmap='RdBu_r'),
                      cax=cax1, label='Temperature')
    cb1.ax.tick_params(labelsize=10)

    # Control intensity colorbar (panel 2)
    cb2 = fig.colorbar(ScalarMappable(norm=norm_u, cmap='YlOrRd'),
                      cax=cax2, label='Control u')
    cb2.ax.tick_params(labelsize=10)

    # Error colorbar (panel 3)
    cb3 = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=error_max),
                                      cmap='hot'),
                      cax=cax3, label='|Error|')
    cb3.ax.tick_params(labelsize=10)

    # Title and time text
    fig.suptitle('2D Heat Equation: Decentralized DPC Control',
                fontsize=16, fontweight='bold', y=0.97)
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=13, fontweight='bold')

    def init():
        im1.set_data(z_unctrl[0])
        im2.set_data(z_ctrl[0])
        im3.set_data(error_ctrl[0])
        scatter2.set_offsets(np.empty((0, 2)))
        scatter3.set_offsets(np.empty((0, 2)))
        line_unctrl.set_data([], [])
        line_ctrl.set_data([], [])
        time_marker.set_xdata([0])
        time_text.set_text('')
        return [im1, im2, im3, scatter2, scatter3, line_unctrl, line_ctrl, time_marker, time_text]

    def animate(frame):
        t = frame_indices[frame]

        # Update field plots
        im1.set_data(z_unctrl[t])
        im2.set_data(z_ctrl[t])
        im3.set_data(error_ctrl[t])

        # Update actuators
        positions = xi_ctrl[t]
        scatter2.set_offsets(positions)
        scatter2.set_array(u_ctrl[t])
        scatter3.set_offsets(positions)

        # Update MSE lines
        line_unctrl.set_data(np.arange(t+1), mse_unctrl[:t+1])
        line_ctrl.set_data(np.arange(t+1), mse_ctrl[:t+1])
        time_marker.set_xdata([t])

        # Update time text
        time_text.set_text(f't = {t} / {T-1}')

        return [im1, im2, im3, scatter2, scatter3, line_unctrl, line_ctrl, time_marker, time_text]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=total_frames, interval=1000/fps, blit=True)

    return fig, anim

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  2D HEAT EQUATION DPC - CENTRALIZED ANIMATION")
    print("=" * 70)

    # Configuration
    n_grid = 32
    n_agents = 16
    T_steps = 300
    fps = 30
    duration = 10  # seconds

    # Setup output directory
    save_dir = Path("figures/images/animations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model and dynamics
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    dynamics = PDEDynamics(policy_apply_fn=model.apply)

    try:
        params = load_params(model, 'decentralized_params_heat2d.msgpack', n_grid, n_agents)
        print(f"✓ Loaded trained parameters ({n_agents} agents)")
    except FileNotFoundError:
        print("✗ Error: decentralized_params_heat2d.msgpack not found")
        return

    # Generate test scenario (same as visualize.py - scenario 1)
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

    print("▶ Running controlled trajectory...")
    z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl, v_traj_ctrl = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )

    print("▶ Running uncontrolled trajectory...")
    z_traj_unctrl, xi_traj_unctrl, u_traj_unctrl, v_traj_unctrl = rollout_uncontrolled(
        z_init, xi_init, T_steps
    )

    print(f"✓ Generated {T_steps} timesteps")

    # Create animation
    print(f"\n▶ Creating animation ({duration}s @ {fps}fps)...")
    fig, anim = create_2x2_animation(
        z_traj_unctrl, z_traj_ctrl, xi_traj_ctrl, u_traj_ctrl,
        z_target, fps=fps, duration=duration
    )

    # Save as GIF
    print("▶ Saving GIF (this may take a few minutes)...")
    gif_path = save_dir / 'heat2d_animation.gif'
    anim.save(gif_path, writer='pillow', fps=fps, dpi=150)
    print(f"✓ Saved: {gif_path}")

    # Save as MP4 (higher quality)
    try:
        print("▶ Saving MP4 (high resolution)...")
        mp4_path = save_dir / 'heat2d_animation.mp4'
        anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=200,
                  extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"✓ Saved: {mp4_path}")
    except Exception as e:
        print(f"⚠ MP4 save failed (ffmpeg may not be installed): {e}")

    plt.close()

    print("\n" + "=" * 70)
    print("  ANIMATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()