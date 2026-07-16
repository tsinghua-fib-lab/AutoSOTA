"""
Animation Script for Heat 1d Decentralized DPC
Creates GIF and MP4 animations showing controlled evolution over time
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import sys
import flax.serialization
from pathlib import Path
from functools import partial

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# --- Path Setup for Imports ---
# We need the project root to import 'dynamics_dual', 'models', etc.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for animation style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 24,
        "font.size": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.titlesize": 26,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
    }
    plt.rcParams.update(tex_fonts)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath, n_pde=100, n_agents=8):
    # Try to find the params file in the current script directory first
    current_dir = Path(__file__).resolve().parent
    file_path_obj = current_dir / filepath
    
    if not file_path_obj.exists():
        # Fallback to just the filename (if run from local dir)
        file_path_obj = Path(filepath)

    with open(file_path_obj, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def rollout_uncontrolled(z_init, xi_init, dynamics, T_steps):
    """Rollout with zero control inputs."""
    # Local import to avoid dependency issues if solver isn't strictly needed globally
    from tesseracts.solverHeat_decentralized import solver
    
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_zero = jnp.zeros_like(xi_curr)
        v_zero = jnp.zeros_like(xi_curr)
        z_next, xi_next = solver.implicit_step(z_curr, xi_curr, u_zero, v_zero)
        return (z_next, xi_next), z_next # Return state for scan
    
    # We only need z trajectory for visualization usually, but here we return z
    # Note: The scan function structure depends on what you need. 
    # Adjusted to return z_traj matching the structure used in main.
    _, z_traj = jax.lax.scan(step_fn, (z_init, xi_init), None, length=T_steps)
    return z_traj, xi_init # xi doesn't change in uncontrolled if v=0

# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_side_by_side_animation(x_grid, z_target, z_traj_ctrl, z_traj_unctrl, 
                                   xi_traj_ctrl, u_traj, fps=30, duration=10):
    """
    Create side-by-side animation comparing controlled vs uncontrolled evolution.
    """
    setup_style()
    
    x = np.array(x_grid).squeeze()
    z_target_np = np.array(z_target).squeeze()
    z_ctrl = np.array(z_traj_ctrl)
    z_unctrl = np.array(z_traj_unctrl)
    xi_ctrl = np.array(xi_traj_ctrl)
    u_np = np.array(u_traj)
    
    T = z_ctrl.shape[0]
    n_agents = xi_ctrl.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    # Calculate frames
    total_frames = fps * duration
    frame_indices = np.linspace(0, T-1, total_frames).astype(int)
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Heat Equation: Decentralized DPC vs Uncontrolled', fontsize=28, fontweight='bold', y=0.98)
    
    # Initialize plots
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    
    # Y-limits based on data
    y_min = min(z_ctrl.min(), z_unctrl.min(), z_target_np.min()) - 0.5
    y_max = max(z_ctrl.max(), z_unctrl.max(), z_target_np.max()) + 0.5
    
    # Panel 1: Uncontrolled
    ax1.set_xlim([0, 1])
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlabel(r'Position $x$')
    ax1.set_ylabel(r'Temperature $z(x,t)$')
    ax1.set_title('Uncontrolled Evolution', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=22)
    line_target1, = ax1.plot(x, z_target_np, 'k--', lw=2.5, label='Target')
    line_unctrl, = ax1.plot([], [], 'b-', lw=3, label='State')
    ax1.legend(loc='upper right', fontsize=18)
    
    # Panel 2: Controlled
    ax2.set_xlim([0, 1])
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlabel(r'Position $x$')
    ax2.set_title('Decentralized DPC Controlled', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=22)
    line_target2, = ax2.plot(x, z_target_np, 'k--', lw=2.5, label='Target')
    line_ctrl, = ax2.plot([], [], 'r-', lw=3, label='State')
    agent_markers = [ax2.scatter([], [], s=100, color=colors[j], edgecolors='black', 
                                  linewidth=1.5, zorder=10) for j in range(n_agents)]
    ax2.legend(loc='upper right', fontsize=18)
    
    # Panel 3: Agent Positions
    ax3.set_xlim([0, T])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_xlabel(r'Time step')
    ax3.set_ylabel(r'Position $\xi_i$')
    ax3.set_title('Agent Trajectories', fontweight='bold')
    ax3.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax3.axhline(y=1, color='gray', ls='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=22)
    agent_lines = [ax3.plot([], [], color=colors[j], lw=2)[0] for j in range(n_agents)]
    time_marker = ax3.axvline(x=0, color='red', lw=2, alpha=0.7)
    
    # Panel 4: MSE Comparison
    mse_ctrl = np.mean((z_ctrl - z_target_np[None, :])**2, axis=1)
    mse_unctrl = np.mean((z_unctrl - z_target_np[None, :])**2, axis=1)
    ax4.set_xlim([0, T])
    ax4.set_ylim([mse_ctrl.min()*0.5, max(mse_ctrl.max(), mse_unctrl.max())*1.2])
    ax4.set_xlabel(r'Time step')
    ax4.set_ylabel(r'MSE')
    ax4.set_title('Tracking Error', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=22)
    line_mse_unctrl, = ax4.plot([], [], 'b-', lw=2.5, label='Uncontrolled')
    line_mse_ctrl, = ax4.plot([], [], 'r-', lw=2.5, label='Decentralized DPC')
    ax4.legend(loc='upper right', fontsize=18)
    
    # Time text
    time_text = fig.text(0.5, 0.01, '', ha='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    def init():
        line_unctrl.set_data([], [])
        line_ctrl.set_data([], [])
        for marker in agent_markers:
            marker.set_offsets(np.empty((0, 2)))
        for line in agent_lines:
            line.set_data([], [])
        time_marker.set_xdata([0])
        line_mse_unctrl.set_data([], [])
        line_mse_ctrl.set_data([], [])
        time_text.set_text('')
        return [line_unctrl, line_ctrl, line_mse_unctrl, line_mse_ctrl, time_marker, time_text] + agent_markers + agent_lines
    
    def animate(frame):
        t = frame_indices[frame]
        
        # Update state lines
        line_unctrl.set_data(x, z_unctrl[t])
        line_ctrl.set_data(x, z_ctrl[t])
        
        # Update agent markers
        for j, marker in enumerate(agent_markers):
            xi = xi_ctrl[t, j]
            idx = int(np.clip(xi * len(x), 0, len(x)-1))
            marker.set_offsets([[xi, z_ctrl[t, idx]]])
        
        # Update agent trajectory lines
        for j, line in enumerate(agent_lines):
            line.set_data(np.arange(t+1), xi_ctrl[:t+1, j])
        
        # Update time marker
        time_marker.set_xdata([t])
        
        # Update MSE lines
        line_mse_unctrl.set_data(np.arange(t+1), mse_unctrl[:t+1])
        line_mse_ctrl.set_data(np.arange(t+1), mse_ctrl[:t+1])
        
        # Update time text
        time_text.set_text(f't = {t} / {T-1}')
        
        return [line_unctrl, line_ctrl, line_mse_unctrl, line_mse_ctrl, time_marker, time_text] + agent_markers + agent_lines
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=1000/fps, blit=True)
    
    return fig, anim

def main():
    print("=" * 60)
    print("  HEAT EQUATION DECENTRALIZED DPC - ANIMATION GENERATOR")
    print("=" * 60)
    
    n_pde, n_agents, T_steps = 100, 8, 300
    fps = 30
    duration = 10  # seconds
    
    # Initialize model directly
    model = DecentralizedControlNet(features=(64, 64))
    
    # Initialize Dynamics
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    try:
        params = load_params(model, 'decentralized_params.msgpack', n_pde, n_agents)
        print(f"✓ Loaded trained parameters ({n_agents} agents)")
    except FileNotFoundError:
        print("✗ Error: decentralized_params.msgpack not found")
        return
    
    x_grid = jnp.linspace(0, 1, n_pde)
    key = jax.random.PRNGKey(42)
    
    print("\n▶ Generating trajectories...")
    key, k1, k2 = jax.random.split(key, 3)
    _, z_init = data_utils.generate_grf(k1, n_points=n_pde, length_scale=0.2)
    _, z_target = data_utils.generate_grf(k2, n_points=n_pde, length_scale=0.4)
    xi_init = jnp.linspace(0.15, 0.85, n_agents)
    
    # Controlled rollout
    z_traj_ctrl, xi_traj_ctrl, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    
    # Uncontrolled rollout
    z_traj_unctrl, _ = rollout_uncontrolled(z_init, xi_init, dynamics, T_steps)
    
    print(f"✓ Generated {T_steps} timesteps")
    
    # Create animation
    print(f"\n▶ Creating animation ({duration}s @ {fps}fps)...")
    fig, anim = create_side_by_side_animation(
        x_grid, z_target, z_traj_ctrl, z_traj_unctrl,
        xi_traj_ctrl, u_traj, fps=fps, duration=duration
    )
    
    # --- SAVE LOGIC START ---
    # Define output directory relative to this script
    current_script_dir = Path(__file__).resolve().parent
    output_dir = current_script_dir / "figures" / "images" / "animations"
    
    print(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- SAVE LOGIC END ---

    # Save as GIF
    print("▶ Saving GIF (this may take a minute)...")
    gif_path = output_dir / 'heat_decentralized_animation.gif'
    anim.save(gif_path, writer='pillow', fps=fps)
    print(f"✓ Saved: {gif_path}")
    
    # Save as MP4 (if ffmpeg is available)
    try:
        print("▶ Saving MP4...")
        mp4_path = output_dir / 'heat_decentralized_animation.mp4'
        anim.save(mp4_path, writer='ffmpeg', fps=fps, 
                  extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"✓ Saved: {mp4_path}")
    except Exception as e:
        print(f"⚠ MP4 save failed (ffmpeg may not be installed): {e}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("  ANIMATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()