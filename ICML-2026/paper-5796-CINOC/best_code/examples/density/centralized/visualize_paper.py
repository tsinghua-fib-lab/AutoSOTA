"""
NS2D Shape Formation - Publication-Quality Visualization

Creates a 2x2 figure matching heat2D style:
- Top-left: Uncontrolled Evolution
- Top-right: DPC Controlled Evolution with actuator positions
- Bottom-left: Tracking Error
- Bottom-right: MSE Tracking Error over time

Also generates GIF and MP4 animations.
"""

import sys
from pathlib import Path

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import flax.serialization

from examples.density.centralized.dynamics import unroll_controlled
from examples.density.centralized.train import (
    N_AGENTS, T_STEPS, PUSH_MAX, SIGMA_PUSH, BUOYANCY, FEATURES
)
from models.policy_ns2d import NS2DControlNet


# =============================================================================
# Academic Style
# =============================================================================

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
    plt.rcParams.update(tex_fonts)


# =============================================================================
# Helper Functions
# =============================================================================

def rollout_uncontrolled(smoke_init, xi_init, rho_target, T_steps, Nx, Ny, dt, buoyancy):
    """Rollout with zero control inputs (natural dynamics only)."""
    from examples.density.centralized.dynamics import ns2d_step_jax
    
    def step_fn(carry, _):
        smoke, xi = carry
        n = xi.shape[0]
        # Zero push velocity (no control)
        push_vel = jnp.zeros((n, 2))
        
        smoke_new = ns2d_step_jax(
            smoke, xi, push_vel,
            dt=dt, buoyancy=buoyancy,
            sigma_push=SIGMA_PUSH,
            Nx=Nx, Ny=Ny
        )
        return (smoke_new, xi), (smoke_new, xi, push_vel)
    
    _, (smoke_traj, xi_traj, v_traj) = jax.lax.scan(
        step_fn, (smoke_init, xi_init), None, length=T_steps
    )
    return smoke_traj, xi_traj, v_traj


def create_animation(
    smoke_unctrl, smoke_ctrl, xi_traj_ctrl, vel_traj_ctrl,
    rho_target, mse_ctrl, mse_unctrl,
    sample_idx=0,
    output_gif=None,
    output_mp4=None,
    fps=10,
    skip_frames=5
):
    """
    Create GIF and MP4 animation with same 2x2 layout as paper figure.
    
    Layout:
    - (0,0) Uncontrolled Evolution
    - (0,1) DPC Controlled Evolution with actuators
    - (1,0) Tracking Error
    - (1,1) MSE Tracking Error over time (with moving marker)
    """
    setup_style()
    
    T, Nx, Ny = smoke_ctrl.shape
    n_agents = xi_traj_ctrl.shape[1]
    
    # Frame indices
    frame_indices = list(range(0, T, skip_frames))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)
    
    # Pre-compute color scales
    vmax = max(float(smoke_ctrl.max()), float(smoke_unctrl.max()), 0.8)
    error_all = np.abs(smoke_ctrl - np.array(rho_target))
    error_max = max(float(error_all.max()), 0.1)
    vel_mag_all = np.sqrt(vel_traj_ctrl[:, :, 0]**2 + vel_traj_ctrl[:, :, 1]**2)
    vel_max = max(float(vel_mag_all.max()), 0.01)
    
    # Grid for contours
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(12, 10))
    
    def animate(frame_num):
        t = frame_indices[frame_num]
        fig.clear()
        
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
        
        smoke_unctrl_t = smoke_unctrl[t]
        smoke_ctrl_t = smoke_ctrl[t]
        xi_t = xi_traj_ctrl[t]
        vel_t = vel_traj_ctrl[t]
        error_t = np.abs(smoke_ctrl_t - np.array(rho_target))
        vel_mag = np.sqrt(vel_t[:, 0]**2 + vel_t[:, 1]**2)
        
        # (0,0) Uncontrolled Evolution - RdBu_r colormap like heat2D
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(smoke_unctrl_t.T, origin='lower', extent=[0, 1, 0, 1.25],
                        cmap='RdBu_r', vmin=0, vmax=vmax, aspect='auto')
        ax1.set_title('Uncontrolled Evolution', fontweight='bold')
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Position y')
        fig.colorbar(im1, ax=ax1, shrink=0.8, label='Smoke Density')
        
        # (0,1) DPC Controlled with actuators - RdBu_r + viridis for control
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(smoke_ctrl_t.T, origin='lower', extent=[0, 1, 0, 1.25],
                        cmap='RdBu_r', vmin=0, vmax=vmax, aspect='auto')
        # Actuators colored by velocity magnitude (viridis for better visibility)
        scatter = ax2.scatter(xi_t[:, 0], xi_t[:, 1], c=vel_mag, cmap='viridis',
                             s=80, edgecolors='white', linewidths=1.5, zorder=10,
                             vmin=0, vmax=vel_max)
        ax2.contour(X, Y, np.array(rho_target), levels=[0.3], colors='lime',
                   linestyles='--', linewidths=2)
        ax2.set_title('DPC Controlled Evolution', fontweight='bold')
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Position y')
        fig.colorbar(scatter, ax=ax2, shrink=0.8, label='Control |v|')
        
        # (1,0) Tracking Error - hot colormap
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(error_t.T, origin='lower', extent=[0, 1, 0, 1.25],
                        cmap='hot', vmin=0, vmax=error_max, aspect='auto')
        # Agent positions in cyan for visibility on hot colormap
        ax3.scatter(xi_t[:, 0], xi_t[:, 1], c='cyan', s=50,
                   edgecolors='black', linewidths=0.8, zorder=10, alpha=0.9)
        ax3.set_title('Tracking Error', fontweight='bold')
        ax3.set_xlabel('Position x')
        ax3.set_ylabel('Position y')
        fig.colorbar(im3, ax=ax3, shrink=0.8, label='|Error|')
        
        # (1,1) MSE over time - PROGRESSIVE (builds up with evolution)
        ax4 = fig.add_subplot(gs[1, 1])
        # Only plot up to current time t (progressive reveal)
        time_arr_current = np.arange(t + 1)
        ax4.plot(time_arr_current, mse_unctrl[:t+1], 'b-', lw=2, label='Uncontrolled', alpha=0.8)
        ax4.plot(time_arr_current, mse_ctrl[:t+1], 'r-', lw=2, label='DPC Controlled', alpha=0.8)
        ax4.fill_between(time_arr_current, mse_ctrl[:t+1], mse_unctrl[:t+1], alpha=0.2, color='green')
        # Current point marker
        ax4.scatter([t], [mse_ctrl[t]], s=120, c='red', zorder=10, edgecolors='white', linewidths=2)
        ax4.scatter([t], [mse_unctrl[t]], s=80, c='blue', zorder=10, edgecolors='white', linewidths=1.5)
        # Set fixed axis limits (full time range)
        ax4.set_xlim([0, len(mse_ctrl)])
        ax4.set_ylim([min(mse_ctrl.min(), mse_unctrl.min()) * 0.5, 
                      max(mse_ctrl.max(), mse_unctrl.max()) * 1.5])
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('MSE')
        ax4.set_title('MSE Tracking Error', fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9, loc='upper right')
        
        fig.suptitle(f'NS2D Shape Formation | Sample {sample_idx+1} | t = {t}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(frame_indices),
                                   interval=1000//fps, blit=False)
    
    # Save GIF
    if output_gif:
        print(f"  Saving GIF: {output_gif}")
        anim.save(output_gif, writer='pillow', fps=fps)
        print(f"  ✓ GIF saved")
    
    # Save MP4 (try multiple writers)
    if output_mp4:
        print(f"  Saving MP4: {output_mp4}")
        saved = False
        for writer_name in ['ffmpeg', 'imagemagick']:
            try:
                anim.save(output_mp4, writer=writer_name, fps=fps)
                print(f"  ✓ MP4 saved (using {writer_name})")
                saved = True
                break
            except Exception:
                continue
        if not saved:
            # Fallback: save frames manually
            try:
                import imageio
                frames = []
                for i in range(len(frame_indices)):
                    animate(i)
                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(img)
                imageio.mimsave(output_mp4, frames, fps=fps)
                print(f"  ✓ MP4 saved (using imageio)")
            except Exception as e:
                print(f"  ✗ MP4 failed: {e}")
    
    plt.close()


# =============================================================================
# Main Paper Visualization (2x2 Grid)
# =============================================================================

def create_paper_figure(
    smoke_unctrl, smoke_ctrl, xi_traj_ctrl, vel_traj_ctrl,
    rho_target, mse_ctrl, mse_unctrl, timestep=-1,
    filename='ns2d_paper_visualization.png'
):
    """
    Create 2x2 publication figure matching heat2D style.
    
    Layout:
    - (0,0) Uncontrolled Evolution
    - (0,1) DPC Controlled Evolution with actuators
    - (1,0) Tracking Error
    - (1,1) MSE Tracking Error over time
    """
    setup_style()
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    
    Nx, Ny = smoke_ctrl[0].shape
    
    # Color scales
    vmin = min(float(smoke_ctrl.min()), float(smoke_unctrl.min()))
    vmax = max(float(smoke_ctrl.max()), float(smoke_unctrl.max()))
    
    # Get the specified timestep data
    t = timestep if timestep >= 0 else len(smoke_ctrl) - 1
    smoke_unctrl_t = smoke_unctrl[t]
    smoke_ctrl_t = smoke_ctrl[t]
    xi_t = xi_traj_ctrl[t]
    vel_t = vel_traj_ctrl[t]
    
    # Error and control intensity
    error = np.abs(smoke_ctrl_t - np.array(rho_target))
    vel_mag = np.sqrt(vel_t[:, 0]**2 + vel_t[:, 1]**2)
    vel_max = max(float(vel_mag.max()), 0.01)
    
    # =========================================================================
    # (0,0) Uncontrolled Evolution - RdBu_r colormap like heat2D
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(smoke_unctrl_t.T, origin='lower', extent=[0, 1, 0, 1.25],
                     cmap='RdBu_r', vmin=0, vmax=vmax, aspect='auto')
    ax1.set_title('Uncontrolled Evolution', fontweight='bold')
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Position y')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Smoke Density')
    
    # =========================================================================
    # (0,1) DPC Controlled Evolution with actuators - RdBu_r + viridis controls
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(smoke_ctrl_t.T, origin='lower', extent=[0, 1, 0, 1.25],
                     cmap='RdBu_r', vmin=0, vmax=vmax, aspect='auto')
    
    # Actuator positions colored by velocity magnitude (viridis for visibility)
    scatter = ax2.scatter(xi_t[:, 0], xi_t[:, 1], c=vel_mag, cmap='viridis',
                         s=80, edgecolors='white', linewidths=1.5, zorder=10,
                         vmin=0, vmax=vel_max)
    
    # Target contour (lime green for visibility on RdBu)
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    ax2.contour(X, Y, np.array(rho_target), levels=[0.3], colors='lime',
               linestyles='--', linewidths=2, alpha=0.9)
    
    ax2.set_title('DPC Controlled Evolution', fontweight='bold')
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Position y')
    cbar2 = fig.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar2.set_label('Control |v|')
    
    # =========================================================================
    # (1,0) Tracking Error - hot colormap with cyan actuators
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    error_max = max(float(error.max()), 0.01)
    im3 = ax3.imshow(error.T, origin='lower', extent=[0, 1, 0, 1.25],
                     cmap='hot', vmin=0, vmax=error_max, aspect='auto')
    
    # Actuator positions (cyan dots visible on hot colormap)
    ax3.scatter(xi_t[:, 0], xi_t[:, 1], c='cyan', s=50,
               edgecolors='black', linewidths=0.8, zorder=10, alpha=0.9)
    
    ax3.set_title('Tracking Error', fontweight='bold')
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Position y')
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('|Error|')
    
    # =========================================================================
    # (1,1) MSE Tracking Error over time
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    time = np.arange(len(mse_ctrl))
    ax4.plot(time, mse_unctrl, 'b-', lw=1.5, label='Uncontrolled', alpha=0.8)
    ax4.plot(time, mse_ctrl, 'r-', lw=1.5, label='DPC Controlled', alpha=0.8)
    ax4.fill_between(time, mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('MSE')
    ax4.set_title('MSE Tracking Error', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # Overall title
    fig.suptitle('NS2D Shape Formation: Centralized DPC Control',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Paper Visualization")
    print("="*60)
    
    setup_style()
    
    # Load config
    data_dir = Path(__file__).parent.parent / 'data'
    config = np.load(data_dir / 'config.npz', allow_pickle=True)
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    n_agents = N_AGENTS
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents}")
    
    # Load test data
    test_data = np.load(data_dir / 'test_data.npz', allow_pickle=True)
    
    # Load model
    model = NS2DControlNet(features=FEATURES, v_max=PUSH_MAX)
    params_path = Path(__file__).parent / 'ns2d_params.msgpack'
    
    if not params_path.exists():
        print(f"Error: {params_path} not found. Run train.py first!")
        return
    
    with open(params_path, 'rb') as f:
        dummy_smoke = jnp.zeros((Nx, Ny))
        dummy_xi = jnp.zeros((n_agents, 2))
        params = model.init(jax.random.PRNGKey(0), dummy_smoke, dummy_smoke, dummy_xi)
        params = flax.serialization.from_bytes(params, f.read())
    
    print("✓ Loaded trained parameters")
    
    # Agent grid
    n_side = int(np.sqrt(n_agents))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    
    T_steps = T_STEPS
    save_dir = Path(__file__).parent
    
    # Process 2 samples
    n_samples = min(2, len(test_data['rho_init']))
    
    for sample_idx in range(n_samples):
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_idx + 1}")
        print("="*60)
        
        smoke_init = jnp.array(test_data['rho_init'][sample_idx])
        rho_target = jnp.array(test_data['rho_target'][sample_idx])
        
        # Controlled trajectory
        print("▶ Running controlled simulation...")
        smoke_ctrl, xi_ctrl, vel_ctrl = unroll_controlled(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
            sigma_push=SIGMA_PUSH, push_max=PUSH_MAX
        )
        
        # Uncontrolled trajectory
        print("▶ Running uncontrolled simulation...")
        smoke_unctrl, xi_unctrl, _ = rollout_uncontrolled(
            smoke_init, xi_init, rho_target, T_steps, Nx, Ny, dt, BUOYANCY
        )
        
        # Convert to numpy
        smoke_ctrl = np.array(smoke_ctrl)
        smoke_unctrl = np.array(smoke_unctrl)
        xi_ctrl = np.array(xi_ctrl)
        vel_ctrl = np.array(vel_ctrl)
        rho_target_np = np.array(rho_target)
        
        # Compute MSE over time
        mse_ctrl = np.mean((smoke_ctrl - rho_target_np)**2, axis=(1, 2))
        mse_unctrl = np.mean((smoke_unctrl - rho_target_np)**2, axis=(1, 2))
        
        # Create paper figure (2x2 grid)
        print("▶ Creating paper figure...")
        create_paper_figure(
            smoke_unctrl, smoke_ctrl, xi_ctrl, vel_ctrl,
            rho_target_np, mse_ctrl, mse_unctrl,
            timestep=-1,  # Final timestep
            filename=str(save_dir / f'ns2d_paper_sample_{sample_idx+1}.png')
        )
        
        # Create animation (same 2x2 layout as PNG)
        print("▶ Creating animation...")
        create_animation(
            smoke_unctrl, smoke_ctrl, xi_ctrl, vel_ctrl,
            rho_target_np, mse_ctrl, mse_unctrl,
            sample_idx=sample_idx,
            output_gif=str(save_dir / f'ns2d_sample_{sample_idx+1}.gif'),
            output_mp4=str(save_dir / f'ns2d_sample_{sample_idx+1}.mp4'),
            fps=9, skip_frames=2
        )
    
    # Print summary
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("\nGenerated files:")
    for i in range(n_samples):
        print(f"  - ns2d_paper_sample_{i+1}.png")
        print(f"  - ns2d_sample_{i+1}.gif")
        print(f"  - ns2d_sample_{i+1}.mp4")


if __name__ == "__main__":
    main()
