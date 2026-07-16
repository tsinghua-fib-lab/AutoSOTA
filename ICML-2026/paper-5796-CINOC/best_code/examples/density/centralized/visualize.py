"""
Visualization for NS2D Shape Formation Control Results

Creates conference-quality figures showing:
- Controlled vs uncontrolled (natural) evolution
- Agent trajectories
- Control signals
- Tracking error comparison
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
import flax.serialization
from examples.density.centralized.dynamics import unroll_controlled
from examples.density.centralized.train import (
    N_AGENTS, T_STEPS, PUSH_MAX, SIGMA_INJECT, SIGMA_PUSH, BUOYANCY, FEATURES
)
from models.policy_ns2d import NS2DControlNet


# =============================================================================
# Academic Style
# =============================================================================

def setup_academic_style():
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 18,
        "font.size": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 20,
        "figure.titlesize": 24,
        "axes.linewidth": 1.5,
    }
    plt.rcParams.update(tex_fonts)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_evolution_comparison(
    smoke_init, smoke_traj, rho_target, xi_traj,
    smoke_traj_unctrl=None,
    filename='ns2d_evolution.png'
):
    """Plot controlled evolution with agent positions."""
    setup_academic_style()
    
    T = smoke_traj.shape[0]
    n_plots = 6
    indices = [int(i * (T - 1) / (n_plots - 1)) for i in range(n_plots)]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.25, hspace=0.3)
    
    vmax = max(float(smoke_traj.max()), float(rho_target.max()), 0.5)
    
    Nx, Ny = smoke_traj.shape[1], smoke_traj.shape[2]
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    levels = np.linspace(0, vmax, 50)
    n_agents = xi_traj.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    cf = None
    for i, idx in enumerate(indices):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        
        smoke_data = np.array(smoke_traj[idx])
        cf = ax.contourf(X, Y, smoke_data, levels=levels, cmap='hot', extend='max')
        ax.contour(X, Y, smoke_data, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        # Agent positions
        for j in range(n_agents):
            xi_pos = xi_traj[idx, j]
            ax.scatter(xi_pos[0], xi_pos[1], s=100, c=[colors[j]], 
                      marker='o', edgecolors='white', linewidth=2, zorder=10)
        
        # Target contour (outline)
        ax.contour(X, Y, np.array(rho_target), levels=[0.3], colors='cyan', 
                  linestyles='--', linewidths=2, alpha=0.7)
        
        ax.set_title(f'$t = {idx}$', fontweight='bold')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_aspect('equal')
    
    # Colorbar
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label(r'Smoke Density $\rho$', fontsize=18)
    
    plt.suptitle('NS2D Shape Formation: Controlled Evolution', 
                fontsize=24, fontweight='bold', y=0.98)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


def plot_control_signals(intensity_traj, vel_traj, xi_traj, filename='ns2d_controls.png'):
    """Plot control signals and agent trajectories."""
    setup_academic_style()
    
    T = intensity_traj.shape[0]
    n_agents = intensity_traj.shape[1]
    time = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Injection intensity
    ax1 = axes[0, 0]
    for j in range(n_agents):
        ax1.plot(time, intensity_traj[:, j], color=colors[j], lw=2, label=f'Agent {j+1}')
    ax1.set_title('Injection Intensity $u_i(t)$', fontweight='bold')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Intensity')
    ax1.grid(True, alpha=0.3)
    
    # Velocity magnitude
    ax2 = axes[0, 1]
    vel_mag = np.linalg.norm(vel_traj, axis=-1)
    for j in range(n_agents):
        ax2.plot(time, vel_mag[:, j], color=colors[j], lw=2)
    ax2.set_title('Velocity Magnitude $|v_i(t)|$', fontweight='bold')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Speed')
    ax2.grid(True, alpha=0.3)
    
    # Agent trajectories (x-y plot)
    ax3 = axes[1, 0]
    for j in range(n_agents):
        ax3.plot(xi_traj[:, j, 0], xi_traj[:, j, 1], color=colors[j], lw=2)
        ax3.scatter(xi_traj[0, j, 0], xi_traj[0, j, 1], s=80, c=[colors[j]], marker='o', zorder=5)
        ax3.scatter(xi_traj[-1, j, 0], xi_traj[-1, j, 1], s=100, c=[colors[j]], 
                   marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax3.set_title('Agent Trajectories $\\xi_i(t)$', fontweight='bold')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.25])
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Agent x-position over time
    ax4 = axes[1, 1]
    for j in range(n_agents):
        ax4.plot(time, xi_traj[:, j, 0], color=colors[j], lw=2, label=f'Agent {j+1}')
    ax4.set_title('Agent Position $x_i(t)$', fontweight='bold')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('$x$ position')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Visualization")
    print("="*60)
    
    # Load grid config from data
    data_dir = Path(__file__).parent.parent / 'data'
    config = np.load(data_dir / 'config.npz')
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    
    # Use imported constants from train.py
    n_agents = N_AGENTS
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {n_agents} (fan-only control)")
    
    test_data = np.load(data_dir / 'test_data.npz')
    
    # Load model with fan-only parameters
    model = NS2DControlNet(features=FEATURES, v_max=PUSH_MAX)
    
    params_path = Path(__file__).parent / 'ns2d_params.msgpack'
    if not params_path.exists():
        print(f"Trained params not found: {params_path}")
        print("Run train.py first!")
        return
    
    with open(params_path, 'rb') as f:
        dummy_smoke = jnp.zeros((Nx, Ny))
        dummy_xi = jnp.zeros((n_agents, 2))
        params = model.init(jax.random.PRNGKey(0), dummy_smoke, dummy_smoke, dummy_xi)
        params = flax.serialization.from_bytes(params, f.read())
    
    print("Loaded trained parameters")
    
    # Agent grid (matching train.py)
    n_side = int(np.sqrt(n_agents))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    
    T_steps = T_STEPS
    save_dir = Path(__file__).parent
    
    # Zero policy for uncontrolled comparison
    def zero_policy(params, smoke, target, xi):
        n = xi.shape[0]
        return jnp.zeros(n), jnp.zeros((n, 2))
    
    # =========================================================================
    # Visualize multiple test samples (at least 5)
    # =========================================================================
    n_test_samples = min(5, len(test_data['rho_init']))
    print(f"\nVisualizing {n_test_samples} test samples...")
    
    # Create multi-sample comparison figure
    fig = plt.figure(figsize=(16, 4 * n_test_samples))
    gs = fig.add_gridspec(n_test_samples, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.3)
    
    vmin, vmax = 0, 1
    
    for sample_idx in range(n_test_samples):
        print(f"  Sample {sample_idx + 1}/{n_test_samples}...")
        
        smoke_init = jnp.array(test_data['rho_init'][sample_idx])
        rho_target = jnp.array(test_data['rho_target'][sample_idx])
        
        # Run controlled simulation
        smoke_traj, xi_traj, intensity_traj, vel_traj = unroll_controlled(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
            sigma_inject=SIGMA_INJECT, sigma_push=SIGMA_PUSH,
            u_max=0.0, push_max=PUSH_MAX
        )
        smoke_traj = np.array(smoke_traj)
        smoke_controlled_final = smoke_traj[-1]
        
        # Plot: Initial | Controlled | Target
        data = [np.array(smoke_init), smoke_controlled_final, np.array(rho_target)]
        labels = ['Initial', 'Controlled', 'Target']
        
        for col_idx, (arr, label) in enumerate(zip(data, labels)):
            ax = fig.add_subplot(gs[sample_idx, col_idx])
            im = ax.imshow(arr.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
            if sample_idx == 0:
                ax.set_title(label, fontsize=16, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'Sample {sample_idx + 1}', fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Shared colorbar
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r'Smoke Density $\rho$', fontsize=14)
    
    plt.suptitle('NS2D Shape Formation: Multiple Test Samples', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(save_dir / 'ns2d_multi_samples.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: ns2d_multi_samples.png")
    
    # =========================================================================
    # Also create detailed visualization for first sample
    # =========================================================================
    print("\nGenerating detailed visualization for sample 0...")
    
    idx = 0
    smoke_init = jnp.array(test_data['rho_init'][idx])
    rho_target = jnp.array(test_data['rho_target'][idx])
    
    smoke_traj, xi_traj, intensity_traj, vel_traj = unroll_controlled(
        smoke_init, xi_init, rho_target, params, model.apply, T_steps,
        Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
        sigma_inject=SIGMA_INJECT, sigma_push=SIGMA_PUSH,
        u_max=0.0, push_max=PUSH_MAX
    )
    
    smoke_traj = np.array(smoke_traj)
    xi_traj = np.array(xi_traj)
    intensity_traj = np.array(intensity_traj)
    vel_traj = np.array(vel_traj)
    
    print(f"Smoke range: [{smoke_traj.min():.3f}, {smoke_traj.max():.3f}]")
    
    plot_evolution_comparison(
        np.array(smoke_init), smoke_traj, np.array(rho_target), xi_traj,
        filename=str(save_dir / 'ns2d_evolution.png')
    )
    
    plot_control_signals(
        intensity_traj, vel_traj, xi_traj,
        filename=str(save_dir / 'ns2d_controls.png')
    )
    
    # Run UNCONTROLLED simulation
    print("Running uncontrolled simulation...")
    smoke_traj_unctrl, _, _, _ = unroll_controlled(
        smoke_init, xi_init, rho_target, None, zero_policy, T_steps,
        Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
        sigma_inject=SIGMA_INJECT, sigma_push=SIGMA_PUSH,
        u_max=0.0, push_max=0.0
    )
    smoke_unctrl_final = np.array(smoke_traj_unctrl)[-1]
    
    # Initial/Uncontrolled/Final/Target comparison
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.15)
    
    labels = ['Initial', 'Uncontrolled', 'Controlled', 'Target']
    data = [np.array(smoke_init), smoke_unctrl_final, smoke_traj[-1], np.array(rho_target)]
    
    for i, (label, arr) in enumerate(zip(labels, data)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(arr.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(label, fontsize=16, fontweight='bold')
        ax.axis('off')
    
    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r'Smoke Density $\rho$', fontsize=14)
    
    plt.suptitle('NS2D Shape Formation Result', fontsize=20, fontweight='bold', y=1.02)
    plt.savefig(save_dir / 'ns2d_result.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: ns2d_result.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
