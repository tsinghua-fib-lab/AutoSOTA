"""
Paper-Quality Visualization for NS2D Density/Smoke Control - CENTRALIZED.
Three rows:
- Top Row: Natural Evolution (Uncontrolled)
- Middle Row: Controlled Evolution (DPC)
- Bottom Row: Metrics (Wasserstein Distance, Avg Speed, Control Intensity)

Same style as decentralized version.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

from examples.density.centralized.dynamics import unroll_controlled, ns2d_step_jax
from examples.density.decentralized.dynamics import compute_wasserstein_loss
from examples.density.centralized.train import (
    N_AGENTS, T_STEPS, PUSH_MAX, SIGMA_PUSH, BUOYANCY, FEATURES
)
from models.policy_ns2d import NS2DControlNet

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'Nx': 64,
    'Ny': 80,
    'n_agents': N_AGENTS,
    'T_steps': T_STEPS,
    'dt': 1.0,
    
    # Time snapshots to display (indices)
    'snapshot_times': [0, 30, 75, 120, 149],  # 5 snapshots
    
    'params_file': 'ns2d_params.msgpack'
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
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
    
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(0)
    dummy_smoke = jnp.zeros((CONFIG['Nx'], CONFIG['Ny']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(key, dummy_smoke, dummy_smoke, dummy_xi)
    return flax.serialization.from_bytes(init_params, serialized_bytes)


def rollout_uncontrolled(smoke_init, xi_init, T_steps, Nx, Ny, dt, buoyancy):
    """Rollout with zero control inputs (natural dynamics only)."""
    
    def step_fn(carry, _):
        smoke, xi = carry
        n = xi.shape[0]
        push_vel = jnp.zeros((n, 2))
        
        smoke_new = ns2d_step_jax(
            smoke, xi, push_vel,
            dt=dt, buoyancy=buoyancy,
            sigma_push=SIGMA_PUSH, Nx=Nx, Ny=Ny
        )
        return (smoke_new, xi), (smoke_new, xi, push_vel)
    
    _, (smoke_traj, xi_traj, v_traj) = jax.lax.scan(
        step_fn, (smoke_init, xi_init), None, length=T_steps
    )
    return smoke_traj, xi_traj, v_traj


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_snapshot_row(ax_list, smoke_traj, xi_traj, timesteps, v_lim, rho_target=None, show_agents=False):
    """Plot a row of smoke field snapshots using contourf."""
    Nx, Ny = CONFIG['Nx'], CONFIG['Ny']
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    levels = np.linspace(0, v_lim, 100)
    
    cf = None
    for i, t in enumerate(timesteps):
        ax = ax_list[i]
        smoke_snap = np.array(smoke_traj[t])
        
        cf = ax.contourf(
            X, Y, smoke_snap,
            levels=levels,
            cmap=cmc.vik,
            extend='max'
        )
        
        # Draw target contour as dotted line
        if rho_target is not None:
            target_np = np.array(rho_target)
            ax.contour(X, Y, target_np, levels=[0.3], colors='lime',
                      linestyles=':', linewidths=1.5, alpha=0.9)
        
        # Optionally show agent positions
        if show_agents and xi_traj is not None:
            xi = xi_traj[t]
            ax.scatter(xi[:, 0], xi[:, 1], c='cyan', s=15, 
                      edgecolors='white', linewidths=0.5, zorder=10, alpha=0.9)
        
        # Set axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.25)
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


def plot_metrics_row(ax_list, wasserstein_ctrl, wasserstein_unctrl, avg_speed, control_intensity):
    """Plot the three metrics subplots with clean styling."""
    time_steps = np.arange(len(wasserstein_ctrl))
    
    # 1. Wasserstein Distance (Tracking Error)
    ax = ax_list[0]
    ax.plot(time_steps, wasserstein_unctrl, color='gray', linestyle='--', lw=1.2, 
            label='Uncontrolled')
    ax.plot(time_steps, wasserstein_ctrl, color='navy', lw=1.5, label='Controlled')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Dist.')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='center left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Agent Speed
    ax = ax_list[1]
    ax.plot(time_steps, avg_speed, color='forestgreen', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(r'Avg $\dot{\boldsymbol{\xi}}$')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. Control Intensity
    ax = ax_list[2]
    ax.plot(time_steps, control_intensity, color='purple', lw=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg |$\mathbf{v}$|')
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_paper_figure(smoke_traj_ctrl, smoke_traj_unctrl, xi_traj_ctrl, vel_traj_ctrl,
                        rho_target, wasserstein_ctrl, wasserstein_unctrl, 
                        avg_speed, control_intensity,
                        save_name="density_centralized_paper_figure.pdf"):
    """
    Create the paper-quality figure with:
    - Row 1: Natural Evolution (Uncontrolled)
    - Row 2: Controlled Evolution (DPC)
    - Row 3: Metrics (Wasserstein, Agent Speed, Control Intensity)
    """
    setup_paper_style()
    
    n_snaps = len(CONFIG['snapshot_times'])
    fig = plt.figure(figsize=(7.5, 5.5))
    
    # Main Layout: 3 rows
    gs_main = gridspec.GridSpec(3, n_snaps + 1, 
                                 width_ratios=[1]*n_snaps + [0.08],
                                 height_ratios=[1, 1, 0.8],
                                 hspace=0.35, wspace=0.03)
    
    # Row 1: Natural Evolution (top)
    ax_row1 = [fig.add_subplot(gs_main[0, j]) for j in range(n_snaps)]
    
    # Row 2: Controlled Evolution
    ax_row2 = [fig.add_subplot(gs_main[1, j]) for j in range(n_snaps)]
    
    # Determine color limits
    v_lim = max(float(jnp.max(smoke_traj_ctrl)),
                float(jnp.max(smoke_traj_unctrl))) * 0.9
    
    # Plot Natural Evolution (top row)
    cf = plot_snapshot_row(ax_row1, smoke_traj_unctrl, None, 
                           CONFIG['snapshot_times'], v_lim, rho_target=rho_target, show_agents=False)
    ax_row1[2].set_title("Natural Evolution", pad=4)
    
    # Plot Controlled Evolution (middle row)
    plot_snapshot_row(ax_row2, smoke_traj_ctrl, xi_traj_ctrl,
                      CONFIG['snapshot_times'], v_lim, rho_target=rho_target, show_agents=True)
    ax_row2[2].set_title("Controlled Evolution", pad=4)
    
    # Add colorbar on the right (spans rows 0-1)
    cax = fig.add_subplot(gs_main[0:2, -1])
    cbar = fig.colorbar(cf, cax=cax, format='%.1f')
    cbar.ax.tick_params(labelsize=10)
    
    # Row 3: Metrics
    gs_metrics = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2, :], 
                                                   wspace=0.6, width_ratios=[0.15, 1, 1, 1])
    ax_metrics = [fig.add_subplot(gs_metrics[0, j]) for j in range(1, 4)]
    
    # Plot metrics
    plot_metrics_row(ax_metrics, wasserstein_ctrl, wasserstein_unctrl, avg_speed, control_intensity)
    
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
    print("NS2D Density Control (CENTRALIZED) - Paper Figure Generation")
    print("=" * 60)
    
    Nx, Ny = CONFIG['Nx'], CONFIG['Ny']
    n_agents = CONFIG['n_agents']
    T_steps = CONFIG['T_steps']
    dt = CONFIG['dt']
    
    # 1. Initialize Model (Centralized)
    model = NS2DControlNet(
        features=FEATURES,
        v_max=PUSH_MAX
    )
    
    # 2. Load Parameters
    try:
        print(f"\nLoading parameters from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
        print("✓ Parameters loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        print("Run training first to generate parameters.")
        sys.exit(1)
    
    # 3. Load test data
    data_dir = Path(__file__).parent.parent / 'data'
    test_data = np.load(data_dir / 'test_data.npz', allow_pickle=True)
    config = np.load(data_dir / 'config.npz', allow_pickle=True)
    
    # Agent grid
    n_side = int(np.sqrt(n_agents))
    xi_init = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    
    # Number of test cases to process
    n_samples = min(30, len(test_data['rho_init']))
    print(f"\nProcessing {n_samples} test cases...")
    
    for sample_idx in range(n_samples):
        print(f"\n{'='*60}")
        print(f"Processing Sample {sample_idx + 1}/{n_samples}")
        print("="*60)
        
        smoke_init = jnp.array(test_data['rho_init'][sample_idx])
        rho_target = jnp.array(test_data['rho_target'][sample_idx])
        
        # 4. Run Simulations
        print(f"▶ Running controlled trajectory ({T_steps} steps)...")
        smoke_traj_ctrl, xi_traj_ctrl, vel_traj_ctrl = unroll_controlled(
            smoke_init, xi_init, rho_target, params, model.apply, T_steps,
            Nx=Nx, Ny=Ny, dt=dt, buoyancy=BUOYANCY,
            sigma_push=SIGMA_PUSH, push_max=PUSH_MAX
        )
        
        print(f"▶ Running uncontrolled trajectory ({T_steps} steps)...")
        smoke_traj_unctrl, xi_traj_unctrl, _ = rollout_uncontrolled(
            smoke_init, xi_init, T_steps, Nx, Ny, dt, BUOYANCY
        )
        
        # 5. Compute Metrics
        print("▶ Computing metrics...")
        
        # Wasserstein distance over time
        def compute_wasserstein_at_t(smoke_t):
            return compute_wasserstein_loss(smoke_t, rho_target)
        
        wasserstein_ctrl = jax.vmap(compute_wasserstein_at_t)(smoke_traj_ctrl)
        wasserstein_unctrl = jax.vmap(compute_wasserstein_at_t)(smoke_traj_unctrl)
        
        # Agent speeds
        speeds_ctrl = jnp.sqrt(jnp.sum(vel_traj_ctrl**2, axis=-1))
        avg_speed = jnp.mean(speeds_ctrl, axis=1)
        
        # Control intensity (velocity magnitude)
        control_intensity = jnp.mean(jnp.sqrt(jnp.sum(vel_traj_ctrl**2, axis=-1)), axis=1)
        
        print(f"  Final Wasserstein (Controlled):   {float(wasserstein_ctrl[-1]):.4f}")
        print(f"  Final Wasserstein (Uncontrolled): {float(wasserstein_unctrl[-1]):.4f}")
        
        # 6. Generate Figure
        print("▶ Generating paper figure...")
        save_name = f"density_centralized_paper_figure_sample_{sample_idx + 1}_nuovo.pdf"
        create_paper_figure(
            smoke_traj_ctrl, smoke_traj_unctrl, xi_traj_ctrl, vel_traj_ctrl,
            rho_target, wasserstein_ctrl, wasserstein_unctrl, 
            avg_speed, control_intensity,
            save_name=save_name
        )
    
    print("\n" + "=" * 60)
    print(f"Done! Generated {n_samples} paper figures.")
    print("=" * 60)