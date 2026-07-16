"""
Conference-Quality Visualization for Heat 1d Decentralized DPC
Style reference: Times New Roman fonts, RdBu_r colormap, clean academic layout
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path
from functools import partial

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")

# --- Path Setup for Imports ---
# We still need the project root for imports, but we won't save images there.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedControlNet
import data_utils

# ═══════════════════════════════════════════════════════════════════════════════
# ACADEMIC STYLING (Times New Roman / Serif)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    """Configure matplotlib for academic/conference style."""
    tex_fonts = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 30,
        "font.size": 28,
        "legend.fontsize": 22,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "axes.titlesize": 32,
        "figure.titlesize": 36,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
        "grid.linewidth": 1.0,
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
    # Localized import
    from tesseracts.solverHeat_decentralized import solver
    
    def step_fn(carry, _):
        z_curr, xi_curr = carry
        u_zero = jnp.zeros_like(xi_curr)
        v_zero = jnp.zeros_like(xi_curr)
        z_next, xi_next = solver.implicit_step(z_curr, xi_curr, u_zero, v_zero)
        return (z_next, xi_next), z_next
    
    _, z_traj = jax.lax.scan(step_fn, (z_init, xi_init), None, length=T_steps)
    return z_traj

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_comparison_figure(x_grid, z_init, z_target, z_traj_ctrl, z_traj_unctrl, 
                             u_traj, v_traj, xi_traj, T_steps, output_dir, example_idx=1):
    """
    Create a 6-panel comparison figure in academic style.
    """
    setup_academic_style()
    
    # Convert to numpy
    x = np.array(x_grid).squeeze()
    z_target_np = np.array(z_target).squeeze()
    z_ctrl = np.array(z_traj_ctrl)
    z_unctrl = np.array(z_traj_unctrl)
    u_np = np.array(u_traj)
    v_np = np.array(v_traj)
    xi_np = np.array(xi_traj)
    
    T = z_ctrl.shape[0]
    step = max(1, T // 12)
    plot_indices = list(range(0, T, step))
    if (T - 1) not in plot_indices:
        plot_indices.append(T - 1)
    
    cmap = plt.get_cmap("RdBu_r")
    n_agents = xi_np.shape[1]
    colors_agents = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    fig, axes = plt.subplots(2, 3, figsize=(36, 20))
    
    label_fs = 40
    title_fs = 42
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 1: Uncontrolled Evolution (Row 1, Col 1)
    # ────────────────────────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    for t in plot_indices:
        color = cmap(t / (T - 1))
        lw = 3.5 if t in [0, T - 1] else 2.0
        alpha = 1.0 if t in [0, T - 1] else 0.6
        ax1.plot(x, z_unctrl[t], color=color, lw=lw, alpha=alpha)
    
    ax1.plot(x, z_target_np, 'k--', lw=3.0, label="Target", zorder=10)
    ax1.set_title(r"Evolution (Control = 0)", fontsize=title_fs, fontweight='bold')
    ax1.set_xlabel(r"Position $x$", fontsize=label_fs)
    ax1.set_ylabel(r"Temperature $z(x,t)$", fontsize=label_fs)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 2: Controlled Evolution (Row 1, Col 2)
    # ────────────────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    for t in plot_indices:
        color = cmap(t / (T - 1))
        lw = 3.5 if t in [0, T - 1] else 2.0
        alpha = 1.0 if t in [0, T - 1] else 0.6
        ax2.plot(x, z_ctrl[t], color=color, lw=lw, alpha=alpha)
    
    ax2.plot(x, z_target_np, 'k--', lw=3.0, label="Target", zorder=10)
    
    # Mark final agent positions
    for j in range(n_agents):
        xi_final = float(xi_np[-1, j])
        idx = int(np.clip(xi_final * len(x), 0, len(x)-1))
        ax2.scatter(xi_final, z_ctrl[-1, idx], s=150, color=colors_agents[j], 
                   edgecolors='black', linewidth=2, zorder=15, marker='o')
    
    ax2.set_title(r"Evolution (Decentralized DPC)", fontsize=title_fs, fontweight='bold')
    ax2.set_xlabel(r"Position $x$", fontsize=label_fs)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 3: Agent Trajectories (Row 1, Col 3)
    # ────────────────────────────────────────────────────────────────────────
    ax3 = axes[0, 2]
    time = np.arange(len(xi_np))
    for j in range(n_agents):
        ax3.plot(time, xi_np[:, j], color=colors_agents[j], lw=2.5, label=f'Agent {j+1}')
        ax3.scatter(0, xi_np[0, j], s=80, color=colors_agents[j], marker='o', zorder=5)
        ax3.scatter(len(time)-1, xi_np[-1, j], s=100, color=colors_agents[j], marker='s',
                   edgecolors='black', linewidth=1.5, zorder=5)
    ax3.axhline(y=0, color='gray', ls='--', alpha=0.5, lw=1.5)
    ax3.axhline(y=1, color='gray', ls='--', alpha=0.5, lw=1.5)
    ax3.set_title(r"Agent Trajectories $\xi_i(t)$", fontsize=title_fs, fontweight='bold')
    ax3.set_xlabel(r"Time step", fontsize=label_fs)
    ax3.set_ylabel(r"Position $\xi_i$", fontsize=label_fs)
    ax3.set_ylim([-0.05, 1.05])
    ax3.grid(True, alpha=0.3)
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 4: Control Intensity u (Row 2, Col 1)
    # ────────────────────────────────────────────────────────────────────────
    ax4 = axes[1, 0]
    T_sig = len(u_np)
    time_sig = np.arange(T_sig)
    
    lines_u = []
    for c in range(n_agents):
        line, = ax4.plot(time_sig, u_np[:, c], lw=2.5, color=colors_agents[c])
        lines_u.append(line)
    
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3, lw=1.5)
    ax4.set_title(r"Control Intensity $u_i(t)$", fontsize=title_fs, fontweight='bold')
    ax4.set_xlabel(r"Time step", fontsize=label_fs)
    ax4.set_ylabel(r"Control $u$", fontsize=label_fs)
    ax4.grid(True, alpha=0.3)
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 5: Velocity v (Row 2, Col 2)
    # ────────────────────────────────────────────────────────────────────────
    ax5 = axes[1, 1]
    for c in range(n_agents):
        ax5.plot(time_sig, v_np[:, c], lw=2.5, color=colors_agents[c])
    
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3, lw=1.5)
    ax5.set_title(r"Agent Velocity $v_i(t)$", fontsize=title_fs, fontweight='bold')
    ax5.set_xlabel(r"Time step", fontsize=label_fs)
    ax5.set_ylabel(r"Velocity $v$", fontsize=label_fs)
    ax5.grid(True, alpha=0.3)
    
    # ────────────────────────────────────────────────────────────────────────
    # Panel 6: Tracking Error Comparison (Row 2, Col 3)
    # ────────────────────────────────────────────────────────────────────────
    ax6 = axes[1, 2]
    mse_ctrl = np.mean((z_ctrl - z_target_np[None, :])**2, axis=1)
    mse_unctrl = np.mean((z_unctrl - z_target_np[None, :])**2, axis=1)
    time_err = np.arange(len(mse_ctrl))
    
    ax6.semilogy(time_err, mse_unctrl, 'b-', lw=3.0, label='Uncontrolled', alpha=0.8)
    ax6.semilogy(time_err, mse_ctrl, 'r-', lw=3.0, label='Decentralized DPC', alpha=0.8)
    ax6.fill_between(time_err, mse_ctrl, mse_unctrl, alpha=0.15, color='green')
    
    ax6.set_title(r"Tracking Error (MSE)", fontsize=title_fs, fontweight='bold')
    ax6.set_xlabel(r"Time step", fontsize=label_fs)
    ax6.set_ylabel(r"MSE (log scale)", fontsize=label_fs)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=40)
    
    # ────────────────────────────────────────────────────────────────────────
    # Common Legend
    # ────────────────────────────────────────────────────────────────────────
    labels_u = [fr"$u_{{{c+1}}}$" for c in range(n_agents)]
    combined_handles = [
        plt.Line2D([], [], color=cmap(0.0), lw=3.5, label="Initial state"),
        plt.Line2D([], [], color=cmap(1.0), lw=3.5, label="Final state"),
        plt.Line2D([], [], color='black', ls='--', lw=3.0, label="Target"),
    ] + lines_u
    combined_labels = ["Initial state", "Final state", "Target"] + labels_u
    
    fig.legend(
        handles=combined_handles,
        labels=combined_labels,
        loc='lower center',
        ncol=len(combined_labels),
        fontsize=40,
        frameon=True,
        fancybox=True,
        shadow=False,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    save_path = output_dir / f'heat_dpc_decentralized_ex{example_idx}__.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return str(save_path)

def create_agent_analysis_figure(xi_traj, u_traj, v_traj, z_traj, z_target, output_dir, example_idx=1):
    """Create detailed agent behavior analysis figure."""
    setup_academic_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    n_agents = xi_traj.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    time = np.arange(len(xi_traj))
    
    label_fs = 30
    title_fs = 32
    
    # Panel 1: Agent Trajectories
    ax1 = axes[0, 0]
    lines = []
    labels = []
    for j in range(n_agents):
        line, = ax1.plot(time, xi_traj[:, j], color=colors[j], lw=2.5)
        lines.append(line)
        labels.append(f'Agent {j+1}')
        ax1.scatter(0, xi_traj[0, j], s=80, color=colors[j], marker='o', zorder=5)
        ax1.scatter(len(time)-1, xi_traj[-1, j], s=100, color=colors[j], marker='s', 
                   edgecolors='black', linewidth=1.5, zorder=5)
    ax1.axhline(y=0, color='gray', ls='--', alpha=0.5, lw=1.5)
    ax1.axhline(y=1, color='gray', ls='--', alpha=0.5, lw=1.5) 
    ax1.set_xlabel('Time step', fontsize=label_fs)
    ax1.set_ylabel(r'Position $\xi_i$', fontsize=label_fs)
    ax1.set_title('Agent Trajectories', fontsize=title_fs, fontweight='bold')
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Control Intensities
    ax2 = axes[0, 1]
    for j in range(n_agents):
        ax2.plot(time, u_traj[:, j], color=colors[j], lw=2.5)
    ax2.axhline(y=0, color='gray', ls='-', alpha=0.3, lw=1.5)
    ax2.set_xlabel('Time step', fontsize=label_fs)
    ax2.set_ylabel(r'Control $u_i$', fontsize=label_fs)
    ax2.set_title('Control Intensity', fontsize=title_fs, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Agent Velocities
    ax3 = axes[1, 0]
    for j in range(n_agents):
        ax3.plot(time, v_traj[:, j], color=colors[j], lw=2.5)
    ax3.axhline(y=0, color='gray', ls='-', alpha=0.3, lw=1.5)
    ax3.set_xlabel('Time step', fontsize=label_fs)
    ax3.set_ylabel(r'Velocity $v_i$', fontsize=label_fs)
    ax3.set_title('Agent Velocity', fontsize=title_fs, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Spacetime Heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(np.array(z_traj).T, aspect='auto', cmap='RdBu_r', origin='lower',
                    extent=[0, len(z_traj), 0, 1])
    # Overlay agent paths
    for j in range(n_agents):
        ax4.plot(time, xi_traj[:, j], color='black', lw=2, alpha=0.7)
    ax4.set_xlabel('Time step', fontsize=label_fs)
    ax4.set_ylabel(r'Position $x$', fontsize=label_fs)
    ax4.set_title('Evolution + Agent Paths', fontsize=title_fs, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label(r'$z(x,t)$', fontsize=label_fs-4)
    cbar.ax.tick_params(labelsize=22)
    
    # Common bottom legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='lower center',
        ncol=n_agents,
        fontsize=28,
        frameon=True,
        fancybox=True,
        shadow=False,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    save_path = output_dir / f'heat_dpc_decentralized_agents_ex{example_idx}___.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return str(save_path)

def main():
    print("=" * 60)
    print("  HEAT EQUATION DECENTRALIZED DPC - CONFERENCE VISUALIZATION")
    print("=" * 60)
    
    n_pde, n_agents, T_steps = 100, 8, 300
    n_examples = 3  # Number of test cases
    
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
    
    # --- SAVE LOGIC START ---
    # Current script directory (where this file lives)
    current_script_dir = Path(__file__).resolve().parent
    
    # Output directory relative to THIS script location
    output_dir = current_script_dir / "figures" / "images" / "conf_viz"
    
    print(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- SAVE LOGIC END ---

    saved_files = []
    
    for i in range(n_examples):
        print(f"\n▶ Generating Example {i+1}/{n_examples}...")
        
        key, k1, k2 = jax.random.split(key, 3)
        # Random GRF setup
        _, z_init = data_utils.generate_grf(k1, n_points=n_pde, length_scale=0.15 + i*0.05)
        _, z_target = data_utils.generate_grf(k2, n_points=n_pde, length_scale=0.35 + i*0.05)
        xi_init = jnp.linspace(0.15, 0.85, n_agents)
        
        # Controlled rollout
        z_traj_ctrl, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        
        # Uncontrolled rollout
        z_traj_unctrl = rollout_uncontrolled(z_init, xi_init, dynamics, T_steps)
        
        # Create comparison figure
        f1 = create_comparison_figure(
            x_grid, z_init, z_target, z_traj_ctrl, z_traj_unctrl,
            u_traj, v_traj, xi_traj, T_steps, 
            output_dir=output_dir,
            example_idx=i+1
        )
        saved_files.append(f1)
        
        # Create agent analysis figure
        f2 = create_agent_analysis_figure(
            xi_traj, u_traj, v_traj, z_traj_ctrl, z_target, 
            output_dir=output_dir,
            example_idx=i+1
        )
        saved_files.append(f2)
    
    print("\n" + "=" * 60)
    print("  VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    for f in saved_files:
        print(f"  • {f}")

if __name__ == "__main__":
    main()