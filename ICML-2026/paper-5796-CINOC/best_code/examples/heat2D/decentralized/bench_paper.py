"""
ICML Paper Figure: Heat 2D Decentralized Control Benchmark
Compares tracking performance across three scenarios:
1. Uncontrolled baseline
2. DPC controlled (obstacle-free)
3. DPC controlled (with obstacles)
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import flax.serialization
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
from data_utils import get_training_data

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Obstacle configuration: [x_center, y_center, radius]
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   # Diagonal line obstacle 1
    [0.50, 0.50, 0.06],   # Diagonal line obstacle 2 (center)
    [0.70, 0.70, 0.06],   # Diagonal line obstacle 3
])
R_SAFE_OBSTACLE = 0.04  # Safety margin around obstacles

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
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Heat2D Paper Benchmark")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--t-steps", type=int, default=300)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--params-file", default="decentralized_params_heat2d.msgpack")
    parser.add_argument("--params-file-obstacles", default="../../heat2D_obstacles/decentralized/decentralized_params_heat2d_obstacles.msgpack")
    parser.add_argument("--dataset-dir", default="../data")
    parser.add_argument("--out-file", default="heat2d_paper_benchmark.pdf")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def build_agent_grid(n_agents):
    """Build agent grid at exact positions [0.2, 0.4, 0.6, 0.8] in both axes."""
    n_side = int(jnp.sqrt(n_agents))
    positions_1d = jnp.array([0.2, 0.4, 0.6, 0.8])[:n_side]
    xi_template = []
    for i in range(n_side):
        for j in range(n_side):
            if len(xi_template) < n_agents:
                xi_template.append([float(positions_1d[i]), float(positions_1d[j])])
    return jnp.array(xi_template)

def load_params(model, params_file, n_grid, n_agents):
    try:
        with open(params_file, "rb") as f:
            serialized_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: '{params_file}' not found. Run training first.")
        sys.exit(1)

    dummy_key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n_grid, n_grid))
    dummy_xi = jnp.zeros((n_agents, 2))
    dummy_params = model.init(dummy_key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(dummy_params, serialized_bytes)

def zero_policy_apply(params, local_z, z_target, local_xi):
    """Dummy policy that returns zero control inputs."""
    n_batch = local_xi.shape[0]
    return jnp.zeros((n_batch,)), jnp.zeros((n_batch, 2))

def draw_obstacles(ax, obstacles, show_safety=True):
    """Draw obstacles with safety margins."""
    for obs in obstacles:
        x, y, r = obs
        # Physical obstacle (solid)
        circle = Circle((x, y), r, facecolor='red', alpha=0.5,
                       edgecolor='darkred', linewidth=1.5, zorder=5)
        ax.add_patch(circle)

        # Safety margin (dashed with gradient)
        if show_safety:
            safety_circle = Circle((x, y), r + R_SAFE_OBSTACLE,
                                  facecolor='red', fill=True, alpha=0.15,
                                  edgecolor='red', linewidth=1.0,
                                  linestyle='--', zorder=4)
            ax.add_patch(safety_circle)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    setup_style()

    print("="*70)
    print("  HEAT 2D DECENTRALIZED CONTROL - ICML PAPER BENCHMARK")
    print("="*70)

    # Configuration
    n_grid = args.n_grid
    n_agents = args.n_agents
    T_steps = args.t_steps
    N_eval = args.n_eval

    # Load model and parameters
    model = DecentralizedHeat2DControlNet(features=(16, 32))

    # Setup output directory
    save_dir = Path("figures/images/obstacle_ablation_viz")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading parameters...")
    params = load_params(model, args.params_file, n_grid, n_agents)
    params_obstacles = load_params(model, args.params_file_obstacles, n_grid, n_agents)
    print(f"✓ Loaded trained parameters for both scenarios")

    # Generate evaluation data
    print(f"\nGenerating {N_eval} evaluation samples...")
    pool_size = max(N_eval, args.pool_size)
    z_init_pool, z_target_pool, _ = get_training_data(
        n_samples=pool_size,
        n_grid=n_grid,
        dataset_dir=args.dataset_dir,
    )

    val_key = jax.random.PRNGKey(args.seed)
    idx = jax.random.randint(val_key, (N_eval,), 0, len(z_init_pool))
    z_init_batch = z_init_pool[idx]
    z_target_batch = z_target_pool[idx]

    # Initialize agents
    xi_init_single = build_agent_grid(n_agents)
    xi_init_batch = jnp.tile(xi_init_single, (N_eval, 1, 1))

    # Run simulations
    print("\nRunning simulations...")

    # Scenario 1: Uncontrolled
    dynamics_unc = PDEDynamics(policy_apply_fn=zero_policy_apply)

    # Scenario 2: Controlled (no obstacles)
    dynamics_ctrl = PDEDynamics(policy_apply_fn=model.apply)

    # Scenario 3: Controlled (with obstacles) - uses same dynamics, different params
    dynamics_ctrl_obs = PDEDynamics(policy_apply_fn=model.apply)

    def run_all_scenarios(z_init, xi_init, z_target):
        # Uncontrolled
        z_u, xi_u, _, _ = dynamics_unc.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        # Controlled (no obstacles)
        z_c, xi_c, _, _ = dynamics_ctrl.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        # Controlled (with obstacles)
        z_co, xi_co, _, _ = dynamics_ctrl_obs.unroll_controlled(
            z_init, xi_init, z_target, params_obstacles, T_steps
        )
        return z_u, xi_u, z_c, xi_c, z_co, xi_co

    results = []
    for start in range(0, N_eval, args.chunk_size):
        end = min(N_eval, start + args.chunk_size)
        chunk_results = jax.vmap(run_all_scenarios)(
            z_init_batch[start:end],
            xi_init_batch[start:end],
            z_target_batch[start:end]
        )
        results.append(chunk_results)

    # Concatenate all chunks
    z_unc_all = jnp.concatenate([r[0] for r in results], axis=0)
    xi_unc_all = jnp.concatenate([r[1] for r in results], axis=0)
    z_ctrl_all = jnp.concatenate([r[2] for r in results], axis=0)
    xi_ctrl_all = jnp.concatenate([r[3] for r in results], axis=0)
    z_ctrl_obs_all = jnp.concatenate([r[4] for r in results], axis=0)
    xi_ctrl_obs_all = jnp.concatenate([r[5] for r in results], axis=0)

    print("✓ Simulations complete")

    # Calculate metrics
    print("\nCalculating metrics...")
    targets_expanded = z_target_batch[:, None, :, :]
    mse_unc = jnp.mean((z_unc_all - targets_expanded)**2, axis=(1, 2, 3))
    mse_ctrl = jnp.mean((z_ctrl_all - targets_expanded)**2, axis=(1, 2, 3))
    mse_ctrl_obs = jnp.mean((z_ctrl_obs_all - targets_expanded)**2, axis=(1, 2, 3))

    print(f"  Uncontrolled MSE:        {jnp.mean(mse_unc):.6f}")
    print(f"  Controlled (no obs) MSE: {jnp.mean(mse_ctrl):.6f}")
    print(f"  Controlled (w/ obs) MSE: {jnp.mean(mse_ctrl_obs):.6f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    print("\nGenerating paper figure...")

    fig = plt.figure(figsize=(15, 5))

    # Select sample for trajectory visualization
    sample_idx = int(jnp.clip(args.sample_idx, 0, N_eval - 1))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1: MSE Distribution (3 boxplots)
    # ─────────────────────────────────────────────────────────────────────────
    ax1 = plt.subplot(1, 3, 1)
    bp = ax1.boxplot(
        [mse_unc, mse_ctrl, mse_ctrl_obs],
        tick_labels=['Uncontrolled', 'DPC\n(obstacle-free)', 'DPC\n(w/ obstacles)'],
        patch_artist=True,
        widths=0.6
    )

    # Color the boxes
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_title(f'Tracking Performance (N={N_eval})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', labelsize=9)

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2: Controlled Trajectories (no obstacles)
    # ─────────────────────────────────────────────────────────────────────────
    ax2 = plt.subplot(1, 3, 2)

    # Draw all trajectories
    for i in range(n_agents):
        traj_x = xi_ctrl_all[sample_idx, :, i, 0]
        traj_y = xi_ctrl_all[sample_idx, :, i, 1]
        ax2.plot(traj_x, traj_y, alpha=0.6, color='blue', linewidth=1.2)

        # Initial position (small x)
        ax2.scatter(traj_x[0], traj_y[0], c='green', s=25, marker='x',
                   linewidths=1.5, zorder=10)
        # Final position (small dot)
        ax2.scatter(traj_x[-1], traj_y[-1], c='blue', s=20, zorder=10)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    ax2.set_title('Agent Trajectories\n(Obstacle-Free)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3: Controlled Trajectories with Obstacles
    # ─────────────────────────────────────────────────────────────────────────
    ax3 = plt.subplot(1, 3, 3)

    # Draw obstacles first (so they appear behind trajectories)
    draw_obstacles(ax3, OBSTACLES, show_safety=True)

    # Draw all trajectories
    for i in range(n_agents):
        traj_x = xi_ctrl_obs_all[sample_idx, :, i, 0]
        traj_y = xi_ctrl_obs_all[sample_idx, :, i, 1]
        ax3.plot(traj_x, traj_y, alpha=0.6, color='blue', linewidth=1.2)

        # Initial position (small x)
        ax3.scatter(traj_x[0], traj_y[0], c='green', s=25, marker='x',
                   linewidths=1.5, zorder=10)
        # Final position (small dot)
        ax3.scatter(traj_x[-1], traj_y[-1], c='blue', s=20, zorder=10)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('x', fontsize=10)
    ax3.set_ylabel('y', fontsize=10)
    ax3.set_title('Agent Trajectories\n(With Obstacles)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Add legend for obstacles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, edgecolor='darkred', label='Obstacle'),
        Patch(facecolor='red', alpha=0.15, edgecolor='red', linestyle='--', label='Safety Zone'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='green',
                  markersize=6, label='Start', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                  markersize=5, label='End'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save figure
    save_path = save_dir / args.out_file
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")

    # Also save PNG version
    png_path = save_dir / Path(args.out_file).with_suffix('.png').name
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")

    plt.close()

    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
