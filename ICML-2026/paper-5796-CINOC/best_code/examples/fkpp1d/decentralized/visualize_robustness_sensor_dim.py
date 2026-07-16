import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import sys
import flax.serialization
from flax import linen as nn
from typing import Sequence
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.ticker as ticker

jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Output Directory
EXPERIMENT_DIR = Path("figures/sensor_dim")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

from dynamics_dual import PDEDynamics 
from data_utils import generate_grf
from models.policy import DecentralizedControlNet

# --- Config ---
SENSOR_RANGES = [0.04, 0.06, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0]
TEST_AGENT_COUNTS = range(20, 101, 20)
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 50 

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings - Times New Roman for papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        
        # Font sizes
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
        
        # Output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

def rollout_uncontrolled(z_init, xi_init, T_steps, nu=0.005, rho=3.0):
    """Rollout with zero control inputs using internal solver."""
    try:
        import tesseracts.solverFKPP_decentralized.solver as solver
        
        def step_fn(carry, _):
            z_curr, xi_curr = carry
            u_zero = jnp.zeros_like(xi_curr)
            v_zero = jnp.zeros_like(xi_curr)
            z_next, xi_next = solver.fkpp_step_1d(z_curr, xi_curr, u_zero, v_zero, nu=nu, rho=rho)
            return (z_next, xi_next), z_next
        
        _, z_traj = jax.lax.scan(step_fn, (z_init, xi_init), None, length=T_steps)
        return z_traj
    except ImportError:
        print("Warning: Could not import solver for uncontrolled rollout. Skipping.")
        return jnp.zeros((T_steps, z_init.shape[0]))

def create_comparison_figure(x_grid, z_init, z_target, z_traj_ctrl, z_traj_unctrl, 
                             u_traj, v_traj, xi_traj, T_steps, s_range, example_idx):
    """Create 6-panel comparison figure (Scaled for Paper Width ~7 inches)."""
    setup_paper_style()
    
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
    if (T - 1) not in plot_indices: plot_indices.append(T - 1)
    
    cmap = plt.get_cmap("RdBu_r")
    n_agents = xi_np.shape[1]
    colors_agents = plt.cm.viridis(np.linspace(0, 1, n_agents))
    
    # Use 7.0 width (full page width) and appropriate height
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5), constrained_layout=True)
    
    # 1. Uncontrolled
    ax1 = axes[0, 0]
    for t in plot_indices:
        color = cmap(t / (T - 1))
        lw = 1.5 if t in [0, T - 1] else 0.8
        alpha = 1.0 if t in [0, T - 1] else 0.6
        ax1.plot(x, z_unctrl[t], color=color, lw=lw, alpha=alpha)
    ax1.plot(x, z_target_np, 'k--', lw=1.2, label="Target", zorder=10)
    ax1.set_title(r"Evolution (Control = 0)")
    ax1.set_xlabel(r"Position $x$")
    ax1.set_ylabel(r"Pop. $z(x,t)$")
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 2. Controlled
    ax2 = axes[0, 1]
    for t in plot_indices:
        color = cmap(t / (T - 1))
        lw = 1.5 if t in [0, T - 1] else 0.8
        alpha = 1.0 if t in [0, T - 1] else 0.6
        ax2.plot(x, z_ctrl[t], color=color, lw=lw, alpha=alpha)
    ax2.plot(x, z_target_np, 'k--', lw=1.2, label="Target", zorder=10)
    
    # Scatter 100 agents (Scaled down size)
    for j in range(n_agents):
        xi_final = float(xi_np[-1, j])
        idx = int(np.clip(xi_final * len(x), 0, len(x)-1))
        # Smaller scatter points (s=15)
        ax2.scatter(xi_final, z_ctrl[-1, idx], s=15, color=colors_agents[j], edgecolors='black', linewidth=0.3, zorder=15)
        
    ax2.set_title(f"Decentralized (Range={s_range})")
    ax2.set_xlabel(r"Position $x$")
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 3. Trajectories
    ax3 = axes[0, 2]
    time = np.arange(len(xi_np))
    for j in range(n_agents):
        ax3.plot(time, xi_np[:, j], color=colors_agents[j], lw=0.8, alpha=0.8)
        
    ax3.set_title(r"Agent Trajectories $\xi_i(t)$")
    ax3.set_xlabel(r"Time step")
    ax3.set_ylabel(r"Pos. $\xi_i$")
    ax3.set_ylim([-0.05, 1.05])
    ax3.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 4. Control u
    ax4 = axes[1, 0]
    for c in range(n_agents):
        ax4.plot(time, u_np[:, c], lw=0.6, color=colors_agents[c], alpha=0.6)
    ax4.set_title(r"Control $u_i(t)$")
    ax4.set_xlabel(r"Time step")
    ax4.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 5. Velocity v
    ax5 = axes[1, 1]
    for c in range(n_agents):
        ax5.plot(time, v_np[:, c], lw=0.6, color=colors_agents[c], alpha=0.6)
    ax5.set_title(r"Velocity $v_i(t)$")
    ax5.set_xlabel(r"Time step")
    ax5.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 6. Error
    ax6 = axes[1, 2]
    mse_ctrl = np.mean((z_ctrl - z_target_np[None, :])**2, axis=1)
    mse_unctrl = np.mean((z_unctrl - z_target_np[None, :])**2, axis=1)
    ax6.semilogy(time, mse_unctrl, 'b-', lw=1.5, label='Uncontrolled', alpha=0.8)
    ax6.semilogy(time, mse_ctrl, 'r-', lw=1.5, label='Decentralized', alpha=0.8)
    ax6.set_title(r"Tracking Error (MSE)")
    ax6.set_xlabel(r"Time step")
    ax6.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    ax6.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    save_name = EXPERIMENT_DIR / f"comparison_range_{s_range}_ex{example_idx}.pdf" # PDF is better for papers
    plt.savefig(save_name)
    plt.close()
    return save_name

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def load_params(model, filepath):
    """Initializes the model with weights from file."""
    if not filepath.exists():
        return None
        
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    
    dummy_init = model.init(
        jax.random.PRNGKey(0), 
        jnp.zeros((N_PDE,)), 
        jnp.zeros((N_PDE,)), 
        jnp.zeros((30,))
    )
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_sensor_dims():
    results = []

    # Pre-generate Test Data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    _, z_init_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.2))(jax.random.split(k1, N_TEST_SAMPLES))
    _, z_target_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.4))(jax.random.split(k2, N_TEST_SAMPLES))
    
    x_grid = jnp.linspace(0, 1, N_PDE)

    for s_range in SENSOR_RANGES:
        print(f"--- Evaluating Sensor Range: {s_range} ---")
        
        model = DecentralizedControlNet(
            features=(64, 64), 
            sensor_range=s_range
        )
        
        param_path = EXPERIMENT_DIR / f"sensor_dim_{s_range}_params.msgpack"
        params = load_params(model, param_path)
        
        if params is None:
            print(f"Skipping range {s_range} (Weights not found)")
            continue

        # Initializing JAX-native dynamics
        dynamics = PDEDynamics(policy_apply_fn=model.apply)

        # Part A: Visualization (100 Agents)
        print(f"   > Generating visualization plots for range {s_range}...")
        viz_key = jax.random.PRNGKey(101)
        n_viz_agents = 100 
        
        for ex_idx in range(1, 4):
            viz_key, sk1, sk2 = jax.random.split(viz_key, 3)
            _, z0_viz = generate_grf(sk1, n_points=N_PDE, length_scale=0.15 + (ex_idx*0.05))
            _, zt_viz = generate_grf(sk2, n_points=N_PDE, length_scale=0.35 + (ex_idx*0.05))
            xi0_viz = jnp.linspace(0.1, 0.9, n_viz_agents)
            
            z_traj_c, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                z0_viz, xi0_viz, zt_viz, params, T_STEPS, key=jax.random.PRNGKey(0)
            )
            z_traj_u = rollout_uncontrolled(z0_viz, xi0_viz, T_STEPS)
            
            create_comparison_figure(x_grid, z0_viz, zt_viz, z_traj_c, z_traj_u, 
                                    u_traj, v_traj, xi_traj, T_STEPS, s_range, ex_idx)
        
        # Part B: MSE Stats
        print(f"   > Calculating MSE stats...")
        for n_agents in TEST_AGENT_COUNTS:
            xi_test = jnp.linspace(0.1, 0.9, n_agents)
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))

            @jax.jit
            def run_single(z_i, z_t, xi_i):
                z_traj, _, _, _ = dynamics.unroll_controlled(
                    z_i, xi_i, z_t, params, T_STEPS, 
                    key=jax.random.PRNGKey(0), noise_u=0, noise_z=0
                )
                return jnp.mean((z_traj[-1] - z_t)**2) 

            final_mses = jax.vmap(run_single)(z_init_test, z_target_test, xi_batch)
            results.append({
                "Sensor Range": s_range,
                "Agents": n_agents,
                "MSE": float(jnp.mean(final_mses)),
                "Std": float(jnp.std(final_mses))
            })

    return pd.DataFrame(results)

def plot_sensor_sensitivity(df):
    setup_paper_style()
    
    # Single column size (approx 5.0 x 3.5 inches)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    
    df_plot = df.copy()
    df_plot["Sensor Range"] = df_plot["Sensor Range"].astype(str)
    unique_ranges = sorted(df["Sensor Range"].unique())
    unique_ranges_str = [str(x) for x in unique_ranges]

    sns.lineplot(
        data=df_plot, 
        x="Agents", 
        y="MSE", 
        hue="Sensor Range", 
        hue_order=unique_ranges_str,
        style="Sensor Range",
        palette="tab10", 
        markers=True, 
        markersize=6, 
        linewidth=1.5,
        ax=ax
    )
    
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training (M=30)")
    ax.set_title(r"Sensor Range Sensitivity")
    ax.set_ylabel("Final Tracking Error (MSE)")
    ax.set_xlabel("Deployment Agent Count")
    
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Compact Legend
    ax.legend(
        title="Range (%)", 
        loc='best', 
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9, 
        ncol=2 # Two columns to make it squarer/smaller
    )
    
    save_path = EXPERIMENT_DIR / "sensor_dimension_sensitivity.pdf"
    plt.savefig(save_path)
    print(f"Saved sensitivity plot to {save_path}")

if __name__ == "__main__":
    print(f"Starting Sensor Dimension Visualization...")
    
    df_results = evaluate_sensor_dims()
    
    if not df_results.empty:
        df_results.to_csv(EXPERIMENT_DIR / "sensor_metrics.csv", index=False)
        plot_sensor_sensitivity(df_results)
    else:
        print("No results generated.")