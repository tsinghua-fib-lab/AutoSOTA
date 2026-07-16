import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import flax.serialization
import optax
from pathlib import Path
from functools import partial
from tqdm import trange
from matplotlib.ticker import ScalarFormatter

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

# --- 1. Setup Directories ---
MODELS_DIR = Path("models/analysis")
FIGURES_DIR = Path("figures/conjecture")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

# --- 2. Training Logic ---
def loss_fn(params, z_init, xi_init, z_target, dynamics, l_effort_weight, n_agents, T_steps, R_safe=0.05):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    # Tracking Error
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    # Control Effort Penalty
    l_effort = jnp.mean(u_traj ** 2) 
    
    # Boundary and Collision Penalties
    margin = 0.02
    l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + 
                       jnp.maximum(0, xi_traj - (1.0 - margin))**2)
    dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 1.0)) ** 2)
    l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)

    return 5.0 * l_track + l_effort_weight * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel, l_track

@partial(jax.jit, static_argnames=('dynamics', 'l_effort_weight', 'n_agents', 'T_steps', 'optimizer'))
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics, l_effort_weight, n_agents, T_steps, optimizer):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None, None))
    def mean_loss_fn(p):
        losses, track_losses = batched_loss_fn(p, z_init_batch, xi_init_batch, z_target_batch, dynamics, l_effort_weight, n_agents, T_steps)
        return jnp.mean(losses), jnp.mean(track_losses)
    (loss, track_l), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, track_l

def train_model(l_weight, n_pde, n_agents, epochs, dynamics, model, optimizer):
    param_path = MODELS_DIR / f"params_lambda_{l_weight}.msgpack"
    if param_path.exists():
        print(f"Skipping training: Model for lambda={l_weight} already exists.")
        return
    
    print(f"Training model for lambda_effort = {l_weight}...")
    key = jax.random.PRNGKey(int(l_weight * 1000) if l_weight > 0 else 42)
    params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    opt_state = optimizer.init(params)
    
    all_keys = jax.random.split(key, 5000)
    _, z_init_all = jax.vmap(partial(data_utils.generate_grf, n_points=n_pde, length_scale=0.2))(all_keys)
    _, z_target_all = jax.vmap(partial(data_utils.generate_grf, n_points=n_pde, length_scale=0.4))(all_keys)
    xi_init_batch = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (32, 1))

    for epoch in trange(epochs):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (32,), 0, 5000)
        params, opt_state, _, _ = train_step(params, opt_state, z_init_all[idx], xi_init_batch, z_target_all[idx], dynamics, l_weight, n_agents, 300, optimizer)

    with open(param_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))

# --- 3. Evaluation with Temporal Windowing (Updated) ---
def run_comparison(n_agents_list, lambda_list, n_pde, T_steps, z_init, z_target, window_ratio=0.7):
    all_results = []
    start_idx = int(T_steps * (1.0 - window_ratio))
    print(f"Analyzing effort from step {start_idx} to {T_steps} (Window: {window_ratio*100:.0f}%)")

    for l_weight in lambda_list:
        param_path = MODELS_DIR / f"params_lambda_{l_weight}.msgpack"
        model = DecentralizedControlNet(features=(64, 64))
        # Initializing native JAX dynamics
        dynamics = PDEDynamics(policy_apply_fn=model.apply)
        
        with open(param_path, 'rb') as f:
            bytes_data = f.read()
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((20,)))
        params = flax.serialization.from_bytes(params, bytes_data)

        for n in n_agents_list:
            print(f"Testing Lambda={l_weight}, N={n}...")
            xi_init = jnp.linspace(0.2, 0.8, n)
            z_traj, _, u_traj, _ = dynamics.unroll_controlled(z_init, xi_init, z_target, params, T_steps)
            
            mse = float(jnp.mean((z_traj[-1] - z_target)**2))
            u_window = u_traj[start_idx:] 
            window_steps = T_steps - start_idx
            
            # 1. Total squared effort: sum_i(u_i^2) averaged over time steps
            total_effort_sq = float(jnp.sum(u_window**2) / window_steps) 
            
            # 2. Total absolute effort: sum_i(|u_i|) averaged over time steps
            total_effort_abs = float(jnp.sum(jnp.abs(u_window)) / window_steps)
            
            all_results.append({
                "lambda": l_weight, 
                "n_agents": n, 
                "mse": mse, 
                "total_effort_sq": total_effort_sq,
                "total_effort_abs": total_effort_abs,
                "window": f"Last {window_ratio*100:.0f}%"
            })
    return pd.DataFrame(all_results)

# --- 4. Plotting (Updated for Paper Style) ---
def plot_conjecture_results_separated(df, window_label):
    setup_paper_style()
    
    # Professional color palette
    colors = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#c0392b', '#d35400']
    
    # Common helper for formatting effort axes
    def format_effort_axis(ax):
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.grid(True, which="both", ls="--", alpha=0.3, linewidth=0.5)
        # Legend outside to save space in small plots
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)

    # --- Plot 1: Tracking MSE ---
    fig1, ax1 = plt.subplots(figsize=(5.0, 3.5))
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax1.semilogy(sub['n_agents'], sub['mse'], marker='o', markersize=5, 
                     label=f'$\lambda_u={l}$', color=colors[i % len(colors)], linewidth=1.5)
    
    # ax1.set_title("Zero-Shot Scalability: Tracking MSE")
    ax1.set_xlabel("Number of Agents ($M$)")
    ax1.set_ylabel("Final $L^2$ Error")
    ax1.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Training $M$')
    ax1.grid(True, which="both", ls="--", alpha=0.3, linewidth=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
    fig1.savefig(FIGURES_DIR / f"scaling_mse_{window_label.replace(' ', '_')}.pdf", bbox_inches='tight')

    # --- Plot 2: Squared Effort ---
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.5))
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax2.loglog(sub['n_agents'], sub['total_effort_sq'], marker='s', markersize=5,
                   label=f'$\lambda_u={l}$', color=colors[i % len(colors)], linewidth=1.5)
    
    # ax2.set_title(r"Steady-State Effort: $\sum u_i^2$")
    ax2.set_xlabel("Number of Agents ($M$)")
    ax2.set_ylabel(r"Mean $\sum u_i^2$")
    format_effort_axis(ax2)
    fig2.savefig(FIGURES_DIR / f"scaling_effort_sq_{window_label.replace(' ', '_')}.pdf", bbox_inches='tight')

    # --- Plot 3: Absolute Effort ---
    fig3, ax3 = plt.subplots(figsize=(5.0, 3.5))
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax3.loglog(sub['n_agents'], sub['total_effort_abs'], marker='^', markersize=5,
                   label=f'$\lambda_u={l}$', color=colors[i % len(colors)], linewidth=1.5)
    
    # ax3.set_title(r"Steady-State Effort: $\sum |u_i|$")
    ax3.set_xlabel("Number of Agents ($M$)")
    ax3.set_ylabel(r"Mean $\sum |u_i|$")
    format_effort_axis(ax3)
    fig3.savefig(FIGURES_DIR / f"scaling_effort_abs_{window_label.replace(' ', '_')}.pdf", bbox_inches='tight')
    
    # Close all figures
    plt.close('all')
    print(f"Three separate PDFs saved to {FIGURES_DIR}")

def main():
    n_pde, T_steps = 100, 300
    lambda_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1]
    n_agents_list = [15, 20, 30, 40, 50, 60]
    WINDOW_RATIO = 0.7

    model = DecentralizedControlNet(features=(64, 64))
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))

    # Using native JAX dynamics wrapper
    dynamics_local = PDEDynamics(policy_apply_fn=model.apply)
    
    for l in lambda_list:
        train_model(l, n_pde, 20, 500, dynamics_local, model, optimizer)

    key = jax.random.PRNGKey(42)
    _, z_init = data_utils.generate_grf(key, n_points=n_pde, length_scale=0.2)
    _, z_target = data_utils.generate_grf(jax.random.PRNGKey(43), n_points=n_pde, length_scale=0.4)
    
    results_df = run_comparison(n_agents_list, lambda_list, n_pde, T_steps, 
                                z_init, z_target, window_ratio=WINDOW_RATIO)
        
    plot_conjecture_results_separated(results_df, f"Last {int(WINDOW_RATIO*100)}%")
    results_df.to_csv(FIGURES_DIR / "conjecture_data_windowed.csv", index=False)

if __name__ == "__main__":
    main()