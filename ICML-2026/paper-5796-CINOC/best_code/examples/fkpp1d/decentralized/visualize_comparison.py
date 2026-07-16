"""
Scalability Analysis Visualization for Decentralized 1D Fisher-KPP Control:
evaluates zero-shot performance of a DecentralizedControlNet policy
as the number of agents varies beyond training conditions with no further fine tuning (Zero-Shot).

Expected Outcome:
* Tracking MSE should remain low and stable as N increases beyond training conditions and goes beyond the underactuated regime (too little agents, 
whose basis linear combinations are insufficient to reach the target).
* Total Control Effort should remain stable as N increases, demonstrating self-normalization effects (bounded control forcing term norm ||B||)
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import flax.serialization
from pathlib import Path

# Import decentralized components
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

def load_params(model, filepath, n_pde=100, n_agents=8):
    """Loads weights from a msgpack file with a dummy initialization."""
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def run_simulation_suite(n_agents_list, params_file, z_init, z_target, n_pde, T_steps):
    """Executes the simulation for each N and returns a DataFrame of metrics."""
    results = {"n_agents": [], "mse": [], "u_effort": []}
    
    for n_agents in n_agents_list:
        print(f">>> Simulating N={n_agents} Actuators...")
        model = DecentralizedControlNet(features=(64, 64))
        # Initializing JAX-native dynamics
        dynamics = PDEDynamics(policy_apply_fn=model.apply)
        
        try:
            params = load_params(model, params_file, n_pde, n_agents)
        except Exception as e:
            print(f"Error loading params for N={n_agents}: {e}")
            continue

        xi_init = jnp.linspace(0.2, 0.8, n_agents)
        # Unroll the controlled trajectory
        z_traj, _, u_traj, _ = dynamics.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )

        # Compute Metrics: MSE and Control Effort 
        final_mse = float(jnp.mean((z_traj[-1] - z_target)**2))
        total_effort = float(jnp.sum(u_traj)) 

        results["n_agents"].append(int(n_agents))
        results["mse"].append(final_mse)
        results["u_effort"].append(total_effort)
        
    return pd.DataFrame(results)

def plot_scalability_analysis(df, output_path, training_n=20):
    """Generates a dual-axis plot."""
    # Use high-quality styles
    plt.style.use('seaborn-v0_8-paper')
    fig, ax1 = plt.subplots(figsize=(8, 5))

    n_vals = df["n_agents"].values
    mse_vals = df["mse"].values
    effort_vals = df["u_effort"].values

    # Axis 1: Tracking Performance (MSE)
    color_mse = '#1f77b4' 
    ax1.set_xlabel('Number of Actuators ($n$)', fontsize=12)
    ax1.set_ylabel('Final Tracking MSE (Log Scale)', color=color_mse, fontsize=12, fontweight='bold')
    ax1.semilogy(n_vals, mse_vals, marker='o', markersize=7, linestyle='-', 
                 color=color_mse, linewidth=2, label='Tracking MSE ($L^2$ error)')
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Axis 2: Resource Cost (Control Effort)
    ax2 = ax1.twinx()
    color_effort = '#e67e22' 
    ax2.set_ylabel('Total Control Effort ($\sum u_i$)', color=color_effort, fontsize=12, fontweight='bold')
    ax2.plot(n_vals, effort_vals, marker='s', markersize=7, linestyle='--', 
             color=color_effort, alpha=0.8, linewidth=2, label='Control Effort ($\mathcal{L}_{force}$)')
    ax2.tick_params(axis='y', labelcolor=color_effort)

    # Highlight the Training Point (Zero-Shot Boundary)
    ax1.axvline(x=training_n, color='#c0392b', linestyle='-', linewidth=2.5, alpha=0.6)
    # ax1.text(training_n + 0.5, ax1.get_ylim()[1] * 0.4, 'Training Boundary ($N=20$)', 
    #          color='#c0392b', rotation=90, fontweight='bold', verticalalignment='top')

    plt.title("Zero-Shot Policy Scalability & Self-Normalization Analysis", fontsize=14, pad=15)
    
    # Combined Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True, fancybox=True, shadow=True)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # --- PDE and Simulation Parameters ---
    n_pde = 100
    T_steps = 300
    training_anchor = 20
    jax.config.update("jax_platform_name", "cpu")
    
    # Path updated to conference folder
    output_dir = Path("figures/images/conference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup PDE State ---
    key = jax.random.PRNGKey(42)
    key, sub1, sub2 = jax.random.split(key, 3)
    _, z_init = data_utils.generate_grf(sub1, n_points=n_pde, length_scale=0.2)
    _, z_target = data_utils.generate_grf(sub2, n_points=n_pde, length_scale=0.4)

    # --- Run Simulations ---
    # Define dense points near training and sparse points for extrapolation
    n_agents_list = list(np.arange(10, 21, 1)) + list(np.arange(25, 65, 5))
    params_file = 'decentralized_params.msgpack'
    
    # Executing simulations without Tesseract runtime
    results_df = run_simulation_suite(
        n_agents_list, params_file, z_init, z_target, n_pde, T_steps
    )

    # --- Save and Visualize ---
    results_df.to_csv(output_dir / "scalability_metrics.csv", index=False)
    plot_scalability_analysis(
        results_df, 
        output_dir / "scalability_plot.pdf",
        training_n=training_anchor
    )
    print(f"\n--- Analysis Complete. Files saved to {output_dir}/scalability_plot.pdf ---")

if __name__ == "__main__":
    main()