import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import flax.serialization
from pathlib import Path
from functools import partial

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- Config ---
MODELS_DIR = Path("figures/noise_experiments/robustness_transfer")
OUTPUT_DIR = Path("figures/noise_experiments/effort")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PDE = 100
T_STEPS = 300
TEST_AGENT_COUNTS = list(range(20, 201, 10)) # [20, 30, ..., 100]

def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    # Initialize dummy params with the training agent count (30) to match structure
    dummy_init = model.init(jax.random.PRNGKey(0), jnp.zeros((N_PDE,)), jnp.zeros((N_PDE,)), jnp.zeros((30,)))
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_effort_scaling():
    """
    Evaluates effort metrics (L1 and L2 norms of control) across varying agent counts.
    Uses native JAX dynamics for steady-state analysis.
    """
    results = []
    
    model = DecentralizedControlNet(features=(64, 64))
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    # Define models to test
    model_files = {
        "Baseline": MODELS_DIR / "baseline_params.msgpack",
        "Low Noise": MODELS_DIR / "low_noise_params.msgpack"
    }

    # Load parameters
    loaded_params = {}
    for name, path in model_files.items():
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        loaded_params[name] = load_params(model, path)
    
    # Generate a fixed evaluation environment
    key = jax.random.PRNGKey(42)
    key_init, key_target = jax.random.split(key)
    # Using a single representative sample for clear trend lines
    _, z_init = generate_grf(key_init, n_points=N_PDE, length_scale=0.2)
    _, z_target = generate_grf(key_target, n_points=N_PDE, length_scale=0.4)

    # Determine window for steady state (last 70%)
    start_step = int(T_STEPS * (1.0 - 0.70)) # Skip first 30%
    print(f"Analyzing metrics from step {start_step} to {T_STEPS} (Steady State)")

    for n_agents in TEST_AGENT_COUNTS:
        print(f"Evaluating M={n_agents}...")
        
        # Interpolate positions for this agent count
        xi_init = jnp.linspace(0.2, 0.8, n_agents)

        for m_name, params in loaded_params.items():
            # Updated API Call: passing key, noise_u, and noise_z
            z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
                z_init, xi_init, z_target, params, T_STEPS, 
                key=jax.random.PRNGKey(0), 
                noise_u=0.0, # Evaluating pure effort scaling without added noise
                noise_z=0.0
            )
            
            # --- Metrics (Windowed) ---
            # u_traj shape: (T_STEPS, n_agents)
            u_steady = u_traj[start_step:, :]
            
            # 1. Total Squared Effort (Energy): mean over time of sum(u^2) over agents
            effort_sq = jnp.mean(jnp.sum(u_steady**2, axis=1))
            
            # 2. Total Absolute Effort (Fuel): mean over time of sum(|u|) over agents
            effort_abs = jnp.mean(jnp.sum(jnp.abs(u_steady), axis=1))

            results.append({
                "Model": m_name,
                "Agents": n_agents,
                "Sum_Sq": float(effort_sq),
                "Sum_Abs": float(effort_abs)
            })

    return pd.DataFrame(results)

# def plot_effort_metrics(df):
#     plt.style.use('seaborn-v0_8-whitegrid')
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Consistent styling
#     palette = {"Baseline": "#2c3e50", "Low Noise": "#e74c3c"}
#     markers = {"Baseline": "o", "Low Noise": "s"}
    
#     # --- Subplot 1: Quadratic Effort ---
#     sns.lineplot(
#         data=df, x="Agents", y="Sum_Sq", hue="Model", style="Model",
#         markers=markers, palette=palette, linewidth=2.5, ax=axes[0], markersize=9
#     )
#     axes[0].set_title(r"Total Quadratic Effort ($\sum u_i^2$)", fontsize=14, fontweight='bold')
#     axes[0].set_ylabel(r"Mean $\sum u_i^2$ (Steady State)", fontsize=12)
#     axes[0].set_xlabel("Number of Agents ($N$)", fontsize=12)
#     axes[0].axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training N=30")
#     axes[0].set_xscale('log')
#     axes[0].set_yscale('log')
    
#     # --- Subplot 2: Absolute Effort ---
#     sns.lineplot(
#         data=df, x="Agents", y="Sum_Abs", hue="Model", style="Model",
#         markers=markers, palette=palette, linewidth=2.5, ax=axes[1], markersize=9
#     )
#     axes[1].set_title(r"Total Absolute Effort ($\sum |u_i|$)", fontsize=14, fontweight='bold')
#     axes[1].set_ylabel(r"Mean $\sum |u_i|$ (Steady State)", fontsize=12)
#     axes[1].set_xlabel("Number of Agents ($N$)", fontsize=12)
#     axes[1].axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training N=30")
#     axes[1].set_xscale('log')
#     axes[1].set_yscale('log')

#     plt.tight_layout()
#     save_path = OUTPUT_DIR / "effort_scaling_log_steady.pdf"
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"Plot saved to {save_path}")

def plot_effort_metrics(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Changed from (1, 2) to a single plot with a standard aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Consistent styling
    palette = {"Baseline": "#2c3e50", "Low Noise": "#e74c3c"}
    markers = {"Baseline": "o", "Low Noise": "s"}
    
    # --- Quadratic Effort Only ---
    sns.lineplot(
        data=df, x="Agents", y="Sum_Sq", hue="Model", style="Model",
        markers=markers, palette=palette, linewidth=2.5, ax=ax, markersize=9
    )
    
    ax.set_title(r"Total Quadratic Effort ($\sum u_i^2$)", fontsize=14, fontweight='bold')
    ax.set_ylabel(r"Mean $\sum u_i^2$ (Steady State)", fontsize=12)
    ax.set_xlabel("Number of Agents ($N$)", fontsize=12)
    
    # Vertical line for training reference
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training $N=30$")
    
    # Set both axes to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Ensure legend is visible
    ax.legend(title="Model", frameon=True)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "effort_scaling_log_steady.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    print("Starting Effort Scaling Analysis (Steady State + Log Scale)...")
    df_results = evaluate_effort_scaling()
    
    # Save CSV for reference
    df_results.to_csv(OUTPUT_DIR / "effort_data_steady.csv", index=False)
    
    plot_effort_metrics(df_results)
    print("Analysis complete.")