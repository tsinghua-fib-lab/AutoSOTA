# We're not using this script actually.
"""
Sensor Dimension Experiment - Multi-Domain Visualization (KS-1D)
Evaluates Decentralized ControlNet across multiple physical configurations.

ADAPTATION NOTE:
This script is configured to match the 'Fixed Density' training run.
- L=22  -> 30 Agents
- L=64  -> 88 Agents (scaled density)
- L=200 -> 272 Agents (scaled density)
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import flax.serialization
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.ticker as ticker

# Force CPU to avoid memory fragmentation during small evals
jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Base Output Directory
BASE_EXPERIMENT_DIR = Path("figures/sensor_dim_ablation")
BASE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# --- KS Specific Imports ---
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions
from models.policy_ks1d import DecentralizedControlNet

# --- Configs ---

# 1. Physics Configurations
# These MUST match the 'n_agents' used in the corrected training script
DOMAIN_CONFIGS = [
    {
        # Target Wavelength: ~52 pixels
        "name": "L22_N128_Original",
        "L_domain": 22.0,
        "N_grid": 128,
        "train_n_agents": 30,
        # Bracket: [Baseline, 0.5x, 0.8x, 1.0x, 1.2x, 1.5x]
        "sensor_dims": [25, 30, 40, 52, 65, 80],
        "test_agent_counts": [15, 20, 25, 30, 35, 40, 42, 45, 47, 50]
    },
    # {
    #     # Target Wavelength: ~36 pixels
    #     "name": "L64_N256_HighRes",
    #     "L_domain": 64.0,
    #     "N_grid": 256,
    #     "train_n_agents": 88,
    #     # Bracket: [Baseline, 0.5x, 0.8x, 1.0x, 1.2x, 2.0x]
    #     "sensor_dims": [18, 30, 36, 45, 72],
    #     "test_agent_counts": [60, 80, 85, 88, 95, 100, 110]
    # },
    # {
    #     # Target Wavelength: ~19 pixels
    #     "name": "L124_N256_Coarse",
    #     "L_domain": 124.0, # Corrected L
    #     "N_grid": 256,     # Corrected N
    #     "train_n_agents": 170,
    #     # Bracket: [Baseline, 0.5x, 1.0x, 1.2x, 1.5x, 2.0x]
    #     "sensor_dims": [10, 19, 24, 30, 40],
    #     "test_agent_counts": [40, 60, 80, 100, 120]
    # }
]

# 2. Evaluation Constants
T_STEPS = 300
N_TEST_SAMPLES = 50 


def load_params(model, filepath, n_grid, n_agents_dummy):
    """Safely loads model parameters."""
    if not filepath.exists(): 
        return None
    
    with open(filepath, 'rb') as f: 
        bytes_data = f.read()
    
    # Init dummy params of the correct shape
    dummy_init = model.init(
        jax.random.PRNGKey(0), 
        jnp.zeros((n_grid,)), 
        jnp.zeros((n_grid,)), 
        jnp.zeros((n_agents_dummy,))
    )
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_single_config(config):
    """
    Runs the full evaluation pipeline for a single domain configuration.
    """
    results = []
    
    # Unpack Config
    config_name = config["name"]
    L = config["L_domain"]
    N = config["N_grid"]
    train_n = config["train_n_agents"]
    sensor_list = config["sensor_dims"]
    test_counts = config["test_agent_counts"]
    
    save_dir = BASE_EXPERIMENT_DIR / config_name

    print(f"\n=== Evaluating Configuration: {config_name} (L={L}, N={N}) ===")
    print(f" > Training Density: {train_n}")
    print(f" > Test Sweep: {test_counts}")
    
    if not save_dir.exists():
        print(f" [!] Directory not found: {save_dir}. Skipping...")
        return pd.DataFrame()

    # 1. Generate Domain-Specific Test Data
    print(f" > Generating Initial Conditions...")
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    u_init_test = get_batch_initial_conditions(subkey, N_TEST_SAMPLES, N, L)
    u_target_test = jnp.zeros_like(u_init_test)

    # 2. Loop through Domain-Specific Sensors
    for s_range in sensor_list:
        # Load weights from the specific subfolder
        param_path = save_dir / f"sensor_{s_range}_params.msgpack"
        
        # Instantiate Model
        model = DecentralizedControlNet(features=(64, 64), L_domain=L, window_size=s_range)
        params = load_params(model, param_path, N, train_n)
        
        if params is None:
            # Silent skip is better here to avoid clutter if some runs are pending
            continue

        print(f"   > Testing Sensor Range: {s_range}")

        # Setup Dynamics
        dynamics = PDEDynamics(policy_apply_fn=model.apply)

        # 3. Calculate MSE for varying Agent Counts
        # JIT-compiled single run wrapper
        @jax.jit
        def run_single(u_i, u_t, xi_i):
            u_traj, _, _, _ = dynamics.unroll_controlled(
                u_i, xi_i, u_t, params, T_STEPS, 
                N_grid=N, L=L, key=jax.random.PRNGKey(0)
            )
            return jnp.mean((u_traj[-1] - u_t)**2) 

        # Batch Vectorization
        batch_run = jax.vmap(run_single, in_axes=(0, 0, 0))

        for n_agents in test_counts:
            # Create equidistant positions for this specific L
            xi_test = jnp.linspace(0.0, L, n_agents, endpoint=False) + (L/n_agents)/2
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))

            final_mses = batch_run(u_init_test, u_target_test, xi_batch)
            
            results.append({
                "Sensor Range": s_range,
                "Agents": n_agents,
                "MSE": float(jnp.mean(final_mses)),
                "Std": float(jnp.std(final_mses))
            })

    return pd.DataFrame(results)

def plot_results(df, config):
    """Generates the plot for a single configuration."""
    if df.empty:
        return

    config_name = config["name"]
    L = config["L_domain"]
    N = config["N_grid"]
    train_n = config["train_n_agents"]
    save_dir = BASE_EXPERIMENT_DIR / config_name

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Plot Logic
    sns.lineplot(
        data=df, 
        x="Agents", 
        y="MSE", 
        hue="Sensor Range", 
        palette="viridis", 
        style="Sensor Range", 
        markers=True, 
        markersize=8, 
        linewidth=2.5,
        dashes=False
    )
    
    # Vertical line for training density
    plt.axvline(x=train_n, color='red', linestyle='--', alpha=0.6, label=f"Training (N={train_n})")
    
    # Dynamic Title based on Config
    plt.title(f"Sensor Sensitivity: L={L}, N={N}", fontsize=16, pad=15)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    plt.yscale('log')
    
    # Grid & Legend
    ax = plt.gca()
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Sensor Range (px)")
    plt.tight_layout()
    
    save_path = save_dir / f"sensitivity_plot_{config_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [Plot Saved] {save_path}")

if __name__ == "__main__":
    print(f"--- Starting Multi-Domain Sensor Analysis ---")
    
    for config in DOMAIN_CONFIGS:
        # 1. Evaluate
        df_results = evaluate_single_config(config)
        
        if not df_results.empty:
            # 2. Save CSV
            save_dir = BASE_EXPERIMENT_DIR / config["name"]
            csv_path = save_dir / "metrics.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"   [Data Saved] {csv_path}")
            
            # 3. Plot
            plot_results(df_results, config)
        else:
            print(f"   [!] No results found for {config['name']}")

    print("\nAll evaluations complete.")