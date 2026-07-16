# We're not using this script actually.
"""
Multi-Scale Generalization Experiment - Visualization Script
Tests if a policy trained on L=32 can zero-shot generalize to L=22...200.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import flax.serialization
from pathlib import Path
import pandas as pd
from functools import partial

# Force CPU to avoid OOM on large domains if GPU memory is limited
# jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Import KS components
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# --- Experiment Config ---
# 1. Source Model (The one trained on L=32)
SOURCE_L = 32.0
TRAIN_DIR = Path("figures/ks_noise_experiments/robustness_transfer_L32")
MODEL_TYPE = "baseline" # Which variant to test? (baseline, low_noise, etc.)

# 2. Target Domains to Test
# Format: (L_domain, N_grid)
# We scale N_grid to keep spatial resolution (dx) roughly constant (~0.125)
DOMAIN_CONFIGS = [
    (22.0, 128),  # Smaller
    (32.0, 256),  # Same (Reference)
    (50.0, 512),  # Medium
    (100.0, 1024), # Large
    (200.0, 2048)  # Massive (Very chaotic)
]

# 3. Test Densities (Agents per Unit Length)
# We use density to compare fairly across scales.
# e.g. Density 1.0 on L=50 means 50 agents.
TEST_DENSITIES = [0.5, 1.0, 1.5, 2, 3]

# General Settings
T_STEPS = 400      # Longer horizon for larger domains
N_TEST_SAMPLES = 20 # Reduced samples for speed on large grids

EXPERIMENT_DIR = Path("figures/ks_generalization")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def load_transfer_params(target_L, target_N, filepath):
    """
    Loads weights from the L=32 model but initializes it 
    for the Target L structure.
    """
    if not filepath.exists():
        print(f"Error: Model file not found at {filepath}")
        return None, None

    # 1. Define Model for TARGET Domain
    # We must instantiate with target_L so internal normalization (x/L) works
    model = DecentralizedControlNet(features=(64, 64), L_domain=target_L)
    
    # 2. Initialize with correct shapes for this domain
    key = jax.random.PRNGKey(0)
    dummy_u = jnp.zeros((target_N,))
    dummy_xi = jnp.zeros((10,)) # Number of agents doesn't impact weight shapes for this arch
    
    dummy_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    
    # 3. Load the binary data (Weights are shape-compatible if arch is decentralized)
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
        
    params = flax.serialization.from_bytes(dummy_params, bytes_data)
    return model, params

def evaluate_generalization():
    results = []
    
    # Path to the specific trained model
    param_path = TRAIN_DIR / f"{MODEL_TYPE}_params.msgpack"
    print(f"--- Generalization Experiment ---")
    print(f"Source Model: {MODEL_TYPE} (Trained on L={SOURCE_L})")
    
    # Loop over different Domain Sizes
    for (L, N) in DOMAIN_CONFIGS:
        print(f"\n>>> Testing Target Domain: L={L}, N_grid={N}")
        
        # 1. Load Model adapted for this L
        model, params = load_transfer_params(L, N, param_path)
        if params is None: continue
        
        # 2. Setup Dynamics for this L
        dynamics = PDEDynamics(policy_apply_fn=model.apply)
        
        # 3. Generate Data (Specific to this L)
        # Larger domains need spin-up to generate valid chaotic states
        key = jax.random.PRNGKey(42)
        u_init_test = get_batch_initial_conditions(key, N_TEST_SAMPLES, N, L)
        u_target_test = jnp.zeros_like(u_init_test)

        # 4. Sweep Densities
        for density in TEST_DENSITIES:
            n_agents = int(L * density)
            # Ensure at least 1 agent
            n_agents = max(1, n_agents)
            
            print(f"   > Density: {density} | Agents: {n_agents}")
            
            # Create equidistant positions
            xi_test = jnp.linspace(0.0, L, n_agents, endpoint=False) + (L/n_agents)/2
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))
            
            # Define Run Function (JIT compiled for this specific N)
            @jax.jit
            def run_single(u_i, u_t, xi_i):
                u_traj, _, _, _ = dynamics.unroll_controlled(
                    u_i, xi_i, u_t, params, T_STEPS, 
                    N_grid=N, L=L, key=jax.random.PRNGKey(0)
                )
                # MSE at final step
                return jnp.mean((u_traj[-1] - u_t)**2)
            
            # Execute Batch
            final_mses = jax.vmap(run_single)(u_init_test, u_target_test, xi_batch)
            
            # Record Stats
            results.append({
                "Domain Size (L)": L,
                "Grid Size (N)": N,
                "Agent Density": density,
                "Agent Count": n_agents,
                "MSE": float(jnp.mean(final_mses)),
                "Std": float(jnp.std(final_mses))
            })

    return pd.DataFrame(results)

def plot_generalization_heatmap(df):
    """Plots a heatmap of MSE vs L and Density."""
    if df.empty: return

    # Pivot for Heatmap
    pivot_table = df.pivot(index="Domain Size (L)", columns="Agent Density", values="MSE")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".2e", 
        cmap="viridis_r", # Reversed so Dark is Low Error (Good)
        norm=plt.matplotlib.colors.LogNorm() # Log scale for color mapping
    )
    plt.title(f"Zero-Shot Generalization: MSE Heatmap\n(Model Trained on L={SOURCE_L})")
    plt.ylabel("Target Domain Size (L)")
    plt.xlabel("Agent Density (Agents/L)")
    
    save_path = EXPERIMENT_DIR / "generalization_heatmap.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")

def plot_generalization_lines(df):
    """Line plot comparison."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Convert L to categorical for discrete colors
    df_plot = df.copy()
    df_plot["Domain Size (L)"] = df_plot["Domain Size (L)"].astype(str)
    
    sns.lineplot(
        data=df_plot,
        x="Agent Density",
        y="MSE",
        hue="Domain Size (L)",
        style="Domain Size (L)",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="flare"
    )
    
    plt.yscale('log')
    plt.title(f"Scalability across Domain Sizes\n(Model Trained on L={SOURCE_L})")
    plt.ylabel("Final Tracking Error (MSE)")
    plt.xlabel("Agent Density (Agents per Unit Length)")
    
    # Mark the training density (approx 30/32 ~= 0.94)
    plt.axvline(x=0.9375, color='gray', linestyle='--', alpha=0.5, label="Training Density")
    plt.legend(title="Domain Size (L)")
    
    save_path = EXPERIMENT_DIR / "generalization_lines.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved line plot to {save_path}")

if __name__ == "__main__":
    df_results = evaluate_generalization()
    
    if not df_results.empty:
        # Save raw data
        df_results.to_csv(EXPERIMENT_DIR / "generalization_metrics.csv", index=False)
        
        # Create Plots
        plot_generalization_heatmap(df_results)
        plot_generalization_lines(df_results)
    else:
        print("Experiment failed to generate results.")