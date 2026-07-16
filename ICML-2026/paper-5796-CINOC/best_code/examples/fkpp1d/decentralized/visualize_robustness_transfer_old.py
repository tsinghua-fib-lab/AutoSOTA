"""
Robustness Transfer Experiment - Visualization Script
Tests the 3 trained models on varying agent counts (Zero-Shot Scalability)
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from tesseract_core import Tesseract
import sys
import flax.serialization
from pathlib import Path
from functools import partial
import pandas as pd

jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

EXPERIMENT_DIR = Path("figures/noise_experiments/robustness_transfer")
# Ensure directory exists
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- Config ---
TEST_AGENT_COUNTS = [20, 30, 60, 100]
MODELS_TO_TEST = ["baseline", "low_noise", "medium_noise", "high_noise"]
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 50 

def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    dummy_init = model.init(jax.random.PRNGKey(0), jnp.zeros((N_PDE,)), jnp.zeros((N_PDE,)), jnp.zeros((30,)))
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def evaluate_scenario(noise_u=0.0, noise_z=0.0, model_subset=None):
    """
    Evaluates models on varying agent counts with specific noise levels.
    """
    solver_ts = Tesseract.from_image("solver_fkpp1d_decentralized:latest")
    results = []
    
    target_models = model_subset if model_subset else MODELS_TO_TEST

    with solver_ts:
        model = DecentralizedControlNet(features=(64, 64))
        dynamics = PDEDynamics(solver_ts, policy_apply_fn=model.apply, use_tesseract=False) 
        
        loaded_params = {}
        for m_name in target_models:
            p_path = EXPERIMENT_DIR / f"{m_name}_params.msgpack"
            if not p_path.exists():
                print(f"Skipping {m_name}, file not found.")
                continue
            loaded_params[m_name] = load_params(model, p_path)

        key = jax.random.PRNGKey(999)
        key, k1, k2 = jax.random.split(key, 3)
        _, z_init_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.2))(jax.random.split(k1, N_TEST_SAMPLES))
        _, z_target_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.4))(jax.random.split(k2, N_TEST_SAMPLES))

        for n_agents in TEST_AGENT_COUNTS:
            print(f"Testing {target_models} | Agents: {n_agents} | Noise U/Z: {noise_u}/{noise_z}...")
            
            xi_test = jnp.linspace(0.1, 0.9, n_agents)
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))
            
            for m_name, params in loaded_params.items():
                
                # JIT Unroll wrapper
                def run_single(z_i, z_t, xi_i):
                    z_traj, _, _, _ = dynamics.unroll_controlled(
                        z_i, xi_i, z_t, params, T_STEPS, 
                        key=jax.random.PRNGKey(0), 
                        noise_u=noise_u, 
                        noise_z=noise_z
                    )
                    return jnp.mean((z_traj[-1] - z_t)**2) 
                
                final_mses = jax.vmap(run_single)(z_init_test, z_target_test, xi_batch)
                
                avg_mse = float(jnp.mean(final_mses))
                std_mse = float(jnp.std(final_mses))
                
                results.append({
                    "Model": m_name,
                    "Agents": n_agents,
                    "MSE": avg_mse,
                    "Std": std_mse,
                    "Scenario": f"u{noise_u}_z{noise_z}"
                })

    return pd.DataFrame(results)

def plot_scalability_curve(df):
    """Original plot showing all models in no noise"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Agents", y="MSE", hue="Model", style="Model", markers=True, markersize=9, linewidth=2.5)
    
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training Scale (N=30)")
    plt.title("Robustness Transfer: Zero-Shot Scalability (No Noise)", fontsize=16)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Model", loc='upper right')
    
    save_path = EXPERIMENT_DIR / "robustness_transfer_curve.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved scalability curve to {save_path}")

def plot_comparison(df, title, filename):
    """Helper to plot Baseline vs Low Noise comparisons"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    # Filter specific palette
    palette = {"baseline": "C0", "low_noise": "C1"}
    
    sns.lineplot(
        data=df, 
        x="Agents", 
        y="MSE", 
        hue="Model", 
        style="Model", 
        markers=True, 
        markersize=10, 
        linewidth=3,
        palette=palette
    )
    
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label="Training Scale (N=30)")
    plt.title(title, fontsize=14)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Model", loc='upper right')
    
    save_path = EXPERIMENT_DIR / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    # 1. Run Standard Evaluation (All models, No Noise)
    print("--- Running Standard Scalability (No Noise) ---")
    df_nonoise = evaluate_scenario(noise_u=0.0, noise_z=0.0)
    df_nonoise.to_csv(EXPERIMENT_DIR / "scalability_metrics.csv", index=False)
    
    # Plot original full comparison
    plot_scalability_curve(df_nonoise)

    # --- New Requested Plots ---

    # Plot 1: Baseline vs Low Noise (No Noise Scenario)
    # Filter the existing no-noise dataframe
    df_subset_clean = df_nonoise[df_nonoise['Model'].isin(['baseline', 'low_noise'])]
    plot_comparison(
        df_subset_clean, 
        "Baseline vs Low Noise (No Deployment Noise)", 
        "comparison_nonoise.pdf"
    )

    # Plot 2: Baseline vs Low Noise (High Noise Scenario)
    print("\n--- Running High Noise Evaluation (u=0.5, z=0.25) ---")
    df_highnoise = evaluate_scenario(
        noise_u=0.5, 
        noise_z=0.25, 
        model_subset=['baseline', 'low_noise']
    )
    
    plot_comparison(
        df_highnoise, 
        "Baseline vs Low Noise (High Deployment Noise)", 
        "comparison_highnoise.pdf"
    )

    print("\nAll experiments completed.")