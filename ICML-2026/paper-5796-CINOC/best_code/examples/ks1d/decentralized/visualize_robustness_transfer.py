# We're not using this script actually.
"""
Robustness Transfer Experiment - Visualization Script (KS-1D L=32)
Tests the trained KS DecentralizedControlNet models on varying agent counts (Zero-Shot Scalability)
and noise levels.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import flax.serialization
from pathlib import Path
from functools import partial
import pandas as pd
from tqdm import tqdm

# Force CPU for evaluation to avoid OOM if running locally, or comment out for GPU
# jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Import KS specific modules
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# --- Config (Must match Training Script) ---
EXPERIMENT_DIR = Path("figures/ks_noise_experiments/robustness_transfer_L32")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Training Constants (for reference and initialization)
L_DOMAIN = 32.0
N_GRID = 256
TRAIN_N_AGENTS = 30

# Testing Config
# We test: 
# 1. Under-actuated (20) 
# 2. Training distribution (30) 
# 3. Over-actuated (40, 50)
TEST_AGENT_COUNTS = [20, 30, 35, 40, 45, 50] 
MODELS_TO_TEST = ["baseline", "low_noise", "medium_noise", "high_noise"]

N_TEST_SAMPLES = 50 
T_STEPS = 300 

def load_params(model, filepath, n_agents_init):
    """Loads parameters into the KS DecentralizedControlNet."""
    if not filepath.exists():
        return None
    
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    
    # Init dummy params to get the structure
    key = jax.random.PRNGKey(0)
    dummy_u = jnp.zeros((N_GRID,))
    dummy_xi = jnp.linspace(0, L_DOMAIN, n_agents_init)
    
    # KS model init expects (rng, u, u_target, xi)
    dummy_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    
    return flax.serialization.from_bytes(dummy_params, bytes_data)

def evaluate_scenario(noise_u=0.0, noise_z=0.0, model_subset=None):
    """
    Evaluates models on varying agent counts with specific noise levels.
    """
    results = []
    target_models = model_subset if model_subset else MODELS_TO_TEST

    # 1. Initialize Model Architecture & Dynamics
    model = DecentralizedControlNet(features=(64, 64), L_domain=L_DOMAIN)
    dynamics = PDEDynamics(policy_apply_fn=model.apply)

    # 2. Load Parameters
    loaded_params = {}
    for m_name in target_models:
        p_path = EXPERIMENT_DIR / f"{m_name}_params.msgpack"
        params = load_params(model, p_path, TRAIN_N_AGENTS)
        if params is None:
            print(f"Warning: Skipping {m_name}, file not found at {p_path}")
            continue
        loaded_params[m_name] = params

    if not loaded_params:
        print("No models loaded. Exiting evaluation.")
        return pd.DataFrame()

    # 3. Generate Evaluation Data (Chaotic States)
    print("Generating chaotic test states (Spin-up)...")
    key = jax.random.PRNGKey(777) # Fixed seed for comparison
    key, subkey = jax.random.split(key)
    
    # Use the specific KS data generator
    u_init_test = get_batch_initial_conditions(subkey, N_TEST_SAMPLES, N_GRID, L_DOMAIN)
    u_target_test = jnp.zeros_like(u_init_test) # Stabilization target

    # 4. Evaluation Loop
    for n_agents in TEST_AGENT_COUNTS:
        print(f"Testing {list(loaded_params.keys())} | Agents: {n_agents} | Noise U/Z: {noise_u}/{noise_z}...")
        
        # Create equidistant agent positions for this test count
        # (Agents distributed evenly across L=32)
        xi_single = jnp.linspace(0.0, L_DOMAIN, n_agents, endpoint=False) + (L_DOMAIN/n_agents)/2
        xi_batch = jnp.tile(xi_single, (N_TEST_SAMPLES, 1))
        
        for m_name, params in loaded_params.items():
            
            # Define single trajectory run
            def run_single(u_i, u_t, xi_i, rng_key):
                # We pass the specific noise config for this evaluation scenario
                # Note: Unroll uses N_grid and L from args or defaults. 
                # We pass them explicitly to be safe.
                u_traj, _, _, _ = dynamics.unroll_controlled(
                    u_i, xi_i, u_t, params, T_STEPS, 
                    N_grid=N_GRID, 
                    L=L_DOMAIN,
                    key=rng_key, 
                    noise_u=noise_u, 
                    noise_z=noise_z
                )
                # Metric: Final MSE (how well did we stabilize?)
                return jnp.mean((u_traj[-1] - u_t)**2)
            
            # Vectorize over batch
            # We split keys so every sample gets unique noise realization
            step_keys = jax.random.split(key, N_TEST_SAMPLES)
            final_mses = jax.vmap(run_single)(u_init_test, u_target_test, xi_batch, step_keys)
            
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
    """Plot showing all models in clean environment"""
    if df.empty: return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df, x="Agents", y="MSE", hue="Model", style="Model", markers=True, markersize=9, linewidth=2.5)
    
    plt.axvline(x=TRAIN_N_AGENTS, color='gray', linestyle='--', alpha=0.5, label=f"Training Scale (N={TRAIN_N_AGENTS})")
    plt.title("KS-1D (L=32): Zero-Shot Scalability (Clean Environment)", fontsize=16)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Model", loc='upper right')
    
    save_path = EXPERIMENT_DIR / "robustness_transfer_curve.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved scalability curve to {save_path}")

def plot_comparison(df, title, filename):
    """Helper to plot Baseline vs Noise comparisons"""
    if df.empty: return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    # Custom palette for clarity
    palette = {
        "baseline": "black", 
        "low_noise": "tab:blue",
        "medium_noise": "tab:orange",
        "high_noise": "tab:red"
    }
    # Filter palette to only keys present in df
    actual_palette = {k: v for k, v in palette.items() if k in df['Model'].unique()}

    sns.lineplot(
        data=df, 
        x="Agents", 
        y="MSE", 
        hue="Model", 
        style="Model", 
        markers=True, 
        markersize=10, 
        linewidth=3,
        palette=actual_palette
    )
    
    plt.axvline(x=TRAIN_N_AGENTS, color='gray', linestyle='--', alpha=0.5, label=f"Train (N={TRAIN_N_AGENTS})")
    plt.title(title, fontsize=14)
    plt.ylabel("Final Tracking Error (MSE)", fontsize=12)
    plt.xlabel("Deployment Agent Count", fontsize=12)
    plt.yscale('log')
    plt.legend(title="Model", loc='upper right')
    
    save_path = EXPERIMENT_DIR / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    # 1. Run Standard Evaluation (All models, No Noise)
    print("--- Running Standard Scalability (No Noise) ---")
    df_nonoise = evaluate_scenario(noise_u=0.0, noise_z=0.0)
    if not df_nonoise.empty:
        df_nonoise.to_csv(EXPERIMENT_DIR / "scalability_metrics.csv", index=False)
        plot_scalability_curve(df_nonoise)

        # Plot 1: Baseline vs Robust variants (No Noise Scenario)
        # Does robustness hurt performance in clean environments?
        plot_comparison(
            df_nonoise, 
            "Performance in Ideal Conditions (Clean)", 
            "comparison_clean_env.png"
        )

    # 2. Run Evaluation with Training-Level Noise (Low)
    print("\n--- Running Evaluation with Low Noise (u=0.05, z=0.025) ---")
    df_lownoise = evaluate_scenario(noise_u=0.05, noise_z=0.025)
    plot_comparison(
        df_lownoise, 
        "Performance under Low Noise", 
        "comparison_low_noise_env.png"
    )

    # 3. Run Evaluation with Extreme Noise
    print("\n--- Running Evaluation with High Noise (u=0.5, z=0.25) ---")
    df_highnoise = evaluate_scenario(noise_u=0.5, noise_z=0.25)
    plot_comparison(
        df_highnoise, 
        "Performance under High Noise", 
        "comparison_high_noise_env.png"
    )

    print("\nAll experiments completed.")