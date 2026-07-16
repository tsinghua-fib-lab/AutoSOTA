"""
Physics Parameter Ablation Study
Tests the 'low_noise' model's robustness to shifts in:
1. Diffusion Coefficient (nu)
2. Growth Rate (rho)
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

jax.config.update("jax_platform_name", "cpu")

# --- Setup Paths ---
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent.parent))

# Output Directory
OUTPUT_DIR = Path("figures/pde_params")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Imports ---
from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- Config ---
# Dense range of agents for the x-axis
TEST_AGENT_COUNTS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
N_PDE = 100
T_STEPS = 300
N_TEST_SAMPLES = 50 

# Parameter Sets
NU_VALUES = [0., 0.001, 0.002, 0.003, 0.004, 0.005]
RHO_VALUES = [3, 4, 5, 6, 7, 8]

# Defaults
DEFAULT_NU = 0.005
DEFAULT_RHO = 3.0

def load_params(model, filepath):
    with open(filepath, 'rb') as f:
        bytes_data = f.read()
    dummy_init = model.init(jax.random.PRNGKey(0), jnp.zeros((N_PDE,)), jnp.zeros((N_PDE,)), jnp.zeros((30,)))
    return flax.serialization.from_bytes(dummy_init, bytes_data)

def run_ablation(dynamics, params, z_init_batch, z_target_batch, nu_list, rho_list, varied_param_name):
    """
    Runs the ablation loop using PDEDynamics wrapper.
    """
    results = []
    
    # Determine which list to iterate
    iter_list = nu_list if varied_param_name == 'nu' else rho_list
    
    print(f"--- Running Ablation for {varied_param_name} ---")
    
    for val in iter_list:
        # Set parameters: one varies, the other stays default
        current_nu = val if varied_param_name == 'nu' else DEFAULT_NU
        current_rho = val if varied_param_name == 'rho' else DEFAULT_RHO
        
        for n_agents in TEST_AGENT_COUNTS:
            print(f"  Param: {val} | Agents: {n_agents}")
            
            # Create agent positions
            xi_test = jnp.linspace(0.1, 0.9, n_agents)
            xi_batch = jnp.tile(xi_test, (N_TEST_SAMPLES, 1))

            # Define JIT-compiled batch runner
            @jax.jit
            def run_batch(z_i, z_t, xi_i):
                def single_run(zi, zt, xii):
                    # Call the wrapper with explicit physics params
                    z_traj, _, _, _ = dynamics.unroll_controlled(
                        zi, xii, zt, params, T_STEPS, 
                        key=jax.random.PRNGKey(0),
                        noise_u=0.0, 
                        noise_z=0.0,
                        nu=current_nu,  # Inject dynamic nu
                        rho=current_rho # Inject dynamic rho
                    )
                    return jnp.mean((z_traj[-1] - zt)**2)
                return jax.vmap(single_run)(z_i, z_t, xi_i)

            mses = run_batch(z_init_batch, z_target_batch, xi_batch)
            
            results.append({
                "Agents": n_agents,
                "MSE": float(jnp.mean(mses)),
                "Value": str(val), # String for categorical legend
                "Parameter": varied_param_name
            })
            
    return pd.DataFrame(results)

def get_style_maps(df, param_col, baseline_val, cmap_name):
    """
    Helper to create palette and dashes so the baseline is always a black solid line.
    """
    unique_vals = sorted(df[param_col].unique())
    baseline_str = str(baseline_val)
    
    # 1. Palette: Map baseline to black, others to colormap
    non_baseline = [v for v in unique_vals if v != baseline_str]
    colors = sns.color_palette(cmap_name, len(non_baseline))
    palette = dict(zip(non_baseline, colors))
    palette[baseline_str] = 'black'
    
    # 2. Dashes: Map baseline to Solid (""), others to Dashed ((2, 2))
    dashes = {v: (2, 2) for v in non_baseline}
    dashes[baseline_str] = "" 

    # 3. Order: Ensure baseline plots last (on top)
    hue_order = non_baseline + [baseline_str]
    
    return palette, dashes, hue_order

if __name__ == "__main__":
    # 1. Initialize Dynamics Wrapper with native JAX solver
    model = DecentralizedControlNet(features=(64, 64))
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    # 2. Load "low_noise" Params
    param_path = Path("figures/noise_experiments/robustness_transfer/low_noise_params.msgpack")
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find low_noise params at {param_path}")
    
    params = load_params(model, param_path)
    
    # 3. Generate Test Data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    _, z_init_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.2))(jax.random.split(k1, N_TEST_SAMPLES))
    _, z_target_test = jax.vmap(partial(generate_grf, n_points=N_PDE, length_scale=0.4))(jax.random.split(k2, N_TEST_SAMPLES))

    # 4. Run Experiments
    # Experiment A: Vary Nu (Rho fixed at 3.0)
    df_nu = run_ablation(dynamics, params, z_init_test, z_target_test, NU_VALUES, [], 'nu')
    
    # Experiment B: Vary Rho (Nu fixed at 0.005)
    df_rho = run_ablation(dynamics, params, z_init_test, z_target_test, [], RHO_VALUES, 'rho')

    # 5. Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- Plot Nu Ablation ---
    # Baseline for Nu is 0.005
    pal_nu, dash_nu, order_nu = get_style_maps(df_nu, "Value", DEFAULT_NU, "viridis")
    
    sns.lineplot(
        ax=axes[0], data=df_nu, x="Agents", y="MSE", 
        hue="Value", style="Value", 
        palette=pal_nu, dashes=dash_nu, hue_order=order_nu, style_order=order_nu,
        markers=True, markersize=8, linewidth=2.5
    )
    axes[0].set_title(f"Sensitivity to Diffusion ($\\nu$)\nTraining $\\nu={DEFAULT_NU}$ (Black Line)", fontsize=14)
    axes[0].set_ylabel("Final Tracking Error (MSE)", fontsize=12)
    axes[0].set_yscale("log")
    axes[0].legend(title="$\\nu$ Value", loc='upper right')

    # --- Plot Rho Ablation ---
    # Baseline for Rho is 3 (integer from list), DEFAULT_RHO is 3.0 (float)
    # The dataframe uses str(val), so we need to match the integer input 3
    pal_rho, dash_rho, order_rho = get_style_maps(df_rho, "Value", 3, "magma")
    
    sns.lineplot(
        ax=axes[1], data=df_rho, x="Agents", y="MSE", 
        hue="Value", style="Value", 
        palette=pal_rho, dashes=dash_rho, hue_order=order_rho, style_order=order_rho,
        markers=True, markersize=8, linewidth=2.5
    )
    axes[1].set_title(f"Sensitivity to Growth Rate ($\\rho$)\nTraining $\\rho={DEFAULT_RHO}$ (Black Line)", fontsize=14)
    axes[1].set_ylabel("") # Shared Y
    axes[1].set_yscale("log")
    axes[1].legend(title="$\\rho$ Value", loc='upper right')

    plt.tight_layout()
    save_path = OUTPUT_DIR / "physics_ablation_lownoise.pdf"
    plt.savefig(save_path)
    print(f"\nSaved combined ablation plot to {save_path}")