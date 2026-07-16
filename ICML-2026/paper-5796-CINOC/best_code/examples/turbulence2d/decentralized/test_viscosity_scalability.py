import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pandas as pd
import sys
import flax.serialization
from pathlib import Path
from functools import partial
import matplotlib.ticker as ticker

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_turb import DecentralizedTurbulenceNet 
from data_utils import get_batch_initial_conditions

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
TRAINED_VISCOSITY = 5e-4
TEST_VISCOSITIES = [2e-4, 3e-4, 4e-4, 5e-4]
TEST_AGENT_COUNTS = [64, 81, 100, 121, 144, 169, 196, 225]  # 8x8 to 14x14

CONFIG = {
    'N_grid': 64,
    'L_domain': 1.0,
    'dt': 0.01,
    'substeps': 5,
    'T_control_steps': 150, 
    'params_file': 'turbulence_params.msgpack',
}

def get_actuator_grid(n_agents):
    grid_dim = int(np.sqrt(n_agents))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1), (grid_dim, grid_dim)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SWEEP EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_scalability_sweep(model, params):
    results = []
    
    for visc in TEST_VISCOSITIES:
        print(f"Sweep - Viscosity: {visc:.1e}")
        # Use same IC for all agent counts within one viscosity for fair comparison
        key = jax.random.PRNGKey(int(visc * 1e7))
        w_hat_init = get_batch_initial_conditions(key, 1, CONFIG['N_grid'], CONFIG['L_domain'], 
                                                  warmup_time=2.0, viscosity=visc)[0]
        
        for n_agents in TEST_AGENT_COUNTS:
            sys.stdout.write(f"\r  Agents: {n_agents}...")
            sys.stdout.flush()
            
            xi_test, grid_shape = get_actuator_grid(n_agents)
            dyn = PDEDynamics2D(policy_apply_fn=model.apply)

            # Run Controlled Simulation
            w_traj, _ = dyn.unroll_controlled(
                w_hat_init, xi_test, params,
                t_steps=CONFIG['T_control_steps'], substeps=CONFIG['substeps'],
                N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
                viscosity=visc, actuator_grid_shape=grid_shape
            )
            
            # Metric: Mean Enstrophy over the last 20% of the trajectory
            final_enstrophy = jnp.mean(jnp.mean(w_traj[-30:]**2, axis=(1,2)))
            
            results.append({
                "Viscosity": f"{visc:.1e}",
                "Agents": n_agents,
                "Enstrophy": float(final_enstrophy)
            })
        print(" Done.")
        
    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAIN & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("figures/images/params/scalability")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Setup ---
    model = DecentralizedTurbulenceNet(features=(32, 64), patch_size=16, 
                                       domain_size=(1.0, 1.0), u_max=150.0)    
    with open(CONFIG['params_file'], 'rb') as f:
        raw_params = f.read()
    
    # We init with any valid grid for loading
    xi_init, _ = get_actuator_grid(64)
    init_params = model.init(jax.random.PRNGKey(0), xi_init, jnp.zeros((1, 64, 64)))
    params = flax.serialization.from_bytes(init_params, raw_params)

    # --- 2. Run Sweep ---
    df_results = run_scalability_sweep(model, params)

    # --- 3. Plotting (Scalability Style) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6)) # Use subplots to get the 'ax' object

    # Create palette
    palette = sns.color_palette("viridis", n_colors=len(TEST_VISCOSITIES))

    sns.lineplot(
        data=df_results, 
        x="Agents", 
        y="Enstrophy", 
        hue="Viscosity", 
        marker="o", 
        markersize=8, 
        linewidth=2.5,
        palette=palette,
        ax=ax
    )

    # --- Y-AXIS TICK CUSTOMIZATION ---
    ax.set_yscale('log')

    # 1. Major Ticks: Standard powers of 10
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

    # 2. Minor Ticks: Adds the small lines between powers of 10
    # 'subs' defines where the minor ticks fall (e.g., 2, 3, 4...9)
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))

    # 3. Formatting: Clean up labels (optional: use ScalarFormatter for 0.01 instead of 10^-2)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation()) 
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter()) # Uncomment for decimal look

    # Visual cues
    ax.axvline(x=64, color='red', linestyle='--', alpha=0.5, label="Training Scale (8x8)")

    ax.set_title(f"Scalability & Robustness: Policy Generalization (Trained @ $\\nu={TRAINED_VISCOSITY}$)", fontsize=14)
    ax.set_ylabel("Residual Enstrophy (Stabilization Error)", fontsize=12)
    ax.set_xlabel("Number of Actuators", fontsize=12)

    # Enable grid for both major and minor ticks
    ax.grid(True, which="major", ls="-", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.2)

    ax.legend(title="Viscosity $\\nu$", loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig("figures/images/params/scalability/turbulence_scalability_viscosity.pdf", dpi=150, bbox_inches='tight')
    print("\n✓ Scalability plot saved to figures/images/params/scalability/turbulence_scalability_viscosity.pdf")