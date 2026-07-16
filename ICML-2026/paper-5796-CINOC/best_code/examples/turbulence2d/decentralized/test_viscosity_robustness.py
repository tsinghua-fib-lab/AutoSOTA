import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path

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
TEST_VISCOSITIES = [2e-4, 3e-4, 4e-4, 5e-4] # Your requested range

CONFIG = {
    'N_grid': 64,
    'L_domain': 1.0,
    'dt': 0.01,
    'substeps': 5,
    'n_agents': 64,
    'grid_shape': (8, 8),
    'T_control_steps': 200,
    'params_file': 'turbulence_params.msgpack',
    'snapshot_times': [0.0, 0.5, 1.0, 2.0] # Seconds
}

def get_actuator_grid():
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_viscosity(visc, model, params):
    """Generates an IC and runs controlled vs baseline comparison."""
    key = jax.random.PRNGKey(int(visc * 1e7)) # Unique seed per viscosity
    
    # 1. Generate chaotic IC for this specific viscosity
    w_hat_init = get_batch_initial_conditions(key, 1, CONFIG['N_grid'], CONFIG['L_domain'], 
                                              warmup_time=2.0, viscosity=visc)[0]
    
    xi_fixed = get_actuator_grid()
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_baseline = PDEDynamics2D(policy_apply_fn=lambda p, x, o: jnp.zeros(CONFIG['n_agents']))

    # 2. Run Simulations
    w_ctrl, _ = dyn_control.unroll_controlled(
        w_hat_init, xi_fixed, params,
        t_steps=CONFIG['T_control_steps'], substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=visc, actuator_grid_shape=CONFIG['grid_shape']
    )

    w_base, _ = dyn_baseline.unroll_controlled(
        w_hat_init, xi_fixed, params,
        t_steps=CONFIG['T_control_steps'], substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=visc, actuator_grid_shape=CONFIG['grid_shape']
    )

    t_axis = np.arange(CONFIG['T_control_steps']) * CONFIG['dt']
    return t_axis, w_ctrl, w_base

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAIN EXECUTION & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("figures/images/params/robustness")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    model = DecentralizedTurbulenceNet(features=(32, 64), patch_size=16, 
                                       domain_size=(1.0, 1.0), u_max=150.0)    
    with open(CONFIG['params_file'], 'rb') as f:
        raw_params = f.read()
    
    xi_grid = get_actuator_grid()
    init_params = model.init(jax.random.PRNGKey(0), xi_grid, jnp.zeros((1, 64, 64)))
    params = flax.serialization.from_bytes(init_params, raw_params)

    # Setup Figure: 4 Viscosities = 4 Rows
    fig = plt.figure(figsize=(15, 3 * len(TEST_VISCOSITIES)))
    outer_grid = gridspec.GridSpec(len(TEST_VISCOSITIES), 2, width_ratios=[2.5, 1], hspace=0.4)

    for row, visc in enumerate(TEST_VISCOSITIES):
        print(f"Testing Viscosity: {visc:.1e}...")
        t, w_ctrl, w_base = evaluate_viscosity(visc, model, params)
        
        # --- Left: Snapshots ---
        gs_snaps = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_grid[row, 0], wspace=0.1)
        max_w = jnp.max(jnp.abs(w_ctrl[0])) * 0.8
        
        for i, t_req in enumerate(CONFIG['snapshot_times']):
            ax = fig.add_subplot(gs_snaps[i])
            idx = (np.abs(t - t_req)).argmin()
            im = ax.imshow(w_ctrl[idx], extent=[0,1,0,1], origin='lower', cmap='RdBu_r', vmin=-max_w, vmax=max_w)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(f"t = {t[idx]:.1f}s")
            if i == 0: ax.set_ylabel(f"$\\nu$={visc:.1e}", fontweight='bold')

        # --- Right: Enstrophy Plot ---
        ax_plot = fig.add_subplot(outer_grid[row, 1])
        e_ctrl = jnp.mean(w_ctrl**2, axis=(1,2))
        e_base = jnp.mean(w_base**2, axis=(1,2))
        
        ax_plot.plot(t, e_base, 'k--', alpha=0.5, label='Natural')
        ax_plot.plot(t, e_ctrl, 'b-', label='Controlled')
        ax_plot.set_yscale('log')
        ax_plot.grid(True, which='both', alpha=0.2)
        if row == 0: ax_plot.legend(fontsize=8)
        ax_plot.set_ylabel("Enstrophy")
        if row == len(TEST_VISCOSITIES)-1: ax_plot.set_xlabel("Time (s)")

    plt.suptitle(f"Robustness Test: Policy Trained at $\\nu$={TRAINED_VISCOSITY}", fontsize=16, y=0.95)
    plt.savefig("figures/images/params/robustness/viscosity_robustness_comparison.pdf", dpi=150, bbox_inches='tight')
    print("✓ Saved combined robustness plot to figures/images/params/robustness/viscosity_robustness_comparison.pdf")