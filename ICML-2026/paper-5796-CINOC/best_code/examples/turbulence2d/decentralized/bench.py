"""
Multi-Sample Comparison: Natural vs. Controlled (Decentralized) Turbulence.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import pickle
import flax.serialization
from pathlib import Path
import matplotlib.ticker as ticker

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_turb import DecentralizedTurbulenceNet 

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 64,          
    'L_domain': 1.0,
    
    # --- Physics ---
    'dt': 0.01,            
    'substeps': 5,         
    'viscosity': 5e-4,     
    
    'n_agents': 64,
    'grid_shape': (8, 8),
    'sigma': 0.05,         
    
    # --- Duration ---
    'T_chaos_steps': 50,    # 0.5s chaos
    'T_control_steps': 200, # 2.0s control
    
    # Snapshot times relative to control start (t=0)
    'snapshot_times': [-0.25, 0.0, 0.5, 1.0, 2.0],
    
    # --- Files ---
    'params_file': 'turbulence_params.msgpack',
    'ic_filename': 'turbulence_chaotic_ics_64_more.pkl', 
    
    # --- Evaluation ---
    'n_eval_samples': 5,    # Number of different I.C.s to test
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA & MODEL LOADING (Unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuator_grid():
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

def load_dataset():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" 
    file_path = data_dir / CONFIG['ic_filename']
    
    if not file_path.exists():
        file_path = Path(CONFIG['ic_filename'])

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {CONFIG['ic_filename']}.")

    print(f"[Data] Loading dataset from: {file_path}")
    with open(file_path, 'rb') as f:
        u_pool = pickle.load(f)
    
    return jnp.array(u_pool) 

def load_params(model, filepath):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
        
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(42)
    xi_fixed = get_actuator_grid()
    dummy_obs = jnp.zeros((1, CONFIG['N_grid'], CONFIG['N_grid']))
    init_params = model.init(key, xi_fixed, dummy_obs)
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def get_zero_policy(n_agents):
    def zero_policy_fn(params, xi, obs):
        return jnp.zeros((n_agents,))
    return zero_policy_fn

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(w0_hat, model, params):
    """Run Chaos -> Branch(Control vs Baseline) for a single IC."""
    n_chaos = CONFIG['T_chaos_steps']
    n_ctrl = CONFIG['T_control_steps']
    xi_fixed = get_actuator_grid()
    
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_baseline = PDEDynamics2D(policy_apply_fn=get_zero_policy(CONFIG['n_agents']))
    
    # 1. Chaos Phase
    w_chaos, _ = dyn_baseline.unroll_controlled(
        w0_hat, xi_fixed, params,
        t_steps=n_chaos, substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'], actuator_grid_shape=CONFIG['grid_shape'],
        sigma=CONFIG['sigma']
    )
    
    # Handoff
    w_handoff_phys = w_chaos[-1]
    w_handoff_hat = jnp.fft.fft2(w_handoff_phys)
    
    # 2. Controlled Branch
    w_ctrl_phase, _ = dyn_control.unroll_controlled(
        w_handoff_hat, xi_fixed, params,
        t_steps=n_ctrl, substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'], actuator_grid_shape=CONFIG['grid_shape'],
        sigma=CONFIG['sigma']
    )

    # 3. Baseline Branch
    w_base_phase, _ = dyn_baseline.unroll_controlled(
        w_handoff_hat, xi_fixed, params,
        t_steps=n_ctrl, substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'], actuator_grid_shape=CONFIG['grid_shape'],
        sigma=CONFIG['sigma']
    )
    
    # Stitch
    w_blue = jnp.concatenate([w_chaos, w_ctrl_phase], axis=0)
    w_grey = jnp.concatenate([w_chaos, w_base_phase], axis=0)
    
    # Time Axis
    dt = CONFIG['dt']
    t_chaos = (np.arange(n_chaos) - n_chaos) * dt 
    t_ctrl = np.arange(n_ctrl) * dt
    t_axis = np.concatenate([t_chaos, t_ctrl])
    
    return t_axis, w_blue, w_grey

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_snapshots_row(ax_list, t, w_data, row_title, v_limit):
    """Plots a single row of snapshots (either Control or Natural)."""
    
    snap_indices = []
    for t_req in CONFIG['snapshot_times']:
        idx = (np.abs(t - t_req)).argmin()
        snap_indices.append(idx)
        
    im = None
    for i, idx in enumerate(snap_indices):
        ax = ax_list[i]
        w_snap = w_data[idx]
        t_snap = t[idx]
        
        im = ax.imshow(w_snap, extent=[0,1,0,1], origin='lower', cmap='RdBu_r', vmin=-v_limit, vmax=v_limit)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border color based on phase
        if t_snap < -1e-4: color = 'firebrick' # Chaos phase
        elif t_snap > 1e-4: color = 'navy'     # Control phase
        else: color = 'black'
            
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
            
        # Top title only for the first row called
        if i == 2: # Center title over row
             ax.set_title(row_title, fontsize=12, pad=10, fontweight='bold')

        # Time labels at the bottom of the snapshots
        ax.set_xlabel(f"t={t_snap:.2f}s", fontsize=9)

    return im

def plot_enstrophy_comparison(ax, t, w_ctrl, w_base):
    """Plots the comparison line graph."""
    e_ctrl = jnp.mean(w_ctrl**2, axis=(1,2))
    e_base = jnp.mean(w_base**2, axis=(1,2))
    
    mask_chaos = t <= 0
    mask_ctrl = t >= 0
    
    # Chaos Phase
    ax.plot(t[mask_chaos], e_base[mask_chaos], color='firebrick', lw=2, label='Chaos Phase')
    
    # Divergence
    ax.plot(t[mask_ctrl], e_base[mask_ctrl], color='grey', linestyle='--', lw=2, label='Natural Evolution')
    ax.plot(t[mask_ctrl], e_ctrl[mask_ctrl], color='navy', lw=2, label='Controlled')
    
    ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"Enstrophy $\langle \omega^2 \rangle$", fontsize=11)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Stabilization Performance", fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"--- Multi-Sample Comparison (N={CONFIG['N_grid']}) ---")
    
    # 1. Init Model
    model = DecentralizedTurbulenceNet(
        features=(32, 64), 
        patch_size=16, 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=150.0 
    )
    
    # 2. Load
    try:
        params = load_params(model, CONFIG['params_file'])
        full_dataset = load_dataset()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Select Multiple Random Indices
    np.random.seed(999) 
    indices = np.random.choice(len(full_dataset), CONFIG['n_eval_samples'], replace=False)
    
    print(f"\nRunning simulation on {len(indices)} samples...")
    print(f"{'Sample ID':<10} | {'Final Enstrophy (Natural)':<28} | {'Final Enstrophy (Controlled)':<28}")
    print("-" * 75)

    last_run_data = None 
    final_ens_base = []
    final_ens_ctrl = []

    for idx in indices:
        w0_hat = full_dataset[idx]
        
        # Run Simulation
        t_axis, w_ctrl, w_base = run_comparison(w0_hat, model, params)
        
        # Calculate Final Enstrophy (at last time step)
        # w is (Time, H, W). Enstrophy is mean(w^2)
        ens_base_final = float(jnp.mean(w_base[-1]**2))
        ens_ctrl_final = float(jnp.mean(w_ctrl[-1]**2))
        
        final_ens_base.append(ens_base_final)
        final_ens_ctrl.append(ens_ctrl_final)
        
        print(f"{idx:<10} | {ens_base_final:.6e}                   | {ens_ctrl_final:.6e}")
        
        # Save data from this run (will overwrite until the last one remains)
        last_run_data = (idx, t_axis, w_ctrl, w_base)

    # Compute Averages
    avg_base = np.mean(final_ens_base)
    avg_ctrl = np.mean(final_ens_ctrl)

    print("-" * 75)
    print(f"{'AVERAGE':<10} | {avg_base:.6e}                   | {avg_ctrl:.6e}")
    print("-" * 75)

    # 4. Plotting (Using the last sample processed)
    if last_run_data:
        idx, t_axis, w_ctrl, w_base = last_run_data
        print(f"\nGenerating plot for last sample (ID: {idx})...")

        # Layout: 2 Rows (Visuals), 1 Column (Graph) spanning both rows
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], wspace=0.1, hspace=0.3)
        
        # -- Row 1: Controlled Visuals --
        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, 0], wspace=0.05)
        ax_row1 = [fig.add_subplot(gs_row1[j]) for j in range(5)]
        
        # -- Row 2: Natural Visuals --
        gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, 0], wspace=0.05)
        ax_row2 = [fig.add_subplot(gs_row2[j]) for j in range(5)]
        
        # -- Right: Comparison Plot --
        ax_plot = fig.add_subplot(gs[:, 1])
        
        # Calculate limits based on initial state for fair comparison
        v_lim = jnp.max(jnp.abs(w_ctrl[0])) * 0.9
        
        # Plot Controlled
        im = plot_snapshots_row(ax_row1, t_axis, w_ctrl, "Controlled (Decentralized)", v_lim)
        
        # Plot Natural
        plot_snapshots_row(ax_row2, t_axis, w_base, "Natural Evolution (Uncontrolled)", v_lim)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax_row1 + ax_row2, fraction=0.02, pad=0.02)
        cbar.set_label(r'Vorticity $\omega$', fontsize=10)
        
        # Plot Graph
        plot_enstrophy_comparison(ax_plot, t_axis, w_ctrl, w_base)

        plt.suptitle(f"Sample Analysis: Index {idx}", fontsize=16, fontweight='bold')
        save_name = "single_sample_comparison.png"
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_name}")