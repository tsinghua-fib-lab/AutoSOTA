"""
Multi-Example Visualization for Trained Turbulence Policy.
Randomly selects 3 examples from the dataset and visualizes stabilization.
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
    # --- 1. Grid Resolution (MUST MATCH TRAINING) ---
    'N_grid': 64,          
    'L_domain': 1.0,
    
    # --- 2. Physics (MUST MATCH TRAINING) ---
    'dt': 0.01,            
    'substeps': 5,         
    'viscosity': 5e-4,     
    
    'n_agents': 64,
    'grid_shape': (8, 8),
    'sigma': 0.05,         
    
    # --- 3. Duration ---
    'T_chaos_steps': 50,    # 0.5s chaos
    'T_control_steps': 200, # 2.0s control
    
    # Times relative to control start (t=0)
    'snapshot_times': [-0.25, 0.0, 0.5, 1.0, 2.0],
    
    # --- 4. Files ---
    'params_file': 'turbulence_params.msgpack',
    # Ensure this matches the file used in training
    'ic_filename': 'turbulence_chaotic_ics_64_more.pkl', 
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuator_grid():
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

def load_dataset():
    """Loads the full dataset from pickle."""
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
    
    return jnp.array(u_pool) # Shape: (pool_size, N, N)

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
# 3. SIMULATION LOOP (Comparison)
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
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

# def plot_single_case(ax_row_snaps, ax_plot, t, w_ctrl, w_base, case_idx):
#     """Helper to plot one case into provided Axes objects."""
    
#     # --- Snapshots ---
#     snap_indices = []
#     for t_req in CONFIG['snapshot_times']:
#         idx = (np.abs(t - t_req)).argmin()
#         snap_indices.append(idx)
        
#     max_w = jnp.max(jnp.abs(w_ctrl[0])) * 0.9 
    
#     for i, idx in enumerate(snap_indices):
#         ax = ax_row_snaps[i]
#         w_snap = w_ctrl[idx]
#         t_snap = t[idx]
        
#         im = ax.imshow(w_snap, extent=[0,1,0,1], origin='lower', cmap='RdBu_r', vmin=-max_w, vmax=max_w)
#         ax.set_xticks([])
#         ax.set_yticks([])
        
#         if t_snap < -1e-4: color = 'firebrick'
#         elif t_snap > 1e-4: color = 'navy'
#         else: color = 'black'
            
#         # Only add titles to the top row
#         if case_idx == 0:
#             ax.set_title(f"t = {t_snap:.2f}s", color=color, fontweight='bold', fontsize=10)
        
#         for spine in ax.spines.values():
#             spine.set_edgecolor(color)
#             spine.set_linewidth(1.5)
            
#         # Add 'Case X' label to the left of the first snapshot
#         if i == 0:
#             ax.set_ylabel(f"Case {case_idx}\nVorticity", fontsize=10)

#     # --- Enstrophy Plot ---
#     e_ctrl = jnp.mean(w_ctrl**2, axis=(1,2))
#     e_base = jnp.mean(w_base**2, axis=(1,2))
    
#     mask_chaos = t <= 0
#     mask_ctrl = t >= 0
    
#     ax_plot.plot(t[mask_chaos], e_base[mask_chaos], color='firebrick', lw=1.5)
#     ax_plot.plot(t[mask_ctrl], e_base[mask_ctrl], color='grey', linestyle='--', label='Uncontrolled', lw=1.5)
#     ax_plot.plot(t[mask_ctrl], e_ctrl[mask_ctrl], color='navy', label='Ours', lw=1.5)
    
#     ax_plot.axvline(x=0, color='k', linestyle=':', alpha=0.5)
#     ax_plot.set_yscale('log')
    
#     if case_idx == 2: # Bottom plot
#         ax_plot.set_xlabel("Time (s)")
#     else:
#         ax_plot.set_xticks([]) # Hide x-ticks for top ones
        
#     ax_plot.set_ylabel(r"Enstrophy")
#     ax_plot.set_xlim(t[0], t[-1])
#     ax_plot.grid(True, which="both", ls="--", alpha=0.3)
    
#     # Legend only on first plot to save space
#     if case_idx == 0:
#         ax_plot.legend(loc='upper right', fontsize=8)

def plot_single_case(ax_row_snaps, ax_plot, t, w_ctrl, w_base, case_idx):
    """Helper to plot one case into provided Axes objects."""
    
    # --- Snapshots ---
    snap_indices = []
    for t_req in CONFIG['snapshot_times']:
        idx = (np.abs(t - t_req)).argmin()
        snap_indices.append(idx)
        
    max_w = jnp.max(jnp.abs(w_ctrl[0])) * 0.9 
    
    im = None # Initialize variable to hold the image object
    
    for i, idx in enumerate(snap_indices):
        ax = ax_row_snaps[i]
        w_snap = w_ctrl[idx]
        t_snap = t[idx]
        
        # Save the 'im' object for the colorbar later
        im = ax.imshow(w_snap, extent=[0,1,0,1], origin='lower', cmap='RdBu_r', vmin=-max_w, vmax=max_w)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if t_snap < -1e-4: color = 'firebrick'
        elif t_snap > 1e-4: color = 'navy'
        else: color = 'black'
            
        # Only add titles to the top row
        if case_idx == 0:
            ax.set_title(f"t = {t_snap:.2f}s", color=color, fontweight='bold', fontsize=10)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
            
        # Add 'Case X' label to the left of the first snapshot
        if i == 0:
            ax.set_ylabel(f"Case {case_idx}\nVorticity", fontsize=10)

    # --- ADDED: Colorbar for the row ---
    # We use the figure object attached to the axes
    # ax=ax_row_snaps tells matplotlib to steal space from the whole row of snapshots
    fig = ax_row_snaps[0].figure
    cbar = fig.colorbar(im, ax=ax_row_snaps, fraction=0.02, pad=0.02)
    cbar.set_label(r'Vorticity $\omega$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # --- Enstrophy Plot ---
    e_ctrl = jnp.mean(w_ctrl**2, axis=(1,2))
    e_base = jnp.mean(w_base**2, axis=(1,2))
    
    mask_chaos = t <= 0
    mask_ctrl = t >= 0
    
    ax_plot.plot(t[mask_chaos], e_base[mask_chaos], color='firebrick', lw=1.5)
    ax_plot.plot(t[mask_ctrl], e_base[mask_ctrl], color='grey', linestyle='--', label='Uncontrolled', lw=1.5)
    ax_plot.plot(t[mask_ctrl], e_ctrl[mask_ctrl], color='navy', label='Ours', lw=1.5)
    
    ax_plot.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    ax_plot.set_yscale('log')
    
    if case_idx == 2: # Bottom plot
        ax_plot.set_xlabel("Time (s)")
    else:
        ax_plot.set_xticks([]) # Hide x-ticks for top ones
        
    ax_plot.set_ylabel(r"Enstrophy")
    ax_plot.set_xlim(t[0], t[-1])
    ax_plot.grid(True, which="both", ls="--", alpha=0.3)
    
    # Legend only on first plot to save space
    if case_idx == 0:
        ax_plot.legend(loc='upper right', fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"--- Visualizing 3 Random Examples (N={CONFIG['N_grid']}) ---")
    
    # Create output directory
    output_dir = Path("figures/images/vanilla")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Init Model
    model = DecentralizedTurbulenceNet(
        features=(32, 64), 
        patch_size=16, # Match 64x64 Training
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=150.0    # Match Training
    )
    
    # 2. Load
    try:
        params = load_params(model, CONFIG['params_file'])
        full_dataset = load_dataset()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 3. Pick 3 Random Indices
    np.random.seed(123) # Fixed seed for reproducibility, or remove for true random
    indices = np.random.choice(len(full_dataset), 3, replace=False)
    print(f"Selected Test Indices: {indices}")

    # 4. Setup Big Plot
    # Layout: 3 Rows. Left side = 5 Snapshots. Right side = 1 Time Series.
    fig = plt.figure(figsize=(16, 8))
    outer_grid = gridspec.GridSpec(3, 2, width_ratios=[2.5, 1], hspace=0.3, wspace=0.15)
    
    for i, idx in enumerate(indices):
        print(f"Processing Case {i} (Index {idx})...")
        w0_hat = full_dataset[idx]
        
        # Run Sim
        t_axis, w_ctrl, w_base = run_comparison(w0_hat, model, params)
        
        # Create Sub-grids for this row
        # Left: Snapshots
        gs_snaps = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer_grid[i, 0], wspace=0.05)
        ax_snaps = [fig.add_subplot(gs_snaps[j]) for j in range(5)]
        
        # Right: Plot
        ax_plot = fig.add_subplot(outer_grid[i, 1])
        
        # Plot
        plot_single_case(ax_snaps, ax_plot, t_axis, w_ctrl, w_base, case_idx=i)

    plt.suptitle("Turbulence Stabilization: 3 Random Test Cases", fontsize=16, fontweight='bold')
    save_name = "figures/images/vanilla/random_3_cases_64x64.png"
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved combined plot to {save_name}")