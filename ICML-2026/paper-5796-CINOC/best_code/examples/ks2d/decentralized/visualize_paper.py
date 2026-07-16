"""
Paper-Quality Visualization for KS-2D Stabilization.
Two-column compatible figure with:
- Top Row: Natural Evolution (Uncontrolled)
- Bottom Row: Controlled Evolution (DPC)
- Right Panel: Stabilization Performance (Energy vs Time)

Fonts: Times New Roman (Serif) for publication quality.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path
import cmcrameri.cm as cmc  # Crameri scientific colormaps

# Force CPU for visualization (avoids OOM on small GPUs during plotting)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet 
from data_utils import get_batch_initial_conditions

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 64,
    'L_domain': 32.0,
    
    # --- CRITICAL: MATCH TRAINING EXACTLY ---
    'dt': 0.005,           # High-res physics
    'substeps': 20,        # 20 physics steps per control step
    'n_agents': 100,       # 10x10 Grid
    'sigma': 1.2,          # Actuator influence radius
    
    # Visualization Timeline 
    # Control Step size = 20 * 0.005 = 0.1s
    'T_chaos_steps': 0,    # No chaos phase - start from t=0
    'T_control_steps': 100, # 10.0 seconds of control
    
    # Snapshots to display (Physical Time, t=0 is control start)
    'snapshot_times': [0.0, 2.0, 5.0, 7.0, 9.0], 
    
    'params_file': 'ks2d_centralized_params.msgpack'
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PLOTTING STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font settings - Times New Roman for papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Math font compatible with Times
        
        # Font sizes for two-column paper
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        
        # Line widths
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        
        # Remove top/right spines for cleaner look
        "axes.spines.top": True,
        "axes.spines.right": True,
        
        # High-quality output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuator_grid():
    """Reconstructs the actuator grid from n_agents."""
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

def load_params(model, filepath):
    """Load trained model parameters."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
        
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(42)
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    init_params = model.init(key, dummy_u, dummy_u, dummy_xi)
    
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def get_zero_policy():
    """Returns a zero-control policy for uncontrolled simulation."""
    def zero_policy_fn(params, u, u_target, xi):
        return jnp.zeros((xi.shape[0],))
    return zero_policy_fn

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison_simulation(key, model, params):
    """
    Run simulation with branching:
    1. Start from a chaotic initial condition
    2. Run both Controlled and Natural Evolution from t=0
    
    Returns:
        t_axis: Time array (starting from t=0)
        u_controlled: State trajectory with control
        u_natural: State trajectory without control (natural evolution)
    """
    n_ctrl = CONFIG['T_control_steps']
    xi_fixed = get_actuator_grid()
    u_target = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    
    # Setup dynamics
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    dyn_natural = PDEDynamics2D(policy_apply_fn=get_zero_policy())
    
    # 1. Generate initial chaotic condition (this IC comes from a warmed-up state)
    print("  [Sim] Generating chaotic initial state...")
    u0 = get_batch_initial_conditions(key, 1, CONFIG['N_grid'], CONFIG['L_domain'])[0]
    
    # 2. Controlled Branch (from t=0)
    print(f"  [Sim] Running controlled phase ({n_ctrl * CONFIG['substeps'] * CONFIG['dt']:.1f}s)...")
    u_controlled, _, _, _ = dyn_control.unroll_controlled(
        u0, xi_fixed, u_target, params,
        t_steps=n_ctrl,
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], 
        dt=CONFIG['dt'], sigma=CONFIG['sigma']
    )
    
    # 3. Natural Evolution Branch (no control, from same IC)
    print(f"  [Sim] Running natural evolution phase ({n_ctrl * CONFIG['substeps'] * CONFIG['dt']:.1f}s)...")
    u_natural, _, _, _ = dyn_natural.unroll_controlled(
        u0, xi_fixed, u_target, params,
        t_steps=n_ctrl,
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], L=CONFIG['L_domain'], 
        dt=CONFIG['dt'], sigma=CONFIG['sigma']
    )
    
    # 4. Time axis (starting from t=0)
    dt_effective = CONFIG['substeps'] * CONFIG['dt']
    t_axis = jnp.arange(n_ctrl) * dt_effective
    
    return t_axis, u_controlled, u_natural

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_snapshot_row(ax_list, t_axis, u_data, v_lim):
    """Plot a row of snapshots at specified time points using contourf."""
    snap_indices = []
    for t_req in CONFIG['snapshot_times']:
        idx = int((np.abs(t_axis - t_req)).argmin())
        snap_indices.append(idx)
    
    # Create coordinate grids for contourf
    N = CONFIG['N_grid']
    L = CONFIG['L_domain']
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # Contour levels for smooth rendering
    levels = np.linspace(-v_lim, v_lim, 100)
    
    cf = None
    for i, idx in enumerate(snap_indices):
        ax = ax_list[i]
        u_snap = np.array(u_data[idx])
        t_snap = float(t_axis[idx])
        
        cf = ax.contourf(
            X, Y, u_snap,
            levels=levels,
            cmap=cmc.vik, 
            extend='both'
            # extend='neither'

        )
        
        # Set axis limits and aspect ratio
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        
        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Consistent border styling
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        
        # Time label below each snapshot
        ax.set_xlabel(f"t={t_snap:.1f}s", fontsize=11)
    
    return cf

def plot_energy_comparison(ax, t_axis, u_controlled, u_natural):
    """Plot energy comparison with three phases: Chaotic Precursor, Natural Evolution, Controlled."""
    # Compute energy (L2 norm squared, averaged over space)
    energy_ctrl = jnp.mean(u_controlled**2, axis=(1, 2))
    energy_natural = jnp.mean(u_natural**2, axis=(1, 2))
    
    # Masks for different phases
    mask_chaos = t_axis <= 0
    mask_ctrl = t_axis >= 0
    
    # Plot Chaotic Precursor (shared phase, red solid)
    ax.plot(t_axis[mask_chaos], energy_natural[mask_chaos], 
            color='firebrick', lw=1.5, label='Chaotic Precursor')
    
    # Plot Natural Evolution (grey dashed)
    ax.plot(t_axis[mask_ctrl], energy_natural[mask_ctrl], 
            color='grey', lw=1.5, linestyle='--', label='Natural Evolution')
    
    # Plot Controlled (blue solid)
    ax.plot(t_axis[mask_ctrl], energy_ctrl[mask_ctrl], 
            color='navy', lw=1.5, label='Controlled')
    
    # Vertical line at t=0
    ax.axvline(x=0, color='k', linestyle=':', alpha=0.5, lw=0.8)
    
    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Energy $\langle u^2 \rangle$")
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_title("Stabilization Performance", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_paper_figure(t_axis, u_controlled, u_natural, save_name="figures/images/paper/ks2d_paper_figure.pdf"):
    """
    Create the paper-quality figure with:
    - Top Row: Natural Evolution (Uncontrolled)
    - Bottom Row: Controlled Evolution (DPC)
    """
    setup_paper_style()
    
    # Figure size for two-column paper
    n_snaps = len(CONFIG['snapshot_times'])
    fig = plt.figure(figsize=(7.0, 3.2))
    
    # Layout: Two rows of snapshots with colorbar on right
    gs_main = gridspec.GridSpec(2, n_snaps + 1, width_ratios=[1]*n_snaps + [0.08], 
                                 hspace=0.45, wspace=0.03)
    
    # Row 1: Natural Evolution (top)
    ax_row1 = [fig.add_subplot(gs_main[0, j]) for j in range(n_snaps)]
    
    # Row 2: Controlled Evolution (bottom)
    ax_row2 = [fig.add_subplot(gs_main[1, j]) for j in range(n_snaps)]
    
    # Determine color limits based on initial state
    v_lim = float(jnp.max(jnp.abs(u_controlled[0]))) * 0.9
    
    # Plot Natural Evolution (top row)
    cf = plot_snapshot_row(ax_row1, t_axis, u_natural, v_lim)
    ax_row1[2].set_title("Natural Evolution", pad=4)
    
    # Plot Controlled Evolution (bottom row)
    plot_snapshot_row(ax_row2, t_axis, u_controlled, v_lim)
    ax_row2[2].set_title("Controlled Evolution", pad=4)
    
    # Add colorbar on the right
    cax = fig.add_subplot(gs_main[:, -1])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    
    # Save figure
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved paper figure to {save_name}")
    
    # Also save PNG version for quick viewing
    png_name = save_name.replace('.pdf', '.png')
    plt.savefig(png_name, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"✓ Saved PNG version to {png_name}")
    
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("KS-2D Paper Figure Generation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("figures/images/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Model
    model = DecentralizedKS2DControlNet(
        features=(64, 128), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=5.0
    )
    
    # 2. Load Parameters
    try:
        print(f"\nLoading parameters from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
        print("✓ Parameters loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
        print("Run training first to generate parameters.")
        sys.exit(1)
    
    # 3. Run Simulation
    print("\nRunning comparison simulation...")
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    t_axis, u_controlled, u_natural = run_comparison_simulation(key, model, params)
    
    print(f"\nSimulation complete:")
    print(f"  Time range: [{float(t_axis[0]):.1f}, {float(t_axis[-1]):.1f}] seconds")
    print(f"  Trajectory shape: {u_controlled.shape}")
    
    # 4. Generate Figure
    print("\nGenerating paper figure...")
    create_paper_figure(t_axis, u_controlled, u_natural)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
