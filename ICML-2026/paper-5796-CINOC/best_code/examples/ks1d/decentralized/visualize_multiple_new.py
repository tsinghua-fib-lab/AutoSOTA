import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {'name': 'Small',  'L': 64.0,  'N': 256,  'n_agents': 30},
    {'name': 'Medium', 'L': 200.0, 'N': 512,  'n_agents': 80},
    {'name': 'Large',  'L': 500.0, 'N': 1024, 'n_agents': 200},
]

COMMON_CONFIG = {
    'dt': 0.05,
    't_steps': 400,        
    'sigma': 1.0,          
    'model_dir': "multiple_experiments/models"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. STYLE SETUP (Larger Fonts)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_plot_style():
    """Configure matplotlib for publication-quality figures with larger fonts."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Increased base sizes
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuators(n_agents, L):
    """Returns fixed actuator positions."""
    return jnp.linspace(0.0, L, n_agents, endpoint=False) + (L/n_agents)/2

def run_scenario_comparison(scenario, model, params):
    L, N, n_agents = scenario['L'], scenario['N'], scenario['n_agents']
    
    # 1. Generate chaotic Initial Condition
    seed = int(L * 100)
    key = jax.random.PRNGKey(seed)
    u0 = get_batch_initial_conditions(key, 1, N, L)[0]
    
    # 2. Setup Dynamics
    zero_policy = lambda p, u, ut, xi: jnp.zeros(n_agents)
    
    dyn_control = PDEDynamics(policy_apply_fn=model.apply)
    dyn_natural = PDEDynamics(policy_apply_fn=zero_policy)
    
    xi_fixed = get_actuators(n_agents, L)
    u_target = jnp.zeros_like(u0)
    
    # 3. Run Simulations
    u_nat, _, _, _ = dyn_natural.unroll_controlled(
        u0, xi_fixed, u_target, params,
        COMMON_CONFIG['t_steps'],   
        N,                                  
        L,                                  
        dt=COMMON_CONFIG['dt'], 
        sigma=COMMON_CONFIG['sigma']
    )
    
    u_ctrl, _, _, _ = dyn_control.unroll_controlled(
        u0, xi_fixed, u_target, params,
        COMMON_CONFIG['t_steps'],   
        N,                                  
        L,                                  
        dt=COMMON_CONFIG['dt'], 
        sigma=COMMON_CONFIG['sigma']
    )
    
    t_axis = np.arange(COMMON_CONFIG['t_steps']) * COMMON_CONFIG['dt']
    return t_axis, u_nat, u_ctrl

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(ax, t, x, u_data, L, title=None, ylabel=False):
    img = u_data.T 
    std_val = np.std(img)
    vmin, vmax = -2.5 * std_val, 2.5 * std_val
    
    im = ax.imshow(img, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[t[0], t[-1], 0, L],
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    if title:
        # rcParams handles size (18), just add bold
        ax.set_title(title, fontweight='bold', pad=12)
    
    ax.set_xlabel("Time (s)")
    if ylabel:
        ax.set_ylabel(r"$x$")
    else:
        ax.set_yticks([])

    return im

def plot_energy_row(ax, t, u_nat, u_ctrl, show_legend=False):
    e_nat = jnp.mean(u_nat**2, axis=1)
    e_ctrl = jnp.mean(u_ctrl**2, axis=1)
    
    ax.plot(t, e_nat, color='grey', linestyle='--', lw=2.0, label='Natural')
    ax.plot(t, e_ctrl, color='navy', lw=2.0, label='Controlled')
    
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=4))
    
    if show_legend:
        # Legend at center right
        ax.legend(loc='center right', framealpha=0.95)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_plot_style() # Apply larger fonts
    print(f"--- KS-1D Multi-Scale Visualization ---")
    
    num_scenarios = len(SCENARIOS)
    
    # Increase figure size slightly to accommodate larger fonts
    # 4.0 inches height per row gives enough breathing room
    fig = plt.figure(figsize=(18, 4.0 * num_scenarios))
    
    # Increased hspace=0.4 to prevent title overlap
    gs_outer = gridspec.GridSpec(num_scenarios, 3, width_ratios=[1, 1, 0.8], 
                                 wspace=0.35, hspace=0.4,
                                 top=0.90) 
    
    ref_im = None
    
    for i, scen in enumerate(SCENARIOS):
        print(f"Processing Scenario {i+1}: {scen['name']} (L={scen['L']}, N={scen['N']})...")
        
        # 1. Initialize & Load
        model = DecentralizedControlNet(features=(64, 64), L_domain=scen['L'])
        param_file = f"ks_params_N{scen['N']}_L{int(scen['L'])}_A{scen['n_agents']}.msgpack"
        full_path = Path(COMMON_CONFIG['model_dir']) / param_file
        
        dummy_u = jnp.zeros((scen['N'],))
        dummy_xi = get_actuators(scen['n_agents'], scen['L'])
        init_params = model.init(jax.random.PRNGKey(0), dummy_u, dummy_u, dummy_xi)
        
        if full_path.exists():
            with open(full_path, 'rb') as f:
                params = flax.serialization.from_bytes(init_params, f.read())
        else:
            print(f"  [Warning] {param_file} not found. Using random weights.")
            params = init_params
            
        # 2. Run
        t, u_nat, u_ctrl = run_scenario_comparison(scen, model, params)
        
        # 3. Plot
        ax_nat = fig.add_subplot(gs_outer[i, 0])
        ax_ctrl = fig.add_subplot(gs_outer[i, 1])
        ax_en = fig.add_subplot(gs_outer[i, 2])
        
        title_nat = f"Natural Evolution (L={int(scen['L'])})" if i == 0 else None
        title_ctrl = "Controlled Evolution" if i == 0 else None
        
        im = plot_heatmap(ax_nat, t, None, u_nat, scen['L'], title=title_nat, ylabel=True)
        if i == 0: ref_im = im
            
        plot_heatmap(ax_ctrl, t, None, u_ctrl, scen['L'], title=title_ctrl)
        
        # Manually set label with scale info
        ax_nat.set_ylabel(f"Scale L={int(scen['L'])}\n$x$", fontweight='bold')
        
        plot_energy_row(ax_en, t, u_nat, u_ctrl, show_legend=(i==0))
        ax_en.set_ylabel(r"Energy $\langle u^2 \rangle$")
        
        if i == 0:
            ax_en.set_title("Stabilization Performance", fontweight='bold', pad=12)

    # --- MANUAL COLORBAR PLACEMENT ---
    # Position: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.15, 0.94, 0.4, 0.02]) 
    
    cb = fig.colorbar(ref_im, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'State field $u(x,t)$', rotation=0, labelpad=5)
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('top')
    
    save_path = Path("figures/images/multi_domain") / "ks1d_multiscale_comparison.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")