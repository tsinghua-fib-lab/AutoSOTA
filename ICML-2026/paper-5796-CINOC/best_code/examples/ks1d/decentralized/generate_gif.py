import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import sys
import flax.serialization
from pathlib import Path
import copy

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
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuators(n_agents, L):
    return jnp.linspace(0.0, L, n_agents, endpoint=False) + (L/n_agents)/2

def run_scenario_comparison(scenario, model, params):
    L, N, n_agents = scenario['L'], scenario['N'], scenario['n_agents']
    
    seed = int(L * 100)
    key = jax.random.PRNGKey(seed)
    u0 = get_batch_initial_conditions(key, 1, N, L)[0]
    
    zero_policy = lambda p, u, ut, xi: jnp.zeros(n_agents)
    
    dyn_control = PDEDynamics(policy_apply_fn=model.apply)
    dyn_natural = PDEDynamics(policy_apply_fn=zero_policy)
    
    xi_fixed = get_actuators(n_agents, L)
    u_target = jnp.zeros_like(u0)
    
    u_nat, _, _, _ = dyn_natural.unroll_controlled(
        u0, xi_fixed, u_target, params,
        COMMON_CONFIG['t_steps'],   
        N, L, dt=COMMON_CONFIG['dt'], sigma=COMMON_CONFIG['sigma']
    )
    
    u_ctrl, _, _, _ = dyn_control.unroll_controlled(
        u0, xi_fixed, u_target, params,
        COMMON_CONFIG['t_steps'],   
        N, L, dt=COMMON_CONFIG['dt'], sigma=COMMON_CONFIG['sigma']
    )
    
    t_axis = np.arange(COMMON_CONFIG['t_steps']) * COMMON_CONFIG['dt']
    return t_axis, u_nat, u_ctrl

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING SETUP UTILS 
# ═══════════════════════════════════════════════════════════════════════════════

def setup_heatmap(ax, t, L, u_data_full, title=None, ylabel=False):
    std_val = np.std(u_data_full)
    vmin, vmax = -2.5 * std_val, 2.5 * std_val
    
    cmap = copy.copy(plt.get_cmap('RdBu_r'))
    cmap.set_bad(color='white') 
    
    img_empty = np.full((u_data_full.shape[1], u_data_full.shape[0]), np.nan)
    
    im = ax.imshow(img_empty, aspect='auto', origin='lower', cmap=cmap,
                   extent=[t[0], t[-1], 0, L],
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=12)
    
    ax.set_xlabel("Time (s)")
    if ylabel:
        ax.set_ylabel(r"$x$")
    else:
        ax.set_yticks([])

    return im

def setup_energy_row(ax, t, u_nat_full, u_ctrl_full, show_legend=False):
    line_nat, = ax.plot([], [], color='grey', linestyle='--', lw=2.0, label='Natural')
    line_ctrl, = ax.plot([], [], color='navy', lw=2.0, label='Controlled')
    
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    
    e_nat_full = jnp.mean(u_nat_full**2, axis=1)
    e_ctrl_full = jnp.mean(u_ctrl_full**2, axis=1)
    min_e = float(min(jnp.min(e_nat_full), jnp.min(e_ctrl_full)))
    max_e = float(max(jnp.max(e_nat_full), jnp.max(e_ctrl_full)))
    
    ax.set_ylim(min_e * 0.5, max_e * 2.0)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=4))
    
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.95)
        
    return line_nat, line_ctrl

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANIMATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_animation(scenarios_to_run, save_name, figsize, is_single=False):
    num_scenarios = len(scenarios_to_run)
    fig = plt.figure(figsize=figsize)
    
    # Adjust top margin to give breathing room for titles and colorbar
    top_margin = 0.75 if is_single else 0.90
    gs_outer = gridspec.GridSpec(num_scenarios, 3, width_ratios=[1, 1, 0.8], 
                                 wspace=0.35, hspace=0.4, top=top_margin) 
    
    scen_data = [] 
    im_nats, im_ctrls = [], []
    line_nats, line_ctrls = [], []
    ref_im = None
    
    for i, scen in enumerate(scenarios_to_run):
        print(f"  Simulating {scen['name']} (L={scen['L']}, N={scen['N']})...")
        
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
            print(f"    [Warning] {param_file} not found. Using random weights.")
            params = init_params
            
        t, u_nat, u_ctrl = run_scenario_comparison(scen, model, params)
        e_nat = jnp.mean(u_nat**2, axis=1)
        e_ctrl = jnp.mean(u_ctrl**2, axis=1)
        
        scen_data.append((t, u_nat, u_ctrl, e_nat, e_ctrl))
        
        ax_nat = fig.add_subplot(gs_outer[i, 0])
        ax_ctrl = fig.add_subplot(gs_outer[i, 1])
        ax_en = fig.add_subplot(gs_outer[i, 2])
        
        title_nat = f"Natural Evolution (L={int(scen['L'])})" if i == 0 and not is_single else ("Natural Evolution" if is_single else None)
        title_ctrl = "Controlled Evolution" if i == 0 else None
        
        im_nat = setup_heatmap(ax_nat, t, scen['L'], u_nat, title=title_nat, ylabel=True)
        if i == 0: ref_im = im_nat
            
        im_ctrl = setup_heatmap(ax_ctrl, t, scen['L'], u_ctrl, title=title_ctrl)
        
        if is_single:
            ax_nat.set_ylabel(r"Spatial domain $x$", fontweight='bold')
            # Pushed suptitle higher up
            fig.suptitle(f"KS-1D Stabilization (L={int(scen['L'])})", fontweight='bold', y=0.96, fontsize=20)
        else:
            ax_nat.set_ylabel(f"Scale L={int(scen['L'])}\n$x$", fontweight='bold')
        
        line_n, line_c = setup_energy_row(ax_en, t, u_nat, u_ctrl, show_legend=(i==0))
        ax_en.set_ylabel(r"Energy $\langle u^2 \rangle$")
        
        if i == 0:
            ax_en.set_title("Stabilization Performance", fontweight='bold', pad=12)
            
        im_nats.append(im_nat)
        im_ctrls.append(im_ctrl)
        line_nats.append(line_n)
        line_ctrls.append(line_c)

    # Smart colorbar placement based on grid size
    if is_single:
        # Nestled comfortably between the suptitle and the subplots
        cbar_ax = fig.add_axes([0.15, 0.84, 0.40, 0.03]) 
    else:
        cbar_ax = fig.add_axes([0.15, 0.94, 0.4, 0.02])
        
    cb = fig.colorbar(ref_im, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'State field $u(x,t)$', rotation=0, labelpad=5)
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.xaxis.set_ticks_position('top')
    
    FRAME_STEP = 4 
    frames = np.arange(1, COMMON_CONFIG['t_steps'] + 1, FRAME_STEP)
    
    def update(frame):
        for i, (t, u_nat, u_ctrl, e_nat, e_ctrl) in enumerate(scen_data):
            img_nat = np.full((u_nat.shape[1], u_nat.shape[0]), np.nan)
            img_nat[:, :frame] = u_nat.T[:, :frame]
            im_nats[i].set_data(img_nat)

            img_ctrl = np.full((u_ctrl.shape[1], u_ctrl.shape[0]), np.nan)
            img_ctrl[:, :frame] = u_ctrl.T[:, :frame]
            im_ctrls[i].set_data(img_ctrl)
            
            line_nats[i].set_data(t[:frame], e_nat[:frame])
            line_ctrls[i].set_data(t[:frame], e_ctrl[:frame])
            
        return im_nats + im_ctrls + line_nats + line_ctrls

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    
    save_path = Path("figures/images/gif") / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    anim.save(save_path, writer=PillowWriter(fps=15)) 
    print(f"✓ Saved animation to {save_path}\n")
    plt.close(fig) # Prevent memory leaks between runs

# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_plot_style() 
    
    # 1. Multi-scale Comparison (All 3 Scenarios)
    print("--- Generating Multi-Scale GIF ---")
    build_animation(
        scenarios_to_run=SCENARIOS, 
        save_name="ks1d_multiscale_comparison.gif", 
        figsize=(18, 4.0 * len(SCENARIOS)),
        is_single=False
    )
    
    # 2. Large Scale Only (L=500)
    print("--- Generating Single L=500 GIF ---")
    large_scenario = [s for s in SCENARIOS if s['L'] == 500.0]
    build_animation(
        scenarios_to_run=large_scenario, 
        save_name="ks1d_L500_comparison.gif", 
        figsize=(14, 9.), 
        is_single=True
    )
    
    print("All tasks completed.")