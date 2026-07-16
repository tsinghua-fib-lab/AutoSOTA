"""
Turbulence Zero-Shot Scaling Ablation Script.
"""

import jax
import jax.numpy as jnp
import optax
import time
import pickle
import flax.serialization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from functools import partial
from pathlib import Path
from tqdm import trange
from matplotlib.ticker import PercentFormatter

# Enable x64
jax.config.update("jax_enable_x64", True)

# --- Local Imports ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_turb import DecentralizedTurbulenceNet 
from data_utils import get_batch_initial_conditions

# --- 1. Configuration ---
CONFIG = {
    'N_grid': 64,         
    'L_domain': 1.0,      
    'dt': 0.01,           
    'viscosity': 5e-4,    
    'substeps': 5,        
    'T_steps': 150,       
    'n_agents_train': 64, # 8x8 Grid
    'batch_size': 4,
    'epochs': 50,
    'pool_size': 500,      
    'ic_filename': 'turbulence_chaotic_ics_64_more.pkl',
    'save_dir': Path("./"),
    'model_name': 'turbulence_params.msgpack'
}

CONFIG['save_dir'].mkdir(parents=True, exist_ok=True)
MODEL_PATH = CONFIG['save_dir'] / CONFIG['model_name']
CSV_PATH = CONFIG['save_dir'] / "turb_zs_results.csv"
PLOT_PATH = CONFIG['save_dir'] / "turb_zs_relative_enstrophy.pdf"

# --- 2. Data Helper ---
def get_or_create_data(config):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / config['ic_filename']

    if file_path.exists():
        print(f"[Data] Found existing ICs at: {file_path}")
        with open(file_path, 'rb') as f:
            u_pool = pickle.load(f)
        if u_pool.shape[1] == config['N_grid']:
            return jnp.array(u_pool)
        print(f"[Data] Resolution mismatch. Regenerating...")

    print(f"[Data] Generating {config['pool_size']} chaotic states...")
    key = jax.random.PRNGKey(42)
    u_pool = get_batch_initial_conditions(
        key, config['pool_size'], config['N_grid'], config['L_domain']
    )
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)
    return u_pool

# --- 3. Grid Helper ---
def get_grid_xi(n_agents, L_domain):
    n_side = int(jnp.ceil(jnp.sqrt(n_agents)))
    x = jnp.linspace(0, L_domain, n_side, endpoint=False) + (L_domain / n_side) / 2.0
    y = jnp.linspace(0, L_domain, n_side, endpoint=False) + (L_domain / n_side) / 2.0
    xv, yv = jnp.meshgrid(x, y) 
    xi = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    return xi[:n_agents]

# --- 4. Loss Function ---
# Added actuator_grid_shape argument
def loss_fn(params, w_hat_init, xi_fixed, dynamics, T_steps, N_grid, L_domain, dt, substeps, viscosity, actuator_grid_shape):
    w_traj, u_ctrl_traj = dynamics.unroll_controlled(
        w_hat_init, 
        xi_fixed, 
        params, 
        t_steps=T_steps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        substeps=substeps,
        viscosity=viscosity,
        actuator_grid_shape=actuator_grid_shape # PASSED EXPLICITLY
    )
    
    l_enstrophy = jnp.mean(w_traj ** 2)
    l_effort = jnp.mean(u_ctrl_traj ** 2)
    total_loss = l_enstrophy + 1e-5 * l_effort
    
    return total_loss, (l_enstrophy, l_effort)

# --- Train Step ---
# Added actuator_grid_shape to static_argnames and args
@partial(jax.jit, static_argnames=('dynamics', 'optimizer', 'T_steps', 'N_grid', 'L_domain', 'dt', 'substeps', 'viscosity', 'actuator_grid_shape'))
def train_step(params, opt_state, w_init_batch, xi_fixed_batch, dynamics, optimizer, T_steps, N_grid, L_domain, dt, substeps, viscosity, actuator_grid_shape):
    
    # Pass arguments to batched function
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, None, None, None, None, None, None, None, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(
            p, w_init_batch, xi_fixed_batch, dynamics, 
            T_steps, N_grid, L_domain, dt, substeps, viscosity, actuator_grid_shape
        )
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 5. Main Execution ---
def main():
    print(f"--- 2D Turbulence Zero-Shot Scaling (Train N={CONFIG['n_agents_train']}) ---")
    key = jax.random.PRNGKey(42)
    
    # 1. Load Data
    w_init_pool = get_or_create_data(CONFIG)
    
    # 2. Setup Model
    model = DecentralizedTurbulenceNet(
        features=(32, 64), 
        patch_size=16,
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=40.0
    )
    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)
    
    # Init
    dummy_xi = get_grid_xi(CONFIG['n_agents_train'], CONFIG['L_domain'])
    dummy_obs = jnp.zeros((1, CONFIG['N_grid'], CONFIG['N_grid']))
    key, init_key = jax.random.split(key)
    init_params = model.init(init_key, dummy_xi, dummy_obs)

    # 3. Training
    if not MODEL_PATH.exists():
        print(f"Training Policy on {CONFIG['n_agents_train']} agents for {CONFIG['epochs']} epochs...")
        
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-4, peak_value=5e-4, warmup_steps=20, 
            decay_steps=CONFIG['epochs'], end_value=1e-6
        )
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
        opt_state = optimizer.init(init_params)
        params = init_params
        
        xi_train_batch = jnp.tile(dummy_xi, (CONFIG['batch_size'], 1, 1))
        
        # Calculate training grid shape (8, 8)
        n_side_train = int(np.sqrt(CONFIG['n_agents_train']))
        train_grid_shape = (n_side_train, n_side_train)

        pbar = trange(CONFIG['epochs'], desc="Training")
        for epoch in pbar:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (CONFIG['batch_size'],), 0, CONFIG['pool_size'])
            w_init_b = w_init_pool[idx]
            
            params, opt_state, loss, aux = train_step(
                params, 
                opt_state, 
                w_init_b, 
                xi_train_batch, 
                dynamics, 
                optimizer,
                CONFIG['T_steps'],
                CONFIG['N_grid'],
                CONFIG['L_domain'],
                CONFIG['dt'],
                CONFIG['substeps'],
                CONFIG['viscosity'],
                train_grid_shape # Passed explicitly (8,8)
            )
            l_enstr, l_eff = aux
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Enstr": f"{l_enstr:.4f}"})
            
        with open(MODEL_PATH, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
        print("Model saved.")
    else:
        print(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            params = flax.serialization.from_bytes(init_params, f.read())

    # --- 4. Zero-Shot Ablation ---
    n_eval_list = [36, 64, 81, 100, 121, 144, 196, 256, 324, 400] 
    n_test = 5 
    w_init_test = w_init_pool[-n_test:]
    
    print(f"\nStarting Zero-Shot Sweep on N={n_eval_list}...")
    results = []

    for n in n_eval_list:
        xi_eval = get_grid_xi(n, CONFIG['L_domain'])
        
        # Calculate shape for this evaluation (e.g., 4x4, 5x5)
        n_side = int(np.sqrt(n))
        eval_grid_shape = (n_side, n_side)
        
        enstrophy_list = []
        effort_list = []
        
        for i in range(n_test):
            w_traj, u_ctrl_traj = dynamics.unroll_controlled(
                w_init_test[i], 
                xi_eval, 
                params, 
                t_steps=CONFIG['T_steps'],
                N_grid=CONFIG['N_grid'],
                L=CONFIG['L_domain'],
                dt=CONFIG['dt'],
                substeps=CONFIG['substeps'],
                viscosity=CONFIG['viscosity'],
                actuator_grid_shape=eval_grid_shape # PASSED EXPLICITLY
            )
            
            final_enstrophy = jnp.mean(w_traj[int(0.8*CONFIG['T_steps']):] ** 2)
            enstrophy_list.append(float(final_enstrophy))
            effort_list.append(float(jnp.mean(u_ctrl_traj**2)))
            
        avg_enstr = np.mean(enstrophy_list)
        avg_effort = np.mean(effort_list)
        
        print(f"N={n:3d} | Enstrophy: {avg_enstr:.5f} | Effort: {avg_effort:.5f}")
        results.append({
            "n_agents": n, 
            "enstrophy": avg_enstr, 
            "effort": avg_effort
        })

    # --- 5. Process & Plot ---
    df = pd.DataFrame(results)
    baseline_enstr = df[df['n_agents'] == CONFIG['n_agents_train']]['enstrophy'].values[0]
    baseline_enstr = max(baseline_enstr, 1e-9)
    
    df['relative_mse'] = (df['enstrophy'] / baseline_enstr) * 100
    df.to_csv(CSV_PATH, index=False)

    plt.style.use('seaborn-v0_8-paper')
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    color = '#16a085'
    ax1.plot(df['n_agents'], df['relative_mse'], marker='o', linestyle='-', 
             color=color, linewidth=2, markersize=8, label='Relative Enstrophy')
    ax1.axvline(x=CONFIG['n_agents_train'], color='#d35400', linestyle='--', alpha=0.8, 
                label=f'Training Size (N={CONFIG["n_agents_train"]})')
    ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel("Number of Agents (N)", fontsize=10)
    ax1.set_ylabel("Relative Enstrophy (%)", color=color, fontsize=10)
    ax1.yaxis.set_major_formatter(PercentFormatter())
    
    ax1.set_title(f"Turbulence Zero-Shot Scalability\n(Train: 8x8 Grid)", 
                  fontsize=12, fontweight='bold')
    
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"\nAnalysis complete. Results saved to {CONFIG['save_dir']}")

if __name__ == "__main__":
    main()