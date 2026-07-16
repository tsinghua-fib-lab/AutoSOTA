"""
Turbulence Zero-Shot Scaling Ablation Script (Ours + DeepONet MAPPO + DeepONet MATD3).
"""

import jax
import jax.numpy as jnp
import optax
import time
import pickle
import flax
import flax.serialization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from functools import partial
from pathlib import Path
from tqdm import trange
from matplotlib.ticker import PercentFormatter

# Enable x64 for spectral stability
jax.config.update("jax_enable_x64", True)

# --- Local Imports ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_turb import DecentralizedTurbulenceNet 
from data_utils import get_batch_initial_conditions

# DeepONet and Solver Imports
import tesseracts.turbulence2d.solver as solver
from examples.turbulence2d.decentralized.bench.models_deeponet_mappo import DeepONetMAPPOActor
from examples.turbulence2d.decentralized.bench.models_deeponet_matd3 import DeepONetActor as DeepONetMATD3Actor

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
    'save_dir': Path("figures/turb_scaling"),
    'model_name_base': 'turbulence_params.msgpack',
    
    # DeepONet Specific Config
    'patch_size': 20,
    'u_max': 75.0,
    'sigma': 0.05
}

CONFIG['save_dir'].mkdir(parents=True, exist_ok=True)
MODEL_PATH_BASE = CONFIG['model_name_base']
MODEL_PATH_MAPPO = Path('bench/models/deeponet_mappo_turb_params.msgpack')
MODEL_PATH_MATD3 = Path('bench/models/deeponet_matd3_params.msgpack')

CSV_PATH = CONFIG['save_dir'] / "turb_zs_results_all.csv"
PLOT_PATH = CONFIG['save_dir'] / "turb_zs_scalability.pdf"

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
        key, config['pool_size'], config['N_grid'], config['L_domain'], viscosity=config['viscosity']
    )
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)
    return u_pool

# --- 3. Grid & Patch Helpers ---
def get_grid_xi(n_agents, L_domain):
    n_side = int(jnp.ceil(jnp.sqrt(n_agents)))
    x = jnp.linspace(0, L_domain, n_side, endpoint=False) + (L_domain / n_side) / 2.0
    y = jnp.linspace(0, L_domain, n_side, endpoint=False) + (L_domain / n_side) / 2.0
    xv, yv = jnp.meshgrid(x, y) 
    xi = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    return xi[:n_agents]

@partial(jax.jit, static_argnames=['N_grid', 'patch_size'])
def get_patches(w_curr, xi_norm_eval, N_grid, patch_size):
    grads = jnp.gradient(w_curr)
    grad_y, grad_x = grads[0], grads[1]
    half_patch = patch_size // 2
    
    w_pad = jnp.pad(w_curr, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')
    gx_pad = jnp.pad(grad_x, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')
    gy_pad = jnp.pad(grad_y, ((half_patch, half_patch), (half_patch, half_patch)), mode='wrap')

    def get_local_obs(xi_single):
        i = (xi_single[1] * N_grid).astype(jnp.int32)
        j = (xi_single[0] * N_grid).astype(jnp.int32)
        p_w = jax.lax.dynamic_slice(w_pad, (i, j), (patch_size, patch_size))
        p_gx = jax.lax.dynamic_slice(gx_pad, (i, j), (patch_size, patch_size))
        p_gy = jax.lax.dynamic_slice(gy_pad, (i, j), (patch_size, patch_size))
        return jnp.stack([p_w, p_gx, p_gy], axis=-1)

    local_patches = jax.vmap(get_local_obs)(xi_norm_eval)
    return (local_patches / 50.0).astype(jnp.float32)

# --- 4. DeepONet Unified Evaluation Step ---
@partial(jax.jit, static_argnames=['max_steps', 'N_grid', 'patch_size', 'substeps', 'model_type', 'mappo_actor', 'matd3_actor'])
def eval_deeponet_step(params, init_state, xi_norm, forcing_hat, kx, ky, k_sq, k_inv,
                       max_steps, N_grid, patch_size, substeps, dt_phys, viscosity, u_max, model_type, mappo_actor, matd3_actor):
    def step_fn(state, _):
        # 1. Extract Patches dynamically based on N agents
        obs_patches = get_patches(state, xi_norm, N_grid, patch_size)[None, ...] # Shape: [1, N, P, P, 3]
        xi_norm_b = xi_norm[None, ...] # Shape: [1, N, 2]

        # 2. Get Actions
        if model_type == 'mappo':
            mean_raw, _ = mappo_actor.apply(params, obs_patches, xi_norm_b)
            env_action = jnp.tanh(mean_raw) * u_max
        elif model_type == 'matd3':
            env_action = matd3_actor.apply(params, obs_patches, xi_norm_b)
        else:
            env_action = jnp.zeros_like(xi_norm_b[..., 0:1])

        act_flat = env_action[0, :, 0].astype(jnp.float64)

        # 3. Physics Step
        w_hat = jnp.fft.fft2(state)
        def rk4_loop(i, w):
            return solver.rk4_step(w, dt_phys, kx, ky, k_sq, k_inv, viscosity, forcing_hat, act_flat)
        w_hat_next = jax.lax.fori_loop(0, substeps, rk4_loop, w_hat)
        next_state = jnp.fft.ifft2(w_hat_next).real

        enstrophy = jnp.mean(next_state**2)
        effort = jnp.mean(act_flat**2)
        return next_state, (enstrophy, effort)

    _, (enstrophies, efforts) = jax.lax.scan(step_fn, init_state, None, length=max_steps)
    return enstrophies, efforts

# --- 5. Main Execution ---
def main():
    print(f"--- 2D Turbulence Zero-Shot Scaling Ablation (Train N={CONFIG['n_agents_train']}) ---")
    key = jax.random.PRNGKey(42)
    
    # 1. Load Data
    w_init_pool = get_or_create_data(CONFIG)
    # Ensure physical domain (real numbers) for DeepONet
    if jnp.iscomplexobj(w_init_pool):
        w_init_pool = jnp.fft.ifft2(w_init_pool).real.astype(jnp.float64)

    # 2. Setup Ours Model
    Ours_model = DecentralizedTurbulenceNet(
        features=(32, 64), patch_size=16,
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']), u_max=40.0
    )
    dynamics = PDEDynamics2D(policy_apply_fn=Ours_model.apply)
    
    dummy_xi = get_grid_xi(CONFIG['n_agents_train'], CONFIG['L_domain'])
    dummy_obs = jnp.zeros((1, CONFIG['N_grid'], CONFIG['N_grid']))
    key, init_key = jax.random.split(key)
    init_params_base = Ours_model.init(init_key, dummy_xi, dummy_obs)

    # Setup DeepONet Models
    dummy_patches = jnp.zeros((1, CONFIG['n_agents_train'], CONFIG['patch_size'], CONFIG['patch_size'], 3), dtype=jnp.float32)
    dummy_xi_b = jnp.zeros((1, CONFIG['n_agents_train'], 2), dtype=jnp.float32)
    
    mappo_actor = DeepONetMAPPOActor(n_agents=CONFIG['n_agents_train'], u_max=CONFIG['u_max'])
    matd3_actor = DeepONetMATD3Actor(u_max=CONFIG['u_max'])
    
    init_params_mappo = mappo_actor.init(init_key, dummy_patches, dummy_xi_b)
    init_params_matd3 = matd3_actor.init(init_key, dummy_patches, dummy_xi_b)

    # 3. Load Parameters
    print("\nLoading Model Weights...")
    try:
        with open(MODEL_PATH_BASE, 'rb') as f:
            params_base = flax.serialization.from_bytes(init_params_base, f.read())
        print(f"[OK] Ours loaded from {MODEL_PATH_BASE}")
    except:
        print(f"[Warning] Ours not found. Using random init.")
        params_base = init_params_base

    try:
        with open(MODEL_PATH_MAPPO, 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
            params_mappo = flax.core.freeze(state_dict['actor'])
        print(f"[OK] MAPPO loaded from {MODEL_PATH_MAPPO}")
    except:
        print(f"[Warning] MAPPO weights not found. Using random init.")
        params_mappo = init_params_mappo

    try:
        with open(MODEL_PATH_MATD3, 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
            params_matd3 = flax.core.freeze(state_dict['actor'])
        print(f"[OK] MATD3 loaded from {MODEL_PATH_MATD3}")
    except:
        print(f"[Warning] MATD3 weights not found. Using random init.")
        params_matd3 = init_params_matd3

    # --- 4. Zero-Shot Sweep ---
    n_eval_list = [36, 64, 81, 100, 121, 144, 196, 256, 324, 400] 
    n_test = 5 
    w_init_test = w_init_pool[-n_test:]
    
    print(f"\nStarting Zero-Shot Sweep on N={n_eval_list}...")
    results = []

    # Pre-compute spectral grid
    kx, ky, k_sq, k_inv = solver.get_spectral_grid(CONFIG['N_grid'], CONFIG['L_domain'])
    dt_phys = CONFIG['dt'] / CONFIG['substeps']

    for n in n_eval_list:
        xi_eval = get_grid_xi(n, CONFIG['L_domain'])
        xi_norm_eval = xi_eval / CONFIG['L_domain']
        
        n_side = int(np.sqrt(n))
        eval_grid_shape = (n_side, n_side)
        
        # Compute dynamic forcing hat for the current grid resolution
        x_c = jnp.linspace(0, CONFIG['L_domain'], n_side, endpoint=False) + CONFIG['L_domain']/(2*n_side)
        y_c = jnp.linspace(0, CONFIG['L_domain'], n_side, endpoint=False) + CONFIG['L_domain']/(2*n_side)
        xv, yv = jnp.meshgrid(x_c, y_c)
        centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)
        forcing_hat = solver.compute_forcing_profile(
            centers_flat[:, 0], centers_flat[:, 1], CONFIG['N_grid'], CONFIG['L_domain'], CONFIG['sigma']
        )
        
        enstr_base_list, enstr_mappo_list, enstr_matd3_list = [], [], []
        eff_base_list, eff_mappo_list, eff_matd3_list = [], [], []
        
        for i in range(n_test):
            w_init = w_init_test[i]
            
            # ---> FIX: Cast to spectral domain for the Ours! <---
            w_hat_init = jnp.fft.fft2(w_init)
            
            # --- 4A. Ours Eval ---
            w_traj_b, u_ctrl_traj_b = dynamics.unroll_controlled(
                w_hat_init, xi_eval, params_base, t_steps=CONFIG['T_steps'], N_grid=CONFIG['N_grid'],
                L=CONFIG['L_domain'], dt=CONFIG['dt'], substeps=CONFIG['substeps'], 
                viscosity=CONFIG['viscosity'], actuator_grid_shape=eval_grid_shape
            )
            enstr_base_list.append(float(jnp.mean(w_traj_b[int(0.8*CONFIG['T_steps']):] ** 2)))
            eff_base_list.append(float(jnp.mean(u_ctrl_traj_b**2)))
            
            # --- 4B. DeepONet MAPPO Eval ---
            enstr_map, eff_map = eval_deeponet_step(
                params_mappo, w_init, xi_norm_eval, forcing_hat, kx, ky, k_sq, k_inv,
                CONFIG['T_steps'], CONFIG['N_grid'], CONFIG['patch_size'], CONFIG['substeps'], 
                dt_phys, CONFIG['viscosity'], CONFIG['u_max'], 'mappo', mappo_actor, matd3_actor
            )
            enstr_mappo_list.append(float(jnp.mean(enstr_map[int(0.8*CONFIG['T_steps']):])))
            eff_mappo_list.append(float(jnp.mean(eff_map)))

            # --- 4C. DeepONet MATD3 Eval ---
            enstr_mat, eff_mat = eval_deeponet_step(
                params_matd3, w_init, xi_norm_eval, forcing_hat, kx, ky, k_sq, k_inv,
                CONFIG['T_steps'], CONFIG['N_grid'], CONFIG['patch_size'], CONFIG['substeps'], 
                dt_phys, CONFIG['viscosity'], CONFIG['u_max'], 'matd3', mappo_actor, matd3_actor
            )
            enstr_matd3_list.append(float(jnp.mean(enstr_mat[int(0.8*CONFIG['T_steps']):])))
            eff_matd3_list.append(float(jnp.mean(eff_mat)))

        avg_base = np.mean(enstr_base_list)
        avg_map = np.mean(enstr_mappo_list)
        avg_mat = np.mean(enstr_matd3_list)
        
        print(f"N={n:3d} | Base: {avg_base:.5f} | MAPPO: {avg_map:.5f} | MATD3: {avg_mat:.5f}")
        
        results.append({"n_agents": n, "model": "Ours", "enstrophy": avg_base, "effort": np.mean(eff_base_list)})
        results.append({"n_agents": n, "model": "MAPPO", "enstrophy": avg_map, "effort": np.mean(eff_mappo_list)})
        results.append({"n_agents": n, "model": "MATD3", "enstrophy": avg_mat, "effort": np.mean(eff_matd3_list)})

    # --- 5. Process & Plot ---
    df = pd.DataFrame(results)
    
    # Calculate Relative MSE for each model individually
    df['relative_error'] = 0.0
    for model in df['model'].unique():
        base_val = df[(df['model'] == model) & (df['n_agents'] == CONFIG['n_agents_train'])]['enstrophy'].values[0]
        base_val = max(base_val, 1e-9)
        mask = df['model'] == model
        df.loc[mask, 'relative_error'] = (df.loc[mask, 'enstrophy'] / base_val) * 100

    df.to_csv(CSV_PATH, index=False)

    plt.style.use('seaborn-v0_8-paper')
    # CHANGED: Use a single subplot instead of 1x2 
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    
    colors = {"Ours": "#e74c3c", "MAPPO": "#3498db", "MATD3": "#2ecc71"}
    markers = {"Ours": "o", "MAPPO": "s", "MATD3": "^"}
    
    for model in df['model'].unique():
        sub_df = df[df['model'] == model]
        # Absolute Plot
        ax1.plot(sub_df['n_agents'], sub_df['enstrophy'], marker=markers[model], 
                 linestyle='-', color=colors[model], linewidth=2, markersize=8, label=model)
        # Relative Plot (Commented Out)
        # ax2.plot(sub_df['n_agents'], sub_df['relative_error'], marker=markers[model], 
        #          linestyle='-', color=colors[model], linewidth=2, markersize=8, label=model)

    # Decorate ax1 (Absolute)
    ax1.axvline(x=CONFIG['n_agents_train'], color='gray', linestyle='--', alpha=0.8, label=f'Train Size (N={CONFIG["n_agents_train"]})')
    ax1.set_xlabel("Number of Agents (M)", fontsize=10)
    ax1.set_ylabel("Absolute Enstrophy", fontsize=10)
    ax1.set_title("Absolute Scalability (Zero-Shot)", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend()

    # Decorate ax2 (Relative) - Commented Out
    # ax2.axvline(x=CONFIG['n_agents_train'], color='gray', linestyle='--', alpha=0.8, label=f'Train Size (N={CONFIG["n_agents_train"]})')
    # ax2.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
    # ax2.set_xlabel("Number of Agents (N)", fontsize=10)
    # ax2.set_ylabel("Relative Enstrophy (%)", fontsize=10)
    # ax2.yaxis.set_major_formatter(PercentFormatter())
    # ax2.set_title("Relative Scalability (Zero-Shot)", fontsize=12, fontweight='bold')
    # ax2.grid(True, which="both", ls="--", alpha=0.3)
    # ax2.legend()
    
    plt.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"\nAnalysis complete. Results saved to {CONFIG['save_dir']}")

if __name__ == "__main__":
    main()