"""
Decentralized Training Script for 2D Kuramoto-Sivashinsky (KS)
Trains a Neural Policy to stabilize chaotic 2D turbulence.
"""

import jax
import jax.numpy as jnp
import optax
import time
import pickle
import flax.serialization
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from tqdm import trange
import sys

jax.config.update("jax_enable_x64", True)

# --- Local Imports ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet 
from data_utils import get_batch_initial_conditions

from tesseracts.ks2d.solver import ks_spectral_step_etdrk4, precompute_etdrk4_coeffs

# --- 1. Configuration ---
CONFIG = {
    'N_grid': 64,         
    'L_domain': 32.0,      
    'dt': 0.005,
    
    # Action Repetition settings
    'substeps': 10,        # Physics steps per Control step
    'T_steps': 50,         # Number of Control steps
    # Total Physics Time = 50 * 10 * 0.005 = 2.5 seconds
    
    # Training
    'n_agents': 100,       
    'batch_size': 4,
    'epochs': 500,
    'pool_size': 500,       
    
    # Files
    'ic_filename': 'ks2d_chaotic_ics_64.pkl',
    'model_save_name': 'ks2d_centralized_params.msgpack'
}

# --- 2. Data Generation (Integrated) ---
def get_or_create_data(config):
    """Manages loading/generating chaotic Initial Conditions."""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
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
    
    import numpy as np
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)
        
    return u_pool

# --- 3. Loss Function (Updated with Substeps) ---
def loss_fn(params, u_init, xi_fixed, u_target, dynamics):
    """
    Computes trajectory loss for 2D KS.
    """
    # Unroll trajectory using the wrapper with SUBSTEPS
    # This keeps the physics stable (dt=0.005) but lets the controller
    # see further into the future (Horizon = 5.0s)
    u_traj, _, u_ctrl_traj, _ = dynamics.unroll_controlled(
        u_init, 
        xi_fixed, 
        u_target, 
        params, 
        t_steps=CONFIG['T_steps'],
        substeps=CONFIG['substeps'],   
        N_grid=CONFIG['N_grid'],
        L=CONFIG['L_domain'],
        dt=CONFIG['dt'],
        sigma=1.2 # !!!!
    )
    
    # 1. Tracking Loss (Stabilize to Target)
    traj_error = u_traj - u_target[None, :, :]
    l_track = jnp.mean(traj_error ** 2)
    
    # 2. Control Effort Loss
    l_effort = jnp.mean(u_ctrl_traj ** 2)
    
    # Total Loss
    total_loss = 100.0 * l_track + 1e-4 * l_effort
    
    return total_loss, (l_track, l_effort)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, u_init_batch, xi_fixed_batch, u_target_batch, dynamics):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, u_init_batch, xi_fixed_batch, u_target_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 4. Main Training Script ---
if __name__ == "__main__":
    print(f"--- 2D KS Control Training (Substeps={CONFIG['substeps']}, Horizon={CONFIG['T_steps']*CONFIG['substeps']*CONFIG['dt']:.2f}s) ---")
    key = jax.random.PRNGKey(42)
    
    # Load Data
    u_init_pool = get_or_create_data(CONFIG)
    u_target_pool = jnp.zeros_like(u_init_pool) 
    
    # Setup Actuators
    grid_dim = int(jnp.sqrt(CONFIG['n_agents']))
    x_lin = jnp.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = jnp.meshgrid(x_lin, x_lin)
    xi_fixed_single = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
    xi_fixed_batch = jnp.tile(xi_fixed_single, (CONFIG['batch_size'], 1, 1))

    # Initialize Model
    model = DecentralizedKS2DControlNet(
        features=(64, 128), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=5.0
    )
    
    key, init_key = jax.random.split(key)
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = jnp.zeros((CONFIG['n_agents'], 2))
    params = model.init(init_key, dummy_u, dummy_u, dummy_xi)
    
    # Optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=5e-4, peak_value=1e-3, warmup_steps=50,
        decay_steps=CONFIG['epochs'], end_value=1e-5
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    opt_state = optimizer.init(params)
    
    # Dynamics Wrapper
    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)

    # Training Loop
    metrics = []
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in trange(CONFIG['epochs']):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (CONFIG['batch_size'],), 0, CONFIG['pool_size'])
        u_init_b = u_init_pool[idx]
        u_target_b = u_target_pool[idx]
        
        params, opt_state, loss, aux = train_step(
            params, opt_state, u_init_b, xi_fixed_batch, u_target_b, dynamics
        )
        
        track_loss, effort_loss = aux
        metrics.append([loss, track_loss, effort_loss])
        
        if epoch % 10 == 0:
            print(f"Ep {epoch} | Loss: {loss:.4f} | Track: {track_loss:.4f} | Effort: {effort_loss:.4f}")

    print(f"Training Complete. Time: {time.time()-start_time:.1f}s")
    
    # Save & Plot
    metrics = jnp.array(metrics)
    with open(CONFIG['model_save_name'], 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics[:, 1], label='Tracking MSE', color='blue')
    plt.plot(metrics[:, 2], label='Control Effort', color='orange', alpha=0.7)
    plt.yscale('log')
    plt.title(f'2D KS Control (L={CONFIG["L_domain"]}, Horizon={CONFIG["T_steps"]*CONFIG["substeps"]*CONFIG["dt"]:.1f}s)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.savefig('ks2d_training_metrics.png')
    # print("Metrics plotted.")