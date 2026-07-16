"""
Decentralized Training Script for 2D Decaying Turbulence.
Trains a Neural Policy to stabilize (kill) chaotic turbulence.
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

# Enable x64 for Spectral Stability
jax.config.update("jax_enable_x64", True)

# --- Local Imports ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
# Swap KS model for Turbulence model
from models.policy_turb import DecentralizedTurbulenceNet 
from data_utils import get_batch_initial_conditions

# --- 1. Configuration ---
CONFIG = {
    'N_grid': 64,         
    'L_domain': 1.0,      # Domain size L=1.0 for Turbulence
    'dt': 0.01,           # Physics timestep
    'viscosity': 5e-4,    # Fluid viscosity
    
    # Action Repetition settings
    'substeps': 5,       # 16 Physics steps per Control step (matches solver stability)
    'T_steps': 150,        # 50 Control steps
    
    # Training
    'n_agents': 64,       # 64 Agents (8x8 Grid)
    'batch_size': 4,
    'epochs': 500,
    'pool_size': 500,      
    
    # Files
    'ic_filename': 'turbulence_chaotic_ics_64_more.pkl',
    'model_save_name': 'turbulence_params___.msgpack'
}

# --- 2. Data Generation ---
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
    # Using Turbulence Data Gen (returns spectral w_hat)
    u_pool = get_batch_initial_conditions(
        key, config['pool_size'], config['N_grid'], config['L_domain']
    )
    
    import numpy as np
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)
        
    return u_pool

# --- 3. Loss Function (Enstrophy) ---
def loss_fn(params, w_hat_init, xi_fixed, dynamics):
    """
    Computes Enstrophy Loss (Vorticity Magnitude).
    Goal: Drive vorticity to zero.
    """
    # Unroll trajectory
    # w_traj is Physical Vorticity
    w_traj, u_ctrl_traj = dynamics.unroll_controlled(
        w_hat_init, 
        xi_fixed, 
        params, 
        t_steps=CONFIG['T_steps'],
        N_grid=CONFIG['N_grid'],
        L=CONFIG['L_domain'],
        dt=CONFIG['dt'],
        substeps=CONFIG['substeps'],
        viscosity=CONFIG['viscosity'],
        actuator_grid_shape=(8, 8) # Sqrt(64)
    )
    
    # 1. Stabilization Loss (Minimize Enstrophy)
    l_enstrophy = jnp.mean(w_traj ** 2)
    
    # 2. Control Effort Loss
    l_effort = jnp.mean(u_ctrl_traj ** 2)
    
    # Total Loss (Enstrophy is usually small, so regularize effort lightly)
    total_loss = l_enstrophy + 1e-5 * l_effort
    
    return total_loss, (l_enstrophy, l_effort)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, w_init_batch, xi_fixed_batch, dynamics):
    # Vectorize loss over batch (axis 0)
    # Note: loss_fn signature is (params, w_hat, xi_fixed, dynamics)
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, w_init_batch, xi_fixed_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 4. Main Training Script ---
if __name__ == "__main__":
    print(f"--- 2D Turbulence Training (Substeps={CONFIG['substeps']}, Horizon={CONFIG['T_steps']*CONFIG['substeps']*CONFIG['dt']:.2f}s) ---")
    key = jax.random.PRNGKey(42)
    
    # Load Data
    w_init_pool = get_or_create_data(CONFIG)
    
    # Setup Actuators (Fixed Grid)
    grid_dim = int(jnp.sqrt(CONFIG['n_agents']))
    # Note: Turbulence solver uses [0, L]
    x_lin = jnp.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = jnp.meshgrid(x_lin, x_lin) # Default 'xy' indexing for meshgrid matches JAX 'xy' logic usually
    
    # Flatten: (n_agents, 2)
    xi_fixed_single = jnp.stack([xv.flatten(), yv.flatten()], axis=-1)
    # Batch: (Batch, n_agents, 2)
    xi_fixed_batch = jnp.tile(xi_fixed_single, (CONFIG['batch_size'], 1, 1))

    # Initialize Model
    model = DecentralizedTurbulenceNet(
        features=(32, 64), 
        patch_size=16,
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=40.0
    )
    
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, CONFIG['N_grid'], CONFIG['N_grid']))
    
    # --- INIT UPDATE ---
    # TurbulenceNet.__call__ expects (xi_fixed, obs)
    params = model.init(init_key, xi_fixed_single, dummy_obs)
    
    # Optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=5e-4, warmup_steps=20,
        decay_steps=CONFIG['epochs'], end_value=1e-6
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
        w_init_b = w_init_pool[idx]
        
        # Train Step
        # Note: We don't pass u_target_batch anymore (Turbulence minimizes w -> 0)
        params, opt_state, loss, aux = train_step(
            params, opt_state, w_init_b, xi_fixed_batch, dynamics
        )
        
        l_enstr, l_effort = aux
        metrics.append([loss, l_enstr, l_effort])
        
        if epoch % 5 == 0:
            print(f"Ep {epoch} | Loss: {loss:.4f} | Enstrophy: {l_enstr:.4f} | Effort: {l_effort:.4f}")

    print(f"Training Complete. Time: {time.time()-start_time:.1f}s")
    
    # Save & Plot
    metrics = jnp.array(metrics)
    with open(CONFIG['model_save_name'], 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics[:, 1], label='Enstrophy (Turbulence)', color='blue')
    plt.plot(metrics[:, 2], label='Control Effort', color='orange', alpha=0.7)
    plt.yscale('log')
    plt.title(f'2D Turbulence Control (L={CONFIG["L_domain"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('turbulence_training_metrics.png')
    print("Metrics plotted.")