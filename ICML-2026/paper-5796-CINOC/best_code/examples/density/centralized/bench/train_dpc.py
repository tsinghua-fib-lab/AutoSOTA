"""
Centralized Deep Policy Control (DPC) Training Script for NS2D Shape Formation Control
Train a global MLP policy to control movable smoke injectors to achieve target shapes.
"""

import sys
from pathlib import Path
import os

# Prevent JAX from preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Add project root
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

import jax
import jax.numpy as jnp
import optax
import time
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import flax.serialization
import numpy as np

from examples.density.centralized.dynamics import unroll_with_full_loss
from models_dpc import CentralizedMLPControlNetNS2D

# =============================================================================
# Hyperparameters
# =============================================================================

N_AGENTS = 9         
T_STEPS = 150         
BATCH_SIZE = 4        
EPOCHS = 1000         

BUOYANCY = 0.0        
SIGMA_PUSH = 0.2      
PUSH_MAX = 0.8        

W_TRACK = 10.0        
W_EFFORT = 0.001      
W_BOUND = 20.0        
W_COLL = 1000.0       
W_ACCEL = 0.05        
R_SAFE = 0.15         

# =============================================================================
# Training
# =============================================================================

def main():
    print("="*60)
    print("NS2D Shape Formation - Centralized DPC Training")
    print("="*60)
    
    # Load data
    data_dir = Path(__file__).parent.parent.parent / 'data'
    if not data_dir.exists():
        print(f"Data not found at {data_dir}")
        return
    
    config = np.load(data_dir / 'config.npz')
    Nx = int(config['Nx'])
    Ny = int(config['Ny'])
    dt = float(config['dt'])
    
    print(f"\nGrid: {Nx}x{Ny}, Agents: {N_AGENTS} (moving)")
    
    train_data = np.load(data_dir / 'train_data.npz')
    pool_size = len(train_data['rho_init'])
    print(f"Training samples: {pool_size}")
    
    # Initialize Centralized DPC Model
    model = CentralizedMLPControlNetNS2D(hidden_dim=256, n_agents=N_AGENTS)
    
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    dummy_smoke = jnp.zeros((Nx, Ny))
    dummy_xi = jnp.zeros((N_AGENTS, 2))
    params = model.init(init_key, dummy_smoke, dummy_smoke, dummy_xi)
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    lr_schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=2000,
        decay_rate=0.5
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule)
    )
    opt_state = optimizer.init(params)
    
    # Loss function
    def loss_fn(params, smoke_init, xi_init, rho_target):
        smoke_final, xi_final, l_track, l_effort, l_bound, l_coll, l_accel = unroll_with_full_loss(
            smoke_init, xi_init, rho_target, params, model.apply, T_STEPS,
            Nx=Nx, Ny=Ny, n_agents=N_AGENTS, dt=dt, buoyancy=BUOYANCY,
            sigma_push=SIGMA_PUSH, push_max=PUSH_MAX, R_safe=R_SAFE
        )
        
        total_loss = W_TRACK * l_track + W_EFFORT * l_effort + \
                     W_BOUND * l_bound + W_COLL * l_coll + W_ACCEL * l_accel
        
        return total_loss, (l_track, l_effort, l_coll)
    
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))
    
    @jax.jit
    def train_step(params, opt_state, smoke_init, xi_init, rho_target):
        def mean_loss(p):
            losses, aux = batched_loss_fn(p, smoke_init, xi_init, rho_target)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, aux)
        
        (loss, aux), grads = jax.value_and_grad(mean_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux
    
    # Initial agent positions (3x3 grid)
    n_side = int(np.sqrt(N_AGENTS))
    xi_template = jnp.stack(jnp.meshgrid(
        jnp.linspace(0.15, 0.85, n_side),
        jnp.linspace(0.15, 1.0, n_side)
    ), axis=-1).reshape(-1, 2)
    
    # Training loop
    metrics = []
    start_time = time.time()
    
    print(f"\nTraining: T={T_STEPS}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
    
    for epoch in trange(EPOCHS):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (BATCH_SIZE,), 0, pool_size)
        
        smoke_batch = jnp.array(train_data['rho_init'][idx])
        target_batch = jnp.array(train_data['rho_target'][idx])
        xi_batch = jnp.tile(xi_template[None], (BATCH_SIZE, 1, 1))
        
        params, opt_state, loss, aux = train_step(
            params, opt_state, smoke_batch, xi_batch, target_batch
        )
        
        if epoch % 1 == 0:
            l_track, l_effort, l_coll = aux
            metrics.append((epoch, float(loss), float(l_track), float(l_effort), float(l_coll)))
            
            if epoch % 1 == 0:
                tqdm.write(f"Ep {epoch} | Loss: {loss:.4f} | Track: {l_track:.4f} | " +
                          f"Effort: {l_effort:.4f} | Coll: {l_coll:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    # Save parameters for benchmarking
    os.makedirs('bench/models', exist_ok=True)
    save_path = Path('models/dpc_ns2d_params.msgpack')
    with open(save_path, 'wb') as f:
        f.write(flax.serialization.to_bytes({'params': params}))
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()