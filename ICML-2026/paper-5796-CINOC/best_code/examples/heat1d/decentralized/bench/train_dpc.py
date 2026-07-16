"""
Centralized Deep Policy Control Training Script for 1D Heat Equation.
Trains a pure MLP policy using native JAX dynamics.
"""
import jax
import jax.numpy as jnp
import sys
import os
from pathlib import Path
import optax
import time
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange
import flax.serialization

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.heat1d.decentralized.dynamics_dual import PDEDynamics
from examples.heat1d.decentralized.data_utils import generate_grf
from models_dpc import CentralizedMLPControlNet

# --- 1. Initialization ---
n_pde, n_agents, batch_size = 100, 8, 32
T_steps = 300
R_safe = 0.05
epochs = 500

model = CentralizedMLPControlNet(hidden_dim=256, n_agents=n_agents)
key = jax.random.PRNGKey(42)

params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

# --- 2. Loss Function ---
def loss_fn(params, z_init, xi_init, z_target, dynamics):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    
    # 1. Tracking Loss
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    
    # 2. Effort Loss
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(v_traj ** 2)
    
    # 3. Boundary Penalty
    margin = 0.02
    l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + 
                       jnp.maximum(0, xi_traj - (1.0 - margin))**2)
    
    # 4. Collision Avoidance
    dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 1.0)) ** 2)
    
    # 5. Damping (Acceleration)
    l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)

    total_loss = 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel
    return total_loss, (l_track, l_effort, l_coll, l_bound)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, z_init_batch, xi_init_batch, z_target_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 3. Training Loop ---
dynamics = PDEDynamics(policy_apply_fn=model.apply)

print("Generating dataset...")
all_keys = jax.random.split(key, 5000)
_, z_init_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.2))(all_keys)
_, z_target_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.4))(all_keys)

xi_init_single = jnp.linspace(0.2, 0.8, n_agents)
xi_init_batch = jnp.tile(xi_init_single, (batch_size, 1))

metrics = []
start_time = time.time()

for epoch in trange(epochs):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (batch_size,), 0, 5000)
    z_init_b, z_target_b = z_init_all[idx], z_target_all[idx]

    params, opt_state, loss, aux = train_step(
        params, opt_state, z_init_b, xi_init_batch, z_target_b, dynamics
    )
    
    if epoch % 10 == 0:
        metrics.append((epoch, loss, *aux))
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Track: {aux[0]:.4f}")

print(f"Training finished in {time.time() - start_time:.2f}s.")

os.makedirs('bench/models', exist_ok=True)
with open('models/dpc_heat_params.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes({'params': params}))
print("Params saved to bench/models/dpc_heat_params.msgpack.")