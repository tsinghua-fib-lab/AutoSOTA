"""
Centralized Deep Policy Control Training Script for 1D Kuramoto-Sivashinsky (KS):
Trains a Centralized MLP policy to stabilize chaotic wave dynamics.
"""
import jax
import jax.numpy as jnp
import optax
import time
import os
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange
import flax.serialization
import sys
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks1d.decentralized.bench.models_dpc import CentralizedMLPControlNet1D_KS

# --- 1. Configuration & Initialization ---
N_grid = 128
L_domain = 22.0
n_agents = 8
T_steps = 200   # Horizon for stabilization
batch_size = 32
epochs = 500

# Initialize Model
model = CentralizedMLPControlNet1D_KS(hidden_dim=256, n_agents=n_agents, u_max=1.0)
key = jax.random.PRNGKey(42)

key, init_key = jax.random.split(key)
dummy_u = jnp.zeros((N_grid,))
dummy_xi = jnp.linspace(0, L_domain, n_agents)
params = model.init(init_key, dummy_u, dummy_u, dummy_xi)

# Optimizer
lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

# --- 2. Loss Function ---
def loss_fn(params, u_init, xi_fixed, u_target, dynamics):
    # Unroll the dynamics
    u_traj, xi_traj, u_ctrl_traj, v_traj = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, T_steps, N_grid, L_domain, dt=0.05, sigma=1.0
    )
    
    # 1. Tracking Loss (Stabilize to 0)
    l_track = jnp.mean((u_traj - u_target[None, :]) ** 2)
    
    # 2. Effort Loss (Control Regularization)
    l_effort = jnp.mean(u_ctrl_traj ** 2) 
    
    # Constraints (Velocity is zero, fixed actuators)
    l_bound = 0.0 
    l_coll = 0.0
    
    # Weighted Sum
    total_loss = 10.0 * l_track + 0.001 * l_effort 
    
    return total_loss, (l_track, l_effort, l_coll, l_bound)

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

# --- 3. Training Loop ---
dynamics = PDEDynamics(policy_apply_fn=model.apply)

print("Generating Chaotic Initial Conditions (Spin-up)...")
pool_size = 1024
key, subkey = jax.random.split(key)
u_init_pool = get_batch_initial_conditions(subkey, pool_size, N_grid, L_domain)

u_target_pool = jnp.zeros_like(u_init_pool)

# Fixed Actuator Positions
xi_fixed_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
xi_fixed_batch = jnp.tile(xi_fixed_single, (batch_size, 1))

metrics = []
start_time = time.time()

print(f"Starting Training on KS-1D (L={L_domain}, Agents={n_agents})...")

for epoch in trange(epochs):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (batch_size,), 0, pool_size)
    u_init_b = u_init_pool[idx]
    u_target_b = u_target_pool[idx]

    params, opt_state, loss, aux = train_step(
        params, opt_state, u_init_b, xi_fixed_batch, u_target_b, dynamics
    )
    
    if epoch % 10 == 0:
        metrics.append((epoch, loss, *aux))
        tqdm_desc = f"Loss: {loss:.4f} | Track: {aux[0]:.4f} | Effort: {aux[1]:.4f}"
        print(f"Epoch {epoch:03d} | {tqdm_desc}")

print(f"Training finished in {time.time() - start_time:.2f}s.")

# --- 4. Plotting and Saving ---
metrics = jnp.array(metrics)
epochs_recorded = metrics[:, 0]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_recorded, metrics[:, 1], color='black', label='Total Loss')
plt.yscale('log')
plt.title('Total Loss (DPC)')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_recorded, metrics[:, 2], label='Tracking Error')
plt.plot(epochs_recorded, metrics[:, 3], label='Control Effort', alpha=0.7)
plt.yscale('log')
plt.title('Stability vs Effort')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('ks_centralized_dpc_training.png')
print("Training metrics plotted and saved.")

os.makedirs('models', exist_ok=True)
save_path = 'models/dpc_ks1d_params.msgpack'
with open(save_path, 'wb') as f:
    f.write(flax.serialization.to_bytes({'params': params}))
print(f"Params saved to {save_path}")