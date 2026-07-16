"""
Centralized Deep Policy Control Training Script for 2D Heat Equation.
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
import argparse

script_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.heat2D.decentralized.dynamics_dual import PDEDynamics
from examples.heat2D.decentralized.data_utils import get_training_data
from models_dpc import CentralizedMLPControlNet2D

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Train 2D Heat Centralized Controller (DPC)')
parser.add_argument('--test', action='store_true', help='Quick test mode (1 sample, 10 epochs)')
args = parser.parse_args()

# --- Configuration ---
n_grid = 32  
n_agents = 16  
R_safe = 0.08  

if args.test:
    print("=" * 50)
    print("RUNNING IN TEST MODE (CENTRALIZED DPC)")
    print("=" * 50)
    n_samples = 1
    batch_size = 1
    T_steps = 100 
    epochs = 10
else:
    n_samples = 5000
    batch_size = 16
    T_steps = 300
    epochs = 500
    
print(f"Config: {n_samples} samples, {epochs} epochs, T={T_steps} steps")

model = CentralizedMLPControlNet2D(hidden_dim=256, n_agents=n_agents)
key = jax.random.PRNGKey(42)

dummy_z = jnp.zeros((n_grid, n_grid))
dummy_xi = jnp.zeros((n_agents, 2))
params = model.init(key, dummy_z, dummy_z, dummy_xi)

lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

# --- Loss Function ---
def loss_fn(params, z_init, xi_init, z_target, dynamics):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )

    # 1. Tracking loss
    l_track = jnp.mean((z_traj - z_target[None, :, :]) ** 2)

    # 2. Effort loss
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(jnp.sum(v_traj ** 2, axis=-1))

    # 3. Boundary penalty (2D)
    margin = 0.02
    x_penalty = jnp.maximum(0, margin - xi_traj[:, :, 0])**2 + \
                jnp.maximum(0, xi_traj[:, :, 0] - (1.0 - margin))**2
    y_penalty = jnp.maximum(0, margin - xi_traj[:, :, 1])**2 + \
                jnp.maximum(0, xi_traj[:, :, 1] - (1.0 - margin))**2
    l_bound = jnp.mean(x_penalty + y_penalty)

    # 4. Collision avoidance (2D Euclidean distance)
    diff = xi_traj[:, :, None, :] - xi_traj[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 10.0)) ** 2)

    # 5. Acceleration penalty
    l_accel = jnp.mean(jnp.sum(jnp.diff(v_traj, axis=0)**2, axis=-1))

    total_loss = 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + \
                 20.0 * l_coll + 0.1 * l_accel

    return total_loss, (l_track, l_effort, l_coll, l_bound)

@partial(jax.jit, static_argnames='dynamics')
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))

    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, z_init_batch, xi_init_batch,
                                        z_target_batch, dynamics)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- Training Loop ---
dynamics = PDEDynamics(policy_apply_fn=model.apply)

print("Loading/Generating 2D dataset...")
z_init_all, z_target_all, n_grid_actual = get_training_data(
    n_samples=n_samples,
    n_grid=n_grid,
    dataset_dir='../../data'
)

if n_grid_actual != n_grid:
    n_grid = n_grid_actual
    print(f"Using n_grid={n_grid} from loaded dataset")
    dummy_z = jnp.zeros((n_grid, n_grid))
    params = model.init(key, dummy_z, dummy_z, dummy_xi)
    opt_state = optimizer.init(params)

print(f"Dataset ready: {z_init_all.shape}")

# Initialize agents in a 4x4 grid pattern
n_side = int(jnp.sqrt(n_agents))
positions_1d = jnp.linspace(0.2, 0.8, n_side)
X, Y = jnp.meshgrid(positions_1d, positions_1d)
xi_init_single = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(jnp.float32)
xi_init_batch = jnp.tile(xi_init_single, (batch_size, 1, 1))

metrics = []
start_time = time.time()

print("\nStarting training...")
for epoch in trange(epochs):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (batch_size,), 0, n_samples)
    z_init_b, z_target_b = z_init_all[idx], z_target_all[idx]

    params, opt_state, loss, aux = train_step(
        params, opt_state, z_init_b, xi_init_batch, z_target_b, dynamics
    )

    if epoch % 10 == 0:
        metrics.append((epoch, loss, *aux))
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Track: {aux[0]:.4f} | " +
                f"Effort: {aux[1]:.4f} | Coll: {aux[2]:.4f} | Bound: {aux[3]:.4f}")

print(f"\nTraining finished in {time.time() - start_time:.2f}s.")

os.makedirs('models', exist_ok=True)
save_path = 'models/dpc_heat2d_params.msgpack'
with open(save_path, 'wb') as f:
    f.write(flax.serialization.to_bytes({'params': params}))
print(f"Parameters saved to {save_path}")