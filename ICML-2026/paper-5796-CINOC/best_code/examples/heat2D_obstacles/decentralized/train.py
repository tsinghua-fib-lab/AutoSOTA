"""Training script for Decentralized 2D Heat Equation Control with Obstacles using JAX and DecentralizedControlNet policy."""
import jax
import jax.numpy as jnp
import sys
from pathlib import Path
import optax
import time
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange
import flax.serialization
import argparse

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
from data_utils import get_training_data

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Train 2D Heat Decentralized Controller with Obstacles')
parser.add_argument('--test', action='store_true', help='Quick test mode (1 sample, 10 epochs)')
args = parser.parse_args()

# --- Configuration ---
n_grid = 32  # Default 32×32 grid (faster training)
n_agents = 16  # 4×4 grid of agents

# Test mode: quick verification
if args.test:
    print("=" * 50)
    print("RUNNING IN TEST MODE (DECENTRALIZED, WITH OBSTACLES)")
    print("=" * 50)
    n_samples = 1
    batch_size = 1
    T_steps = 100  # Shorter horizon
    epochs = 10
    print(f"Config: {n_samples} sample, {epochs} epochs, T={T_steps} steps")
else:
    n_samples = 5000
    batch_size = 16
    T_steps = 300
    epochs = 500
    print(f"Config: {n_samples} samples, {epochs} epochs, T={T_steps} steps")

R_safe = 0.08  # Collision radius for agent-agent in 2D
R_safe_obstacle = 0.04  # Safety distance for obstacles

# Obstacle configuration: [x_center, y_center, radius]
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   # Diagonal line obstacle 1
    [0.50, 0.50, 0.06],   # Diagonal line obstacle 2 (center)
    [0.70, 0.70, 0.06],   # Diagonal line obstacle 3
])

model = DecentralizedHeat2DControlNet(features=(16, 32))
key = jax.random.PRNGKey(0)

# Initialize with dummy 2D inputs
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

    # 4a. Agent-agent collision avoidance (2D Euclidean distance)
    diff = xi_traj[:, :, None, :] - xi_traj[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll_agents = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 10.0)) ** 2)

    # 4b. Agent-obstacle collision avoidance
    # Compute distances from agents to obstacles: (T, M, K, 2)
    # xi_traj: (T, M, 2), OBSTACLES: (K, 3) where [:, :2] are centers
    obstacle_centers = OBSTACLES[:, :2]  # (K, 2)
    obstacle_radii = OBSTACLES[:, 2]     # (K,)

    # Broadcast: (T, M, 1, 2) - (1, 1, K, 2) -> (T, M, K, 2)
    diff_obstacles = xi_traj[:, :, None, :] - obstacle_centers[None, None, :, :]
    dists_obstacles = jnp.sqrt(jnp.sum(diff_obstacles**2, axis=-1) + 1e-8)  # (T, M, K)

    # Collision threshold: agent must be at least (R_safe_obstacle + obstacle_radius) away
    # For each agent-obstacle pair, penalize if distance < R_safe_obstacle + r_obstacle
    safety_distances = R_safe_obstacle + obstacle_radii[None, None, :]  # (1, 1, K)
    l_coll_obstacles = jnp.mean(jnp.maximum(0, safety_distances - dists_obstacles) ** 2)

    # 5. Acceleration penalty
    l_accel = jnp.mean(jnp.sum(jnp.diff(v_traj, axis=0)**2, axis=-1))

    total_loss = 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + \
                 20.0 * l_coll_agents + 100.0 * l_coll_obstacles + 0.1 * l_accel

    return total_loss, (l_track, l_effort, l_coll_agents, l_coll_obstacles, l_bound)

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

# Load or generate dataset
print("Loading/Generating 2D dataset...")
z_init_all, z_target_all, n_grid_actual = get_training_data(
    n_samples=n_samples,
    n_grid=n_grid,
    dataset_dir='../../heat2D/data'
)

# Update n_grid if it changed during loading
if n_grid_actual != n_grid:
    n_grid = n_grid_actual
    print(f"Using n_grid={n_grid} from loaded dataset")

    # Re-initialize model with correct grid size
    dummy_z = jnp.zeros((n_grid, n_grid))
    params = model.init(key, dummy_z, dummy_z, dummy_xi)
    opt_state = optimizer.init(params)

print(f"Dataset ready: {z_init_all.shape}")

# Initialize agents in grid pattern at exact positions [0.2, 0.4, 0.6, 0.8]
n_side = int(jnp.sqrt(n_agents))
positions_1d = jnp.array([0.2, 0.4, 0.6, 0.8])[:n_side]
xi_template = []
for i in range(n_side):
    for j in range(n_side):
        if len(xi_template) < n_agents:
            xi_template.append([float(positions_1d[i]), float(positions_1d[j])])
xi_init_single = jnp.array(xi_template)
xi_init_batch = jnp.tile(xi_init_single, (batch_size, 1, 1))

metrics = []
start_time = time.time()

print("\nStarting training...")
print(f"Obstacles at: {OBSTACLES.tolist()}")
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
              f"Effort: {aux[1]:.4f} | CollA: {aux[2]:.4f} | CollO: {aux[3]:.4f} | Bound: {aux[4]:.4f}")

print(f"\nTraining finished in {time.time() - start_time:.2f}s.")

# Save parameters
with open('decentralized_params_heat2d_obstacles.msgpack', 'wb') as f:
    f.write(flax.serialization.to_bytes(params))
print("Parameters saved to decentralized_params_heat2d_obstacles.msgpack")

# Plot metrics
if metrics:
    metrics = jnp.array(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].plot(metrics[:, 0], metrics[:, 1])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_yscale('log')

    axes[0, 1].plot(metrics[:, 0], metrics[:, 2])
    axes[0, 1].set_title('Tracking Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_yscale('log')

    axes[0, 2].plot(metrics[:, 0], metrics[:, 3])
    axes[0, 2].set_title('Effort Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_yscale('log')

    axes[1, 0].plot(metrics[:, 0], metrics[:, 4])
    axes[1, 0].set_title('Agent-Agent Collision Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_yscale('log')

    axes[1, 1].plot(metrics[:, 0], metrics[:, 5])
    axes[1, 1].set_title('Agent-Obstacle Collision Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_yscale('log')

    axes[1, 2].plot(metrics[:, 0], metrics[:, 6])
    axes[1, 2].set_title('Boundary Loss')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')

    plt.tight_layout()
    # plt.savefig('training_metrics_heat2d_obstacles_decentralized_new.png')
    # print("Training metrics saved to training_metrics_heat2d_obstacles_decentralized.png")
