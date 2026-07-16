"""
Train decentralized controllers for multiple Kuramoto-Sivashinsky (KS) 1D scenarios
with varying domain sizes, grid resolutions, and number of actuators.
Each scenario is trained separately, and the resulting models are saved for later evaluation
and visualization.
"""
import jax
import jax.numpy as jnp
import optax
import time
from functools import partial
import matplotlib.pyplot as plt
from tqdm import trange
import flax.serialization
import sys
import os
from pathlib import Path

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# --- 1. Multi-Scenario Configuration ---
n_agents_list = [200, 30, 80]
L_domain_list = [500.0, 64.0, 200.0]
N_grid_list = [1024, 256, 512] 

# Training Hyperparameters
batch_size = 64
epochs = 1000
save_path = "multiple_experiments/models"
os.makedirs(save_path, exist_ok=True)

# Define Loss and Train Step (General versions)
def loss_fn(params, u_init, xi_fixed, u_target, dynamics, n_grid, l_domain, t_steps=200):
    u_traj, xi_traj, u_ctrl_traj, v_traj = dynamics.unroll_controlled(
        u_init, xi_fixed, u_target, params, t_steps, n_grid, l_domain, dt=0.05, sigma=1.0
    )
    l_track = jnp.mean((u_traj - u_target[None, :]) ** 2)
    l_effort = jnp.mean(u_ctrl_traj ** 2) 
    total_loss = 10.0 * l_track + 0.001 * l_effort 
    return total_loss, (l_track, l_effort, 0.0, 0.0) # padding aux for consistency

@partial(jax.jit, static_argnames=['dynamics', 'n_grid', 'l_domain'])
def train_step(params, opt_state, u_init_batch, xi_fixed_batch, u_target_batch, dynamics, n_grid, l_domain):
    batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None))
    
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, u_init_batch, xi_fixed_batch, u_target_batch, dynamics, n_grid, l_domain)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    
    # We apply the updates using the 'optimizer' defined in the outer scope
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# --- 2. Main Experiment Loop ---
for i in range(len(n_agents_list)):
    n_agents = n_agents_list[i]
    L_domain = L_domain_list[i]
    N_grid = N_grid_list[i]
    
    print(f"\n{'='*30}")
    print(f"Scenario {i+1}: N={N_grid}, L={L_domain}, Agents={n_agents}")
    print(f"{'='*30}")

    # Initialize Model & Optimizer for this specific scenario
    model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    # Dynamics for current scenario
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    dummy_u = jnp.zeros((N_grid,))
    dummy_xi = jnp.linspace(0, L_domain, n_agents)
    params = model.init(init_key, dummy_u, dummy_u, dummy_xi)

    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    opt_state = optimizer.init(params)

    # Data Generation (Pre-warm chaotic states for specific L and N)
    pool_size = 2500
    key, subkey = jax.random.split(key)
    u_init_pool = get_batch_initial_conditions(subkey, pool_size, N_grid, L_domain)
    u_target_pool = jnp.zeros_like(u_init_pool)

    # Fixed Actuator Positions for this scale
    xi_fixed_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
    xi_fixed_batch = jnp.tile(xi_fixed_single, (batch_size, 1))

    # Training Loop for Scenario
    scenario_metrics = []
    for epoch in trange(epochs, desc=f"Training Exp {i+1}"):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, pool_size)
        u_init_b = u_init_pool[idx]
        u_target_b = u_target_pool[idx]

        params, opt_state, loss, aux = train_step(
            params, opt_state, u_init_b, xi_fixed_batch, u_target_b, dynamics, N_grid, L_domain
        )
        
        if epoch % 10 == 0:
            scenario_metrics.append((epoch, loss, *aux))
            print(f"Exp {i+1} | Epoch {epoch:03d} | Loss: {loss:.4f} | Track: {aux[0]:.4f} | Effort: {aux[1]:.4f}")

    # --- 3. Save Scenario Results ---
    model_filename = f"ks_params_N{N_grid}_L{int(L_domain)}_A{n_agents}.msgpack"
    with open(os.path.join(save_path, model_filename), 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    
    print(f"Saved weights to {save_path}/{model_filename}")

print("\nAll experiments complete.")