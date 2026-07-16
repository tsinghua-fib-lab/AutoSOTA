import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
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
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Import your specific KS components
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

def train(
    n_pde=1024,       # Pool size for initial conditions
    n_agents=8, 
    batch_size=32, 
    T_steps=200, 
    L_domain=22.0,    # Domain size
    N_grid=128,       # Grid resolution
    epochs=500, 
    noise_u=0.0,      # Actuator Noise
    noise_z=0.0,      # Sensor Noise
    sensor_range=4,    # Sensor Patch Size (in grid points)
    save_repo="./", 
    plot_filename='ks_decentralized_training', 
    net_params_filename='ks_decentralized_params', 
    plot_metrics=True
):
    """
    Trains a decentralized DecentralizedControlNet policy to stabilize the 1D Kuramoto-Sivashinsky equation.
    """
    
    # Configuration dict to pass into JIT-ed functions easily if needed
    CONFIG = {
        "noise_u": noise_u, 
        "noise_z": noise_z,
        "L": L_domain,
        "N": N_grid
    }

    print(f"--- Initializing KS Training (L={L_domain}, N={N_grid}, Noise_U={noise_u}) ---")

    # --- 1. Initialization ---
    # KS Policy (Outputting intensity only)
    model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain, window_size=sensor_range)
    key = jax.random.PRNGKey(42)

    # Init params
    key, init_key = jax.random.split(key)
    dummy_u = jnp.zeros((N_grid,))
    dummy_xi = jnp.linspace(0, L_domain, n_agents)
    params = model.init(init_key, dummy_u, dummy_u, dummy_xi)

    # Optimizer
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    opt_state = optimizer.init(params)

    # Initialize Dynamics Wrapper
    # We pass the model's apply function here
    dynamics = PDEDynamics(policy_apply_fn=model.apply)

    # --- 2. Loss Function ---
    def loss_fn(params, u_init, xi_fixed, u_target, key, dynamics):
        """
        Calculates loss over a full trajectory unroll with noise.
        """
        # Unroll the dynamics using the wrapper
        # We pass the noise configuration and the random key here
        u_traj, xi_traj, u_ctrl_traj, _ = dynamics.unroll_controlled(
            u_init, 
            xi_fixed, 
            u_target, 
            params, 
            T_steps, 
            N_grid=CONFIG["N"],
            L=CONFIG["L"],
            key=key,
            noise_u=CONFIG["noise_u"], 
            noise_z=CONFIG["noise_z"],
            dt=0.05,
            sigma=1.0
        )
        
        # 1. Tracking Loss (Stabilize chaotic waves to 0)
        # We want the field u(x,t) to be close to u_target (usually 0)
        l_track = jnp.mean((u_traj - u_target[None, :]) ** 2)
        
        # 2. Effort Loss (Penalize high actuation intensity)
        l_effort = jnp.mean(u_ctrl_traj ** 2) 
        
        # Weighted Sum
        # High weight on tracking because KS chaos is hard to suppress
        total_loss = 10.0 * l_track + 0.001 * l_effort 
        
        return total_loss, (l_track, l_effort)

    @partial(jax.jit, static_argnames='dynamics')
    def train_step(params, opt_state, u_init_batch, xi_fixed_batch, u_target_batch, key, dynamics):
        # Split key for the batch (one key per simulation in the batch)
        keys = jax.random.split(key, u_init_batch.shape[0])
        
        # Vmap over batch dimension (0), passing unique keys
        batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, None))
        
        def mean_loss_fn(p):
            losses, auxs = batched_loss_fn(p, u_init_batch, xi_fixed_batch, u_target_batch, keys, dynamics)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

        (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, aux

    # --- 3. Data Generation (Spin-up) ---
    print("Generating Chaotic Initial Conditions...")
    # 
    # We need to run the solver for a while to get "mature" chaos, otherwise
    # the initial conditions are too smooth and easy to control.
    key, subkey = jax.random.split(key)
    u_init_pool = get_batch_initial_conditions(subkey, n_pde, N_grid, L_domain)
    
    # Target is stabilization -> Zero state
    u_target_pool = jnp.zeros_like(u_init_pool)

    # Fixed Actuator Positions (Equispaced)
    xi_fixed_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
    xi_fixed_batch = jnp.tile(xi_fixed_single, (batch_size, 1))

    # --- 4. Training Loop ---
    metrics = []
    start_time = time.time()
    
    pbar = trange(epochs, desc="Training")
    for epoch in pbar:
        # Sample batch indices
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, n_pde)
        u_init_b = u_init_pool[idx]
        u_target_b = u_target_pool[idx]

        # Generate a fresh key for the noise injection in this step
        key, step_key = jax.random.split(key)

        params, opt_state, loss, aux = train_step(
            params, opt_state, u_init_b, xi_fixed_batch, u_target_b, step_key, dynamics
        )
        
        # Logging
        if epoch % 10 == 0:
            l_track_val = aux[0]
            l_effort_val = aux[1]
            metrics.append((epoch, loss, l_track_val, l_effort_val))
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Track": f"{l_track_val:.4f}"})

    print(f"Training finished in {time.time() - start_time:.2f}s.")     
            
    # --- 5. Saving ---
    if save_repo:
        save_dir = Path(save_repo)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_plot_path = save_dir / f'{plot_filename}.png'
        full_param_path = save_dir / f'{net_params_filename}.msgpack'
    else:
        full_plot_path = f'{plot_filename}.png'
        full_param_path = f'{net_params_filename}.msgpack'

    # Save Parameters
    with open(full_param_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"Params saved at {full_param_path}.")

    # --- 6. Plotting ---
    if plot_metrics:
        metrics = jnp.array(metrics)
        epochs_recorded = metrics[:, 0]
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Total Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_recorded, metrics[:, 1], color='black', label='Total Loss')
        plt.yscale('log')
        plt.title('Total Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.legend()

        # Subplot 2: Tracking vs Effort
        plt.subplot(1, 2, 2)
        plt.plot(epochs_recorded, metrics[:, 2], label='Tracking Error')
        plt.plot(epochs_recorded, metrics[:, 3], label='Control Effort', alpha=0.7)
        plt.yscale('log')
        plt.title('Stability vs Effort')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig(full_plot_path) 
        print(f"Training metrics plotted and saved to {full_plot_path}")

if __name__ == "__main__":
    # Example Run
    train(
        n_pde=1024,
        n_agents=8,
        epochs=500,
        L_domain=64.0,  
        noise_u=0.0,   
        noise_z=0.0    
    )