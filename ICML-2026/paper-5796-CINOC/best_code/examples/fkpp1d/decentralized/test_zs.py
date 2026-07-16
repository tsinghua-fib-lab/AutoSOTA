import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import flax.serialization
import optax
from pathlib import Path
from functools import partial
from tqdm import trange
from matplotlib.ticker import ScalarFormatter, PercentFormatter

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# --- 1. Setup Directories ---
SAVE_DIR = Path("figures/zs-comparisons")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "trained_policy_n30.msgpack"
CSV_PATH = SAVE_DIR / "zs_scalability_results.csv"

# --- 2. Training & Eval Logic Imports ---
from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

def loss_fn(params, z_init, xi_init, z_target, dynamics, n_agents, T_steps):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    l_track = jnp.mean((z_traj[-1] - z_target)**2) 
    l_effort = jnp.mean(u_traj ** 2) 
    
    margin = 0.02
    l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + jnp.maximum(0, xi_traj - 0.98)**2)
    dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, 0.05 - (dists + mask)) ** 2)
    
    return 10.0 * l_track + 0.01 * l_effort + 100.0 * l_bound + 1.0 * l_coll

@partial(jax.jit, static_argnames=('dynamics', 'n_agents', 'T_steps', 'optimizer'))
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics, n_agents, T_steps, optimizer):
    def mean_loss(p):
        losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None))(
            p, z_init_batch, xi_init_batch, z_target_batch, dynamics, n_agents, T_steps)
        return jnp.mean(losses)
    loss, grads = jax.value_and_grad(mean_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# --- 3. Execution Pipeline ---

def main():
    n_pde = 100
    n_train = 30  
    n_eval_list = [15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180]
    epochs = 500
    
    model = DecentralizedControlNet(features=(64, 64))
    optimizer = optax.adam(1e-3)
    
    # Dynamics wrapper now defaults to the internal JAX logic
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    # Initialize params template (needed for either loading or training)
    key = jax.random.PRNGKey(42)
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_train,)))

    # --- Training Phase / Model Loading ---
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Starting training on N={n_train}...")
        params = init_params
        opt_state = optimizer.init(params)
        
        # Generate dataset
        _, z_inits = jax.vmap(partial(data_utils.generate_grf, n_points=n_pde))(jax.random.split(key, 1000))
        _, z_targets = jax.vmap(partial(data_utils.generate_grf, n_points=n_pde))(jax.random.split(key, 1000) + 1)
        xi_init_train = jnp.tile(jnp.linspace(0.1, 0.9, n_train), (32, 1))

        for _ in trange(epochs):
            idx = jax.random.randint(key, (32,), 0, 1000)
            params, opt_state, _ = train_step(
                params, opt_state, z_inits[idx], xi_init_train, 
                z_targets[idx], dynamics, n_train, 300, optimizer
            )
        
        # Save the trained model
        with open(MODEL_PATH, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
        print(f"Training complete. Model saved to {MODEL_PATH}")
    else:
        print(f"Model found. Loading existing model from {MODEL_PATH}...")
        with open(MODEL_PATH, 'rb') as f:
            params = flax.serialization.from_bytes(init_params, f.read())

    # --- Evaluation Phase ---
    results = []
    # Fixed test scenario
    _, test_z_init = data_utils.generate_grf(jax.random.PRNGKey(99), n_points=n_pde)
    _, test_z_target = data_utils.generate_grf(jax.random.PRNGKey(100), n_points=n_pde)

    for n in n_eval_list:
        print(f"Evaluating Zero-Shot N={n}...")
        xi_init = jnp.linspace(0.1, 0.9, n)
        z_traj, _, _, _ = dynamics.unroll_controlled(test_z_init, xi_init, test_z_target, params, 300)
        mse = jnp.mean((z_traj[-1] - test_z_target)**2)
        results.append({"n_agents": n, "mse": float(mse)})

    df = pd.DataFrame(results)
    
    # --- Relative MSE Calculation ---
    baseline_mse = df[df['n_agents'] == 30]['mse'].values[0]
    df['relative_mse'] = (df['mse'] / baseline_mse) * 100

    # --- Save Data to CSV ---
    df.to_csv(CSV_PATH, index=False)
    print(f"Evaluation data saved to {CSV_PATH}")

    # --- Plotting (Generating PDF again) ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['n_agents'], df['relative_mse'], marker='o', linestyle='-', 
            color='#2980b9', linewidth=2, markersize=8, label='Policy Performance')
    
    ax.axvline(x=30, color='#e74c3c', linestyle='--', alpha=0.8, label='Training Cardinality ($N=30$)')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    ax.set_title("Zero-Shot Scalability: Relative Tracking MSE\n(Trained on $N=30$)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Number of Agents ($N$)", fontsize=10)
    ax.set_ylabel("Relative MSE (%)", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = SAVE_DIR / "zs_scalability_relative_mse.pdf"
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()