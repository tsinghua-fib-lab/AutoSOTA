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
from matplotlib.ticker import PercentFormatter

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# --- 1. Setup Directories ---
SAVE_DIR = Path("figures/zs-comparisons")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "heat2d_policy_n16.msgpack"
CSV_PATH = SAVE_DIR / "heat2d_zs_results.csv"
PLOT_PATH = SAVE_DIR / "heat2d_zs_relative_mse.pdf"

# --- 2. Logic Imports ---
from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
from data_utils import get_training_data

# --- 3. Helper: Agent Grid Initializer ---
def get_grid_xi(n_agents):
    """Generates a grid of initial positions for n_agents in [0,1]^2."""
    n_side = int(jnp.ceil(jnp.sqrt(n_agents)))
    x = jnp.linspace(0.2, 0.8, n_side)
    y = jnp.linspace(0.2, 0.8, n_side)
    xv, yv = jnp.meshgrid(x, y)
    xi = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    return xi[:n_agents]

# --- 4. Loss Function ---
def loss_fn(params, z_init, xi_init, z_target, dynamics, n_agents, T_steps):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    
    # 1. Final state tracking error
    l_track = jnp.mean((z_traj[-1] - z_target)**2) 
    
    # 2. Effort loss
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(jnp.sum(v_traj ** 2, axis=-1))
    
    # 3. Boundary penalty (2D)
    margin = 0.02
    b_penalty = jnp.maximum(0, margin - xi_traj)**2 + jnp.maximum(0, xi_traj - (1.0 - margin))**2
    l_bound = jnp.mean(b_penalty)
    
    # 4. Collision avoidance
    diff = xi_traj[:, :, None, :] - xi_traj[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, 0.08 - (dists + mask * 10.0)) ** 2)

    return 5.0 * l_track + 0.1 * l_effort + 100.0 * l_bound + 20.0 * l_coll

@partial(jax.jit, static_argnames=('dynamics', 'n_agents', 'T_steps', 'optimizer'))
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics, n_agents, T_steps, optimizer):
    def mean_loss(p):
        losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None))(
            p, z_init_batch, xi_init_batch, z_target_batch, dynamics, n_agents, T_steps)
        return jnp.mean(losses)
    loss, grads = jax.value_and_grad(mean_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# --- 5. Execution Pipeline ---
def main():
    n_grid = 32
    n_train = 16  # Baseline: 4x4 grid
    epochs = 500
    batch_size = 16
    T_steps = 300
    
    # Sweep range for Zero-Shot Evaluation
    n_eval_list = [9, 16, 25, 36, 49, 64, 81, 100]
    
    model = DecentralizedHeat2DControlNet(features=(16, 32))
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))

    # Initialization
    key = jax.random.PRNGKey(42)
    xi_init_train = get_grid_xi(n_train)
    dummy_z = jnp.zeros((n_grid, n_grid))
    init_params = model.init(key, dummy_z, dummy_z, xi_init_train)

    # --- Training Phase ---
    if not MODEL_PATH.exists():
        print(f"Training Heat 2D on N={n_train} agents...")
        params = init_params
        opt_state = optimizer.init(params)
        
        # Load dataset
        z_init_all, z_target_all, _ = get_training_data(n_samples=5000, n_grid=n_grid, dataset_dir='../data')
        xi_init_batch = jnp.tile(xi_init_train, (batch_size, 1, 1))

        for _ in trange(epochs):
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (batch_size,), 0, 5000)
            params, opt_state, _ = train_step(params, opt_state, z_init_all[idx], xi_init_batch, z_target_all[idx], dynamics, n_train, T_steps, optimizer)
        
        with open(MODEL_PATH, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
    else:
        print(f"Loading Heat 2D model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            params = flax.serialization.from_bytes(init_params, f.read())

    # --- Zero-Shot Evaluation Phase ---
    results = []
    # Test on a specific sample
    z_init_test, z_target_test, _ = get_training_data(n_samples=1, n_grid=n_grid, dataset_dir='../data')
    z_init_test, z_target_test = z_init_test[0], z_target_test[0]

    for n in n_eval_list:
        print(f"Evaluating Zero-Shot N={n}...")
        xi_init = get_grid_xi(n)
        z_traj, _, _, _ = dynamics.unroll_controlled(z_init_test, xi_init, z_target_test, params, T_steps)
        mse = float(jnp.mean((z_traj[-1] - z_target_test)**2))
        results.append({"n_agents": n, "mse": mse})

    df = pd.DataFrame(results)
    
    # --- Relative Normalization (N=16 is 100%) ---
    baseline_mse = df[df['n_agents'] == 16]['mse'].values[0]
    df['relative_mse'] = (df['mse'] / baseline_mse) * 100
    df.to_csv(CSV_PATH, index=False)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['n_agents'], df['relative_mse'], marker='D', linestyle='-', 
            color='#8e44ad', linewidth=2, markersize=8, label='Heat 2D Policy')
    
    ax.axvline(x=16, color='#e67e22', linestyle='--', alpha=0.8, label='Training Size ($N=16$)')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    ax.set_title("Heat 2D Zero-Shot Scalability: Relative MSE\n(Trained on $4\\times4$ Agent Grid)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Number of Agents ($N$)", fontsize=10)
    ax.set_ylabel("Relative MSE (%)", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"Analysis complete. Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()