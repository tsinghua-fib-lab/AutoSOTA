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
SAVE_DIR = Path("figures/ks_zs_scaling")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "ks1d_policy_n30.msgpack"
CSV_PATH = SAVE_DIR / "ks1d_zs_results.csv"
PLOT_PATH = SAVE_DIR / "ks1d_zs_relative_mse.pdf"

# --- 2. Logic Imports (KS Specific) ---
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# --- 3. Helper: Agent Initializer ---
def get_equispaced_xi(n_agents, L_domain):
    """Generates n_agents equispaced positions on [0, L]."""
    # Centered intervals: |  .  |  .  |  .  |
    return jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain / n_agents) / 2.0

# --- 4. Loss Function ---
def loss_fn(params, u_init, xi_fixed, u_target, key, dynamics, T_steps, N_grid, L_domain, noise_u, noise_z):
    # Unroll trajectory
    u_traj, _, u_ctrl_traj, _ = dynamics.unroll_controlled(
        u_init, 
        xi_fixed, 
        u_target, 
        params, 
        T_steps, 
        N_grid=N_grid,
        L=L_domain,
        key=key,
        noise_u=noise_u, 
        noise_z=noise_z,
        dt=0.05,
        sigma=1.0
    )
    
    # 1. Final state tracking error (Stabilize to target)
    # KS is chaotic, so we take the mean over the last 10% of timesteps
    l_track = jnp.mean((u_traj[int(0.9*T_steps):] - u_target[None, :])**2)
    
    # 2. Effort loss
    l_effort = jnp.mean(u_ctrl_traj ** 2)
    
    # Weighted Sum 
    return 10.0 * l_track + 0.001 * l_effort

@partial(jax.jit, static_argnames=('dynamics', 'T_steps', 'N_grid', 'L_domain', 'optimizer'))
def train_step(params, opt_state, u_init_batch, xi_batch, u_target_batch, key, dynamics, T_steps, N_grid, L_domain, optimizer):
    keys = jax.random.split(key, u_init_batch.shape[0])
    
    def mean_loss(p):
        losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, None, None, None, None, None, None))(
            p, u_init_batch, xi_batch, u_target_batch, keys, dynamics, T_steps, N_grid, L_domain, 0.05, 0.025)
        return jnp.mean(losses)
    
    loss, grads = jax.value_and_grad(mean_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# --- 5. Execution Pipeline ---
def main():
    # Physics Config (Fixed L=32)
    L_domain = 32.0
    N_grid = 256
    
    # Training Config
    n_train = 30      # Training on 30 Agents (~1.0 spacing)
    epochs = 500
    batch_size = 32
    T_steps = 300
    
    # Zero-Shot Sweep: From Half Density (15) to 3x Density
    n_eval_list = [15, 20, 25, 30, 40, 50, 60, 90, 100, 120, 150, 180]
    
    # Model Setup
    model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain, window_size=4)
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))

    # Initialization
    key = jax.random.PRNGKey(42)
    dummy_u = jnp.zeros((N_grid,))
    dummy_xi = get_equispaced_xi(n_train, L_domain)
    init_key, key = jax.random.split(key)
    
    # Initialize parameters with n_train agents
    init_params = model.init(init_key, dummy_u, dummy_u, dummy_xi)

    # --- Training Phase ---
    if not MODEL_PATH.exists():
        print(f"Training KS 1D Policy on N={n_train} agents (L={L_domain})...")
        params = init_params
        opt_state = optimizer.init(params)
        
        # Load/Generate chaotic data
        print("Generating Chaotic Initial Conditions...")
        key, subkey = jax.random.split(key)
        u_init_pool = get_batch_initial_conditions(subkey, 1024, N_grid, L_domain)
        u_target_pool = jnp.zeros_like(u_init_pool)
        
        # Batch of fixed positions for training
        xi_train_batch = jnp.tile(dummy_xi, (batch_size, 1))

        pbar = trange(epochs, desc="Training")
        for _ in pbar:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (batch_size,), 0, 1024)
            key, step_key = jax.random.split(key)
            
            params, opt_state, loss = train_step(
                params, opt_state, 
                u_init_pool[idx], xi_train_batch, u_target_pool[idx], 
                step_key, dynamics, T_steps, N_grid, L_domain, optimizer
            )
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
        
        with open(MODEL_PATH, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
    else:
        print(f"Loading KS model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            params = flax.serialization.from_bytes(init_params, f.read())

    # --- Zero-Shot Evaluation Phase ---
    results = []
    
    # Create a consistent, difficult test set
    print("Generating Test Case...")
    n_test_samples = 5
    key, subkey = jax.random.split(key)
    u_init_test = get_batch_initial_conditions(subkey, n_test_samples, N_grid, L_domain)
    u_target_test = jnp.zeros_like(u_init_test)

    # Longer horizon for evaluation to ensure stability holds
    T_eval = 400 

    for n in n_eval_list:
        print(f"Evaluating Zero-Shot N={n}...")
        
        # 1. Setup new agent configuration
        xi_eval = get_equispaced_xi(n, L_domain)
        
        # 2. Run simulation (averaged over test samples)
        mse_list = []
        for i in range(n_test_samples):
             # Zero noise for deterministic evaluation curve
            u_traj, _, _, _ = dynamics.unroll_controlled(
                u_init_test[i], xi_eval, u_target_test[i], params, T_eval,
                N_grid=N_grid, L=L_domain, key=jax.random.PRNGKey(0),
                noise_u=0.0, noise_z=0.0, dt=0.05, sigma=1.0
            )
            # Final state error
            mse_val = float(jnp.mean((u_traj[-1] - u_target_test[i])**2))
            mse_list.append(mse_val)
            
        avg_mse = np.mean(mse_list)
        results.append({"n_agents": n, "mse": avg_mse})

    df = pd.DataFrame(results)
    
    # --- Relative Normalization ---
    baseline_mse = df[df['n_agents'] == n_train]['mse'].values[0]
    # Prevent div by zero
    baseline_mse = max(baseline_mse, 1e-9)
    
    df['relative_mse'] = (df['mse'] / baseline_mse) * 100
    df.to_csv(CSV_PATH, index=False)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot line
    ax.plot(df['n_agents'], df['relative_mse'], marker='o', linestyle='-', 
            color='#2980b9', linewidth=2, markersize=8, label='KS 1D Policy')
    
    # Mark Training Point
    ax.axvline(x=n_train, color='#e74c3c', linestyle='--', alpha=0.8, label=f'Training Size ($N={n_train}$)')
    # Mark 100% line
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    ax.set_title(f"KS 1D Zero-Shot Scalability: Relative MSE\n(Trained on N={n_train}, $L={L_domain}$)", fontsize=12, fontweight='bold')
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