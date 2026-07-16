import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import flax.serialization
import optax
import pickle
from pathlib import Path
from functools import partial
from tqdm import trange
from matplotlib.ticker import PercentFormatter

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# --- 1. Setup Directories ---
SAVE_DIR = Path("figures/zs_scaling")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "ks2d_policy_n100.msgpack"
CSV_PATH = SAVE_DIR / "ks2d_zs_results.csv"
PLOT_PATH = SAVE_DIR / "ks2d_zs_relative_mse.pdf"

# --- 2. Logic Imports (KS 2D Specific) ---
from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet
from data_utils import get_batch_initial_conditions

# --- 3. Helper: Agent Grid Initializer ---
def get_grid_xi(n_agents, L_domain):
    """Generates a regular grid of initial positions for n_agents in [0, L]x[0, L]."""
    n_side = int(jnp.ceil(jnp.sqrt(n_agents)))
    # Create grid points centered in their cells
    # linspace with endpoint=False and offset helps center them
    x = jnp.linspace(0, L_domain, n_side, endpoint=False) + (L_domain / n_side) / 2.0
    y = jnp.linspace(0, L_domain, n_side, endpoint=False) + (L_domain / n_side) / 2.0
    
    xv, yv = jnp.meshgrid(x, y)
    xi = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    
    # If n_agents isn't a perfect square, we just take the first n (though we usually use squares)
    return xi[:n_agents]

# --- 4. Loss Function (KS 2D with Substeps) ---
def loss_fn(params, u_init, xi_fixed, u_target, dynamics, T_steps, substeps, N_grid, L_domain, dt):
    # Unroll trajectory (Control Step = substeps * dt)
    u_traj, _, u_ctrl_traj, _ = dynamics.unroll_controlled(
        u_init, 
        xi_fixed, 
        u_target, 
        params, 
        t_steps=T_steps,
        substeps=substeps,   
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        sigma=1.2 # Fixed influence width
    )
    
    # 1. Final state tracking error (Stabilize to target)
    # Using mean over the last 20% of trajectory for robustness in 2D chaos
    l_track = jnp.mean((u_traj[int(0.8*T_steps):] - u_target[None, :, :])**2)
    
    # 2. Effort loss
    l_effort = jnp.mean(u_ctrl_traj ** 2)
    
    # Weighted Sum 
    return 50.0 * l_track + 5e-3 * l_effort

@partial(jax.jit, static_argnames=('dynamics', 'T_steps', 'substeps', 'N_grid', 'L_domain', 'dt', 'optimizer'))
def train_step(params, opt_state, u_init_batch, xi_batch, u_target_batch, dynamics, T_steps, substeps, N_grid, L_domain, dt, optimizer):
    def mean_loss(p):
        losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None, None, None, None, None, None))(
            p, u_init_batch, xi_batch, u_target_batch, dynamics, T_steps, substeps, N_grid, L_domain, dt)
        return jnp.mean(losses)
    
    loss, grads = jax.value_and_grad(mean_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# --- 5. Data Helper ---
def get_or_create_data(n_samples, N_grid, L_domain):
    """
    Manages loading/generating chaotic Initial Conditions.
    Aligned with training script paths.
    """
    # 1. Resolve path relative to THIS script (examples/ks2d/decentralized/)
    current_dir = Path(__file__).resolve().parent
    # 2. Target the same data folder (examples/ks2d/data/)
    data_dir = current_dir.parent / "data" 
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Use the EXACT same filename as the training script
    filename = f"ks2d_chaotic_ics_{N_grid}.pkl"
    file_path = data_dir / filename

    if file_path.exists():
        print(f"[Data] Found existing ICs at: {file_path}")
        with open(file_path, 'rb') as f:
            u_pool = pickle.load(f)
            u_pool = jnp.array(u_pool)
            
        # Check if we have compatible resolution
        if u_pool.shape[1] != N_grid:
            raise ValueError(f"Resolution mismatch in {filename}. Expected {N_grid}, got {u_pool.shape[1]}")

        # Check if we have ENOUGH samples
        if u_pool.shape[0] >= n_samples:
            print(f"[Data] Loaded {u_pool.shape[0]} samples. Using first {n_samples}.")
            return u_pool[:n_samples]
        else:
            print(f"[Data] Existing file has {u_pool.shape[0]} samples, but {n_samples} needed. Regenerating...")

    # If file doesn't exist or isn't big enough
    print(f"[Data] Generating {n_samples} chaotic states...")
    key = jax.random.PRNGKey(42)
    u_pool = get_batch_initial_conditions(key, n_samples, N_grid, L_domain)
    
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)
        
    return u_pool

# --- 6. Execution Pipeline ---
def main():
    # Physics Config (KS 2D)
    CONFIG = {
        'N_grid': 64,         
        'L_domain': 32.0,      
        'dt': 0.005,
        'substeps': 20,       
        'T_steps': 50,        
    }
    
    # Training Config
    n_train = 196     # Baseline: 10x10 grid
    epochs = 50
    batch_size = 4    # Small batch size for 2D due to memory
    pool_size = 100   # Pool of initial conditions
    
    # Zero-Shot Sweep (Perfect Squares for regular grids)
    # 36 (6x6), 64 (8x8), 100 (10x10), 144 (12x12), 196 (14x14), 256 (16x16)
    n_eval_list = [144, 196, 256, 324, 400, 484, 576, 676, 784, 900, 1024]
    
    # Model Setup
    model = DecentralizedKS2DControlNet(
        features=(64, 128), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=5.0
    )
    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)
    
    # Learning Rate with Warmup
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=5e-4, peak_value=1e-3, warmup_steps=10,
        decay_steps=epochs, end_value=1e-5
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))

    # Initialization
    key = jax.random.PRNGKey(42)
    dummy_u = jnp.zeros((CONFIG['N_grid'], CONFIG['N_grid']))
    dummy_xi = get_grid_xi(n_train, CONFIG['L_domain'])
    init_key, key = jax.random.split(key)
    
    init_params = model.init(init_key, dummy_u, dummy_u, dummy_xi)

    # --- Training Phase ---
    if not MODEL_PATH.exists():
        print(f"Training KS 2D Policy on N={n_train} agents...")
        params = init_params
        opt_state = optimizer.init(params)
        
        # Load Data
        u_init_pool = get_or_create_data(pool_size, CONFIG['N_grid'], CONFIG['L_domain'])
        u_target_pool = jnp.zeros_like(u_init_pool)
        
        # Batch of fixed positions
        xi_train_batch = jnp.tile(dummy_xi, (batch_size, 1, 1))

        pbar = trange(epochs, desc="Training")
        for _ in pbar:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (batch_size,), 0, pool_size)
            
            params, opt_state, loss = train_step(
                params, opt_state, 
                u_init_pool[idx], xi_train_batch, u_target_pool[idx], 
                dynamics, 
                CONFIG['T_steps'], CONFIG['substeps'], CONFIG['N_grid'], CONFIG['L_domain'], CONFIG['dt'],
                optimizer
            )
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
        
        with open(MODEL_PATH, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
    else:
        print(f"Loading KS 2D model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            params = flax.serialization.from_bytes(init_params, f.read())

    # --- Zero-Shot Evaluation Phase ---
    results = []
    
    # Create Test Set (Small due to computation time)
    print("Generating Test Case...")
    n_test = 2
    u_init_test = get_or_create_data(n_test + pool_size, CONFIG['N_grid'], CONFIG['L_domain'])[-n_test:]
    u_target_test = jnp.zeros_like(u_init_test)

    # Use same horizon as training for consistency in this check
    T_eval = CONFIG['T_steps'] 

    for n in n_eval_list:
        print(f"Evaluating Zero-Shot N={n}...")
        
        xi_eval = get_grid_xi(n, CONFIG['L_domain'])
        
        mse_list = []
        for i in range(n_test):
            u_traj, _, _, _ = dynamics.unroll_controlled(
                u_init_test[i], xi_eval, u_target_test[i], params, 
                t_steps=T_eval,
                substeps=CONFIG['substeps'],
                N_grid=CONFIG['N_grid'],
                L=CONFIG['L_domain'],
                dt=CONFIG['dt'],
                sigma=1.2
            )
            # MSE of final state
            mse_val = float(jnp.mean((u_traj[-1] - u_target_test[i])**2))
            mse_list.append(mse_val)
            
        avg_mse = np.mean(mse_list)
        results.append({"n_agents": n, "mse": avg_mse})

    df = pd.DataFrame(results)
    
    # --- Relative Normalization ---
    baseline_mse = df[df['n_agents'] == n_train]['mse'].values[0]
    baseline_mse = max(baseline_mse, 1e-9)
    
    df['relative_mse'] = (df['mse'] / baseline_mse) * 100
    df.to_csv(CSV_PATH, index=False)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['n_agents'], df['relative_mse'], marker='s', linestyle='-', 
            color='#16a085', linewidth=2, markersize=8, label='KS 2D Policy')
    
    ax.axvline(x=n_train, color='#d35400', linestyle='--', alpha=0.8, label=f'Training Size ($N={n_train}$)')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    ax.set_title(f"KS 2D Zero-Shot Scalability: Relative MSE\n(Trained on {int(np.sqrt(n_train))}x{int(np.sqrt(n_train))} Grid, $L={int(CONFIG['L_domain'])}$)", fontsize=12, fontweight='bold')
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