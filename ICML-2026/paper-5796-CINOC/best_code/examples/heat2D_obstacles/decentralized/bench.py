"""
Evaluation Script: Heat 2D Control with Obstacles
Compares the trained Decentralized Policy against a zero-control baseline,
checking for tracking performance and obstacle avoidance.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
import sys
import argparse
from pathlib import Path
from tesseract_core import Tesseract

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedHeat2DControlNet
from data_utils import get_training_data

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Heat2D Obstacles Decentralized Controller")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--t-steps", type=int, default=300)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--params-file", default="decentralized_params_heat2d_obstacles.msgpack")
    parser.add_argument("--dataset-dir", default="../../heat2D/data")
    parser.add_argument("--out-file", default="figures/images/bench/heat2d_obstacles_results.png")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

def build_agent_grid(n_agents):
    """Build agent grid at exact positions [0.2, 0.4, 0.6, 0.8] in both axes."""
    n_side = int(jnp.sqrt(n_agents))
    positions_1d = jnp.array([0.2, 0.4, 0.6, 0.8])[:n_side]
    xi_template = []
    for i in range(n_side):
        for j in range(n_side):
            if len(xi_template) < n_agents:
                xi_template.append([float(positions_1d[i]), float(positions_1d[j])])
    return jnp.array(xi_template)

def load_params(model, params_file, n_grid, n_agents):
    try:
        with open(params_file, "rb") as f:
            serialized_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: '{params_file}' not found. Run training script first.")
        sys.exit(1)

    dummy_key = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((n_grid, n_grid))
    dummy_xi = jnp.zeros((n_agents, 2))
    dummy_params = model.init(dummy_key, dummy_z, dummy_z, dummy_xi)
    return flax.serialization.from_bytes(dummy_params, serialized_bytes)

# --- 1. Configuration ---
args = parse_args()
if args.cpu:
    jax.config.update("jax_platform_name", "cpu")

n_grid = args.n_grid
n_agents = args.n_agents
T_steps = args.t_steps
N_eval = args.n_eval
R_safe = 0.08
R_safe_obstacle = 0.04

# Obstacle Config (Matches training)
# [x, y, radius]
OBSTACLES = jnp.array([
    [0.30, 0.30, 0.06],   # Diagonal line obstacle 1
    [0.50, 0.50, 0.06],   # Diagonal line obstacle 2 (center)
    [0.70, 0.70, 0.06],   # Diagonal line obstacle 3
])

model = DecentralizedHeat2DControlNet(features=(16, 32))

# --- 2. Helper: Zero Policy ---
def zero_policy_apply(params, local_z, z_target, local_xi):
    """
    Dummy policy: returns zero forcing and zero velocity.
    """
    n_batch = local_xi.shape[0]
    return jnp.zeros((n_batch,)), jnp.zeros((n_batch, 2))

# --- 3. Data Generation & Loading ---
print(f"Loading/Generating {N_eval} Evaluation Samples...")

# Load dataset (using same utility as training)
pool_size = max(N_eval, args.pool_size)
z_init_pool, z_target_pool, _ = get_training_data(
    n_samples=pool_size,
    n_grid=n_grid,
    dataset_dir=args.dataset_dir,
)

# Pick random validation subset
val_key = jax.random.PRNGKey(args.seed)
idx = jax.random.randint(val_key, (N_eval,), 0, len(z_init_pool))
z_init_batch = z_init_pool[idx]
z_target_batch = z_target_pool[idx]

# Initialize Agents (Grid Pattern)
xi_init_single = build_agent_grid(n_agents)
xi_init_batch = jnp.tile(xi_init_single, (N_eval, 1, 1))

# Load Parameters
print("Loading trained parameters...")
params = load_params(model, args.params_file, n_grid, n_agents)

# --- 4. Evaluation Loop ---
# A. Controlled
dynamics_ctrl = PDEDynamics(policy_apply_fn=model.apply)
# B. Uncontrolled
dynamics_unc = PDEDynamics(policy_apply_fn=zero_policy_apply)

print("Running simulations...")

def run_comparison(z_init, xi_init, z_target):
    z_c, xi_c, _, _ = dynamics_ctrl.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    z_u, xi_u, _, _ = dynamics_unc.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    return z_c, xi_c, z_u, xi_u

z_ctrl_chunks = []
xi_ctrl_chunks = []
z_unc_chunks = []
xi_unc_chunks = []

for start in range(0, N_eval, args.chunk_size):
    end = min(N_eval, start + args.chunk_size)
    z_init_chunk = z_init_batch[start:end]
    xi_init_chunk = xi_init_batch[start:end]
    z_target_chunk = z_target_batch[start:end]

    z_c, xi_c, z_u, xi_u = jax.vmap(run_comparison)(
        z_init_chunk, xi_init_chunk, z_target_chunk
    )
    z_ctrl_chunks.append(z_c)
    xi_ctrl_chunks.append(xi_c)
    z_unc_chunks.append(z_u)
    xi_unc_chunks.append(xi_u)

z_ctrl_all = jnp.concatenate(z_ctrl_chunks, axis=0)
xi_ctrl_all = jnp.concatenate(xi_ctrl_chunks, axis=0)
z_unc_all = jnp.concatenate(z_unc_chunks, axis=0)
xi_unc_all = jnp.concatenate(xi_unc_chunks, axis=0)

# --- 5. Analysis ---
print("Calculating metrics...")

# MSE Calculation
targets_expanded = z_target_batch[:, None, :, :]
mse_ctrl = jnp.mean((z_ctrl_all - targets_expanded)**2, axis=(1, 2, 3))
mse_unc = jnp.mean((z_unc_all - targets_expanded)**2, axis=(1, 2, 3))

print(f"Average MSE (Controlled):   {jnp.mean(mse_ctrl):.6f}")
print(f"Average MSE (Uncontrolled): {jnp.mean(mse_unc):.6f}")
print(f"Median MSE (Controlled):    {jnp.median(mse_ctrl):.6f}")
print(f"Median MSE (Uncontrolled):  {jnp.median(mse_unc):.6f}")

if args.no_plot:
    sys.exit(0)

# --- 6. Visualization ---
plt.figure(figsize=(16, 10))

# Helper to draw obstacles
def draw_obstacles(ax):
    for obs in OBSTACLES:
        # Draw physical obstacle (Red)
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)
        ax.add_patch(circle)
        # Draw safety margin (Dotted Red)
        margin = plt.Circle((obs[0], obs[1]), obs[2] + R_safe_obstacle, color='red', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(margin)

# 1. Error Distribution
plt.subplot(2, 2, 1)
plt.boxplot([mse_ctrl, mse_unc], labels=['Controlled', 'Uncontrolled'])
plt.title(f'Tracking MSE Distribution (N={N_eval})')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Sample Index for visualization
sample_idx = int(jnp.clip(args.sample_idx, 0, N_eval - 1))

vmin = float(jnp.min(jnp.array([
    jnp.min(z_target_batch[sample_idx]),
    jnp.min(z_ctrl_all[sample_idx, -1]),
    jnp.min(z_unc_all[sample_idx, -1]),
])))
vmax = float(jnp.max(jnp.array([
    jnp.max(z_target_batch[sample_idx]),
    jnp.max(z_ctrl_all[sample_idx, -1]),
    jnp.max(z_unc_all[sample_idx, -1]),
])))

# 2. Agent Trajectories (Controlled)
ax_traj = plt.subplot(2, 2, 2)
draw_obstacles(ax_traj) # Draw obstacles!
for i in range(n_agents):
    # Plot path
    plt.plot(xi_ctrl_all[sample_idx, :, i, 0], xi_ctrl_all[sample_idx, :, i, 1], alpha=0.6, color='blue')
    # Plot start/end
    plt.scatter(xi_ctrl_all[sample_idx, 0, i, 0], xi_ctrl_all[sample_idx, 0, i, 1], c='green', s=10, marker='x')
    plt.scatter(xi_ctrl_all[sample_idx, -1, i, 0], xi_ctrl_all[sample_idx, -1, i, 1], c='blue', s=20)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Controlled Trajectories vs Obstacles')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# 3. Target State
plt.subplot(2, 3, 4)
plt.imshow(
    z_target_batch[sample_idx],
    origin="lower",
    extent=[0, 1, 0, 1],
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)
draw_obstacles(plt.gca())
plt.title('Target Field')
plt.colorbar()

# 4. Controlled Final State
plt.subplot(2, 3, 5)
plt.imshow(
    z_ctrl_all[sample_idx, -1],
    origin="lower",
    extent=[0, 1, 0, 1],
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)
draw_obstacles(plt.gca())
# Overlay Final Agent Positions
plt.scatter(xi_ctrl_all[sample_idx, -1, :, 0], xi_ctrl_all[sample_idx, -1, :, 1], 
            c='cyan', s=30, edgecolors='white', label='Agents')
plt.title(f'Controlled Final (MSE={mse_ctrl[sample_idx]:.4f})')
plt.colorbar()

# 5. Uncontrolled Final State
plt.subplot(2, 3, 6)
plt.imshow(
    z_unc_all[sample_idx, -1],
    origin="lower",
    extent=[0, 1, 0, 1],
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)
draw_obstacles(plt.gca())
plt.scatter(xi_unc_all[sample_idx, -1, :, 0], xi_unc_all[sample_idx, -1, :, 1], 
            c='grey', s=30, edgecolors='white', alpha=0.5)
plt.title(f'Uncontrolled Final (MSE={mse_unc[sample_idx]:.4f})')
plt.colorbar()

plt.tight_layout()

# Create output directory
output_dir = Path(args.out_file).parent
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(args.out_file)
print(f"Comparison plot saved to '{args.out_file}'")
