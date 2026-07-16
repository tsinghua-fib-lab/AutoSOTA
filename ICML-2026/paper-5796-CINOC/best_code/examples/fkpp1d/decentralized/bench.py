"""
Evaluation Script: Controlled vs Uncontrolled Performance
Compares the trained Decentralized ControlNet against a zero-control baseline
on 100 new initial conditions.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
import sys
from pathlib import Path
from functools import partial

# Add project root to sys.path (Same as training)
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# --- 1. Setup & Configuration ---
BENCH_DIR = Path("figures/images/bench")
BENCH_DIR.mkdir(parents=True, exist_ok=True)

n_pde, n_agents = 100, 20
T_steps = 300
N_eval = 100  # Number of evaluation samples

# Load Model Structure (to get shapes right)
model = DecentralizedControlNet(features=(64, 64))

# --- 2. Helper: Zero Policy for Uncontrolled Baseline ---
# --- 2. Helper: Zero Policy for Uncontrolled Baseline ---
def zero_policy_apply(params, local_z, z_target, local_xi):
    """
    Dummy policy that returns zero control inputs.
    Arguments must match the call signature in solver.py: 
    (params, z_observed, z_target, xi_curr)
    """
    n_batch = local_xi.shape[0]
    # Return zeros for forcing (u) and velocity (v)
    return jnp.zeros((n_batch,)), jnp.zeros((n_batch,))

# --- 3. Data Generation & Loading ---
print(f"Generating {N_eval} Evaluation Initial Conditions...")
key = jax.random.PRNGKey(42) # Different seed than training for validation

# Generate 100 samples
key, subkey1, subkey2 = jax.random.split(key, 3)
_, z_init_batch = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.2))(jax.random.split(subkey1, N_eval))
_, z_target_batch = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.4))(jax.random.split(subkey2, N_eval))

# Agents start uniform
xi_init_single = jnp.linspace(0.2, 0.8, n_agents)
xi_init_batch = jnp.tile(xi_init_single, (N_eval, 1))

# Load Trained Parameters
print("Loading trained parameters...")
with open('decentralized_params.msgpack', 'rb') as f:
    serialized_bytes = f.read()

# Initialize dummy params to define structure, then restore
dummy_params = model.init(jax.random.PRNGKey(0), jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
params = flax.serialization.from_bytes(dummy_params, serialized_bytes)

# --- 4. Evaluation Loop ---
# Initializing native JAX dynamics
dynamics_ctrl = PDEDynamics(policy_apply_fn=model.apply)

# Uncontrolled Dynamics (Uses zero policy)
dynamics_unc = PDEDynamics(policy_apply_fn=zero_policy_apply)

print("Running simulations...")

# Define the unroll function for a single batch element
def run_comparison(z_init, xi_init, z_target):
    # Controlled run
    z_c, xi_c, u_c, v_c = dynamics_ctrl.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    # Uncontrolled run (params=None usually works if policy ignores it, 
    # but we pass params to be safe; the zero_policy ignores them)
    z_u, xi_u, u_u, v_u = dynamics_unc.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps
    )
    return (z_c, xi_c), (z_u, xi_u)

# Vmap over the 100 I.C.s
(traj_ctrl, traj_unc) = jax.vmap(run_comparison)(z_init_batch, xi_init_batch, z_target_batch)

# Unpack results: Shapes are (N_eval, T_steps, ...)
z_ctrl_all, xi_ctrl_all = traj_ctrl
z_unc_all, xi_unc_all = traj_unc

# --- 5. Analysis & Visualization ---
print("Calculating metrics...")

# Calculate Tracking Error (MSE against Target)
# Target is (N_eval, n_pde), Trajectory is (N_eval, T_steps, n_pde)
# We expand target to match time dimension for broadcasting
targets_expanded = z_target_batch[:, None, :]

mse_ctrl = jnp.mean((z_ctrl_all - targets_expanded)**2, axis=(1, 2))
mse_unc = jnp.mean((z_unc_all - targets_expanded)**2, axis=(1, 2))

print(f"Average MSE (Controlled):   {jnp.mean(mse_ctrl):.6f}")
print(f"Average MSE (Uncontrolled): {jnp.mean(mse_unc):.6f}")

# --- Plotting ---
plt.figure(figsize=(15, 10))

# 1. Comparison of MSE
plt.subplot(2, 2, 1)
plt.boxplot([mse_ctrl, mse_unc], labels=['Controlled', 'Uncontrolled'])
plt.title(f'Tracking Error Distribution (N={N_eval})')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

# 2. Field Heatmap (Sample #0)
# Show the final state of the field for the first sample
sample_idx = 0
x_grid = jnp.linspace(0, 1, n_pde)

plt.subplot(2, 2, 2)
plt.plot(x_grid, z_target_batch[sample_idx], 'k--', label='Target', linewidth=2)
plt.plot(x_grid, z_ctrl_all[sample_idx, -1, :], 'b-', label='Controlled Final')
plt.plot(x_grid, z_unc_all[sample_idx, -1, :], 'r-', label='Uncontrolled Final')
plt.plot(x_grid, z_init_batch[sample_idx], 'g:', label='Initial', alpha=0.5)
plt.title(f'Field State @ T={T_steps} (Sample {sample_idx})')
plt.legend()
plt.grid(True)

# 3. Agent Trajectories (Controlled)
plt.subplot(2, 2, 3)
for i in range(n_agents):
    plt.plot(xi_ctrl_all[sample_idx, :, i], color='blue', alpha=0.5)
plt.title('Controlled Agent Trajectories')
plt.xlabel('Time Step')
plt.ylabel('Position (x)')
plt.ylim(0, 1)

# 4. Agent Trajectories (Uncontrolled)
plt.subplot(2, 2, 4)
for i in range(n_agents):
    plt.plot(xi_unc_all[sample_idx, :, i], color='red', alpha=0.5)
plt.title('Uncontrolled Agent Trajectories')
plt.xlabel('Time Step')
plt.ylabel('Position (x)')
plt.ylim(0, 1)

plt.tight_layout()
save_path = BENCH_DIR / 'comparison_results.png'
plt.savefig(save_path)
print(f"Comparison plot saved to '{save_path}'")