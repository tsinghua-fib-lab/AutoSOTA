"""
Evaluation Script: KS-1D Stabilization
Compares the trained Policy against Uncontrolled Chaos on 100 initial conditions.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.serialization
import sys
from pathlib import Path
from functools import partial

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Import KS specific modules (adjusting for the centralized folder structure)
from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# --- 1. Configuration ---
N_grid = 128
L_domain = 22.0
n_agents = 8
T_steps = 200
N_eval = 100

model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)

# --- 2. Helper: Zero Policy (Uncontrolled) ---
def zero_policy_apply(params, u_obs, u_target, xi_fixed):
    """
    Dummy policy for KS with Fixed Actuators.
    Returns ONLY zero forcing (stabilization inactive).
    Arguments must match solver interface: (params, u, u_target, xi)
    """
    # xi_fixed has shape (n_agents,) inside the vmap
    n_agents = xi_fixed.shape[0]
    
    # RETURN SINGLE ARRAY (Forcing only), not a tuple.
    return jnp.zeros((n_agents,))

# --- 3. Data Generation ---
print(f"Generating {N_eval} Chaotic Initial Conditions...")
key = jax.random.PRNGKey(1234) # Validation Seed

# KS "Spin-up" to get realistic chaotic states
key, subkey = jax.random.split(key)
u_init_batch = get_batch_initial_conditions(subkey, N_eval, N_grid, L_domain)
u_target_batch = jnp.zeros_like(u_init_batch) # Stabilization target is 0

# Fixed Actuators
xi_fixed_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
xi_fixed_batch = jnp.tile(xi_fixed_single, (N_eval, 1))

# --- 4. Load Model ---
print("Loading trained parameters...")
try:
    with open('ks_centralized_params.msgpack', 'rb') as f:
        serialized_bytes = f.read()
except FileNotFoundError:
    print("Error: 'ks_centralized_params.msgpack' not found.")
    sys.exit(1)

# Init dummy structure
dummy_key = jax.random.PRNGKey(0)
dummy_u = jnp.zeros((N_grid,))
dummy_xi = jnp.linspace(0, L_domain, n_agents)
dummy_params = model.init(dummy_key, dummy_u, dummy_u, dummy_xi)

params = flax.serialization.from_bytes(dummy_params, serialized_bytes)

# --- 5. Evaluation Loop ---
# Initialize Dynamics (Pure JAX, no Tesseract needed for this KS implementation)
dynamics_ctrl = PDEDynamics(policy_apply_fn=model.apply)
dynamics_unc = PDEDynamics(policy_apply_fn=zero_policy_apply)

print("Running simulations...")

@jax.jit
def run_comparison(u_init, xi_fixed, u_target):
    # Controlled
    u_c, _, _, _ = dynamics_ctrl.unroll_controlled(
        u_init, xi_fixed, u_target, params, T_steps, N_grid, L_domain, dt=0.05, sigma=1.0
    )
    # Uncontrolled
    u_u, _, _, _ = dynamics_unc.unroll_controlled(
        u_init, xi_fixed, u_target, params, T_steps, N_grid, L_domain, dt=0.05, sigma=1.0
    )
    return u_c, u_u

# Batch execution
u_ctrl_all, u_unc_all = jax.vmap(run_comparison)(u_init_batch, xi_fixed_batch, u_target_batch)

# --- 6. Metrics ---
print("Calculating metrics...")

# Calculate Energy (L2 Norm squared) over time
# Shape: (N_eval, T_steps, N_grid)
energy_ctrl = jnp.mean(u_ctrl_all**2, axis=2) # Average over space
energy_unc = jnp.mean(u_unc_all**2, axis=2)

final_energy_ctrl = energy_ctrl[:, -1]
final_energy_unc = energy_unc[:, -1]

print(f"Mean Final Energy (Controlled):   {jnp.mean(final_energy_ctrl):.6f}")
print(f"Mean Final Energy (Uncontrolled): {jnp.mean(final_energy_unc):.6f}")

# --- 7. Plotting ---
plt.figure(figsize=(15, 12))

# 1. Energy Distribution (Boxplot)
plt.subplot(2, 2, 1)
plt.boxplot([final_energy_ctrl, final_energy_unc], labels=['Controlled', 'Uncontrolled'])
plt.yscale('log')
plt.title(f'Final System Energy (Log Scale, N={N_eval})')
plt.ylabel('Mean Squared Field Value')
plt.grid(True, alpha=0.3)

# 2. Energy Evolution (Mean over batch)
plt.subplot(2, 2, 2)
time_axis = jnp.arange(T_steps) * 0.05
plt.plot(time_axis, jnp.mean(energy_ctrl, axis=0), 'b-', label='Controlled (Mean)', linewidth=2)
plt.plot(time_axis, jnp.mean(energy_unc, axis=0), 'r-', label='Uncontrolled (Mean)', linewidth=2)
plt.fill_between(time_axis, 
                 jnp.percentile(energy_ctrl, 25, axis=0), 
                 jnp.percentile(energy_ctrl, 75, axis=0), color='blue', alpha=0.1)
plt.fill_between(time_axis, 
                 jnp.percentile(energy_unc, 25, axis=0), 
                 jnp.percentile(energy_unc, 75, axis=0), color='red', alpha=0.1)
plt.title('System Energy Evolution')
plt.xlabel('Time (s)')
plt.ylabel('Energy (L2)')
plt.legend()
plt.grid(True)

# 3. Space-Time Heatmap (Controlled - Sample 0)
sample_idx = 0
plt.subplot(2, 2, 3)
plt.imshow(u_ctrl_all[sample_idx].T, aspect='auto', origin='lower', 
           extent=[0, T_steps*0.05, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
plt.colorbar(label='u(x,t)')
plt.title('Controlled Dynamics (Space-Time)')
plt.xlabel('Time (s)')
plt.ylabel('Domain (x)')

# 4. Space-Time Heatmap (Uncontrolled - Sample 0)
plt.subplot(2, 2, 4)
plt.imshow(u_unc_all[sample_idx].T, aspect='auto', origin='lower', 
           extent=[0, T_steps*0.05, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
plt.colorbar(label='u(x,t)')
plt.title('Uncontrolled Chaos (Space-Time)')
plt.xlabel('Time (s)')
plt.ylabel('Domain (x)')

plt.tight_layout()
save_path = Path("figures/images/bench") / "ks_comparison_results.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path)
print(f"Comparison plot saved to '{save_path}'")